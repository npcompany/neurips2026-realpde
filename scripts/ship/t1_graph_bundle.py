#!/usr/bin/env python
"""Serve a clean Track-1 bundle's whole predict() as ONE CUDA graph per call size.

    uv run python scripts/ship/t1_graph_bundle.py <clean_bundle_dir> <out_dir> [--fp32-out]

``--fp32-out`` returns float32 arrays: the fp16 results are upcast inside the graph (lossless, so
the scores are unchanged), because a numpy fp16->fp32 conversion on the host costs ~1 ms/sample.

The board calls predict() on 5 windows per fresh process (ingestion log 2026-09-24), after an
untimed same-shape warm-up call. At that size our call is launch-bound: ~12 ms on the A10G for
~4 ms of kernels, and only 43% of it is the backbone (why graphing the backbone alone did not
move the board, 940734). This wraps the bundle's FastT1 path so that the device prefix
(operating-point estimate, freestream scale, normalisation), the forward, the interval head,
the width rule, the second-moment gain and pressure zeroing replay as one graph, and only the
input copy in and the three fp16 outputs out stay on the host, as plain pageable copies. Pinned
staging was tried (FINDINGS 236) and taken out again (237): in the board's own shape -- a fresh
process, one untimed call, one timed call -- the timed call's pinned buffers are fresh
allocations and cost 2.3 ms per call of 5, against 0.08 ms saved in a long-lived process.

Generated, not hand-written: ``graph_serve.run_core`` is FastT1.run from the bundle's own
``fast_serve.py`` with three mechanical edits (the host-built second-moment static planes become
an argument computed on the device, the fp16 finiteness check becomes a returned flag instead of a
host sync, and autocast's weight cache is off for capture). The graph is captured on the FIRST
call at a size -- the harness's untimed one -- and used only if its replay reproduces the
reference predict's output on that same input element for element; any failure, a mismatch or a
non-finite flag on a later call serves the reference predict. The device prefix of the graph
(``rpde_t1/models/graph_prefix.py``, copied from the source tree) is the operating-point
estimate, the freestream scale, the normalisation and the second-moment head's constant planes
with their scalar tails fused into three kernels; its values equal the bundle's ``op_estimate``
element for element (``tests/test_graph_prefix.py``), and the first-call comparison holds it to
that on every call size.
"""

from __future__ import annotations

import ast
import shutil
import sys
import textwrap
from pathlib import Path

PREFIX_SRC = Path(__file__).resolve().parents[2] / "src/rpde_t1/models/graph_prefix.py"

GRAPH_CLASS = '''

class GraphT1:
    """predict() for one call size as a single replayed CUDA graph, falling back to `ref`."""

    OUT_FP32 = False

    def __init__(self, fast, ref, h: int = 32, w: int = 64) -> None:
        import numpy as np

        from ..geometry import sdf_table

        angles, stack = sdf_table(h, w)
        self.fast, self.ref = fast, ref
        self.angles = torch.as_tensor(np.asarray(angles, dtype=np.float64), device="cuda")
        self.stack = torch.as_tensor(np.asarray(stack, dtype=np.float32), device="cuda")
        self.graphs: dict[int, Any] = {}

    def _core(self, x_dev: torch.Tensor):
        from .graph_prefix import device_prefix

        cond_dev, ub, xb, sp = device_prefix(x_dev, self.angles, self.stack)
        # float32 outputs come out of the last kernels already rounded to fp16, in fp32
        outs, ok = run_core(
            self.fast, xb, cond_dev, ub, sp, torch.float16, torch.float16, round16=self.OUT_FP32
        )
        if self.OUT_FP32:
            outs = outs.float()
        return outs, ok

    def _ref(self, x, metadata):
        import numpy as np

        out = self.ref(x, metadata)
        if not self.OUT_FP32:
            return out
        if isinstance(out, dict):
            return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}
        return np.ascontiguousarray(out, dtype=np.float32)

    def _build(self, x, ref_out):
        import numpy as np

        x_static = torch.empty(x.shape, dtype=torch.float32, device="cuda")
        x_static.copy_(torch.from_numpy(x))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(2):
                self._core(x_static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            outs, ok = self._core(x_static)
        with torch.inference_mode():
            graph.replay()
        got = self._to_host(outs)
        same = bool(ok.item()) and all(
            np.array_equal(got[i], np.asarray(ref_out[k]))
            for i, k in enumerate(("prediction", "lower", "upper"))
        )
        return (graph, x_static, outs, ok) if same else None

    @staticmethod
    def _to_host(outs):
        """The outputs to a fresh pageable numpy array (a synchronous copy).

        Not a pinned buffer: a fresh pinned buffer per call is a fresh host allocation in a
        process that makes only one or two calls, and that allocation costs more than the
        faster transfer saves."""
        return outs.cpu().numpy()

    def __call__(self, input_array, metadata=None):
        import numpy as np

        x = np.ascontiguousarray(np.asarray(input_array, dtype=np.float32))
        n = x.shape[0]
        if n not in self.graphs:
            ref_out = self._ref(x, metadata)
            try:
                self.graphs[n] = self._build(x, ref_out)
            except Exception:  # noqa: BLE001 - the reference predict is always correct
                self.graphs[n] = None
            return ref_out
        g = self.graphs[n]
        if g is None:
            return self._ref(x, metadata)
        graph, x_static, outs, ok = g
        try:
            x_static.copy_(torch.from_numpy(x))
            with torch.inference_mode():
                graph.replay()
            got = self._to_host(outs)
            if not bool(ok.item()):
                return self._ref(x, metadata)
        except Exception:  # noqa: BLE001
            self.graphs[n] = None
            return self._ref(x, metadata)
        return {"prediction": got[0], "lower": got[1], "upper": got[2]}
'''


def run_core_source(fast_serve_src: str) -> str:
    tree = ast.parse(fast_serve_src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastT1")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    lines = fast_serve_src.splitlines()
    start = min([d.lineno for d in fn.decorator_list] + [fn.lineno]) - 1
    src = textwrap.dedent("\n".join(lines[start : fn.end_lineno]))
    edits = [
        (
            "def run(self, xb, cb, ub, cond_np, autocast_dtype, out_dtype, round16=False):",
            "def run_core(self, xb, cb, ub, sp, autocast_dtype, out_dtype, round16=False):",
        ),
        ("    from .tke_head import tke_static_planes\n", ""),
        (
            'torch.autocast("cuda", dtype=autocast_dtype)',
            'torch.autocast("cuda", dtype=autocast_dtype, cache_enabled=False)',
        ),
        ("    sp = tke_static_planes(cond_np, hh_, ww_, dev)\n", ""),
        (
            "    if amp and (\n"
            "        not bool(torch.stack((torch.isfinite(pred).all(), torch.isfinite(wdt).all())).all())\n"  # noqa: E501
            "    ):\n"
            "        return None\n",
            "    ok = torch.stack((torch.isfinite(pred).all(), torch.isfinite(wdt).all())).all()\n",
        ),
        ("    return outs[0], outs[1], outs[2]", "    return outs, ok"),
    ]
    for old, new in edits:
        if src.count(old) != 1:
            raise SystemExit(f"FastT1.run no longer matches the expected text: {old!r}")
        src = src.replace(old, new)
    return src


def _literal(node: ast.expr) -> object:
    """A constant, or the constant argument of a one-argument call such as `_blend_of(0.0)`."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        return _literal(node.args[0])
    raise SystemExit(f"not a literal: {ast.dump(node)}")


def _fast_kw_value(kw: ast.keyword) -> object:
    """A FastT1 keyword's value as the GRAPH path serves it.

    `half_weights` is the one keyword predictor.py spells as an expression,
    `autocast_dtype is torch.float16`; the graph runs the core at fp16 (`run_core`'s dtype
    arguments), so it is True there. Every other keyword is a literal. This is the check the
    2026-09-25 night chain lacked: `FastT1` gained the required `half_weights` keyword (the
    fp16 fallback fix of 6a17c1c) and the template did not pass it, so `_graph_for` raised
    inside its try, the bundle silently served the eager reference path, and the daemon check
    caught it as "graph state none" -- after the whole chain had run.
    """
    if kw.arg == "half_weights":
        v = kw.value
        if (
            isinstance(v, ast.Compare)
            and isinstance(v.left, ast.Name)
            and v.left.id == "autocast_dtype"
            and len(v.ops) == 1
            and isinstance(v.ops[0], ast.Is)
            and ast.unparse(v.comparators[0]) == "torch.float16"
        ):
            return True
        if isinstance(v, ast.Constant) and isinstance(v.value, bool):
            return v.value
        raise SystemExit(f"FastT1(half_weights=...) is not the fp16 expression: {ast.unparse(v)}")
    return _literal(kw.value)


def fast_kwargs(predictor_src: str) -> dict[str, object]:
    """The `FastT1(...)` keyword values the clean bundle's `make_predict` serves with.

    The clean conversion folds the packaged constants (`--tke-shrink`, blend, centre shrink)
    into `predictor.py`, so the graph path must read them from there rather than carry its own
    copy: a hard-coded value that differs from the bundle's makes the first-call identity check
    fail and the bundle silently serve the slow reference path. Every `shrink=` in the file
    must agree, or this refuses.
    """
    tree = ast.parse(predictor_src)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == "FastT1")
            or (isinstance(n.func, ast.Name) and n.func.id == "FastT1")
        )
    ]
    if len(calls) != 1:
        raise SystemExit(f"expected one FastT1(...) call in predictor.py, found {len(calls)}")
    wanted = ("blend", "shrink", "uses_tau", "centre_shrink", "half_weights")
    kw = {k.arg: _fast_kw_value(k) for k in calls[0].keywords if k.arg in wanted}
    for name in wanted:
        if name not in kw:
            raise SystemExit(f"FastT1(...) in predictor.py has no {name}=")
    shrinks = {
        _literal(k.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg == "shrink"
    }
    if shrinks != {kw["shrink"]}:
        raise SystemExit(f"predictor.py serves more than one shrink value: {sorted(shrinks)}")
    return kw


def undefined_free_names(mod_src: str, fn_name: str = "run_core") -> list[str]:
    """Names `fn_name` reads that the module neither imports nor defines.

    The extracted `run_core` is `FastT1.run` moved into a new module, so every module-level
    name it uses must be imported there too. On 2026-09-25 night `run` gained `F.linear` /
    `F.conv3d` (FINDINGS 236) and this module did not import `F`: `_core` raised NameError
    inside `_build`'s try, every call size fell back to the eager path, and only the daemon
    check at the end of the chain said so. Checked here so the build refuses instead.
    """
    import builtins  # noqa: PLC0415

    tree = ast.parse(mod_src)
    defined = set(dir(builtins))
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Assign):
            defined |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            local.add(n.id)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            local |= {(a.asname or a.name).split(".")[0] for a in n.names}
    free = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(free - local - defined)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fp32_out = "--fp32-out" in sys.argv[1:]
    src_dir, out_dir = Path(args[0]), Path(args[1])
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists")
    kw = fast_kwargs((src_dir / "rpde_t1/models/predictor.py").read_text())
    print(f"FastT1 settings read from the bundle: {kw}")
    shutil.copytree(src_dir, out_dir, ignore=shutil.ignore_patterns("__pycache__"))
    fs = (out_dir / "rpde_t1/models/fast_serve.py").read_text()
    body = run_core_source(fs)
    mod = (
        '"""The Track-1 predict() replayed as one CUDA graph per call size."""\n\n'
        "from __future__ import annotations\n\nfrom typing import Any\n\nimport torch\n"
        "import torch.nn.functional as F  # noqa: N812\n\n"
        "from .fast_serve import _grad_energy, _kernel, _nullctx  # noqa: F401\n\n\n"
        + body
        + "\n"
        + GRAPH_CLASS
    )
    missing = undefined_free_names(mod)
    if missing:
        raise SystemExit(f"run_core reads names graph_serve.py does not define: {missing}")
    if fp32_out:
        mod = mod.replace("    OUT_FP32 = False\n", "    OUT_FP32 = True\n", 1)
    (out_dir / "rpde_t1/models/graph_serve.py").write_text(mod)
    # the device prefix (operating point, scale, normalisation, constant planes) in fewer
    # kernels: a serving twin of the bundle's op_estimate, copied from the source tree
    shutil.copy(PREFIX_SRC, out_dir / "rpde_t1/models/graph_prefix.py")
    sub = out_dir / "submission.py"
    s = sub.read_text()
    edits = [
        (
            '_state = {"fn": None, "calls": 0}',
            '_state = {"fn": None, "calls": 0, "graph": None}',
        ),
        (
            '        _state["fn"] = make_predict(\n'
            "            model,\n"
            "            width_rule=rule,\n"
            "            tke_head=head,\n"
            "        )\n",
            '        _state["fn"] = make_predict(\n'
            "            model,\n"
            "            width_rule=rule,\n"
            "            tke_head=head,\n"
            "        )\n"
            '        _state["graph"] = _graph_for(model, rule, head, _state["fn"])\n',
        ),
        (
            "    fn = _get_predict()\n    out = fn(x, metadata)\n",
            '    _get_predict()\n    fn = _state["graph"] or _state["fn"]\n    out = fn(x, metadata)\n',  # noqa: E501
        ),
        (
            "warm_geometry_tables()\n",
            "def _graph_for(model, rule, head, ref):\n"
            '    """The single-graph predict over the same FastT1 configuration as `ref`, or None."""\n'  # noqa: E501
            "    try:\n"
            "        from rpde_t1.models import fast_serve\n"
            "        from rpde_t1.models.graph_serve import GraphT1\n\n"
            "        if not rule.is_uniform:\n"
            "            rule = rule.to_uniform()\n"
            "        if not fast_serve.t1_supported(model, tke_head=head, width_rule=rule):\n"
            "            return None\n"
            "        fast = fast_serve.FastT1(\n"
            f"            model, width_rule=rule, tke_head=head, blend={kw['blend']!r}, "
            f"shrink={kw['shrink']!r},\n"
            f"            uses_tau={kw['uses_tau']!r}, centre_shrink={kw['centre_shrink']!r},\n"
            f"            half_weights={kw['half_weights']!r},\n"
            "        )\n"
            "        return GraphT1(fast, ref)\n"
            "    except Exception:  # noqa: BLE001 - the reference predict is always correct\n"
            "        return None\n\n\n"
            "warm_geometry_tables()\n",
        ),
    ]
    for old, new in edits:
        if s.count(old) != 1:
            raise SystemExit(f"submission.py no longer matches the expected text: {old!r}")
        s = s.replace(old, new)
    sub.write_text(s)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
