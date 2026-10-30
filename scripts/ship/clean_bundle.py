#!/usr/bin/env python
"""Turn a staged Track-1 submission bundle into a minimal, readable one with identical outputs.

    uv run python scripts/ship/clean_bundle.py <staged_bundle_dir> --name <out_name> \
        [--out-root submissions] [--work-dir DIR] [--reuse-trace] [--skip-verify]

The staged bundle is what ``python -m rpde_t1.submit --run ...`` leaves under
``submissions/<name>/``: a generated ``submission.py``, the weights, the head files, the
starting-kit payload and the whole inference subset of ``rpde_t1``. Most of it is never
executed by the shipped configuration. This tool

1. **traces** the staged bundle in an isolated interpreter (``python -I``, bundle root first on
   ``sys.path``, the eval-image-missing packages blocked) over real windows at several call
   sizes, recording every imported file, every executed line, every opened data file and the
   argument values of every call;
2. **prunes** the package to the modules that were imported, folds configuration-dependent
   branches of the construction-time functions (``make_predict``, the factory, every
   ``__init__``) against the recorded argument values, removes functions, methods, classes and
   constants that neither executed nor are referenced, strips every comment, rewrites the
   docstrings to short neutral summaries and reformats;
3. **generates** a short ``submission.py`` from the staged one's constants (parsed generically
   with ``ast``) with the same construction, the same import-time warm-up and the same
   ``predict``;
4. **verifies**: clean-room run, bit-identity against the staged bundle on the same inputs and
   call sizes, a determinism control, the daemonic-import check, the extracted-size cap and a
   forbidden-word scan over every text member.

GPU work goes through ``scripts/gpu.sh`` in two invocations (trace, verify). ``--reuse-trace``
skips the first when the work directory already holds a trace of the same source bundle.

Folding trusts the traced argument values only for functions that ran exclusively during
construction (import), and every fold is protected by the bit-identity check. Where a branch
cannot be decided statically it is kept with its prose removed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rpde_t1.submit.package import (  # noqa: E402
    _BLOCKER_SRC,
    MAX_EXTRACTED_BYTES,
    MISSING_IN_EVAL_IMAGE,
    Bundle,
    verify_bundle,
    zip_bundle,
)
from rpde_t2.submit.package import (  # noqa: E402
    MISSING_IN_EVAL_IMAGE_T2,
    verify_ttt_bundle,
)

GPU_SH = REPO / "scripts" / "gpu.sh"
T2_DAEMON_VERIFY = REPO / "scripts" / "t2" / "t2_daemon_verify.py"
T2_IMAGE_EVAL = REPO / "scripts" / "t2" / "t2_image_local_eval.sh"
DAEMON_VERIFY = REPO / "scripts" / "eval" / "t1_daemon_verify.py"

# --------------------------------------------------------------------------------------------
# Policy constants
# --------------------------------------------------------------------------------------------

#: The final scan over every text member must return nothing for this.
FORBIDDEN_RE = re.compile(
    r"FINDINGS|§|STATUS\.md|ledger|board|Bruno|Benjamin|Codabench|2026-|W&B|wandb|tmux|FIXME|TODO"
)
#: Words that disqualify a derived docstring sentence (case-insensitive, whole words).
NON_NEUTRAL_RE = re.compile(
    r"\b(we|our|ours|us|measured|measure|measures|findings|protocol|see|retired|day|notebook|"
    r"campaign|submission[s]? (was|were)|ship(ped|s)?|leaderboard|glossary|deliberately|"
    r"reason|because|cost[s]?|worth|bought|buys|pays|against|track[- ]?2|track[- ]?1|"
    r"organizers|organisers|vendored|kit|scoring\.py|ltta|stream)\b",
    re.IGNORECASE,
)
#: Upper-case emphasis a derived sentence must not carry (acronyms such as GPU or TKE are fine).
EMPHASIS_WORDS = {
    "THIS",
    "NOT",
    "ONLY",
    "ALL",
    "EVERY",
    "BEFORE",
    "AFTER",
    "CONSTANT",
    "BACKBONE",
    "TRAINING",
    "FITTED",
    "REPORTED",
    "MEASURED",
    "CHECKED",
    "EAGER",
    "FIRST",
    "LAST",
    "NEVER",
    "ALWAYS",
    "MUST",
    "SAME",
    "ONE",
    "TWO",
    "AND",
    "OR",
    "IS",
    "ARE",
    "WAS",
    "WERE",
    "BOTH",
    "WITH",
    "WITHOUT",
    "INSIDE",
    "OUTSIDE",
    "PER",
    "ANY",
    "NO",
    "YES",
    "OFF",
    "ON",
    "IN",
    "AT",
    "TO",
    "OF",
    "FOR",
    "BY",
    "AS",
    "HERE",
    "THERE",
    "THEN",
    "THAN",
    "WHAT",
    "WHICH",
    "WHERE",
    "EXACTLY",
    "DOES",
    "DO",
    "DID",
    "DONE",
    "BIT",
    "REJECTED",
    "RETIRED",
}
UNIT_RE = re.compile(r"\d+(\.\d+)?\s*(ms|MB|GB|kB|s|%|x|TFLOPS)\b|\bx\d|->|\*\*|~\d")

#: Attribute chains that name distinct singletons; ``is`` between two of them is decidable.
DTYPE_SYMS = {
    "torch.float16",
    "torch.bfloat16",
    "torch.float32",
    "torch.float64",
    "torch.int32",
    "torch.int64",
    "torch.bool",
    "np.float16",
    "np.float32",
    "np.float64",
}

#: Construction-time functions whose traced argument values may be folded into their bodies,
#: in addition to every ``__init__``. A function called during ``predict`` is never folded.
FOLD_QUALNAMES = {
    "rpde_t1/models/predictor.py": {"make_predict", "make_gpu_predict"},
    "rpde_t1/models/factory.py": {
        "load_for_inference",
        "build_for_inference",
        "build_backbone",
        "wrap_model",
    },
    "rpde_t1/models/fast_serve.py": {"t1_supported"},
}

#: Calls that evaluate to a constant under a documented assumption. The kit's template states
#: that ``metadata`` is an empty dictionary on scored calls, so the labelled-metadata route of
#: the predictor can never be taken on the platform.
ASSUMED_CALLS = {("rpde_t1/models/predictor.py", "_metadata_has_labels"): False}

#: Statements removed by pattern: (module, enclosing function qualname, regex over the
#: statement's first unparsed line). Used for hooks that only a compiled build reads.
PRUNE_RULES = [
    (
        "rpde_t1/models/predictor.py",
        "make_predict",
        r"^predict\.(demote_to_eager|uses_compiled|restore_compiled) = ",
    ),
]

#: Methods never removed even when unexecuted: called implicitly by Python or by torch.
IMPLICIT_METHODS = {"forward", "extra_repr", "reset_parameters"}

#: Pure call targets a module-level constant may use and still be dropped when unreferenced.
PURE_CALLS = {
    "np.array",
    "np.asarray",
    "np.geomspace",
    "np.linspace",
    "np.arange",
    "np.float32",
    "np.float64",
    "np.int64",
    "re.compile",
    "frozenset",
    "set",
    "dict",
    "tuple",
    "list",
    "range",
    "float",
    "int",
    "str",
    "bool",
}

#: Neutral docstrings for the public names that survive. Keys are ``<module>:<qualname>``;
#: ``<module>:`` is the module docstring. Anything not listed gets the first sentence of its
#: original docstring when that sentence is neutral, and no docstring otherwise.
DOCSTRINGS = {
    "rpde_t1/__init__.py:": (
        "Track-1 constants: tensor layout, channel names and the interval scale."
    ),
    "rpde_t1/models/__init__.py:": "Model code for the submission.",
    "rpde_t1/features.py:": "Operating-point features estimated from an input window.",
    "rpde_t1/geometry.py:": (
        "Airfoil geometry tables used by the second-moment head's static planes."
    ),
    "rpde_t1/models/factory.py:": "Rebuild the model described by a checkpoint's configuration.",
    "rpde_t1/models/predictor.py:": (
        "The predict() closure: normalisation, forward, intervals and post-processing."
    ),
    "rpde_t1/models/fast_serve.py:": (
        "Fused serving path that reproduces the reference chunk in fewer kernels."
    ),
    "rpde_t1/models/stformer.py:": "Spatio-temporal transformer backbone.",
    "rpde_t1/models/transformer.py:": "Attention and feed-forward building blocks.",
    "rpde_t1/models/bounds.py:": "Interval head and the bounded-model wrapper.",
    "rpde_t1/models/wrappers.py:": "Residual wrapper and model-call helpers.",
    "rpde_t1/models/tke_head.py:": "Second-moment correction head.",
    "rpde_t1/models/width_rule.py:": (
        "Fitted width rule that sizes the interval from the head's error scale."
    ),
    "rpde_t1/models/op_estimate.py:": "Device-side operating-point estimation.",
    "rpde_t1/models/fluctuations.py:": "Fluctuation gain applied to a predicted window.",
    "rpde_t1/models/interval.py:": "Interval constants.",
    "rpde_t1/models/flow.py:": "Kinetic-energy map of a window, on the device.",
    "submission.py:": (
        "RealPDE Track 2 submission: a frozen forecaster with an interval head and a "
        "second-moment correction head, served through the streaming contract."
    ),
    "rpde_t2/__init__.py:": "Track-2 constants and the evaluator-space normalisation helpers.",
    "rpde_t2/ttt.py:": "The streaming contract: get_ttt_model, reset_ttt_state and ttt_step.",
    "rpde_t2/ttt.py:FrozenPort": "A frozen forecaster served through the streaming contract.",
    "rpde_t2/ttt.py:FrozenPort.ttt_step": (
        "One streaming step: normalised input in, prediction and bounds out."
    ),
    "rpde_t2/ttt.py:FrozenPort.reset_ttt_state": "Forget the trajectory state at a boundary.",
    "rpde_t1/models/predictor.py:make_gpu_predict": (
        "Build the batch-1 device-resident predict closure used by the streaming contract."
    ),
    "rpde_t1/models/gpu_tail.py:": "Device-side tail of the streaming pipeline.",
    "rpde_t1/models/causal_history_v2.py:": "Causal trajectory statistics carried between steps.",
    "rpde_t1/models/aux.py:": "Auxiliary-head helpers.",
    "rpde_t1/models/predictor.py:make_predict": (
        "Build a predict(input_array, metadata=None) closure around a model."
    ),
    "rpde_t1/models/predictor.py:warm_up_predict": (
        "Run synthetic predict calls so the timed calls meet a warm device."
    ),
    "rpde_t1/models/predictor.py:_spin_gpu": (
        "Keep the GPU busy with dense GEMMs for the given number of seconds."
    ),
    "rpde_t1/models/factory.py:load_for_inference": (
        "Load a training checkpoint and return (model, config)."
    ),
    "rpde_t1/models/factory.py:build_for_inference": (
        "Reconstruct the architecture described by a saved training config."
    ),
    "rpde_t1/models/factory.py:build_backbone": (
        "The bare architecture named by the config, before wrapping."
    ),
    "rpde_t1/models/factory.py:wrap_model": (
        "Apply the residual wrapper and the bound head around a backbone."
    ),
    "rpde_t1/models/fast_serve.py:FastT1": (
        "One Track-1 chunk (forward, width rule, guard, second-moment gain, zero pressure) in "
        "fewer kernels."
    ),
    "rpde_t1/models/fast_serve.py:FastT1.run": (
        "Compute the three output blocks for one chunk, or None when the fp16 forward is "
        "non-finite."
    ),
    "rpde_t1/models/fast_serve.py:t1_supported": (
        "Whether FastT1 reproduces the reference chunk for this model and head configuration."
    ),
    "rpde_t1/models/fast_serve.py:FastSTFormer": (
        "The STFormer forward with fused elementwise kernels."
    ),
    "rpde_t1/models/width_rule.py:WidthRule": (
        "Per-bin multipliers on the head's half-widths, indexed by predicted log-error, channel "
        "and lead."
    ),
    "rpde_t1/models/tke_head.py:TKEHead": (
        "Convolutional head predicting a per-pixel log energy-ratio correction."
    ),
    "rpde_t1/models/bounds.py:BoundHead": (
        "Convolutional head predicting the interval scale from prediction features."
    ),
    "rpde_t1/models/bounds.py:BoundedModel": "A forecaster paired with an interval head.",
    "rpde_t1/models/stformer.py:STFormer": "Spatio-temporal transformer forecaster.",
}

# --------------------------------------------------------------------------------------------
# 1. Constants of the staged submission.py
# --------------------------------------------------------------------------------------------

_CONST_NAME_RE = re.compile(r"_?[A-Z][A-Z0-9_]*")


def _parse_channels(spec: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in str(spec).split(",") if s.strip())


def parse_constants(text: str, strict: bool = True) -> dict[str, Any]:
    """Evaluate every module-level ``UPPER_NAME = <expr>`` of a generated submission.py.

    Expressions are evaluated in a namespace of the constants seen so far plus ``range`` and
    ``tuple`` and the two channel parsers; ``_CKPT_PATH = os.path.join(_HERE, "<file>")`` is
    recorded as ``_CKPT_NAME``. Anything that cannot be evaluated is an error, so a new kind of
    constant fails loudly instead of being dropped.
    """
    tree = ast.parse(text)
    ns: dict[str, Any] = {
        "range": range,
        "tuple": tuple,
        "parse_geom_channels": _parse_channels,
        "parse_flow_channels": _parse_channels,
    }
    out: dict[str, Any] = {}
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        name = node.targets[0].id
        if not _CONST_NAME_RE.fullmatch(name) or name == "_HERE":
            continue
        if name == "_CKPT_PATH":
            call = node.value
            if (
                isinstance(call, ast.Call)
                and len(call.args) == 2
                and isinstance(call.args[1], ast.Constant)
            ):
                out["_CKPT_NAME"] = call.args[1].value
                continue
            raise ValueError("unrecognised _CKPT_PATH form: " + ast.unparse(node))
        try:
            value = eval(  # noqa: S307 - our own generated file, restricted namespace
                compile(ast.Expression(node.value), "<const>", "eval"),
                {"__builtins__": {}},
                {**ns, **out},
            )
        except Exception as exc:  # noqa: BLE001
            if not strict:
                continue
            raise ValueError(f"cannot evaluate constant {name}: {ast.unparse(node)}") from exc
        out[name] = value
    return out


# --------------------------------------------------------------------------------------------
# 2. A small partial evaluator over ast expressions
# --------------------------------------------------------------------------------------------


class Known:
    """A value known exactly (a Python literal, possibly a container of literals).

    ``subst`` False means the value may decide branches but must not replace the expression
    that produced it (a parameter every call site passes explicitly stays a parameter).
    """

    __slots__ = ("value", "subst", "node")

    def __init__(self, value: Any, subst: bool = True, node: ast.AST | None = None) -> None:
        self.value = value
        self.subst = subst
        self.node = node  # an expression to substitute instead of the literal (a named default)

    def __repr__(self) -> str:
        return f"Known({self.value!r})"


class Sym:
    """A value known to be a specific singleton named by an attribute chain."""

    __slots__ = ("chain",)

    def __init__(self, chain: str) -> None:
        self.chain = chain

    def __repr__(self) -> str:
        return f"Sym({self.chain})"


class _Opaque:
    """A value known only to be something other than None."""

    def __repr__(self) -> str:
        return "OPAQUE"


OPAQUE = _Opaque()
UNKNOWN = None

_SIMPLE = (type(None), bool, int, float, str)
_PURE_STR_METHODS = {
    "startswith",
    "endswith",
    "lower",
    "upper",
    "strip",
    "lstrip",
    "rstrip",
    "split",
    "replace",
    "partition",
    "isdigit",
    "count",
    "join",
}
_PURE_DICT_METHODS = {"get", "keys", "values", "items"}
_READ_METHODS = _PURE_DICT_METHODS | _PURE_STR_METHODS | {"copy", "index"}
_PURE_BUILTINS: dict[str, Any] = {
    "bool": bool,
    "str": str,
    "int": int,
    "float": float,
    "len": len,
    "tuple": tuple,
    "list": list,
    "dict": dict,
    "set": set,
    "max": max,
    "min": min,
    "abs": abs,
    "sorted": sorted,
    "round": round,
    "any": any,
    "all": all,
    "sum": sum,
}
#: Calls whose result is never None (tensor/array/module constructors and tensor methods).
_OPAQUE_CALL_PREFIXES = (
    "nn.",
    "torch.nn.",
    "torch.from_numpy",
    "torch.as_tensor",
    "torch.tensor",
    "torch.zeros",
    "torch.empty",
    "torch.ones",
    "torch.arange",
    "torch.linspace",
    "np.asarray",
    "np.ascontiguousarray",
    "np.zeros",
    "np.empty",
    "np.array",
    "np.ones",
    "torch.full",
)
_OPAQUE_METHODS = {
    "to",
    "float",
    "half",
    "double",
    "cuda",
    "cpu",
    "view",
    "reshape",
    "abs",
    "mean",
    "sum",
    "clamp_min",
    "clamp",
    "contiguous",
    "eval",
    "permute",
    "transpose",
    "numpy",
    "long",
    "exp",
    "log",
    "sqrt",
    "std",
    "unsqueeze",
    "squeeze",
    "expand",
    "flatten",
    "detach",
    "clone",
}
_MAX_SUBST_STR = 40
_PURE_EXTRA_NAMES = {
    "range",
    "enumerate",
    "zip",
    "isinstance",
    "ValueError",
    "TypeError",
    "KeyError",
    "Exception",
    "reversed",
    "map",
    "filter",
}


def attr_chain(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def truthy(v: Any) -> bool | None:
    if isinstance(v, Known):
        try:
            return bool(v.value)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(v, (Sym, _Opaque)):
        return True if isinstance(v, Sym) else None
    return None


def is_none(v: Any) -> bool | None:
    if isinstance(v, Known):
        return v.value is None
    if isinstance(v, (Sym, _Opaque)):
        return False
    return None


def is_pure(node: ast.AST) -> bool:
    """Whether evaluating ``node`` can have no side effect (attribute loads included)."""
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
            return False
    return True


def substitutable(v: Any) -> bool:
    """Whether a known value may replace the expression that produced it, in place."""
    if not isinstance(v, Known) or not v.subst:
        return False
    x = v.value
    if isinstance(x, str):
        return len(x) <= _MAX_SUBST_STR and "\n" not in x and not path_like(x)
    return isinstance(x, _SIMPLE)


def path_like(s: Any) -> bool:
    return isinstance(s, str) and ("/" in s or "\\" in s)


def literal_node(value: Any) -> ast.AST | None:
    """An expression node for a literal value, or None when it has no literal form."""
    if isinstance(value, _SIMPLE):
        return ast.Constant(value=value)
    if isinstance(value, tuple):
        elts = [literal_node(v) for v in value]
        return None if any(e is None for e in elts) else ast.Tuple(elts=elts, ctx=ast.Load())
    if isinstance(value, list):
        elts = [literal_node(v) for v in value]
        return None if any(e is None for e in elts) else ast.List(elts=elts, ctx=ast.Load())
    if isinstance(value, dict):
        ks = [literal_node(k) for k in value]
        vs = [literal_node(v) for v in value.values()]
        if any(k is None for k in ks) or any(v is None for v in vs):
            return None
        return ast.Dict(keys=ks, values=vs)
    return None


class Context:
    """What an evaluator needs to see beyond the local environment."""

    def __init__(self, module: str, cleaner: Cleaner | None = None):
        self.module = module
        self.cleaner = cleaner

    def module_env(self, rel: str) -> dict[str, Any]:
        if self.cleaner is None:
            return {}
        return self.cleaner.module_envs.get(rel, {})

    def imported_value(self, rel: str, name: str) -> Any:
        """The known value of a symbol imported into ``rel`` from another kept module."""
        if self.cleaner is None:
            return UNKNOWN
        target = self.cleaner.imports.get(rel, {}).get(name)
        if target is None or target[0] is None or target[1] is None:
            return UNKNOWN
        return self.cleaner.module_envs.get(target[0], {}).get(target[1], UNKNOWN)

    def alias_target(self, rel: str, name: str) -> str | None:
        if self.cleaner is None:
            return None
        return self.cleaner.imports.get(rel, {}).get(name, (None, None))[0]

    def pure_call(self, rel: str, name: str, args: list, kwargs: dict) -> Any:
        if self.cleaner is None or self.cleaner.pure_funcs is None:
            return UNKNOWN
        target = self.cleaner.resolve(rel, name) if rel in self.cleaner.modules else None
        if target is None:
            return UNKNOWN
        return self.cleaner.pure_funcs.evaluate(target[0], target[1], args, kwargs)


class Evaluator:
    def __init__(
        self,
        env: dict[str, Any],
        ctx: Context,
        self_name: str | None = None,
        self_env: dict[str, Any] | None = None,
        fn_bound: set[str] | None = None,
    ):
        self.env = env
        self.ctx = ctx
        self.module = ctx.module
        self.self_name = self_name
        self.self_env = self_env or {}
        self.fn_bound = fn_bound or set()

    def copy(self, env: dict[str, Any]) -> Evaluator:
        return Evaluator(env, self.ctx, self.self_name, self.self_env, self.fn_bound)

    def ev(self, node: ast.AST) -> Any:
        m = getattr(self, "ev_" + type(node).__name__, None)
        if m is None:
            return UNKNOWN
        try:
            return m(node)
        except Exception:  # noqa: BLE001 - anything odd is simply unknown
            return UNKNOWN

    def ev_Constant(self, node):
        return Known(node.value)

    def ev_Name(self, node):
        if node.id in self.env:
            return self.env[node.id]
        if node.id in self.fn_bound:
            return UNKNOWN
        return self.ctx.imported_value(self.module, node.id)

    def ev_Tuple(self, node):
        vals = [self.ev(e) for e in node.elts]
        if all(isinstance(v, Known) for v in vals):
            return Known(tuple(v.value for v in vals))
        return UNKNOWN

    def ev_List(self, node):
        vals = [self.ev(e) for e in node.elts]
        if all(isinstance(v, Known) for v in vals):
            return Known([v.value for v in vals])
        return UNKNOWN

    def ev_Dict(self, node):
        out = {}
        for k, v in zip(node.keys, node.values, strict=True):
            if k is None:
                return UNKNOWN
            kv, vv = self.ev(k), self.ev(v)
            if not (isinstance(kv, Known) and isinstance(vv, Known)):
                return UNKNOWN
            out[kv.value] = vv.value
        return Known(out)

    def ev_Attribute(self, node):
        chain = attr_chain(node)
        if chain in DTYPE_SYMS:
            return Sym(chain)
        if isinstance(node.value, ast.Name):
            base = node.value.id
            if self.self_name and base == self.self_name and node.attr in self.self_env:
                return self.self_env[node.attr]
            if base not in self.env and base not in self.fn_bound:
                target = self.ctx.alias_target(self.module, base)
                if target is not None:
                    return self.ctx.module_env(target).get(node.attr, UNKNOWN)
        return UNKNOWN

    def ev_UnaryOp(self, node):
        v = self.ev(node.operand)
        if isinstance(node.op, ast.Not):
            t = truthy(v)
            return UNKNOWN if t is None else Known(not t)
        if isinstance(v, Known) and isinstance(v.value, (int, float)):
            if isinstance(node.op, ast.USub):
                return Known(-v.value, subst=v.subst)
            if isinstance(node.op, ast.UAdd):
                return Known(+v.value, subst=v.subst)
        return UNKNOWN

    def ev_BoolOp(self, node):
        vals = [self.ev(v) for v in node.values]
        if not all(isinstance(v, Known) for v in vals):
            return UNKNOWN
        if isinstance(node.op, ast.And):
            for v in vals:
                if not v.value:
                    return v
            return vals[-1]
        for v in vals:
            if v.value:
                return v
        return vals[-1]

    def ev_IfExp(self, node):
        t = truthy(self.ev(node.test))
        if t is None:
            return UNKNOWN
        return self.ev(node.body if t else node.orelse)

    def ev_BinOp(self, node):
        left, right = self.ev(node.left), self.ev(node.right)
        if not (isinstance(left, Known) and isinstance(right, Known)):
            return UNKNOWN
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a**b,
        }
        fn = ops.get(type(node.op))
        if fn is None:
            return UNKNOWN
        return Known(fn(left.value, right.value), subst=left.subst and right.subst)

    def ev_Compare(self, node):
        if len(node.ops) != 1:
            return UNKNOWN
        op = node.ops[0]
        left, right = self.ev(node.left), self.ev(node.comparators[0])
        if isinstance(op, (ast.Is, ast.IsNot)):
            res = self._is(left, right)
            if res is None:
                return UNKNOWN
            return Known(res if isinstance(op, ast.Is) else not res)
        if isinstance(op, (ast.In, ast.NotIn)):
            res = self._in(left, node.comparators[0], right)
            if res is None:
                return UNKNOWN
            return Known(res if isinstance(op, ast.In) else not res)
        if isinstance(op, (ast.Eq, ast.NotEq)) and not (
            isinstance(left, Known) and isinstance(right, Known)
        ):
            same = self._eq(left, right)
            if same is None:
                return UNKNOWN
            return Known(same if isinstance(op, ast.Eq) else not same)
        if not (isinstance(left, Known) and isinstance(right, Known)):
            return UNKNOWN
        ops = {
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
        }
        fn = ops.get(type(op))
        return Known(bool(fn(left.value, right.value))) if fn else UNKNOWN

    @staticmethod
    def _is(left, right) -> bool | None:
        ln, rn = is_none(left), is_none(right)
        if ln is None or rn is None:
            return None
        if ln or rn:
            return ln and rn
        if isinstance(left, Sym) and isinstance(right, Sym):
            if left.chain == right.chain:
                return True
            if left.chain in DTYPE_SYMS and right.chain in DTYPE_SYMS:
                return False
            return None
        if isinstance(left, Known) and isinstance(right, Known):
            if isinstance(left.value, bool) and isinstance(right.value, bool):
                return left.value is right.value
            if type(left.value) is not type(right.value) and (
                isinstance(left.value, bool) or isinstance(right.value, bool)
            ):
                return False
        return None

    def _in(self, left, right_node, right) -> bool | None:
        if isinstance(left, Known) and isinstance(right, Known):
            try:
                return left.value in right.value
            except Exception:  # noqa: BLE001
                return None
        if isinstance(right_node, (ast.Tuple, ast.List, ast.Set)) and left is not UNKNOWN:
            found_unknown = False
            for elt in right_node.elts:
                res = self._eq(left, self.ev(elt))
                if res is True:
                    return True
                if res is None:
                    found_unknown = True
            return None if found_unknown else False
        return None

    @staticmethod
    def _eq(a, b) -> bool | None:
        if isinstance(a, Known) and isinstance(b, Known):
            return a.value == b.value
        if isinstance(a, Sym) and isinstance(b, Sym):
            if a.chain == b.chain:
                return True
            return False if (a.chain in DTYPE_SYMS and b.chain in DTYPE_SYMS) else None
        for x, y in ((a, b), (b, a)):
            if isinstance(x, (Sym, _Opaque)) and isinstance(y, Known) and y.value is None:
                return False
        return None

    def ev_Subscript(self, node):
        if _slicing(node.slice):
            base = self.ev(node.value)
            return OPAQUE if isinstance(base, _Opaque) else UNKNOWN
        key = self.ev(node.slice)
        if not isinstance(key, Known):
            return UNKNOWN
        if isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values, strict=True):
                if k is None:
                    return UNKNOWN
                kv = self.ev(k)
                if isinstance(kv, Known) and kv.value == key.value:
                    return self.ev(v)
            return UNKNOWN
        base = self.ev(node.value)
        if isinstance(base, Known):
            return Known(base.value[key.value], subst=base.subst and key.subst)
        return UNKNOWN

    def ev_Call(self, node):
        if any(k.arg is None for k in node.keywords) or any(
            isinstance(a, ast.Starred) for a in node.args
        ):
            return UNKNOWN
        chain = attr_chain(node.func)
        if chain is not None and chain.startswith(_OPAQUE_CALL_PREFIXES):
            return OPAQUE
        args = [self.ev(a) for a in node.args]
        kwargs = {k.arg: self.ev(k.value) for k in node.keywords}
        all_known = all(isinstance(a, Known) for a in args) and all(
            isinstance(v, Known) for v in kwargs.values()
        )
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if (self.module, name) in ASSUMED_CALLS:
                return Known(ASSUMED_CALLS[(self.module, name)])
            if name in self.env or name in self.fn_bound:
                return UNKNOWN
            fn = _PURE_BUILTINS.get(name)
            if fn is not None:
                if all_known:
                    return Known(
                        fn(*[a.value for a in args], **{k: v.value for k, v in kwargs.items()}),
                        subst=all(a.subst for a in args) and all(v.subst for v in kwargs.values()),
                    )
                return UNKNOWN
            if all_known:
                return self.ctx.pure_call(
                    self.module,
                    name,
                    [a.value for a in args],
                    {k: v.value for k, v in kwargs.items()},
                )
            return UNKNOWN
        if isinstance(node.func, ast.Attribute):
            # ``locals().get("name")`` for a name the function never binds.
            if (
                node.func.attr == "get"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "locals"
                and args
                and isinstance(args[0], Known)
                and isinstance(args[0].value, str)
                and args[0].value not in self.fn_bound
            ):
                return Known(kwargs.get("default", args[1] if len(args) > 1 else Known(None)).value)
            recv = self.ev(node.func.value)
            if isinstance(recv, _Opaque) and node.func.attr in _OPAQUE_METHODS:
                return OPAQUE
            if isinstance(node.func.value, ast.Name) and node.func.value.id not in self.env:
                target = self.ctx.alias_target(self.module, node.func.value.id)
                if target is not None and all_known:
                    return self.ctx.pure_call(
                        target,
                        node.func.attr,
                        [a.value for a in args],
                        {k: v.value for k, v in kwargs.items()},
                    )
            if not isinstance(recv, Known) or not all_known:
                return UNKNOWN
            a = [x.value for x in args]
            kw = {k: v.value for k, v in kwargs.items()}
            if isinstance(recv.value, str) and node.func.attr in _PURE_STR_METHODS:
                return Known(getattr(recv.value, node.func.attr)(*a, **kw))
            if isinstance(recv.value, dict) and node.func.attr in _PURE_DICT_METHODS:
                out = getattr(recv.value, node.func.attr)(*a, **kw)
                return Known(list(out) if node.func.attr != "get" else out)
        return UNKNOWN


def _slicing(node: ast.AST) -> bool:
    """A subscript that slices (keeps a tensor a tensor) rather than indexing one element."""
    if isinstance(node, ast.Slice):
        return True
    if isinstance(node, ast.Tuple):
        return any(isinstance(e, ast.Slice) for e in node.elts) or any(
            isinstance(e, ast.Constant) and e.value is Ellipsis for e in node.elts
        )
    return False


def _const_node(value: Any) -> ast.Constant:
    return ast.Constant(value=value)


class Simplifier:
    """Rewrite an expression under a partial environment: fold decidable tests and lookups."""

    def __init__(self, evaluator: Evaluator):
        self.E = evaluator

    def env(self) -> dict[str, Any]:
        return self.E.env

    def run(self, node: ast.AST, test: bool = False) -> ast.AST:
        if node is None:
            return None
        if isinstance(node, ast.Lambda):
            return self._scoped(node, {a.arg for a in _all_args(node.args)})
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            names = set()
            for gen in node.generators:
                names |= {n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name)}
            return self._scoped(node, names)
        if isinstance(node, ast.BoolOp):
            return self._boolop(node, test)
        if isinstance(node, ast.IfExp):
            node.test = self.run(node.test, test=True)
            t = truthy(self.E.ev(node.test))
            if t is True:
                return self.run(node.body, test)
            if t is False:
                return self.run(node.orelse, test)
            node.body = self.run(node.body, test)
            node.orelse = self.run(node.orelse, test)
            return node
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Dict):
            node.slice = self.run(node.slice)
            key = self.E.ev(node.slice)
            if isinstance(key, Known):
                for k, v in zip(node.value.keys, node.value.values, strict=True):
                    kv = self.E.ev(k) if k is not None else UNKNOWN
                    if isinstance(kv, Known) and kv.value == key.value:
                        return self.run(v)
            node.value = self._children(node.value)
            return node
        if isinstance(node, ast.Slice):
            node = self._children(node)
            if (
                isinstance(node.lower, ast.Constant)
                and node.lower.value == 0
                and node.upper is None
                and node.step is None
            ):
                node.lower = None
            return node
        # A name that is None at trace time must not become the literal ``None`` where the code
        # subscripts, calls or takes an attribute of it: ``None["x"]`` / ``None(x)`` is a
        # SyntaxWarning at import (FINDINGS 232, the disabled interval calibrator). The branch
        # is unreachable in that case, so the name is simply kept.
        base = None
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            base = node.value
        elif isinstance(node, ast.Call):
            base = node.func
        if base is not None:
            base._keep_if_none = True  # type: ignore[attr-defined]
        if isinstance(node, ast.Call):
            node = self._children(node)
            node = self._expand_star_kwargs(node)
        else:
            node = self._children(node)
        if isinstance(node, ast.Constant):
            return node
        if isinstance(node, ast.Subscript) and _is_full_slice(node.slice):
            return node.value
        if isinstance(node, ast.BinOp):
            folded = self._binop_identity(node)
            if folded is not None:
                return folded
        keep_none = getattr(node, "_keep_if_none", False)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            v = self.E.ev(node)
            if keep_none and isinstance(v, Known) and v.value is None:
                return node
            if isinstance(v, Known) and v.node is not None:
                return copy.deepcopy(v.node)
            return _const_node(v.value) if substitutable(v) else node
        if isinstance(
            node, (ast.Compare, ast.UnaryOp, ast.Call, ast.BinOp, ast.Subscript, ast.Attribute)
        ):
            v = self.E.ev(node)
            if keep_none and isinstance(v, Known) and v.value is None:
                return node
            if isinstance(node, (ast.BinOp, ast.UnaryOp)) and isinstance(v, Known):
                if isinstance(v.value, float):
                    return node  # keep float arithmetic as written
            if substitutable(v) and (is_pure(node) or isinstance(node, ast.Call)):
                return _const_node(v.value)
        return node

    def _binop_identity(self, node: ast.BinOp) -> ast.AST | None:
        for side, other in ((node.right, node.left), (node.left, node.right)):
            v = self.E.ev(side)
            if not (isinstance(v, Known) and type(v.value) in (int, float)):
                continue
            if isinstance(node.op, ast.Add) and type(v.value) is int and v.value == 0:
                return other
            if isinstance(node.op, ast.Mult) and v.value == 1:
                return other
        return None

    def _expand_star_kwargs(self, node: ast.Call) -> ast.Call:
        new_kw = []
        for k in node.keywords:
            if k.arg is None:
                v = self.E.ev(k.value)
                if (
                    isinstance(v, Known)
                    and isinstance(v.value, dict)
                    and all(isinstance(key, str) and key.isidentifier() for key in v.value)
                ):
                    nodes = {key: literal_node(val) for key, val in v.value.items()}
                    if all(n is not None for n in nodes.values()) and not (
                        set(nodes) & {kw.arg for kw in node.keywords if kw.arg}
                    ):
                        new_kw.extend(ast.keyword(arg=key, value=n) for key, n in nodes.items())
                        continue
            new_kw.append(k)
        node.keywords = new_kw
        return node

    def _scoped(self, node, shadow: set[str]) -> ast.AST:
        saved = {n: self.E.env.pop(n) for n in shadow if n in self.E.env}
        try:
            return self._children(node)
        finally:
            self.E.env.update(saved)

    def _children(self, node: ast.AST) -> ast.AST:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                new = []
                for item in value:
                    if isinstance(item, ast.expr):
                        new.append(self.run(item))
                    elif isinstance(item, ast.keyword):
                        item.value = self.run(item.value)
                        new.append(item)
                    elif isinstance(item, ast.comprehension):
                        item.iter = self.run(item.iter)
                        item.ifs = [self.run(i, test=True) for i in item.ifs]
                        new.append(item)
                    else:
                        new.append(item)
                setattr(node, field, new)
            elif isinstance(value, ast.expr) and field != "ctx":
                setattr(node, field, self.run(value))
        return node

    def _boolop(self, node: ast.BoolOp, test: bool) -> ast.AST:
        is_and = isinstance(node.op, ast.And)
        operands = [self.run(v, test=test) for v in node.values]
        kept: list[ast.AST] = []
        last_dropped: ast.AST | None = None
        for i, operand in enumerate(operands):
            t = truthy(self.E.ev(operand))
            neutral_op = t is not None and (t if is_and else not t)
            if neutral_op:
                # ``True and X`` / ``False or X``: decides nothing. In a value context the LAST
                # operand is the expression's value when every other one is neutral, so it
                # stays unless we are in a test.
                if i == len(operands) - 1 and not test:
                    last_dropped = operand
                continue
            if t is not None:
                # A deciding operand. Exact when nothing unknown precedes it; in a test
                # context, also when everything preceding is side-effect free.
                if not kept:
                    return operand
                if test and all(is_pure(k) for k in kept):
                    return _const_node(not is_and)
                kept.append(operand)
                break
            kept.append(operand)
        if last_dropped is not None:
            kept.append(last_dropped)
        if not kept:
            return _const_node(is_and)
        if len(kept) == 1:
            return kept[0]
        node.values = kept
        return node


def _is_full_slice(node: ast.AST) -> bool:
    def empty(s):
        return isinstance(s, ast.Slice) and s.lower is None and s.upper is None and s.step is None

    if isinstance(node, ast.Tuple):
        return bool(node.elts) and all(empty(e) for e in node.elts)
    return empty(node)


def _all_args(args: ast.arguments) -> list[ast.arg]:
    out = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        out.append(args.vararg)
    if args.kwarg:
        out.append(args.kwarg)
    return out


# --------------------------------------------------------------------------------------------
# 3. Statement-level folding of one function
# --------------------------------------------------------------------------------------------


def _base_name(target: ast.AST) -> str | None:
    while isinstance(target, (ast.Attribute, ast.Subscript)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def stores_in(node: ast.AST, include_nested: bool = False) -> set[str]:
    """Names bound anywhere in ``node`` (not descending into nested scopes unless asked)."""
    out: set[str] = set()

    def visit(n: ast.AST) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            if not include_nested:
                for d in n.decorator_list:
                    visit(d)
                return
        if isinstance(n, ast.Lambda) and not include_nested:
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(
            n.ctx, (ast.Store, ast.Del)
        ):
            base = _base_name(n)
            if base:
                out.add(base)
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return out


def nonlocal_names(fn: ast.AST) -> set[str]:
    """Names any nested function rebinds through ``nonlocal``/``global``."""
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.Nonlocal, ast.Global)):
            out.update(n.names)
    return out


def loads_in(node: ast.AST) -> Counter:
    c: Counter = Counter()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            c[n.id] += 1
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            c.update(n.names)
    return c


def name_binding_counts(fn: ast.AST) -> Counter:
    """How many times each name is bound anywhere in ``fn`` (nested scopes included)."""
    counts: Counter = Counter()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            counts[n.id] += 1
        elif isinstance(n, (ast.Nonlocal, ast.Global)):
            counts.update(n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[n.name] += 1
        elif isinstance(n, ast.ExceptHandler) and n.name:
            counts[n.name] += 1
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                counts[(a.asname or a.name).split(".")[0]] += 1
    return counts


def parent_map(root: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for n in ast.walk(root):
        for child in ast.iter_child_nodes(n):
            parents[id(child)] = n
    return parents


def container_safe(root: ast.AST, name: str, value: Known, E: Evaluator, is_param: bool) -> bool:
    """Whether a name bound to a mutable container is only ever read within ``root``.

    Reads are subscripts, ``.get``-style method calls, membership tests and truthiness. A
    parameter may also be passed on to other functions (configuration dictionaries are not
    mutated by their readers). A store that writes the value already held is a read.
    """
    parents = parent_map(root)
    for n in ast.walk(root):
        if not (isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)):
            continue
        p = parents.get(id(n))
        if isinstance(p, ast.Subscript) and p.value is n:
            if isinstance(p.ctx, ast.Load):
                continue
            gp = parents.get(id(p))
            if (
                isinstance(p.ctx, ast.Store)
                and isinstance(gp, ast.Assign)
                and gp.targets == [p]
                and isinstance(value.value, dict)
            ):
                key, new = E.ev(p.slice), E.ev(gp.value)
                if (
                    isinstance(key, Known)
                    and isinstance(new, Known)
                    and key.value in value.value
                    and value.value[key.value] == new.value
                    and type(value.value[key.value]) is type(new.value)
                ):
                    continue
            return False
        if isinstance(p, ast.Attribute) and p.value is n:
            gp = parents.get(id(p))
            if isinstance(gp, ast.Call) and gp.func is p and p.attr in _READ_METHODS:
                continue
            return False
        if isinstance(p, ast.Compare):
            if n in p.comparators and all(isinstance(op, (ast.In, ast.NotIn)) for op in p.ops):
                continue
            if p.left is n and all(isinstance(op, (ast.Is, ast.IsNot)) for op in p.ops):
                continue
            return False
        if isinstance(p, (ast.If, ast.While, ast.IfExp)) and p.test is n:
            continue
        if isinstance(p, ast.BoolOp) or (isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not)):
            continue
        if (
            isinstance(p, ast.Call)
            and is_param
            and (n in p.args or any(k.value is n for k in p.keywords))
        ):
            continue
        if isinstance(p, ast.keyword) and is_param:
            gp = parents.get(id(p))
            if isinstance(gp, ast.Call):
                continue
        return False
    return True


_TERMINAL = (ast.Return, ast.Raise, ast.Break, ast.Continue)


class FunctionFolder:
    """Fold one function body flow-sensitively; then its nested functions with a closure env."""

    def __init__(self, ctx: Context, module_env: dict[str, Any]):
        self.ctx = ctx
        self.module_env = module_env

    def fold(
        self,
        fn: ast.FunctionDef,
        param_env: dict[str, Any],
        outer_env: dict[str, Any],
        self_env: dict[str, Any] | None = None,
    ) -> None:
        self.root = fn
        params = {a.arg for a in _all_args(fn.args)}
        bound = stores_in(fn, include_nested=False) | params | nonlocal_names(fn)
        env = {k: v for k, v in {**self.module_env, **outer_env}.items() if k not in bound}
        rebound_outside = nonlocal_names(fn)
        self_name = None
        if self_env and fn.args.args and fn.args.args[0].arg == "self":
            self_name = "self"
        fn_bound = stores_in(fn, include_nested=True) | params
        E = Evaluator(env, self.ctx, self_name=self_name, self_env=self_env, fn_bound=fn_bound)
        for name, value in param_env.items():
            if name in rebound_outside or name not in params:
                continue
            self._record(name, value, env, False, E, is_param=True)
        S = Simplifier(E)
        self.top_values: dict[str, Any] = {}
        fn.body = self._stmts(fn.body, S, top=True) or [ast.Pass()]
        counts = name_binding_counts(fn)
        closure = {k: v for k, v in env.items() if k in params and counts[k] == 0}
        for name, value in self.top_values.items():
            if counts[name] == 1 and value is not UNKNOWN:
                closure[name] = value
        for nested in nested_functions(fn.body):
            sub = FunctionFolder(self.ctx, self.module_env)
            sub.fold(nested, {}, {**outer_env, **closure}, self_env=None)

    # -- statements -------------------------------------------------------------------------

    def _stmts(self, stmts: list[ast.stmt], S: Simplifier, top: bool = False) -> list[ast.stmt]:
        env = S.env()
        out: list[ast.stmt] = []
        for st in stmts:
            out.extend(self._stmt(st, S, env, top))
            if out and isinstance(out[-1], _TERMINAL):
                break
        if len(out) > 1:
            out = [s for s in out if not isinstance(s, ast.Pass)] or [ast.Pass()]
        return out

    def _record(
        self, name: str, value: Any, env: dict, top: bool, E: Evaluator, is_param=False
    ) -> None:
        if isinstance(value, Known) and isinstance(value.value, (dict, list, set)):
            if not container_safe(self.root, name, value, E, is_param):
                value = OPAQUE
        if value is UNKNOWN:
            env.pop(name, None)
        else:
            env[name] = value
        if top:
            self.top_values[name] = value

    def _stmt(self, st: ast.stmt, S: Simplifier, env: dict, top: bool) -> list[ast.stmt]:
        E = S.E
        if isinstance(st, ast.Assign):
            st.value = S.run(st.value)
            for t in st.targets:
                if isinstance(t, ast.Subscript):
                    t.slice = S.run(t.slice)
            if (
                len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and isinstance(st.value, ast.Name)
                and st.value.id == st.targets[0].id
            ):
                return []  # ``x = x``
            if len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                self._record(st.targets[0].id, E.ev(st.value), env, top, E)
            elif all(isinstance(t, ast.Name) for t in st.targets):
                v = E.ev(st.value)
                for t in st.targets:
                    self._record(t.id, v, env, top, E)
            elif (
                len(st.targets) == 1
                and isinstance(st.targets[0], ast.Tuple)
                and isinstance(st.value, ast.Tuple)
                and len(st.targets[0].elts) == len(st.value.elts)
                and all(isinstance(t, ast.Name) for t in st.targets[0].elts)
            ):
                for t, v in zip(st.targets[0].elts, st.value.elts, strict=True):
                    self._record(t.id, E.ev(v), env, top, E)
            else:
                for t in st.targets:
                    for n in stores_in(t):
                        env.pop(n, None)
                        if top:
                            self.top_values[n] = UNKNOWN
            return [st]
        if isinstance(st, ast.AnnAssign):
            if st.value is not None:
                st.value = S.run(st.value)
            if isinstance(st.target, ast.Name):
                self._record(
                    st.target.id, E.ev(st.value) if st.value is not None else UNKNOWN, env, top, E
                )
            else:
                for n in stores_in(st.target):
                    env.pop(n, None)
            return [st]
        if isinstance(st, ast.AugAssign):
            st.value = S.run(st.value)
            if isinstance(st.target, ast.Name):
                cur = env.get(st.target.id, UNKNOWN)
                new = E.ev(
                    ast.BinOp(
                        left=ast.Name(id=st.target.id, ctx=ast.Load()), op=st.op, right=st.value
                    )
                )
                if isinstance(cur, Known) and substitutable(new):
                    assign = ast.Assign(targets=[st.target], value=_const_node(new.value))
                    self._record(st.target.id, new, env, top, E)
                    return [assign]
            for n in stores_in(st.target):
                env.pop(n, None)
                if top:
                    self.top_values[n] = UNKNOWN
            return [st]
        if isinstance(st, ast.If):
            st.test = S.run(st.test, test=True)
            t = truthy(E.ev(st.test))
            if t is True:
                return self._stmts(st.body, S, top)
            if t is False:
                return self._stmts(st.orelse, S, top)
            body = self._stmts(st.body, self._branch(S), False)
            orelse = self._stmts(st.orelse, self._branch(S), False) if st.orelse else []
            self._invalidate(body + orelse, env, top)
            body = [s for s in body if not isinstance(s, ast.Pass)]
            orelse = [s for s in orelse if not isinstance(s, ast.Pass)]
            if not body and not orelse:
                return [] if is_pure(st.test) else [ast.Expr(value=st.test)]
            if not body:
                st.test = ast.UnaryOp(op=ast.Not(), operand=st.test)
                st.body, st.orelse = orelse, []
                return [st]
            st.body, st.orelse = body, orelse
            return [st]
        if isinstance(st, (ast.For, ast.AsyncFor, ast.While)):
            self._invalidate([st], env, top)
            if isinstance(st, ast.While):
                st.test = S.run(st.test, test=True)
            else:
                st.iter = S.run(st.iter)
            st.body = self._stmts(st.body, self._branch(S), False)
            st.orelse = self._stmts(st.orelse, self._branch(S), False) if st.orelse else []
            return [st]
        if isinstance(st, (ast.With, ast.AsyncWith)):
            for item in st.items:
                item.context_expr = S.run(item.context_expr)
                if item.optional_vars is not None:
                    for n in stores_in(item.optional_vars):
                        env.pop(n, None)
            st.body = self._stmts(st.body, S, top)
            return [st]
        if isinstance(st, ast.Try):
            st.body = self._stmts(st.body, self._branch(S), False)
            for h in st.handlers:
                if h.type is not None:
                    h.type = S.run(h.type)
                h.body = self._stmts(h.body, self._branch(S), False)
            st.orelse = self._stmts(st.orelse, self._branch(S), False) if st.orelse else []
            st.finalbody = self._stmts(st.finalbody, self._branch(S), False) if st.finalbody else []
            self._invalidate([st], env, top)
            return [st]
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            env.pop(st.name, None)
            if top:
                self.top_values[st.name] = UNKNOWN
            return [st]
        if isinstance(st, ast.Return):
            if st.value is not None:
                st.value = S.run(st.value)
            return [st]
        if isinstance(st, ast.Expr):
            st.value = S.run(st.value)
            return [st]
        if isinstance(st, ast.Raise):
            if st.exc is not None:
                st.exc = S.run(st.exc)
            if st.cause is not None:
                st.cause = S.run(st.cause)
            return [st]
        if isinstance(st, ast.Assert):
            st.test = S.run(st.test, test=True)
            if truthy(E.ev(st.test)) is True:
                return []
            if st.msg is not None:
                st.msg = S.run(st.msg)
            return [st]
        self._invalidate([st], env, top)
        return [st]

    @staticmethod
    def _branch(S: Simplifier) -> Simplifier:
        return Simplifier(S.E.copy(dict(S.E.env)))

    def _invalidate(self, stmts: list[ast.stmt], env: dict, top: bool) -> None:
        names: set[str] = set()
        for s in stmts:
            names |= stores_in(s, include_nested=False) | nonlocal_names(s)
        for n in names:
            env.pop(n, None)
            if top:
                self.top_values[n] = UNKNOWN


def _with_named_defaults(
    fn: ast.FunctionDef, env: dict[str, Any], ctx: Context, menv: dict[str, Any]
) -> dict[str, Any]:
    """Substitute a parameter by its default's *name* when the folded value is that default."""
    positional = fn.args.posonlyargs + fn.args.args
    defaults = [None] * (len(positional) - len(fn.args.defaults)) + list(fn.args.defaults)
    pairs = list(zip(positional, defaults, strict=True)) + list(
        zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)
    )
    E = Evaluator(dict(menv), ctx)
    for a, d in pairs:
        v = env.get(a.arg)
        if d is None or not isinstance(v, Known) or not v.subst:
            continue
        if isinstance(d, (ast.Name, ast.Attribute)) and attr_chain(d) not in DTYPE_SYMS:
            dv = E.ev(d)
            if isinstance(dv, Known) and dv.value == v.value and type(dv.value) is type(v.value):
                env[a.arg] = Known(v.value, subst=True, node=d)
    return env


def nested_functions(stmts: list[ast.stmt]) -> list[ast.FunctionDef]:
    """Functions defined directly inside ``stmts`` (at any statement depth, not inside them)."""
    out = []

    def visit(n):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(child)
            elif isinstance(child, ast.ClassDef):
                continue
            elif isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case)):
                visit(child)

    for s in stmts:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(s)
        elif not isinstance(s, ast.ClassDef):
            visit(s)
    return out


def all_nested_functions(fn: ast.FunctionDef) -> list[ast.FunctionDef]:
    out = []
    stack = nested_functions(fn.body)
    while stack:
        f = stack.pop()
        out.append(f)
        stack.extend(nested_functions(f.body))
    return out


def _pure_value(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            chain = attr_chain(sub.func)
            if not (
                (isinstance(sub.func, ast.Name) and sub.func.id in _PURE_BUILTINS)
                or (isinstance(sub.func, ast.Attribute) and sub.func.attr in _READ_METHODS)
                or chain in PURE_CALLS
            ):
                return False
        elif isinstance(sub, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
            return False
    return True


def remove_dead_stores(fn: ast.FunctionDef) -> None:
    """Drop top-level pure stores that are never read, or overwritten before any read."""
    loads = loads_in(fn)
    body = fn.body
    keep = [True] * len(body)
    for i, st in enumerate(body):
        if not isinstance(st, (ast.Assign, ast.AnnAssign)):
            continue
        targets = st.targets if isinstance(st, ast.Assign) else [st.target]
        value = st.value
        if value is None or not _pure_value(value):
            continue
        if all(isinstance(t, ast.Name) for t in targets):
            names = [t.id for t in targets]
            if all(loads[n] == 0 for n in names):
                keep[i] = False
                continue
            if len(names) == 1 and _overwritten_before_read(body, i, names[0]):
                keep[i] = False
                continue
        if (
            isinstance(st, ast.Assign)
            and len(targets) == 1
            and isinstance(targets[0], ast.Tuple)
            and isinstance(value, ast.Tuple)
            and len(targets[0].elts) == len(value.elts)
            and all(isinstance(t, ast.Name) for t in targets[0].elts)
        ):
            pairs = [
                (t, v)
                for t, v in zip(targets[0].elts, value.elts, strict=True)
                if not (loads[t.id] == 0 and _pure_value(v))
            ]
            if not pairs:
                keep[i] = False
            elif len(pairs) == 1:
                st.targets = [pairs[0][0]]
                st.value = pairs[0][1]
            else:
                targets[0].elts = [p[0] for p in pairs]
                value.elts = [p[1] for p in pairs]
    fn.body = [s for s, k in zip(body, keep, strict=True) if k] or [ast.Pass()]


def _param_overwritten(fn: ast.FunctionDef, name: str) -> bool:
    """A parameter rebound by a top-level assignment before anything reads it."""
    for j, st in enumerate(fn.body):
        if isinstance(st, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in st.targets
        ):
            if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(st.value)):
                return False
            return not any(
                isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and name in loads_in(o)
                for k, o in enumerate(fn.body)
                if k < j
            )
        if name in loads_in(st) or name in stores_in(st, include_nested=True):
            return False
    return False


def _overwritten_before_read(body: list[ast.stmt], i: int, name: str) -> bool:
    """The store at ``body[i]`` is overwritten by a later top-level store before any read.

    Nested functions that read the name must all be defined after the overwriting store,
    so that no call made in between can observe the first value.
    """
    for j in range(i + 1, len(body)):
        st = body[j]
        if isinstance(st, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in st.targets
        ):
            if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(st.value)):
                return False
            for k, other in enumerate(body):
                if isinstance(other, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if k < j and name in loads_in(other):
                        return False
            return True
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if name in loads_in(st):
                return False
            continue
        if name in loads_in(st) or name in stores_in(st, include_nested=True):
            return False
    return False


# --------------------------------------------------------------------------------------------
# 4. Module-level machinery: the trace, module constants, references, docstrings
# --------------------------------------------------------------------------------------------


def deser(tagged: Any) -> Any:
    """Rebuild a traced argument value from the driver's tagged JSON."""
    kind = tagged["k"]
    if kind == "lit":
        v = tagged["v"]
        t = tagged.get("t")
        if t == "tuple":
            return Known(tuple(deser(x).value for x in v))
        if t == "list":
            return Known([deser(x).value for x in v])
        if t == "dict":
            return Known({k: deser(x).value for k, x in v.items()})
        return Known(v)
    if kind == "sym":
        return Sym(tagged["v"])
    return OPAQUE


def _foldable_recorded(v: Any) -> bool:
    """Traced argument values that may be folded: no filesystem paths, nothing huge."""
    if isinstance(v, Known):
        x = v.value
        if isinstance(x, str):
            return not path_like(x) and len(x) <= 200
        if isinstance(x, (tuple, list)):
            return len(x) <= 64
        if isinstance(x, dict):
            return len(x) <= 400
    return True


class ModuleInfo:
    def __init__(self, rel: str, tree: ast.Module, executed: set[int], calls: dict):
        self.rel = rel
        self.tree = tree
        self.executed = executed
        self.calls = calls  # firstlineno -> record

    def qualname_map(self) -> dict[int, tuple[str, ast.AST, ast.ClassDef | None]]:
        """``lineno -> (qualname, node, owning class)`` for every def, keyed by header and by
        first decorator line (CPython's ``co_firstlineno`` uses the decorator)."""
        out = {}

        def visit(body, prefix, owner):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    q = prefix + n.name
                    out[n.lineno] = (q, n, owner)
                    for d in n.decorator_list:
                        out.setdefault(d.lineno, (q, n, owner))
                    visit(n.body, q + ".", None)
                elif isinstance(n, ast.ClassDef):
                    q = prefix + n.name
                    visit(n.body, q + ".", n)
                elif hasattr(n, "body") and isinstance(getattr(n, "body", None), list):
                    visit(n.body, prefix, owner)
                    for extra in ("orelse", "finalbody"):
                        visit(getattr(n, extra, []) or [], prefix, owner)
                    for h in getattr(n, "handlers", []) or []:
                        visit(h.body, prefix, owner)

        visit(self.tree.body, "", None)
        return out

    def top_defs(self) -> dict[str, ast.AST]:
        return {
            n.name: n
            for n in self.tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }


def module_constants(tree: ast.Module, ctx: Context, mutable_attrs: set[str]) -> dict[str, Any]:
    """Names bound exactly once at module level to a known value.

    ``UPPER_CASE`` names qualify by convention. Other names qualify only when nothing in the
    bundle assigns them as an attribute of the module (``mutable_attrs``), so a module-level
    override slot that some other file may set stays dynamic.
    """
    counts = Counter()
    globals_declared = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Global):
            globals_declared.update(n.names)
    for st in tree.body:
        for n in stores_in(st):
            counts[n] += 1
    env: dict[str, Any] = {}
    E = Evaluator(env, ctx)
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1:
            t, v = st.targets[0], st.value
        elif isinstance(st, ast.AnnAssign) and st.value is not None:
            t, v = st.target, st.value
        else:
            continue
        if not isinstance(t, ast.Name) or t.id == "__all__":
            continue
        if counts[t.id] != 1 or t.id in globals_declared:
            continue
        if not _CONST_NAME_RE.fullmatch(t.id) and t.id in mutable_attrs:
            continue
        val = E.ev(v)
        if (
            isinstance(val, Known)
            and isinstance(val.value, (dict, list, set))
            and not container_safe(tree, t.id, val, E, False)
        ):
            # A module-level container that some function writes (``_state["fn"] = ...``) is
            # state, not a constant: its initial entries hold only until the first write.
            # Folding ``if _state["fn"] is None:`` against them dropped the build-once guard
            # of a Track-2 submission (FINDINGS 222d).
            continue
        if isinstance(val, Known):
            env[t.id] = Known(val.value, subst=False)  # a named constant reads better
        elif isinstance(val, Sym):
            env[t.id] = val
    return env


def executed_in(node: ast.AST, executed: set[int]) -> bool:
    if isinstance(node, ast.ClassDef):
        return any(
            executed_in(m, executed)
            for m in node.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    if not node.body:
        return False
    start = node.body[0].lineno
    return any(start <= ln <= node.end_lineno for ln in executed)


def _pure_constant_value(node: ast.AST) -> bool:
    return _pure_value(node)


def rewrite_all(tree: ast.Module, keep: set[str]) -> None:
    for st in tree.body:
        if (
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Name)
            and st.targets[0].id == "__all__"
            and isinstance(st.value, (ast.List, ast.Tuple))
        ):
            st.value.elts = [
                e for e in st.value.elts if isinstance(e, ast.Constant) and e.value in keep
            ]


def module_defined_names(tree: ast.Module) -> set[str]:
    names = set()
    for st in tree.body:
        names |= stores_in(st)
    return names


def first_sentence(doc: str) -> str | None:
    text = doc.strip()
    if not text:
        return None
    para = re.split(r"\n\s*\n", text, maxsplit=1)[0]
    para = re.sub(r"-\s*\n\s*", "-", para)
    para = " ".join(para.split())
    para = re.sub(
        r":(?:func|class|data|mod|attr|meth|obj):`~?([^`]*)`",
        lambda m: m.group(1).split(".")[-1],
        para,
    )
    para = para.replace("``", "`")
    m = re.match(r"(.+?[.!?])(\s|$)", para)
    sentence = m.group(1) if m else para
    sentence = sentence.strip()
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


def neutral(sentence: str | None) -> bool:
    if not sentence or len(sentence) < 8 or len(sentence) > 140:
        return False
    if FORBIDDEN_RE.search(sentence) or NON_NEUTRAL_RE.search(sentence) or UNIT_RE.search(sentence):
        return False
    if sentence.count("`") % 2 or sentence.count("(") != sentence.count(")"):
        return False
    if any(w in EMPHASIS_WORDS for w in re.findall(r"\b[A-Z]{3,}\b", sentence)):
        return False
    return not sentence.startswith(("(", "*", "-"))


def rewrite_docstrings(tree: ast.Module, rel: str) -> None:
    def fix(node, key, allow: bool = True):
        body = node.body
        has = (
            bool(body)
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )
        custom = DOCSTRINGS.get(key)
        if custom is None and has and allow:
            s = first_sentence(body[0].value.value)
            custom = s if neutral(s) else None
        if has:
            body.pop(0)
        if custom:
            body.insert(0, ast.Expr(value=ast.Constant(value=custom)))
        if not body:
            body.append(ast.Pass())

    fix(tree, f"{rel}:")

    def visit(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + n.name
                fix(n, f"{rel}:{q}", allow=not n.name.startswith("__"))
                visit(n.body, q + ".")
            elif hasattr(n, "body") and isinstance(getattr(n, "body", None), list):
                visit(n.body, prefix)
                for extra in ("orelse", "finalbody"):
                    visit(getattr(n, extra, []) or [], prefix)
                for h in getattr(n, "handlers", []) or []:
                    visit(h.body, prefix)

    visit(tree.body, "")


class PureFunctions:
    """Module-level functions that compute a literal from literal arguments, run at clean time.

    A function qualifies when its body only calls pure builtins, read-only string/dict methods
    and other qualifying functions of the same module, reads nothing but its arguments, its
    locals and known module constants, and imports nothing. ``parse_arch`` and ``aux_weights``
    are the motivating cases: their results decide which constructor branches a checkpoint
    takes.
    """

    def __init__(self, cleaner: Cleaner):
        self.cleaner = cleaner
        self.namespaces: dict[str, dict[str, Any]] = {}
        self.names: dict[str, set[str]] = {}
        for rel, info in cleaner.modules.items():
            consts = dict(cleaner.module_envs.get(rel, {}))
            for name, (mod, sym) in cleaner.imports.get(rel, {}).items():
                if mod is not None and sym is not None:
                    v = cleaner.module_envs.get(mod, {}).get(sym)
                    if isinstance(v, Known):
                        consts[name] = v
            self._build(rel, info.tree, consts)

    def _build(self, rel: str, tree: ast.Module, consts: dict[str, Any]) -> None:
        defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        pure: set[str] = set()
        changed = True
        while changed:
            changed = False
            for name, fn in defs.items():
                if name not in pure and self._is_pure(fn, pure, consts):
                    pure.add(name)
                    changed = True
        if not pure:
            return
        ns: dict[str, Any] = {"__builtins__": dict(_PURE_BUILTINS)}
        ns["__builtins__"].update(
            {
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "isinstance": isinstance,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "Exception": Exception,
                "reversed": reversed,
                "map": map,
                "filter": filter,
                "print": lambda *a, **k: None,
                "Any": Any,
            }
        )
        for k, v in consts.items():
            if isinstance(v, Known):
                ns[k] = v.value
        src = "\n".join(ast.unparse(copy.deepcopy(defs[n])) for n in pure)
        try:
            exec(compile(src, f"<pure:{rel}>", "exec"), ns)  # noqa: S102 - our own parsers
        except Exception:  # noqa: BLE001
            return
        self.namespaces[rel] = ns
        self.names[rel] = pure

    @staticmethod
    def _is_pure(fn: ast.FunctionDef, pure: set[str], consts: dict[str, Any]) -> bool:
        if fn.decorator_list or fn.args.vararg or fn.args.kwarg:
            return False
        local = {a.arg for a in _all_args(fn.args)} | stores_in(fn, include_nested=True)
        callable_locals: set[str] = set()
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.For)
                and isinstance(n.target, ast.Name)
                and isinstance(n.iter, (ast.Tuple, ast.List))
                and all(isinstance(e, ast.Name) and e.id in _PURE_BUILTINS for e in n.iter.elts)
            ):
                callable_locals.add(n.target.id)
        callables = set(_PURE_BUILTINS) | pure | callable_locals | _PURE_EXTRA_NAMES
        allowed_names = (
            local | callables | {"Any"} | {k for k, v in consts.items() if isinstance(v, Known)}
        )
        banned = (
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Yield,
            ast.YieldFrom,
            ast.Await,
            ast.With,
            ast.Lambda,
            ast.ClassDef,
        )
        call_funcs = {id(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
        for n in ast.walk(fn):
            if isinstance(n, banned):
                return False
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    if n.func.id not in callables:
                        return False
                elif isinstance(n.func, ast.Attribute):
                    if n.func.attr not in _READ_METHODS:
                        return False
                else:
                    return False
            elif isinstance(n, ast.Attribute) and id(n) not in call_funcs:
                return False
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id not in allowed_names:
                    return False
        return True

    def evaluate(self, rel: str, name: str, args: list, kwargs: dict) -> Any:
        if name not in self.names.get(rel, set()):
            return UNKNOWN
        try:
            result = self.namespaces[rel][name](*copy.deepcopy(args), **copy.deepcopy(kwargs))
        except Exception:  # noqa: BLE001
            return UNKNOWN
        return Known(result) if _json_like(result) else UNKNOWN


def _json_like(v: Any) -> bool:
    if isinstance(v, _SIMPLE):
        return True
    if isinstance(v, (tuple, list)):
        return all(_json_like(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _json_like(x) for k, x in v.items())
    return False


# --------------------------------------------------------------------------------------------
# 5. The cleaner: ties the passes together over the kept modules
# --------------------------------------------------------------------------------------------

_LITERAL_LIKE = (ast.Constant,)


def _literal_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_literal_like(e) for e in node.elts)
    if isinstance(node, ast.Attribute):
        return attr_chain(node) in DTYPE_SYMS
    return False


class Cleaner:
    def __init__(
        self, root: Path, trace: dict, protected_globals: set[str] = frozenset(), log=print
    ):
        self.root = root
        self.trace = trace
        self.log = log
        self.modules: dict[str, ModuleInfo] = {}
        self.extra_refs: Counter = Counter()  # references from the generated submission.py
        self.protected_globals = set(protected_globals)
        self.notes: list[str] = []
        self.imports: dict[str, dict[str, tuple[str | None, str | None]]] = {}
        self.module_envs: dict[str, dict[str, Any]] = {}
        self.pure_funcs: PureFunctions | None = None
        self.construction_only: set[tuple[str, str]] = set()
        self.explicit_params: dict[tuple[str, str], set[str]] = {}
        self._sites_cache: dict[tuple[str, str], list[dict]] = {}
        self._self_envs_cache: dict = {}

    # -- selection ---------------------------------------------------------------------------

    def select(self, include_submission: bool = False) -> list[str]:
        root = str(self.root)
        kept = set()
        skip = set() if include_submission else {"submission.py"}
        for f in self.trace["modules"]:
            if f.startswith(root):
                rel = os.path.relpath(f, root)
                if rel.endswith(".py") and rel not in skip:
                    kept.add(rel)
        for f in self.trace["executed"]:
            rel = os.path.relpath(f, root)
            if rel.endswith(".py") and rel not in skip and not rel.startswith("_verify"):
                kept.add(rel)
        for rel in sorted(kept):
            path = self.root / rel
            tree = ast.parse(path.read_text(encoding="utf-8"))
            executed = set(self.trace["executed"].get(str(path), []))
            calls = {}
            for key, rec in self.trace["calls"].items():
                fname, _, line = key.rpartition(":")
                if fname == str(path):
                    calls[int(line)] = rec
            self.modules[rel] = ModuleInfo(rel, tree, executed, calls)
        return sorted(kept)

    def data_files(self) -> list[str]:
        root = str(self.root)
        out = []
        for f in self.trace["opened"]:
            if f.startswith(root):
                rel = os.path.relpath(f, root)
                if not rel.endswith((".py", ".pyc")) and "__pycache__" not in rel:
                    out.append(rel)
        return sorted(set(out))

    # -- module-aware name resolution --------------------------------------------------------

    def _module_of(self, rel: str, level: int, module: str | None) -> str | None:
        """Resolve a relative import to a kept module path (``a/b.py`` or ``a/b/__init__.py``)."""
        if level == 0:
            parts = (module or "").split(".")
        else:
            base = Path(rel).parent
            for _ in range(level - 1):
                base = base.parent
            parts = list(base.parts) + ((module or "").split(".") if module else [])
        cand = "/".join(p for p in parts if p)
        for c in (cand + ".py", cand + "/__init__.py"):
            if c in self.modules:
                return c
        return None

    def rebuild_imports(self) -> None:
        self.imports = {}
        for rel, info in self.modules.items():
            table: dict[str, tuple[str | None, str | None]] = {}
            for n in ast.walk(info.tree):
                if isinstance(n, ast.ImportFrom):
                    pkg = self._module_of(rel, n.level, n.module)
                    for a in n.names:
                        local = a.asname or a.name
                        # ``from . import x`` may name a module rather than a symbol.
                        sub = None
                        if pkg is not None and pkg.endswith("__init__.py"):
                            sub = self._module_of(
                                rel, n.level, (n.module + "." if n.module else "") + a.name
                            )
                        elif pkg is None:
                            sub = self._module_of(
                                rel, n.level, (n.module + "." if n.module else "") + a.name
                            )
                        if sub is not None:
                            table[local] = (sub, None)
                        else:
                            table[local] = (pkg, a.name)
                elif isinstance(n, ast.Import):
                    for a in n.names:
                        target = self._module_of(rel, 0, a.name)
                        table[(a.asname or a.name).split(".")[0]] = (target, None)
            self.imports[rel] = table

    def resolve(self, rel: str, name: str) -> tuple[str, str] | None:
        info = self.modules[rel]
        if (
            name in info.top_defs()
            or name in module_defined_names(info.tree)
            and name not in self.imports.get(rel, {})
        ):
            return (rel, name)
        target = self.imports.get(rel, {}).get(name)
        if target is None or target[0] is None:
            return None
        if target[1] is None:
            return None  # a module alias, not a symbol
        return (target[0], target[1])

    def rebuild_envs(self) -> None:
        mutable_attrs = self._attribute_store_names() | self.protected_globals
        for _ in range(3):  # constants defined in terms of imported constants
            for rel, info in self.modules.items():
                self.module_envs[rel] = module_constants(
                    info.tree, Context(rel, self), mutable_attrs
                )

    def _attribute_store_names(self) -> set[str]:
        out: set[str] = set()
        for info in self.modules.values():
            for n in ast.walk(info.tree):
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
                    out.add(n.attr)
                elif (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in ("setattr", "delattr")
                    and len(n.args) >= 2
                    and isinstance(n.args[1], ast.Constant)
                ):
                    out.add(str(n.args[1].value))
        return out

    # -- passes ------------------------------------------------------------------------------

    def run(self) -> None:
        self.rebuild_imports()
        self.rebuild_envs()
        self.pure_funcs = PureFunctions(self)
        self.apply_prune_rules()
        for _round in range(20):
            before = self._digest()
            self.rebuild_imports()
            self.rebuild_envs()
            self.pure_funcs = PureFunctions(self)
            self.fold_module_ifs()
            self._construction_analysis()
            self.fold_all()
            self.fold_static_params()
            self.inline_constant_functions()
            self.remove_dead()
            self.prune_parameters()
            if self._digest() == before:
                break
        self.remove_dead()
        for info in self.modules.values():
            rewrite_docstrings(info.tree, info.rel)
            ast.fix_missing_locations(info.tree)

    def fold_module_ifs(self) -> None:
        """Fold module-level ``if`` statements decided by the module's own constants."""
        for rel, info in self.modules.items():
            ctx = Context(rel, self)
            E = Evaluator(dict(self.module_envs.get(rel, {})), ctx)
            S = Simplifier(E)
            new_body = []
            changed = False
            for st in info.tree.body:
                if isinstance(st, ast.If):
                    st.test = S.run(st.test, test=True)
                    t = truthy(E.ev(st.test))
                    if t is True:
                        new_body.extend(st.body)
                        changed = True
                        continue
                    if t is False:
                        new_body.extend(st.orelse)
                        changed = True
                        continue
                new_body.append(st)
            if changed:
                info.tree.body = new_body
                self.notes.append(f"{rel}: folded module-level conditionals")

    def apply_prune_rules(self) -> None:
        for rel, qual, pattern in PRUNE_RULES:
            info = self.modules.get(rel)
            if info is None:
                continue
            rx = re.compile(pattern)
            for _, (q, node, _owner) in info.qualname_map().items():
                if q != qual:
                    continue
                before = len(node.body)
                node.body = [
                    s for s in node.body if not rx.search(ast.unparse(s).splitlines()[0])
                ] or [ast.Pass()]
                if len(node.body) != before:
                    self.notes.append(
                        f"{rel}:{qual}: removed {before - len(node.body)} statement(s) "
                        f"matching {pattern!r}"
                    )

    def _traced_env(self, rel: str, qual: str, node, rec: dict | None) -> dict[str, Any]:
        if rec is None or "predict" in rec.get("phases", []):
            return {}
        full = qual in FOLD_QUALNAMES.get(rel, set())
        allowed = full or (rel, qual) in self.construction_only
        if not allowed:
            return {}
        explicit = set(self.explicit_params.get((rel, qual.split(".")[0]), set()))
        positional = [a.arg for a in node.args.posonlyargs + node.args.args]
        if positional and positional[0] == "self":
            positional = positional[1:]
        for i, name in enumerate(positional):
            if f"#pos{i}" in explicit:
                explicit.add(name)
        env = {}
        for name, values in rec["args"].items():
            if name == "self" or len(values) != 1:
                continue
            v = deser(values[0])
            if not _foldable_recorded(v):
                continue
            if isinstance(v, Known) and not full and name in explicit:
                v = Known(v.value, subst=False)
            env[name] = v
        return _with_named_defaults(node, env, Context(rel, self), self.module_envs.get(rel, {}))

    def _construction_analysis(self) -> None:
        """Functions reachable only from construction-time code, and explicitly passed params.

        A function is construction-only when the trace saw it run during import only and
        every resolved call site sits inside a function that is itself construction-only
        (the ``make_predict``/factory allow-list and import-only ``__init__`` methods seed
        the set). Their traced argument values may then be folded like a constructor's.
        """
        sites = self._resolved_call_sites()
        self.explicit_params = {}
        for target, ss in sites.items():
            names = set()
            for site in ss:
                call = site["call"]
                names |= {k.arg for k in call.keywords if k.arg}
                names |= {f"#pos{i}" for i in range(len(call.args))}
            self.explicit_params[target] = names
        import_only: set[tuple[str, str]] = set()
        for rel, info in self.modules.items():
            for lineno, (qual, node, _owner) in info.qualname_map().items():
                if lineno != node.lineno:
                    continue
                rec = self._record_of(info, node)
                if rec is not None and rec.get("phases") == ["import"]:
                    import_only.add((rel, qual))
        cons = {(rel, q) for rel, qs in FOLD_QUALNAMES.items() for q in qs}
        while True:
            added = False
            for key in import_only - cons:
                rel, qual = key
                parts = qual.split(".")
                if len(parts) == 2 and parts[1] == "__init__":
                    ss = sites.get((rel, parts[0]))
                elif len(parts) == 1:
                    ss = sites.get((rel, qual))
                else:
                    continue  # methods and nested functions: call sites not resolvable
                if not ss:
                    continue
                if all(
                    s["enclosing"] is None
                    or s["enclosing"] in cons
                    or (s["enclosing"][0], s["enclosing"][1] + ".__init__") in cons
                    for s in ss
                ):
                    cons.add(key)
                    added = True
            if not added:
                break
        self.construction_only = cons

    def _record_of(self, info: ModuleInfo, node) -> dict | None:
        rec = info.calls.get(node.lineno)
        for d in node.decorator_list:
            rec = rec or info.calls.get(d.lineno)
        return rec

    def fold_all(self) -> None:
        """Fold every function against its traced/known values: the allow-listed construction
        functions first (their folding rewrites constructor call sites), then everything else
        against the call sites as they then stand."""
        for pass_ in ("allowlist", "rest"):
            if pass_ == "rest":
                self._construction_analysis()
            self_envs = self._instance_constants()
            for rel, info in self.modules.items():
                ctx = Context(rel, self)
                menv = self.module_envs.get(rel, {})
                for lineno, (qual, node, owner) in sorted(info.qualname_map().items()):
                    if lineno != node.lineno or ("." in qual and owner is None):
                        continue
                    listed = qual in FOLD_QUALNAMES.get(rel, set())
                    if (pass_ == "allowlist") != listed:
                        continue
                    penv = self._traced_env(rel, qual, node, self._record_of(info, node))
                    cls_key = (rel, owner.name) if owner is not None else None
                    senv = self_envs.get(cls_key) if cls_key else None
                    FunctionFolder(ctx, menv).fold(node, penv, {}, self_env=senv)
                    remove_dead_stores(node)
                    for nested in all_nested_functions(node):
                        remove_dead_stores(nested)

    # -- static call-site folding ---------------------------------------------------------

    def _string_refs(self) -> set[str]:
        out = set()
        for info in self.modules.values():
            for n in ast.walk(info.tree):
                if (
                    isinstance(n, ast.Constant)
                    and isinstance(n.value, str)
                    and n.value.isidentifier()
                ):
                    out.add(n.value)
        return out

    def _resolved_call_sites(self) -> dict[tuple[str, str], list[dict]]:
        """``(module, def name) -> call sites`` for module-level functions and classes."""
        out: dict[tuple[str, str], list[dict]] = {}
        for rel, info in self.modules.items():
            enclosing: dict[int, tuple[str, str] | None] = {}

            def mark(body, qual, enclosing=enclosing, rel=rel):
                for st in body:
                    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        q = (qual + "." if qual else "") + st.name
                        for d in getattr(st, "decorator_list", []):
                            for n in ast.walk(d):
                                if isinstance(n, ast.Call):
                                    enclosing[id(n)] = (rel, qual) if qual else None
                        mark(st.body, q, enclosing, rel)
                        continue
                    stack = [st]
                    while stack:
                        n = stack.pop()
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            mark([n], qual, enclosing, rel)
                            continue
                        if isinstance(n, ast.Call):
                            enclosing[id(n)] = (rel, qual) if qual else None
                        stack.extend(ast.iter_child_nodes(n))

            mark(info.tree.body, "")
            for n in ast.walk(info.tree):
                if not isinstance(n, ast.Call):
                    continue
                target = None
                if isinstance(n.func, ast.Name):
                    target = self.resolve(rel, n.func.id)
                elif isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
                    alias = self.imports.get(rel, {}).get(n.func.value.id)
                    if alias and alias[0] is not None and alias[1] is None:
                        target = (alias[0], n.func.attr)
                if target is None:
                    continue
                out.setdefault(target, []).append(
                    {
                        "call": n,
                        "module": rel,
                        "enclosing": enclosing.get(id(n)),
                        "npos": len(n.args),
                        "star": any(isinstance(a, ast.Starred) for a in n.args)
                        or any(k.arg is None for k in n.keywords),
                    }
                )
        return out

    def _class_hierarchy(self, rel: str, name: str) -> set[tuple[str, str]]:
        """Classes related to ``(rel, name)`` by inheritance within the bundle."""
        bases: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for r, info in self.modules.items():
            for n in info.tree.body:
                if isinstance(n, ast.ClassDef):
                    bs = set()
                    for b in n.bases:
                        if isinstance(b, ast.Name):
                            res = self.resolve(r, b.id)
                            if res:
                                bs.add(res)
                    bases[(r, n.name)] = bs
        related = {(rel, name)}
        while True:
            more = set()
            for c, bs in bases.items():
                if c in related:
                    more |= bs
                if bs & related:
                    more.add(c)
            more -= related
            if not more:
                return related
            related |= more

    def fold_static_params(self) -> None:
        """Fold parameters that every call site in the bundle passes as the same literal."""
        sites = self._resolved_call_sites()
        self._sites_cache = sites
        strings = self._string_refs()
        self_envs = self._instance_constants()
        self._self_envs_cache = self_envs
        for rel, info in self.modules.items():
            ctx = Context(rel, self)
            menv = self.module_envs.get(rel, {})
            for node in info.tree.body:
                targets: list[tuple[ast.FunctionDef, list[dict], ast.ClassDef | None]] = []
                if isinstance(node, ast.FunctionDef):
                    if node.name in strings or (rel, node.name) in ASSUMED_CALLS:
                        continue
                    targets.append((node, sites.get((rel, node.name), []), None))
                elif isinstance(node, ast.ClassDef):
                    init = next(
                        (
                            m
                            for m in node.body
                            if isinstance(m, ast.FunctionDef) and m.name == "__init__"
                        ),
                        None,
                    )
                    if init is None or node.name in strings:
                        continue
                    if self._class_hierarchy(rel, node.name) != {(rel, node.name)}:
                        continue  # subclasses may call super().__init__ with other values
                    targets.append((init, sites.get((rel, node.name), []), node))
                for fn, fsites, owner in targets:
                    if not fsites or any(s_["star"] for s_ in fsites):
                        continue
                    env = self._static_env(fn, fsites, owner is not None, ctx)
                    rec = self._record_of(info, fn)
                    if rec is not None:
                        for name in list(env):
                            recorded = rec["args"].get(name)
                            if recorded and (
                                len(recorded) != 1
                                or repr(
                                    deser(recorded[0]).value
                                    if isinstance(deser(recorded[0]), Known)
                                    else deser(recorded[0])
                                )
                                != repr(
                                    env[name].value if isinstance(env[name], Known) else env[name]
                                )
                            ):
                                env.pop(name)
                    if not env:
                        continue
                    senv = self_envs.get((rel, owner.name)) if owner is not None else None
                    FunctionFolder(ctx, menv).fold(fn, env, {}, self_env=senv)
                    remove_dead_stores(fn)
                    for nested in all_nested_functions(fn):
                        remove_dead_stores(nested)

    def _static_env(
        self,
        fn: ast.FunctionDef,
        fsites: list[dict],
        is_method: bool,
        ctx: Context,
        depth: int = 0,
    ) -> dict[str, Any]:
        """Parameter values agreed on by every call site, decide-only when the call sites pass
        them explicitly. An argument counts when it is a literal or evaluates to one in its
        caller (a ``self.x`` instance constant, a known module constant)."""
        self_envs = getattr(self, "_self_envs_cache", None) or {}
        positional = fn.args.posonlyargs + fn.args.args
        offset = 1 if is_method else 0
        n_defaults = len(fn.args.defaults)
        env: dict[str, Any] = {}

        def passed(call: ast.Call, index: int | None, name: str):
            if index is not None and index < len(call.args):
                return call.args[index], True
            for k in call.keywords:
                if k.arg == name:
                    return k.value, True
            return None, False

        params = [
            (
                a,
                i - offset,
                fn.args.defaults[i - (len(positional) - n_defaults)]
                if i >= len(positional) - n_defaults
                else None,
            )
            for i, a in enumerate(positional)
            if i >= offset
        ]
        params += [
            (a, None, d) for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)
        ]
        for a, index, default in params:
            dumps, explicit, nodes = [], False, []
            ok = True
            for s_ in fsites:
                node_v, was_passed = passed(s_["call"], index, a.arg)
                if node_v is None:
                    node_v = default
                if node_v is None:
                    ok = False
                    break
                if _literal_like(node_v):
                    v = Evaluator({}, ctx).ev(node_v)
                elif depth < 2:
                    v = self._eval_in_caller(node_v, s_, self_envs, depth + 1)
                else:
                    v = UNKNOWN
                if not isinstance(v, (Known, Sym)) or not _foldable_recorded(v):
                    ok = False
                    break
                explicit |= was_passed
                dumps.append(repr(v.value) if isinstance(v, Known) else repr(v))
                nodes.append(v)
            if not ok or not dumps or len(set(dumps)) != 1:
                continue
            v = nodes[0]
            if isinstance(v, Known):
                v = Known(v.value, subst=not explicit)
            env[a.arg] = v
        return _with_named_defaults(fn, env, ctx, self.module_envs.get(ctx.module, {}))

    def _eval_in_caller(self, node: ast.AST, site: dict, self_envs: dict, depth: int = 0) -> Any:
        """Evaluate a call-site argument with the caller's module and instance constants, and
        with the caller's own statically known parameters."""
        if not is_pure(node):
            return UNKNOWN
        enclosing = site.get("enclosing")
        rel = site["module"]
        self_env = None
        env = dict(self.module_envs.get(rel, {}))
        if enclosing is not None:
            parts = enclosing[1].split(".")
            if len(parts) >= 2:
                self_env = self_envs.get((rel, parts[0]))
            if len(parts) == 1 or (len(parts) == 2 and parts[1] == "__init__"):
                caller = self._function_node(rel, enclosing[1])
                if caller is not None:
                    csites = self._sites_cache.get((rel, parts[0]), [])
                    if csites and not any(c["star"] for c in csites):
                        env.update(
                            self._static_env(
                                caller, csites, len(parts) == 2, Context(rel, self), depth
                            )
                        )
        E = Evaluator(
            env, Context(rel, self), self_name="self" if self_env else None, self_env=self_env
        )
        return E.ev(node)

    def _function_node(self, rel: str, qual: str) -> ast.FunctionDef | None:
        info = self.modules.get(rel)
        if info is None:
            return None
        for lineno, (q, node, _owner) in info.qualname_map().items():
            if q == qual and lineno == node.lineno and isinstance(node, ast.FunctionDef):
                return node
        return None

    def inline_constant_functions(self) -> None:
        """Replace calls to functions whose whole body is ``return <literal>`` by the literal."""
        strings = self._string_refs()
        sites = self._resolved_call_sites()
        constants: dict[tuple[str, str], ast.AST] = {}
        for rel, info in self.modules.items():
            ctx = Context(rel, self)
            for node in info.tree.body:
                if not isinstance(node, ast.FunctionDef) or node.decorator_list:
                    continue
                if node.name in strings or (rel, node.name) in ASSUMED_CALLS:
                    continue
                if node.args.vararg or node.args.kwarg:
                    continue
                body = [
                    st
                    for st in node.body
                    if not (
                        isinstance(st, ast.Expr)
                        and isinstance(st.value, ast.Constant)
                        and isinstance(st.value.value, str)
                    )
                ]
                if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
                    ret = body[0].value
                    if _literal_like(ret):
                        constants[(rel, node.name)] = ret
                        continue
                    fsites = sites.get((rel, node.name), [])
                    if not fsites or any(s_["star"] for s_ in fsites):
                        continue
                    env = {
                        k: (Known(v.value) if isinstance(v, Known) else v)
                        for k, v in self._static_env(node, fsites, False, ctx).items()
                    }
                    env.update(self.module_envs.get(rel, {}))
                    v = Evaluator(env, ctx).ev(ret)
                    if isinstance(v, Known) and (is_pure(ret) or isinstance(ret, ast.Call)):
                        lit = literal_node(v.value)
                        if lit is not None and len(ast.dump(lit)) < 400:
                            constants[(rel, node.name)] = lit
        if not constants:
            return
        cleaner = self

        class Inliner(ast.NodeTransformer):
            def __init__(self, rel):
                self.rel = rel

            def visit_Call(self, node):
                self.generic_visit(node)
                if any(isinstance(a, ast.Starred) for a in node.args) or any(
                    k.arg is None for k in node.keywords
                ):
                    return node
                if not all(is_pure(a) for a in node.args) or not all(
                    is_pure(k.value) for k in node.keywords
                ):
                    return node
                target = None
                if isinstance(node.func, ast.Name):
                    target = cleaner.resolve(self.rel, node.func.id)
                if target in constants:
                    cleaner.notes.append(f"{self.rel}: inlined constant function {target[1]}()")
                    return copy.deepcopy(constants[target])
                return node

        for rel, info in self.modules.items():
            for node in info.tree.body:
                if isinstance(node, ast.FunctionDef) and (rel, node.name) in constants:
                    continue
                Inliner(rel).visit(node)

    def _instance_constants(self) -> dict[tuple[str, str], dict[str, Any]]:
        """``(module, class) -> {attr: value}`` for attributes stored exactly once across the
        class's hierarchy, at the top level of its own ``__init__``, with a known value; a
        store elsewhere that writes the same value does not disqualify it."""
        self_stores: Counter = Counter()  # (module, class, attr)
        other_stores: dict[str, list[tuple[str, ast.AST]]] = {}  # attr -> [(module, value node)]
        setattr_strings: set[str] = set()
        for rel, info in self.modules.items():
            for cls in [n for n in info.tree.body if isinstance(n, ast.ClassDef)]:
                for n in ast.walk(cls):
                    if (
                        isinstance(n, ast.Attribute)
                        and isinstance(n.ctx, (ast.Store, ast.Del))
                        and isinstance(n.value, ast.Name)
                        and n.value.id == "self"
                    ):
                        self_stores[(rel, cls.name, n.attr)] += 1
            parents = parent_map(info.tree)
            for n in ast.walk(info.tree):
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
                    if isinstance(n.value, ast.Name) and n.value.id == "self":
                        continue
                    p = parents.get(id(n))
                    value = (
                        p.value
                        if isinstance(p, (ast.Assign, ast.AnnAssign))
                        and getattr(p, "value", None) is not None
                        and (not isinstance(p, ast.Assign) or p.targets == [n])
                        else None
                    )
                    other_stores.setdefault(n.attr, []).append((rel, value))
                elif (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in ("setattr", "delattr")
                    and len(n.args) >= 2
                    and isinstance(n.args[1], ast.Constant)
                ):
                    setattr_strings.add(str(n.args[1].value))
        out: dict[tuple[str, str], dict[str, Any]] = {}
        sites = self._resolved_call_sites()
        self._sites_cache = sites
        for rel, info in self.modules.items():
            ctx = Context(rel, self)
            menv = self.module_envs.get(rel, {})
            for cls in [n for n in info.tree.body if isinstance(n, ast.ClassDef)]:
                if any(
                    isinstance(m, ast.FunctionDef) and m.name in ("__setattr__", "__getattr__")
                    for m in cls.body
                ):
                    continue
                init = next(
                    (
                        m
                        for m in cls.body
                        if isinstance(m, ast.FunctionDef) and m.name == "__init__"
                    ),
                    None,
                )
                if init is None or not init.args.args or init.args.args[0].arg != "self":
                    continue
                hierarchy = self._class_hierarchy(rel, cls.name)
                init_env = dict(menv)
                init_env.update(
                    self._traced_env(rel, f"{cls.name}.__init__", init, self._record_of(info, init))
                )
                if hierarchy == {(rel, cls.name)}:
                    init_env.update(
                        self._static_env(init, sites.get((rel, cls.name), []), True, ctx)
                    )
                E = Evaluator(init_env, ctx, fn_bound=stores_in(init, include_nested=True))
                consts = {}
                for st in init.body:
                    if isinstance(st, ast.Assign) and len(st.targets) == 1:
                        t, value = st.targets[0], st.value
                    elif isinstance(st, ast.AnnAssign) and st.value is not None:
                        t, value = st.target, st.value
                    else:
                        continue
                    if not (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                    ):
                        continue
                    attr = t.attr
                    if attr in setattr_strings:
                        continue
                    if sum(self_stores[(r, c, attr)] for r, c in hierarchy) != 1:
                        continue
                    v = E.ev(value)
                    if isinstance(v, Known) and not isinstance(v.value, _SIMPLE):
                        continue
                    if v is UNKNOWN:
                        continue
                    if isinstance(v, Known):
                        v = Known(v.value, subst=False)  # ``self.x`` reads better than a literal
                    ok = True
                    for other_rel, other_value in other_stores.get(attr, []):
                        if other_value is None:
                            ok = False
                            break
                        ov = Evaluator(
                            dict(self.module_envs.get(other_rel, {})), Context(other_rel, self)
                        ).ev(other_value)
                        if not (
                            isinstance(v, Known)
                            and isinstance(ov, Known)
                            and ov.value == v.value
                            and type(ov.value) is type(v.value)
                        ):
                            ok = False
                            break
                    if ok:
                        consts[attr] = v
                if consts:
                    out[(rel, cls.name)] = consts
        return out

    def _digest(self) -> str:
        h = hashlib.sha1()
        for rel in sorted(self.modules):
            h.update(ast.dump(self.modules[rel].tree).encode())
        return h.hexdigest()

    # -- dead code ---------------------------------------------------------------------------

    def _references(self) -> tuple[Counter, Counter, Counter]:
        """Resolved symbol references, attribute-name references and identifier strings."""
        symbols: Counter = Counter()
        attrs: Counter = Counter()
        strings: Counter = Counter(self.extra_refs)
        for rel, info in self.modules.items():
            all_node = None
            for st in info.tree.body:
                if (
                    isinstance(st, ast.Assign)
                    and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name)
                    and st.targets[0].id == "__all__"
                ):
                    all_node = st
            skip = {id(n) for n in ast.walk(all_node)} if all_node is not None else set()
            for st in info.tree.body:
                own = (
                    st.name
                    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    else None
                )
                for n in ast.walk(st):
                    if id(n) in skip:
                        continue
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        res = self.resolve(rel, n.id)
                        if res is not None and not (own is not None and res == (rel, own)):
                            symbols[res] += 1
                    elif isinstance(n, ast.Attribute):
                        if isinstance(n.value, ast.Name):
                            alias = self.imports.get(rel, {}).get(n.value.id)
                            if alias and alias[0] is not None and alias[1] is None:
                                symbols[(alias[0], n.attr)] += 1
                                continue
                        attrs[n.attr] += 1
                    elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                        if n.value.isidentifier():
                            strings[n.value] += 1
                    elif isinstance(n, (ast.Global, ast.Nonlocal)):
                        for name in n.names:
                            symbols[(rel, name)] += 1
        for name, count in self.extra_refs.items():
            for rel in self.modules:
                symbols[(rel, name)] += count
        return symbols, attrs, strings

    def remove_dead(self) -> None:
        for _round in range(30):
            self.rebuild_imports()
            symbols, attrs, strings = self._references()
            changed = False
            for info in self.modules.values():
                changed |= self._remove_dead_in(info, symbols, attrs, strings)
            changed |= self._prune_dict_constants()
            if not changed:
                break

    def _prune_dict_constants(self) -> bool:
        """Drop entries of module-level string-keyed dict constants whose key never appears as a
        string anywhere else in the bundle (kernel-source tables keyed by kernel name)."""
        counts: Counter = Counter()
        dict_nodes: list[tuple[ast.Dict, str]] = []
        for rel, info in self.modules.items():
            for st in info.tree.body:
                if (
                    isinstance(st, ast.Assign)
                    and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name)
                    and isinstance(st.value, ast.Dict)
                    and st.value.keys
                    and all(
                        isinstance(k, ast.Constant) and isinstance(k.value, str)
                        for k in st.value.keys
                    )
                ):
                    dict_nodes.append((st.value, rel))
            for n in ast.walk(info.tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    counts[n.value] += 1
        changed = False
        for node, rel in dict_nodes:
            own = Counter(k.value for k in node.keys)
            keep = [
                (k, v)
                for k, v in zip(node.keys, node.values, strict=True)
                if counts[k.value] - own[k.value] > 0
            ]
            if len(keep) != len(node.keys):
                for k, _v in zip(node.keys, node.values, strict=True):
                    if counts[k.value] - own[k.value] <= 0:
                        self.notes.append(f"{rel}: dropped unused table entry {k.value!r}")
                node.keys = [k for k, _ in keep]
                node.values = [v for _, v in keep]
                changed = True
        return changed

    def _remove_dead_in(
        self, info: ModuleInfo, symbols: Counter, attrs: Counter, strings: Counter
    ) -> bool:
        changed = False
        rel = info.rel

        def unreferenced(name: str, as_method: bool) -> bool:
            if strings[name]:
                return False
            if as_method:
                return attrs[name] == 0
            return symbols[(rel, name)] == 0

        def prune_body(body, is_class: bool, top: bool) -> list:
            nonlocal changed
            out = []
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if is_class:
                        removable = (
                            not (n.name.startswith("__") or n.name in IMPLICIT_METHODS)
                            and not executed_in(n, info.executed)
                            and unreferenced(n.name, True)
                        )
                    elif top:
                        removable = unreferenced(n.name, False)
                    else:
                        removable = False
                    if removable:
                        changed = True
                        self.notes.append(
                            f"{rel}: dropped {'method' if is_class else 'function'} {n.name}"
                        )
                        continue
                    if not is_class:
                        n.body = prune_body(n.body, False, False)
                elif isinstance(n, ast.ClassDef):
                    if top and not executed_in(n, info.executed) and unreferenced(n.name, False):
                        changed = True
                        self.notes.append(f"{rel}: dropped class {n.name}")
                        continue
                    n.body = prune_body(n.body, True, False) or [ast.Pass()]
                elif top and isinstance(n, (ast.Assign, ast.AnnAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    flat = [
                        e for t in targets for e in (t.elts if isinstance(t, ast.Tuple) else [t])
                    ]
                    names = [e.id for e in flat if isinstance(e, ast.Name)]
                    if "__all__" in names or (
                        n.value is not None
                        and len(names) == len(flat)
                        and _pure_constant_value(n.value)
                        and all(unreferenced(name, False) for name in names)
                    ):
                        changed = True
                        continue
                elif (
                    hasattr(n, "body")
                    and isinstance(getattr(n, "body", None), list)
                    and not is_class
                ):
                    n.body = prune_body(n.body, False, False) or [ast.Pass()]
                    if getattr(n, "orelse", None):
                        n.orelse = prune_body(n.orelse, False, False)
                    if getattr(n, "finalbody", None):
                        n.finalbody = prune_body(n.finalbody, False, False)
                    for h in getattr(n, "handlers", []) or []:
                        h.body = prune_body(h.body, False, False) or [ast.Pass()]
                out.append(n)
            return out

        info.tree.body = prune_body(info.tree.body, False, True)
        return changed

    # -- unused parameters --------------------------------------------------------------------

    def prune_parameters(self) -> None:
        """Drop parameters no function body reads, rewriting every call site accordingly.

        A call site that passed a dropped parameter positionally has its later positional
        arguments converted to keywords, so nothing shifts. Functions with ``*args``/``**kw``,
        with call sites that unpack, referenced by name in a string, or whose class has
        subclasses in the bundle are left alone.
        """
        self.rebuild_imports()
        sites = self._resolved_call_sites()
        strings = self._string_refs()
        for rel, info in self.modules.items():
            for node in info.tree.body:
                fn, owner = None, None
                if isinstance(node, ast.FunctionDef):
                    fn = node
                    if node.name in strings:
                        continue
                    fsites = sites.get((rel, node.name), [])
                elif isinstance(node, ast.ClassDef):
                    owner = node
                    fn = next(
                        (
                            m
                            for m in node.body
                            if isinstance(m, ast.FunctionDef) and m.name == "__init__"
                        ),
                        None,
                    )
                    if fn is None or node.name in strings:
                        continue
                    if self._class_hierarchy(rel, node.name) != {(rel, node.name)}:
                        continue
                    fsites = sites.get((rel, node.name), [])
                else:
                    continue
                if fn.args.kwarg is not None or fn.args.vararg is not None or fn.args.posonlyargs:
                    continue
                if any(s_["star"] for s_ in fsites):
                    continue
                loads = loads_in(fn)
                positional = list(fn.args.args)
                offset = 1 if owner is not None else 0
                defaults_offset = len(positional) - len(fn.args.defaults)
                drop: set[str] = set()
                for i, a in enumerate(positional):
                    if i < offset or (loads[a.arg] and not _param_overwritten(fn, a.arg)):
                        continue
                    if i < defaults_offset and not fsites:
                        continue  # a required parameter of an externally called function
                    drop.add(a.arg)
                for a in fn.args.kwonlyargs:
                    if not loads[a.arg]:
                        drop.add(a.arg)
                if not drop:
                    continue
                names = [a.arg for a in positional[offset:]]
                for s_ in fsites:
                    call = s_["call"]
                    new_args, new_kw = [], []
                    convert = False
                    for j, arg in enumerate(call.args):
                        pname = names[j] if j < len(names) else None
                        if pname in drop:
                            convert = True
                            continue
                        if convert and pname is not None:
                            new_kw.append(ast.keyword(arg=pname, value=arg))
                        else:
                            new_args.append(arg)
                    new_kw += [k for k in call.keywords if k.arg not in drop]
                    call.args, call.keywords = new_args, new_kw
                fn.args.defaults = [
                    d
                    for a, d in zip(positional[defaults_offset:], fn.args.defaults, strict=True)
                    if a.arg not in drop
                ]
                fn.args.args = [a for a in positional if a.arg not in drop]
                kw = [
                    (a, d)
                    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)
                    if a.arg not in drop
                ]
                fn.args.kwonlyargs = [a for a, _ in kw]
                fn.args.kw_defaults = [d for _, d in kw]
                qual = f"{owner.name}.__init__" if owner is not None else fn.name
                self.notes.append(
                    f"{rel}:{qual}: dropped unused parameter(s) {', '.join(sorted(drop))}"
                )

    # -- output ------------------------------------------------------------------------------

    def signature(self, rel: str, qual: str) -> ast.arguments | None:
        info = self.modules.get(rel)
        if info is None:
            return None
        for lineno, (q, node, _owner) in info.qualname_map().items():
            if q == qual and lineno == node.lineno:
                return node.args
        return None

    def canonicalize_imports(self) -> None:
        """Point ``from .x import name`` at the module that defines ``name``.

        A module that only re-exports a name would lose that import to the unused-import
        cleanup; importing from the defining module keeps every import a real use.
        """
        self.rebuild_imports()

        def origin(rel: str, name: str, depth: int = 0) -> tuple[str, str] | None:
            info = self.modules.get(rel)
            if info is None or depth > 10:
                return None
            if name in module_defined_names(info.tree) and name not in self.imports.get(rel, {}):
                return (rel, name)
            target = self.imports.get(rel, {}).get(name)
            if target is None or target[0] is None or target[1] is None:
                return None
            return origin(target[0], target[1], depth + 1)

        def dotted(rel: str) -> str:
            path = rel[: -len("/__init__.py")] if rel.endswith("/__init__.py") else rel[:-3]
            return path.replace("/", ".")

        for rel, info in self.modules.items():
            new_body = []
            for st in info.tree.body:
                if isinstance(st, ast.ImportFrom) and st.level > 0:
                    pkg = self._module_of(rel, st.level, st.module)
                    moved: dict[str, list[ast.alias]] = {}
                    kept = []
                    for a in st.names:
                        o = origin(pkg, a.name) if pkg else None
                        if o is None or o[0] == pkg:
                            kept.append(a)
                        else:
                            moved.setdefault(o[0], []).append(
                                ast.alias(
                                    name=o[1],
                                    asname=a.asname or (a.name if a.name != o[1] else None),
                                )
                            )
                    if kept:
                        st.names = kept
                        new_body.append(st)
                    for target_rel, aliases in moved.items():
                        new_body.append(
                            ast.ImportFrom(module=dotted(target_rel), names=aliases, level=0)
                        )
                    continue
                new_body.append(st)
            info.tree.body = new_body

    def strip_unused_imports(self) -> None:
        """Drop module-level imports whose bound names are never read in the module."""
        for info in self.modules.values():
            loads = loads_in(info.tree)
            attrs_bases = {
                n.value.id
                for n in ast.walk(info.tree)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            }
            new_body = []
            for st in info.tree.body:
                if isinstance(st, ast.ImportFrom) and st.module != "__future__":
                    st.names = [
                        a
                        for a in st.names
                        if loads[a.asname or a.name] or (a.asname or a.name) in attrs_bases
                    ]
                    if not st.names:
                        continue
                elif isinstance(st, ast.Import):
                    st.names = [
                        a
                        for a in st.names
                        if loads[(a.asname or a.name).split(".")[0]]
                        or (a.asname or a.name).split(".")[0] in attrs_bases
                    ]
                    if not st.names:
                        continue
                new_body.append(st)
            info.tree.body = new_body

    def drop_empty_modules(self) -> None:
        """Remove modules left with nothing but a docstring and imports that nobody imports."""
        while True:
            self.strip_unused_imports()
            self.rebuild_imports()
            imported = set()
            for table in self.imports.values():
                for mod, _sym in table.values():
                    if mod is not None:
                        imported.add(mod)
            gone = []
            for rel, info in self.modules.items():
                if rel.endswith("__init__.py") or rel in imported:
                    continue
                body = [
                    st
                    for st in info.tree.body
                    if not isinstance(st, (ast.Import, ast.ImportFrom))
                    and not (
                        isinstance(st, ast.Expr)
                        and isinstance(st.value, ast.Constant)
                        and isinstance(st.value.value, str)
                    )
                ]
                if not body:
                    gone.append(rel)
            if not gone:
                return
            for rel in gone:
                self.notes.append(f"{rel}: dropped (empty after cleaning)")
                del self.modules[rel]

    def write(self, out_root: Path) -> None:
        self.canonicalize_imports()
        self.drop_empty_modules()
        for rel, info in self.modules.items():
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(unparse(info.tree) + "\n", encoding="utf-8")


class _Unparser(ast._Unparser):
    """``ast.unparse`` with triple-quoted multi-line strings and bare tuples in assignments."""

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, str) and "\n" in v and '"""' not in v and not v.endswith('"'):
            if "\\" not in v:
                self.write('"""' + v + '"""')
                return
            if not v.endswith("\\"):
                self.write('r"""' + v + '"""')
                return
        super().visit_Constant(node)

    def _bare(self, node):
        if isinstance(node, ast.Tuple) and node.elts:
            self.interleave(lambda: self.write(", "), self.traverse, node.elts)
            if len(node.elts) == 1:
                self.write(",")
        else:
            self.traverse(node)

    def visit_Assign(self, node):
        if not any(isinstance(t, ast.Tuple) for t in node.targets):
            return super().visit_Assign(node)
        self.fill()
        for target in node.targets:
            self._bare(target)
            self.write(" = ")
        self._bare(node.value)

    def visit_Return(self, node):
        self.fill("return")
        if node.value:
            self.write(" ")
            self._bare(node.value)


def unparse(tree: ast.AST) -> str:
    return _Unparser().visit(tree)


# --------------------------------------------------------------------------------------------
# 6. The generated submission.py
# --------------------------------------------------------------------------------------------

_HEADER = '''"""RealPDE Track 1 submission: a forecaster with an interval head and a second-moment
correction head.

Model construction, checkpoint loading and the warm-up run at import time; ``predict`` runs
the forecaster on the given windows and returns the prediction with its interval bounds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
'''


def _params_of(args: ast.arguments | None) -> set[str] | None:
    if args is None:
        return None
    return {a.arg for a in _all_args(args)}


def _call_ok(cleaner: Cleaner, rel: str, qual: str, kwargs: list[str], npos: int) -> None:
    """Refuse to emit a call the cleaned signature cannot accept."""
    args = cleaner.signature(rel, qual)
    if args is None:
        raise ValueError(f"{rel}:{qual} is missing from the cleaned package")
    names = {a.arg for a in _all_args(args)}
    positional = [a.arg for a in args.posonlyargs + args.args]
    bad = [k for k in kwargs if k not in names and args.kwarg is None]
    if bad or npos > len(positional):
        raise ValueError(f"{rel}:{qual} does not accept {bad or npos} after cleaning")


def generate_submission(staged_text: str, consts: dict[str, Any], cleaner: Cleaner) -> str:
    """Emit the clean submission.py for ``consts`` against the cleaned package."""
    tree = ast.parse(staged_text)
    get_predict = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_get_predict"),
        None,
    )
    if get_predict is None:
        raise ValueError("the staged submission.py has no _get_predict()")
    mp_kwargs: list[ast.keyword] = []
    for n in ast.walk(get_predict):
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "mp_kwargs"
            and isinstance(n.value, ast.Call)
        ):
            mp_kwargs = list(n.value.keywords)
    if not mp_kwargs:
        for n in ast.walk(get_predict):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "make_predict"
            ):
                mp_kwargs = [k for k in n.keywords if k.arg is not None]
    if not mp_kwargs:
        raise ValueError("could not find the make_predict keyword arguments in _get_predict()")

    compile_mode = consts.get("COMPILE_MODE", "")
    compile_dynamic = bool(consts.get("COMPILE_DYNAMIC", False))
    cudagraph = consts.get("CUDAGRAPH_BUCKETS", "")
    re_corners = bool(consts.get("RE_CORNERS", False))
    geom = tuple(consts.get("GEOM_CHANNELS", ()))
    flow = tuple(consts.get("FLOW_CHANNELS", ()))
    tke_head = consts.get("TKE_HEAD_NAME", "")
    rule = consts.get("WIDTH_RULE_NAME", "")
    anchor = consts.get("LEAD_ANCHOR_NAME", "")
    warmup = tuple(consts["WARMUP_SIZES"])
    batch = int(consts["_BATCH"])
    if warmup[-1] != batch:
        raise ValueError("WARMUP_SIZES must end with _BATCH")

    mp_params = _params_of(cleaner.signature("rpde_t1/models/predictor.py", "make_predict"))
    lfi_params = _params_of(cleaner.signature("rpde_t1/models/factory.py", "load_for_inference"))
    if mp_params is None or lfi_params is None:
        raise ValueError("make_predict or load_for_inference is missing from the cleaned package")
    kept_kwargs = [k for k in mp_kwargs if k.arg in mp_params]
    dropped_kwargs = [k.arg for k in mp_kwargs if k.arg not in mp_params]
    kw_text = ", ".join(ast.unparse(k.value) for k in kept_kwargs)
    used = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", kw_text))
    local_names = {"device", "config", "rule", "head", "head_tau", "anchor", "model"}
    keywords = {"os", "_HERE", "None", "True", "False", "or", "and", "not", "path", "join", "get"}
    unknown = used - set(consts) - local_names - keywords
    if unknown:
        raise ValueError(
            f"make_predict kwargs reference names the generator does not define: {sorted(unknown)}"
        )

    # -- _get_predict -----------------------------------------------------------------------
    body = [
        "def _get_predict():",
        '    """Build the predict closure once, at import."""',
        '    if _state["fn"] is None:',
    ]
    need_device = "device" in used or "device" in lfi_params or bool(tke_head) or bool(cudagraph)
    if need_device:
        body.append('        device = "cuda" if torch.cuda.is_available() else "cpu"')
    lfi_kw = []
    if "kit_dir" in lfi_params:
        lfi_kw.append("kit_dir=_HERE")
    if "device" in lfi_params:
        lfi_kw.append("device=device")
    _call_ok(
        cleaner,
        "rpde_t1/models/factory.py",
        "load_for_inference",
        [k.split("=")[0] for k in lfi_kw],
        1,
    )
    extra = "".join(", " + k for k in lfi_kw)
    body.append(f"        model, config = load_for_inference(_CKPT_PATH{extra})")
    body.append('        if RETURN_BOUNDS and not config.get("bounds", False):')
    body.append(
        '            raise RuntimeError("RETURN_BOUNDS is on but the checkpoint has no bound head")'
    )
    if tke_head and "head" in used:
        _call_ok(cleaner, "rpde_t1/models/tke_head.py", "load_head", ["device"], 1)
        body.append("        head = load_head(os.path.join(_HERE, TKE_HEAD_NAME), device=device)")
    if tke_head and "head_tau" in used:
        _call_ok(cleaner, "rpde_t1/models/tke_head.py", "head_uses_tau", [], 1)
        body.append("        head_tau = head_uses_tau(os.path.join(_HERE, TKE_HEAD_NAME))")
    if rule and "rule" in used:
        _call_ok(cleaner, "rpde_t1/models/width_rule.py", "WidthRule.from_json", [], 2)
        body.append("        rule = WidthRule.from_json(Path(_HERE) / WIDTH_RULE_NAME)")
    if anchor and "anchor" in used:
        body.append("        anchor = LeadAnchor.from_json(Path(_HERE) / LEAD_ANCHOR_NAME)")
    if cudagraph:
        body.append("        mp_kwargs = dict(")
        for k in kept_kwargs:
            body.append(f"            {k.arg}={ast.unparse(k.value)},")
        body.append("        )")
        body.append(
            "        model = wrap_for_inference(model, CUDAGRAPH_BUCKETS, _BATCH, PRECISION, "
            "make_predict, mp_kwargs, device=device)"
        )
        call = "make_predict(model, **mp_kwargs)"
    else:
        call = (
            "make_predict(\n            model,\n"
            + "".join(f"            {k.arg}={ast.unparse(k.value)},\n" for k in kept_kwargs)
            + "        )"
        )
    if compile_mode and compile_dynamic:
        body.append("        _compile = torch.compile")
        body.append(
            '        torch.compile = lambda m, **kw: _compile(m, **{**kw, "dynamic": True})'
        )
        body.append("        try:")
        body.append(f'            _state["fn"] = {call}')
        body.append("        finally:")
        body.append("            torch.compile = _compile")
    else:
        body.append(f'        _state["fn"] = {call}')
    body.append('    return _state["fn"]')
    get_predict_text = "\n".join(body) + "\n"

    # -- import-time work -------------------------------------------------------------------
    tail = []
    if re_corners:
        tail.append("_features._re_override = _features.corners3_reynolds\n")
    if tke_head or consts.get("TKE_BLEND") or geom:
        tail.append("warm_geometry_tables()\n")
    warm = []
    if compile_mode:
        warm.append("_warmed = True")
        warm.append("for _size in WARMUP_SIZES:")
        warm.append("    _spin = DEFAULT_SPIN_S if _size == WARMUP_SIZES[-1] else 0.0")
        warm.append(
            "    _warmed = warm_up_predict(_get_predict(), _size, reps=3, spin_s=_spin) and _warmed"
        )
        warm.append("if not _warmed:")
        warm.append("    _fn = _get_predict()")
        warm.append('    if hasattr(_fn, "demote_to_eager"):')
        warm.append("        _fn.demote_to_eager()")
        _call_ok(cleaner, "rpde_t1/models/predictor.py", "warm_up_predict", ["reps", "spin_s"], 2)
    else:
        warm.append("for _size in WARMUP_SIZES:")
        warm.append("    _spin = DEFAULT_SPIN_S if _size == WARMUP_SIZES[-1] else 0.0")
        warm.append("    warm_up_predict(_get_predict(), _size, spin_s=_spin)")
        _call_ok(cleaner, "rpde_t1/models/predictor.py", "warm_up_predict", ["spin_s"], 2)
    tail.append("\n".join(warm) + "\n")

    pred = ["def predict(input_array, metadata=None):"]
    pred.append(
        '    """Track 1 API: (N, T_in, H, W, C) raw fields -> dict(prediction, lower, upper)."""'
    )
    pred.append("    x = np.asarray(input_array, dtype=np.float32)")
    pred.append("    fn = _get_predict()")
    if compile_mode and compile_dynamic:
        pred.append('    if x.shape[0] == 1 and hasattr(fn, "demote_to_eager"):')
        pred.append("        fn.demote_to_eager()")
        pred.append("        try:")
        pred.append("            out = fn(x, metadata)")
        pred.append("        finally:")
        pred.append("            fn.restore_compiled()")
        pred.append("    else:")
        pred.append("        out = fn(x, metadata)")
    else:
        pred.append("    out = fn(x, metadata)")
    pred.append("    if RETURN_BOUNDS and not isinstance(out, dict):")
    pred.append('        raise RuntimeError("bounds requested but predict returned a bare array")')
    pred.append("    return out")
    tail.append("\n".join(pred) + "\n")
    code_text = get_predict_text + "\n".join(tail)

    # -- constants: only the ones the emitted code reads ------------------------------------
    referenced = set(re.findall(r"\b_?[A-Z][A-Z0-9_]*\b", code_text)) | {"_BATCH"}
    const_lines = [f'_CKPT_PATH = os.path.join(_HERE, "{consts["_CKPT_NAME"]}")']
    first = ["_BATCH", "RETURN_BOUNDS", "PRECISION"]
    order = first + sorted(set(consts) - set(first))
    for name in order:
        if name not in consts or name in ("_CKPT_NAME", "WARMUP_SIZES") or name not in referenced:
            continue
        if name in ("GEOM_CHANNELS", "FLOW_CHANNELS"):
            fn = "parse_geom_channels" if name == "GEOM_CHANNELS" else "parse_flow_channels"
            const_lines.append(f'{name} = {fn}("{",".join(consts[name])}")')
        else:
            const_lines.append(f"{name} = {consts[name]!r}")
    const_lines.append(f"WARMUP_SIZES = {_warmup_literal(warmup, batch)}")

    # -- imports ------------------------------------------------------------------------------
    imports = []
    if compile_mode:
        imports.append('os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"\n')
    imports.append("import torch  # noqa: E402\n")
    if geom or flow:
        imports.append("from rpde_t1.features import parse_flow_channels  # noqa: E402")
        imports.append(
            "from rpde_t1.geometry import parse_geom_channels, warm_geometry_tables  # noqa: E402"
        )
    elif "warm_geometry_tables()" in code_text:
        imports.append("from rpde_t1.geometry import warm_geometry_tables  # noqa: E402")
    imports.append("from rpde_t1.models.factory import load_for_inference  # noqa: E402")
    imports.append(
        "from rpde_t1.models.predictor import DEFAULT_SPIN_S, make_predict, warm_up_predict"
        "  # noqa: E402"
    )
    tke_imports = [n for n in ("head_uses_tau", "load_head") if n in code_text]
    if tke_imports:
        imports.append(
            f"from rpde_t1.models.tke_head import {', '.join(tke_imports)}  # noqa: E402"
        )
    if "WidthRule" in code_text:
        imports.append("from rpde_t1.models.width_rule import WidthRule  # noqa: E402")
    if "LeadAnchor" in code_text:
        imports.append("from rpde_t1.models.lead_anchor import LeadAnchor  # noqa: E402")
    if cudagraph:
        imports.append("from rpde_t1.models.cudagraph import wrap_for_inference  # noqa: E402")
    if re_corners:
        imports.append("from rpde_t1 import features as _features  # noqa: E402")

    header = _HEADER
    if "Path(" not in code_text:
        header = header.replace("from pathlib import Path\n", "")
    text = "\n\n".join(
        [
            header,
            "\n".join(imports) + "\n",
            "\n".join(const_lines) + "\n",
            '_state = {"fn": None}\n',
            code_text,
        ]
    )
    if dropped_kwargs:
        cleaner.notes.append(
            "submission.py: make_predict kwargs folded into the package and dropped: "
            + ", ".join(dropped_kwargs)
        )
    return text


def _warmup_literal(sizes: tuple[int, ...], batch: int) -> str:
    body = list(sizes[:-1])
    # Compress runs into range() for readability when long.
    if len(body) >= 6 and body == list(range(body[0], body[-1] + 1)):
        return f"tuple(range({body[0]}, {body[-1] + 1})) + (_BATCH,)"
    for split in range(len(body), 0, -1):
        head = body[:split]
        if len(head) >= 6 and head == list(range(head[0], head[-1] + 1)):
            tail = "".join(f"{n}, " for n in body[split:])
            return f"tuple(range({head[0]}, {head[-1] + 1})) + ({tail}_BATCH)"
    return "(" + "".join(f"{n}, " for n in body) + "_BATCH)"


# --------------------------------------------------------------------------------------------
# 7. The isolated driver: trace / run predict on saved inputs
# --------------------------------------------------------------------------------------------

_DRIVER_SRC = r"""
import builtins, io, json, os, sys, time
ARGS = json.loads(sys.argv[1])
ROOT = ARGS["root"]
sys.path.insert(0, ROOT)
{blocker}

_opened = set()
_real_open = builtins.open


class _RecOpen:
    # A callable object, not a function: stored on a class (pathlib's accessor) it must not
    # bind as a method.
    def __call__(self, file, *a, **k):
        try:
            p = os.path.abspath(os.fspath(file))
            if p.startswith(ROOT):
                _opened.add(p)
        except Exception:
            pass
        return _real_open(file, *a, **k)


_rec_open = _RecOpen()
builtins.open = _rec_open
io.open = _rec_open

executed = {{}}
calls = {{}}
phase = {{"v": "import"}}


def _ser(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return {{"k": "lit", "v": v}}
    if isinstance(v, (tuple, list)):
        items = [_ser(x) for x in v]
        if all(i["k"] == "lit" for i in items):
            return {{"k": "lit", "v": items, "t": "tuple" if isinstance(v, tuple) else "list"}}
        return {{"k": "opaque"}}
    if isinstance(v, dict):
        if all(isinstance(k, str) for k in v):
            items = {{k: _ser(x) for k, x in v.items()}}
            if all(i["k"] == "lit" for i in items.values()):
                return {{"k": "lit", "v": items, "t": "dict"}}
        return {{"k": "opaque"}}
    t = type(v)
    if t.__module__ == "torch" and t.__name__ == "dtype":
        return {{"k": "sym", "v": str(v)}}
    return {{"k": "opaque"}}


def _tracer(frame, event, arg):
    if event != "call":
        return None
    code = frame.f_code
    fn = code.co_filename
    if not fn.startswith(ROOT):
        return None
    key = fn + ":" + str(code.co_firstlineno)
    rec = calls.get(key)
    if rec is None:
        rec = calls[key] = {{"name": code.co_name, "phases": [], "args": {{}}, "n": 0}}
    rec["n"] += 1
    if phase["v"] not in rec["phases"]:
        rec["phases"].append(phase["v"])
    nargs = code.co_argcount + code.co_kwonlyargcount
    loc = frame.f_locals
    for name in code.co_varnames[:nargs]:
        if name.startswith(".") or name not in loc:
            continue
        s = _ser(loc[name])
        lst = rec["args"].setdefault(name, [])
        if s not in lst and len(lst) < 8:
            lst.append(s)
    lines = executed.get(fn)
    if lines is None:
        lines = executed[fn] = set()

    def _local(frame, event, arg):
        if event == "line":
            lines.add(frame.f_lineno)
        return _local

    return _local


if ARGS["trace"]:
    sys.settrace(_tracer)

import numpy as np

t0 = time.perf_counter()
import submission
import_s = time.perf_counter() - t0
phase["v"] = "predict"

results = {{"import_s": import_s, "calls": []}}
os.makedirs(ARGS["out_dir"], exist_ok=True)
for split, path in ARGS["inputs"].items():
    x = np.load(path)
    for size in ARGS["sizes"]:
        n = x.shape[0] if size == "all" else int(size)
        if n > x.shape[0]:
            continue
        preds, los, his = [], [], []
        secs = 0.0
        for start in range(0, x.shape[0], n):
            xb = np.ascontiguousarray(x[start : start + n])
            t0 = time.perf_counter()
            out = submission.predict(xb)
            secs += time.perf_counter() - t0
            p = np.asarray(out["prediction"], dtype=np.float32)
            lo = np.asarray(out["lower"], dtype=np.float32)
            hi = np.asarray(out["upper"], dtype=np.float32)
            results["calls"].append(
                {{
                    "split": split,
                    "size": size,
                    "n": int(xb.shape[0]),
                    "shape_ok": p.shape == (xb.shape[0], 20, 32, 64, 3),
                    "dtype": str(p.dtype),
                    "finite": bool(
                        np.isfinite(p).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
                    ),
                    "p_zero": bool(
                        (p[..., 2] == 0).all()
                        and (lo[..., 2] == 0).all()
                        and (hi[..., 2] == 0).all()
                    ),
                    "ordered": bool((lo <= hi).all()),
                }}
            )
            preds.append(p)
            los.append(lo)
            his.append(hi)
        np.savez(
            os.path.join(ARGS["out_dir"], "%s_%s.npz" % (split, size)),
            prediction=np.concatenate(preds),
            lower=np.concatenate(los),
            upper=np.concatenate(his),
        )
        results.setdefault("seconds", {{}})["%s_%s" % (split, size)] = secs

sys.settrace(None)
if ARGS["trace"]:
    mods = sorted(
        {{
            m.__file__
            for m in list(sys.modules.values())
            if getattr(m, "__file__", None) and m.__file__.startswith(ROOT)
        }}
    )
    trace = {{
        "modules": mods,
        "executed": {{k: sorted(v) for k, v in executed.items()}},
        "calls": calls,
        "opened": sorted(_opened),
    }}
    with _real_open(ARGS["trace_out"], "w") as f:
        json.dump(trace, f)
with _real_open(os.path.join(ARGS["out_dir"], "results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("DRIVER_OK")
"""


_DRIVER_HEAD = _DRIVER_SRC.split("import numpy as np\n")[0]

_DRIVER_T2_TAIL = r"""
import numpy as np
import torch

t0 = time.perf_counter()
import submission
model = submission.get_ttt_model(ROOT, "cuda")
import_s = time.perf_counter() - t0
phase["v"] = "predict"


def to_normalized(x):
    return np.ascontiguousarray(x, dtype=np.float32)  # the parent saved evaluator-space arrays


results = {{"import_s": import_s, "calls": []}}
os.makedirs(ARGS["out_dir"], exist_ok=True)


def _run_stream(name, trajectories):
    preds, los, his = [], [], []
    secs = 0.0
    for windows, targets in trajectories:
        model.reset_ttt_state()
        prev = None
        for i in range(len(windows)):
            xn = torch.from_numpy(to_normalized(windows[i][None])).to("cuda")
            t0 = time.perf_counter()
            out, info = model.ttt_step(xn, prev)
            torch.cuda.synchronize()
            secs += time.perf_counter() - t0
            p = torch.as_tensor(out).detach().float().cpu().numpy()
            lo = torch.as_tensor(info["lower"]).detach().float().cpu().numpy()
            hi = torch.as_tensor(info["upper"]).detach().float().cpu().numpy()
            results["calls"].append(
                {{
                    "split": name,
                    "n": 1,
                    "shape_ok": tuple(p.shape) == (1, 20, 32, 64, 3),
                    "dtype": str(p.dtype),
                    "finite": bool(
                        np.isfinite(p).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
                    ),
                    "ordered": bool((lo <= hi).all()),
                    "adapt_loss_ok": info.get("adapt_loss") is None
                    or isinstance(info.get("adapt_loss"), float),
                }}
            )
            preds.append(p)
            los.append(lo)
            his.append(hi)
            if targets is not None:
                prev = torch.from_numpy(to_normalized(targets[i][None])).to("cuda")
    np.savez(
        os.path.join(ARGS["out_dir"], "%s_stream.npz" % name),
        prediction=np.concatenate(preds),
        lower=np.concatenate(los),
        upper=np.concatenate(his),
    )
    results.setdefault("seconds", {{}})[name] = secs


for split, spec in ARGS["streams"].items():
    x = np.load(spec["inputs"])
    y = np.load(spec["targets"]) if spec.get("targets") else None
    trajs = [(x[g], y[g] if y is not None else None) for g in spec["groups"]]
    _run_stream(split, trajs)

sys.settrace(None)
if ARGS["trace"]:
    mods = sorted(
        {{
            m.__file__
            for m in list(sys.modules.values())
            if getattr(m, "__file__", None) and m.__file__.startswith(ROOT)
        }}
    )
    trace = {{
        "modules": mods,
        "executed": {{k: sorted(v) for k, v in executed.items()}},
        "calls": calls,
        "opened": sorted(_opened),
    }}
    with _real_open(ARGS["trace_out"], "w") as f:
        json.dump(trace, f)
with _real_open(os.path.join(ARGS["out_dir"], "results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("DRIVER_OK")
"""


def run_driver(
    bundle_root: Path,
    inputs: dict[str, Path],
    sizes: list[str],
    out_dir: Path,
    trace_out: Path | None,
    work_dir: Path,
    gpu_log: str,
    use_gpu_lock: bool,
    track2: dict | None = None,
) -> dict:
    """Run the isolated driver against ``bundle_root`` (through gpu.sh unless told not to).

    ``track2`` switches to the streaming driver: ``{"streams": {name: {inputs, targets,
    groups}}, "synthetic_traj": n, "synthetic_steps": n}``.
    """
    args = {
        "root": str(bundle_root),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "sizes": sizes,
        "out_dir": str(out_dir),
        "trace": trace_out is not None,
        "trace_out": str(trace_out) if trace_out else "",
    }
    if track2 is not None:
        args.update(track2)
        blocker = _BLOCKER_SRC.format(
            blocked=list(MISSING_IN_EVAL_IMAGE_T2), vendored=[], root=str(bundle_root)
        )
        src = (_DRIVER_HEAD + _DRIVER_T2_TAIL).format(blocker=blocker)
    else:
        blocker = _BLOCKER_SRC.format(
            blocked=list(MISSING_IN_EVAL_IMAGE), vendored=["einops"], root=str(bundle_root)
        )
        src = _DRIVER_SRC.format(blocker=blocker)
    script = work_dir / f"driver_{gpu_log}.py"
    script.write_text(src, encoding="utf-8")
    cmd = [sys.executable, "-I", str(script), json.dumps(args)]
    if use_gpu_lock:
        cmd = [str(GPU_SH), gpu_log] + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work_dir))
    if "DRIVER_OK" not in proc.stdout:
        raise RuntimeError(
            f"driver failed (exit {proc.returncode})\n--- stdout ---\n{proc.stdout[-6000:]}\n"
            f"--- stderr ---\n{proc.stderr[-6000:]}"
        )
    return json.loads((out_dir / "results.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# 8. Verification helpers
# --------------------------------------------------------------------------------------------

TEXT_SUFFIXES = {".py", ".json", ".txt", ".md", ".cfg", ".toml", ".yaml", ".yml", ".ini", ""}


def forbidden_scan(root: Path) -> list[tuple[str, int, str]]:
    hits = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN_RE.search(line):
                hits.append((str(path.relative_to(root)), i, line.strip()[:120]))
    return hits


def _fmt_count(v: int | None, present: bool) -> str:
    return "-" if not present else ("bin" if v is None else str(v))


def member_listing(root: Path) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(root))
        if path.suffix in (".py", ".json", ".txt", ".md"):
            try:
                out[rel] = len(path.read_text(encoding="utf-8").splitlines())
                continue
            except UnicodeDecodeError:
                pass
        out[rel] = None
    return out


def compare_outputs(ref_dir: Path, new_dir: Path) -> list[dict]:
    import numpy as np

    rows = []
    for ref_file in sorted(ref_dir.glob("*.npz")):
        new_file = new_dir / ref_file.name
        row = {"case": ref_file.stem, "identical": False}
        if not new_file.exists():
            row["error"] = "missing"
            rows.append(row)
            continue
        with np.load(ref_file) as a, np.load(new_file) as b:
            ok = True
            for key in ("prediction", "lower", "upper"):
                same = np.array_equal(a[key], b[key])
                row[key] = bool(same)
                if not same:
                    ok = False
                    d = np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))
                    row[key + "_max_abs_diff"] = float(np.nanmax(d))
                    row[key + "_n_diff"] = int((a[key] != b[key]).sum())
            row["identical"] = ok
        rows.append(row)
    return rows


def cpu_import_smoke(root: Path, modules: list[str]) -> None:
    """Import every kept module in an isolated interpreter (no GPU, no model build)."""
    names = [m[:-3].replace("/", ".") for m in modules if m != "submission.py"]
    body = (
        f"import sys\nsys.path.insert(0, {str(root)!r})\nimport importlib\n"
        f"for name in {names!r}:\n    importlib.import_module(name)\nprint('IMPORT_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", body], capture_output=True, text=True, cwd=str(root.parent)
    )
    if "IMPORT_OK" not in proc.stdout:
        raise RuntimeError(f"cleaned package fails to import:\n{proc.stderr[-4000:]}")


def source_digest(root: Path) -> str:
    h = hashlib.sha1()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# 9. Main
# --------------------------------------------------------------------------------------------


def prepare_inputs(work_dir: Path, splits: list[str]) -> dict[str, Path]:
    import numpy as np

    from rpde_t1.score.harness import load_eval_set

    out = {}
    arrays = []
    for split in splits:
        dst = work_dir / f"inputs_{split}.npy"
        if not dst.exists():
            x, _t, _m = load_eval_set(split)
            np.save(dst, np.ascontiguousarray(np.asarray(x, dtype=np.float32)))
        out[split] = dst
        arrays.append(np.load(dst, mmap_mode="r"))
    if len(arrays) > 1:
        dst = work_dir / "inputs_concat.npy"
        if not dst.exists():
            np.save(dst, np.concatenate([np.asarray(a) for a in arrays]))
        out["concat"] = dst
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("staged", type=Path, help="staged bundle directory (submissions/<name>/)")
    ap.add_argument("--name", required=True, help="output bundle name")
    ap.add_argument("--out-root", type=Path, default=REPO / "submissions")
    ap.add_argument(
        "--work-dir", type=Path, default=None, help="scratch directory (inputs, traces, outputs)"
    )
    ap.add_argument("--splits", default="val_extrap,hard5")
    ap.add_argument("--sizes", default="1,12,13,all")
    ap.add_argument(
        "--reuse-trace", action="store_true", help="reuse the work dir's trace of this source"
    )
    ap.add_argument("--skip-verify", action="store_true", help="stop after writing the zip")
    ap.add_argument(
        "--no-gpu-lock",
        action="store_true",
        help="run GPU children directly (caller holds the lock)",
    )
    ap.add_argument(
        "--track2",
        action="store_true",
        help="the bundle implements the Track-2 streaming contract (get_ttt_model): trace and "
        "verify it as a stream and clean submission.py in place",
    )
    ap.add_argument(
        "--reference-eval-log",
        type=Path,
        default=None,
        help="Track 2: a t2_image_local_eval log of the source bundle whose five subscores "
        "the cleaned bundle must reproduce",
    )
    ap.add_argument(
        "--t2-eval-stream",
        type=Path,
        default=None,
        help="Track 2: run the kit's local_eval.py on this stream directory in this interpreter "
        "instead of inside the pinned image (for a run that is already inside the evaluation "
        "image, e.g. the Decision-phase training container); the pass rule is unchanged",
    )
    args = ap.parse_args(argv)
    if args.track2:
        return main_track2(args)

    staged = args.staged.resolve()
    if not (staged / "submission.py").is_file():
        ap.error(f"{staged} has no submission.py")
    out_dir = (args.out_root / args.name).resolve()
    zip_path = out_dir.with_suffix(".zip")
    if out_dir.exists() and not (out_dir / ".clean_bundle").exists():
        ap.error(f"{out_dir} exists and was not written by this tool; choose another --name")
    work = (args.work_dir or (args.out_root / ".clean_bundle_work" / args.name)).resolve()
    work.mkdir(parents=True, exist_ok=True)
    splits = [s for s in args.splits.split(",") if s]
    sizes = [s for s in args.sizes.split(",") if s]
    lock = not args.no_gpu_lock

    print(f"source bundle : {staged}")
    print(f"output bundle : {out_dir}  ->  {zip_path}")
    print(f"work dir      : {work}")

    # 1. inputs and the trace of the source bundle (GPU run 1)
    inputs = prepare_inputs(work, splits)
    digest = source_digest(staged)
    trace_path = work / "trace.json"
    ref_out = work / "ref_outputs"
    stamp = work / "trace.digest"
    if args.reuse_trace and trace_path.exists() and stamp.exists() and stamp.read_text() == digest:
        print("trace         : reused")
    else:
        print("trace         : running the source bundle under the GPU lock (this may queue) ...")
        ref_res = run_driver(
            staged,
            inputs,
            sizes,
            ref_out,
            trace_path,
            work,
            f"clean_bundle_trace_{args.name}",
            lock,
        )
        stamp.write_text(digest)
        print(f"trace         : done, import {ref_res['import_s']:.1f} s")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    # 2. clean
    staged_text = (staged / "submission.py").read_text(encoding="utf-8")
    consts = parse_constants(staged_text)
    protected = {"_re_override"} if consts.get("RE_CORNERS") else set()
    cleaner = Cleaner(staged, trace, protected_globals=protected)
    cleaner.select()
    named = {consts["_CKPT_NAME"]} | {
        v for k, v in consts.items() if k.endswith("_NAME") and isinstance(v, str) and v
    }
    data_files = sorted(set(cleaner.data_files()) | {n for n in named if (staged / n).is_file()})
    # Names the generated submission.py may call are protected while cleaning; once it is
    # generated, only the names it actually uses stay protected and dead code is swept again.
    cleaner.extra_refs.update(
        [
            "load_for_inference",
            "make_predict",
            "warm_up_predict",
            "DEFAULT_SPIN_S",
            "warm_geometry_tables",
            "load_head",
            "head_uses_tau",
            "WidthRule",
            "from_json",
            "LeadAnchor",
            "wrap_for_inference",
            "parse_geom_channels",
            "parse_flow_channels",
            "corners3_reynolds",
            "_re_override",
        ]
    )
    cleaner.run()
    submission_text = generate_submission(staged_text, consts, cleaner)
    cleaner.extra_refs = Counter(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", submission_text))
    cleaner.remove_dead()
    submission_text = generate_submission(staged_text, consts, cleaner)
    ast.parse(submission_text)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / ".clean_bundle").write_text(f"source={staged}\n")
    cleaner.write(out_dir)
    (out_dir / "submission.py").write_text(submission_text, encoding="utf-8")
    for rel in data_files:
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged / rel, dst)
    py_files = [str(p) for p in out_dir.rglob("*.py")]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "F401,I",
            "--fix",
            "--quiet",
            "--no-cache",
            *py_files,
        ],
        cwd=str(REPO),
        check=False,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--quiet", "--no-cache", *py_files],
        cwd=str(REPO),
        check=True,
        capture_output=True,
    )
    cpu_import_smoke(out_dir, sorted(cleaner.modules))

    # Listing and scans, before zipping (the marker file is excluded from the zip).
    before = member_listing(staged)
    after = {k: v for k, v in member_listing(out_dir).items() if k != ".clean_bundle"}
    hits = forbidden_scan(out_dir)
    (out_dir / ".clean_bundle").unlink()
    bundle = zip_bundle(out_dir, zip_path)
    (out_dir / ".clean_bundle").write_text(f"source={staged}\n")

    print("\n== members (lines) before -> after ==")
    for rel in sorted(set(before) | set(after)):
        b, a = before.get(rel), after.get(rel)
        marker = "" if rel in after else "   (dropped)"
        print(
            f"  {rel:70s} {_fmt_count(b, rel in before):>6} -> "
            f"{_fmt_count(a, rel in after):>6}{marker}"
        )
    tot_b = sum(v for v in before.values() if v)
    tot_a = sum(v for v in after.values() if v)
    print(f"  members {len(before)} -> {len(after)}; text lines {tot_b} -> {tot_a}")
    print(
        f"  extracted {bundle.extracted_mb:.1f} MB (cap {MAX_EXTRACTED_BYTES / 1024 / 1024:.0f}), "
        f"zip {bundle.zip_mb:.1f} MB"
    )
    print("\n== cleaning notes ==")
    for n in cleaner.notes:
        print("  " + n)
    print("\n== forbidden-word scan ==")
    if hits:
        for rel, ln, line in hits:
            print(f"  {rel}:{ln}: {line}")
    else:
        print("  clean")
    if args.skip_verify:
        print("\nverification skipped (--skip-verify)")
        return 1 if hits else 0

    # 3. verify (GPU run 2): clean-room, cleaned outputs, determinism control, daemon check
    print("\nverify        : running under the GPU lock (this may queue) ...")
    verify_json = work / "verify.json"
    vargs = {
        "zip": str(zip_path),
        "staged_src": str(staged),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "sizes": sizes,
        "work": str(work),
        "out": str(verify_json),
    }
    cmd = [sys.executable, str(Path(__file__).resolve()), "--phase-verify", json.dumps(vargs)]
    if lock:
        cmd = [str(GPU_SH), f"clean_bundle_verify_{args.name}"] + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if not verify_json.exists():
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit("verification phase failed")
    v = json.loads(verify_json.read_text(encoding="utf-8"))

    ok = not hits
    print("\n== clean-room run (extracted zip, python -I, eval-image-missing packages blocked) ==")
    cr = v["clean_room"]
    print(
        f"  shape_ok={cr['shape_ok']} dtype={cr['dtype']} finite={cr['finite']} "
        f"p_zero={cr['p_channel_zero']} keys={cr['keys']} seconds={cr['seconds']:.2f}"
    )
    ok &= bool(cr["shape_ok"] and cr["finite"] and cr["p_channel_zero"])
    print("\n== bit-identity against the source bundle ==")
    rows = compare_outputs(ref_out, work / "new_outputs")
    for r in rows:
        flag = "IDENTICAL" if r["identical"] else "DIFFERENT"
        extra = (
            ""
            if r["identical"]
            else " " + json.dumps({k: v for k, v in r.items() if k not in ("case", "identical")})
        )
        print(f"  {r['case']:24s} {flag}{extra}")
        ok &= r["identical"]
    calls = json.loads((work / "new_outputs" / "results.json").read_text())["calls"]
    bad = [c for c in calls if not (c["shape_ok"] and c["finite"] and c["p_zero"] and c["ordered"])]
    print(
        f"  cleaned bundle predict() calls: {len(calls)}, "
        f"all shape/finite/zero-p/ordered: {not bad}"
    )
    ok &= not bad
    print("\n== determinism control (source bundle re-run) ==")
    for r in v["control"]:
        print(f"  {r['case']:24s} {'IDENTICAL' if r['identical'] else 'DIFFERENT'}")
    print("\n== daemonic-import check ==")
    print("  " + v["daemon"]["summary"])
    ok &= v["daemon"]["ok"]
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  {zip_path}")
    return 0 if ok else 1


def _synthetic_window(k: int):
    """A per-trajectory-distinct physical window (as scripts/t2/cudagraph_multitraj_parity.py)."""
    import numpy as np

    x = np.zeros((1, 20, 32, 64, 3), np.float32)
    x[..., 0] = 0.10 + 0.03 * k
    r0 = 10 + (k % 4)
    x[:, :, r0 : r0 + 4, 4 + 2 * k : 24 + 2 * k, :2] = 0.0
    return x


def _t2_streams(work: Path, ntraj: int = 5, tsteps: int = 60) -> dict:
    """The real val_extrap stream (grouped into trajectories, revealed targets included) and
    the synthetic multi-trajectory stream, both already in evaluator space."""
    import numpy as np

    from rpde_t1.score.harness import load_eval_set
    from rpde_t2 import to_normalized

    xi, yi = work / "t2_inputs_val_extrap.npy", work / "t2_targets_val_extrap.npy"
    gi = work / "t2_groups_val_extrap.json"
    if not (xi.exists() and yi.exists() and gi.exists()):
        x, y, meta = load_eval_set("val_extrap")
        np.save(xi, to_normalized(np.asarray(x, dtype=np.float32)))
        np.save(yi, to_normalized(np.asarray(y, dtype=np.float32)))
        groups: dict[int, list[int]] = {}
        for i, (traj, _off) in enumerate(meta["origins"]):
            groups.setdefault(int(traj), []).append(i)
        gi.write_text(json.dumps([sorted(g) for g in groups.values()]))
    si = work / "t2_inputs_synthetic.npy"
    if not si.exists():
        steps = []
        for k in range(ntraj):
            base = _synthetic_window(k)
            for i in range(tsteps):
                b = base.copy()
                b[..., 0] *= 1.0 + 0.1 * float(np.sin((i + k) / 5.0))
                steps.append(b[0])
        np.save(si, to_normalized(np.stack(steps)))
    return {
        "streams": {
            "val_extrap": {
                "inputs": str(xi),
                "targets": str(yi),
                "groups": json.loads(gi.read_text()),
            },
            "synthetic": {
                "inputs": str(si),
                "targets": None,
                "groups": [list(range(k * tsteps, (k + 1) * tsteps)) for k in range(ntraj)],
            },
        }
    }


def _parse_subscores(text: str) -> dict[str, float]:
    out = {}
    for m in re.finditer(r"\[local_eval\]\s+(\w+_score)\s+([0-9.]+)", text):
        out[m.group(1)] = float(m.group(2))
    return out


def _ruff_available() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True, check=False
    )
    return proc.returncode == 0


def main_track2(args) -> int:
    staged = args.staged.resolve()
    if not (staged / "submission.py").is_file():
        raise SystemExit(f"{staged} has no submission.py")
    out_dir = (args.out_root / args.name).resolve()
    zip_path = out_dir.with_suffix(".zip")
    if out_dir.exists() and not (out_dir / ".clean_bundle").exists():
        raise SystemExit(
            f"{out_dir} exists and was not written by this tool; choose another --name"
        )
    work = (args.work_dir or (args.out_root / ".clean_bundle_work" / args.name)).resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = not args.no_gpu_lock
    print(f"source bundle : {staged}")
    print(f"output bundle : {out_dir}  ->  {zip_path}")
    print(f"work dir      : {work}")

    t2 = _t2_streams(work)
    digest = source_digest(staged)
    trace_path = work / "trace.json"
    ref_out = work / "ref_outputs"
    stamp = work / "trace.digest"
    if args.reuse_trace and trace_path.exists() and stamp.exists() and stamp.read_text() == digest:
        print("trace         : reused")
    else:
        print("trace         : running the source bundle's stream under the GPU lock ...")
        ref_res = run_driver(
            staged,
            {},
            [],
            ref_out,
            trace_path,
            work,
            f"clean_bundle_trace_{args.name}",
            lock,
            track2=t2,
        )
        stamp.write_text(digest)
        print(f"trace         : done, import {ref_res['import_s']:.1f} s")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    staged_text = (staged / "submission.py").read_text(encoding="utf-8")
    consts = parse_constants(staged_text, strict=False)
    cleaner = Cleaner(staged, trace)
    cleaner.select(include_submission=True)
    named = {consts["_CKPT_NAME"]} | {
        v for k, v in consts.items() if k.endswith("_NAME") and isinstance(v, str) and v
    }
    data_files = sorted(set(cleaner.data_files()) | {n for n in named if (staged / n).is_file()})
    cleaner.extra_refs.update(
        ["get_ttt_model", "reset_ttt_state", "ttt_step", "adapt_loss", "predict"]
    )
    cleaner.run()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / ".clean_bundle").write_text(f"source={staged}\n")
    cleaner.write(out_dir)
    for rel in data_files:
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged / rel, dst)
    py_files = [str(p) for p in out_dir.rglob("*.py")]
    if _ruff_available():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--select",
                "F401,I",
                "--fix",
                "--quiet",
                "--no-cache",
                *py_files,
            ],
            cwd=str(REPO),
            check=False,
            capture_output=True,
        )
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", "--quiet", "--no-cache", *py_files],
            cwd=str(REPO),
            check=True,
            capture_output=True,
        )
    else:
        # Without ruff the pruned modules keep their original import lines, and an import of a
        # name the pruning removed from another module (e.g. a constant only dead code used)
        # fails at import time; the import smoke / verification below then fails and the
        # caller keeps the staged bundle. Formatting is cosmetic; the F401 fix is not.
        cleaner.notes.append("ruff not installed: unused-import fix and formatting skipped "
                             "(the pruned bundle may fail to import; verification decides)")
    cpu_import_smoke(out_dir, [m for m in sorted(cleaner.modules) if m != "submission.py"])
    before = member_listing(staged)
    after = {k: v for k, v in member_listing(out_dir).items() if k != ".clean_bundle"}
    hits = forbidden_scan(out_dir)
    (out_dir / ".clean_bundle").unlink()
    bundle = zip_bundle(out_dir, zip_path)
    (out_dir / ".clean_bundle").write_text(f"source={staged}\n")

    print("\n== members (lines) before -> after ==")
    for rel in sorted(set(before) | set(after)):
        b, a = before.get(rel), after.get(rel)
        marker = "" if rel in after else "   (dropped)"
        print(
            f"  {rel:70s} {_fmt_count(b, rel in before):>6} -> "
            f"{_fmt_count(a, rel in after):>6}{marker}"
        )
    tot_b = sum(v for v in before.values() if v)
    tot_a = sum(v for v in after.values() if v)
    print(f"  members {len(before)} -> {len(after)}; text lines {tot_b} -> {tot_a}")
    print(
        f"  extracted {bundle.extracted_mb:.1f} MB (cap {MAX_EXTRACTED_BYTES / 1024 / 1024:.0f}), "
        f"zip {bundle.zip_mb:.1f} MB"
    )
    print("\n== cleaning notes ==")
    for n in cleaner.notes:
        print("  " + n)
    print("\n== forbidden-word scan ==")
    if hits:
        for rel, ln, line in hits:
            print(f"  {rel}:{ln}: {line}")
    else:
        print("  clean")
    if args.skip_verify:
        print("\nverification skipped (--skip-verify)")
        return 1 if hits else 0

    print("\nverify        : running under the GPU lock ...")
    verify_json = work / "verify.json"
    vargs = {
        "zip": str(zip_path),
        "out_dir": str(out_dir),
        "staged_src": str(staged),
        "work": str(work),
        "out": str(verify_json),
        "t2": t2,
        "eval_stream": str(args.t2_eval_stream.resolve()) if args.t2_eval_stream else "",
    }
    cmd = [sys.executable, str(Path(__file__).resolve()), "--phase-verify-t2", json.dumps(vargs)]
    if lock:
        cmd = [str(GPU_SH), f"clean_bundle_verify_{args.name}"] + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if not verify_json.exists():
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit("verification phase failed")
    v = json.loads(verify_json.read_text(encoding="utf-8"))
    ok = not hits
    print("\n== clean-room run (extracted zip, python -I, Track-2 contract, 3 steps) ==")
    cr = v["clean_room"]
    print("  " + json.dumps(cr))
    ok &= bool(cr.get("finite") and cr.get("shapes_stable") and cr.get("bounds_ordered"))
    print("\n== bit-identity against the source bundle (every streaming step) ==")
    rows = compare_outputs(ref_out, work / "new_outputs")
    for r in rows:
        flag = "IDENTICAL" if r["identical"] else "DIFFERENT"
        extra = (
            ""
            if r["identical"]
            else " " + json.dumps({k: x for k, x in r.items() if k not in ("case", "identical")})
        )
        print(f"  {r['case']:24s} {flag}{extra}")
        ok &= r["identical"]
    calls = json.loads((work / "new_outputs" / "results.json").read_text())["calls"]
    bad = [
        c
        for c in calls
        if not (c["shape_ok"] and c["finite"] and c["ordered"] and c["adapt_loss_ok"])
    ]
    print(
        f"  cleaned bundle ttt_step() calls: {len(calls)}, "
        f"all shape/finite/ordered/adapt_loss: {not bad}"
    )
    ok &= not bad
    print("\n== determinism control (source bundle re-run) ==")
    for r in v["control"]:
        print(f"  {r['case']:24s} {'IDENTICAL' if r['identical'] else 'DIFFERENT'}")
    print("\n== daemonic-import check (scripts/t2/t2_daemon_verify.py) ==")
    print("  " + v["daemon"]["summary"])
    ok &= v["daemon"]["ok"]
    if args.t2_eval_stream:
        print(
            "\n== local evaluation (the kit's local_eval.py in this image, "
            f"{args.t2_eval_stream}) =="
        )
    else:
        print("\n== in-image local evaluation (scripts/t2/t2_image_local_eval.sh, val_extrap) ==")
    img = v["image_eval"]
    scores = _parse_subscores(img.get("stdout", ""))
    ref_scores = {}
    if args.reference_eval_log and args.reference_eval_log.is_file():
        ref_scores = _parse_subscores(args.reference_eval_log.read_text(encoding="utf-8"))
    for k in ("rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score"):
        ref = ref_scores.get(k)
        same = (
            ""
            if ref is None
            else ("  same" if k == "time_score" or scores.get(k) == ref else "  DIFFERENT")
        )
        print(f"  {k:14s} clean={scores.get(k)}  source={ref}{same}")
        if ref is not None and k != "time_score":
            ok &= scores.get(k) == ref
    if not scores:
        print("  " + (img.get("stdout", "")[-1500:] or img.get("stderr", "")[-1500:]))
        ok = False
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  {zip_path}")
    return 0 if ok else 1


def phase_verify_t2(spec: dict) -> None:
    zip_path = Path(spec["zip"])
    work = Path(spec["work"])
    out: dict[str, Any] = {}
    bundle = Bundle(zip_path, zip_path.with_suffix(""), 0, 0)
    try:
        out["clean_room"] = verify_ttt_bundle(bundle, n_steps=3)
    except Exception as exc:  # noqa: BLE001
        out["clean_room"] = {"error": str(exc)[-2000:]}
    with tempfile.TemporaryDirectory(prefix="clean_bundle_") as tmp:
        root = Path(tmp) / "extracted"
        root.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        new_out = work / "new_outputs"
        if new_out.exists():
            shutil.rmtree(new_out)
        run_driver(root, {}, [], new_out, None, work, "verify_new", False, track2=spec["t2"])
    ctrl_out = work / "ctrl_outputs"
    if ctrl_out.exists():
        shutil.rmtree(ctrl_out)
    run_driver(
        Path(spec["staged_src"]),
        {},
        [],
        ctrl_out,
        None,
        work,
        "verify_ctrl",
        False,
        track2=spec["t2"],
    )
    out["control"] = compare_outputs(ctrl_out, work / "ref_outputs")
    proc = subprocess.run(
        [sys.executable, str(T2_DAEMON_VERIFY), spec["out_dir"]],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    out["daemon"] = {
        "ok": proc.returncode == 0,
        "summary": (lines[-1] if lines else proc.stderr[-300:]) + f" (exit {proc.returncode})",
    }
    if spec.get("eval_stream"):
        kit = REPO / "vendor" / "realpde_t2_starting_kit_v6"
        proc = subprocess.run(
            [
                sys.executable,
                str(kit / "local_eval.py"),
                "--submission",
                spec["out_dir"],
                "--data",
                spec["eval_stream"],
            ],
            capture_output=True,
            text=True,
            cwd=str(kit),
        )
    else:
        proc = subprocess.run(
            ["bash", str(T2_IMAGE_EVAL), spec["out_dir"], "val_extrap"],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
    out["image_eval"] = {
        "rc": proc.returncode,
        "stdout": proc.stdout[-6000:],
        "stderr": proc.stderr[-3000:],
    }
    Path(spec["out"]).write_text(json.dumps(out, indent=1), encoding="utf-8")


def phase_verify(spec: dict) -> None:
    """GPU phase 2, run as one child so the lock is taken once."""
    zip_path = Path(spec["zip"])
    work = Path(spec["work"])
    out: dict[str, Any] = {}
    bundle = Bundle(zip_path, zip_path.with_suffix(""), 0, 0)
    out["clean_room"] = verify_bundle(bundle, n_windows=2)
    with tempfile.TemporaryDirectory(prefix="clean_bundle_") as tmp:
        root = Path(tmp) / "extracted"
        root.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        new_out = work / "new_outputs"
        if new_out.exists():
            shutil.rmtree(new_out)
        run_driver(
            root,
            {k: Path(v) for k, v in spec["inputs"].items()},
            spec["sizes"],
            new_out,
            None,
            work,
            "verify_new",
            False,
        )
    # Determinism control: the source bundle again on one case.
    ctrl_out = work / "ctrl_outputs"
    if ctrl_out.exists():
        shutil.rmtree(ctrl_out)
    first_split = next(iter(spec["inputs"]))
    run_driver(
        Path(spec["staged_src"]),
        {first_split: Path(spec["inputs"][first_split])},
        ["13"],
        ctrl_out,
        None,
        work,
        "verify_ctrl",
        False,
    )
    out["control"] = compare_outputs(ctrl_out, work / "ref_outputs")
    # Daemonic import check.
    proc = subprocess.run(
        [sys.executable, str(DAEMON_VERIFY), str(zip_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    out["daemon"] = {
        "ok": proc.returncode == 0,
        "summary": (lines[-1] if lines else proc.stderr[-300:]) + f" (exit {proc.returncode})",
        "stdout": proc.stdout[-3000:],
    }
    Path(spec["out"]).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--phase-verify":
        phase_verify(json.loads(sys.argv[2]))
        raise SystemExit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "--phase-verify-t2":
        phase_verify_t2(json.loads(sys.argv[2]))
        raise SystemExit(0)
    raise SystemExit(main())
