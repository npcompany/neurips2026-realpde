#!/usr/bin/env python
"""Import a Track-1 bundle in a DAEMONIC worker and exercise predict, matching the board.

The platform runs each participant in a daemonic subprocess. A daemonic process may not
create children, and `torch.compile` -> inductor's `AsyncCompile.warm_pool()` ->
`ProcessPoolExecutor` does exactly that: FINDINGS 76c caught a Track-2 bundle failing
ingestion at IMPORT with `AssertionError: daemonic processes are not allowed to have
children`, and identifies it as the likely real cause of Track 1's "compile is 2-for-2
against us" -- a crash, not a regression. A normal interpreter does NOT reproduce it, which
is why `scripts/t2/t2_daemon_verify.py` exists; this is its Track-1 twin, and until now
Track 1 had none because Track 1 shipped nothing compiled.

It also sweeps a ladder of batch sizes, because the platform calls predict() with a small,
unknown N many times over (FINDINGS 49) and a compiled bundle must not recompile on an
unseen shape inside a scored call.

    uv run python scripts/eval/t1_daemon_verify.py submissions/<name>.zip
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import tempfile
import time
import zipfile
from pathlib import Path

#: Sizes to exercise. 1 and 2 are the degenerate cases a dynamic graph is most likely to
#: specialise away from; 210 is what the bundle warms at; the rest bracket FINDINGS 49's
#: "per-call N below 37".
LADDER = (210, 37, 12, 13, 5, 2, 1, 105, 64)


def _graph_state(module) -> str:
    """'captured'/'fallback' per size for a t1_graph_bundle.py bundle, else 'n/a'."""
    st = getattr(module, "_state", None)
    if not isinstance(st, dict) or "graph" not in st:
        return "n/a"
    g = st["graph"]
    if g is None:
        return "none"
    graphs = getattr(g, "graphs", {})
    if not graphs:
        return "not-built"
    return ",".join(f"{n}:{'captured' if v is not None else 'fallback'}" for n, v in graphs.items())


def _run(bundle_dir: str, queue, platform_warmup: bool = False) -> None:
    try:
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "submission", os.path.join(bundle_dir, "submission.py")
        )
        module = importlib.util.module_from_spec(spec)
        t0 = time.perf_counter()
        spec.loader.exec_module(module)          # construction + warm-up live here
        import_s = time.perf_counter() - t0

        rows = {}
        rng = np.random.default_rng(0)
        for n in LADDER:
            x = np.full((n, 20, 32, 64, 3), 0.15, dtype=np.float32)
            if platform_warmup:
                # The board's harness answers one UNTIMED synthetic call of the same shape in
                # every fresh process before the timed one (Track-1 FAQ); a bundle with no
                # import-time warm-up relies on it, and a graph bundle captures on it.
                module.predict(rng.standard_normal(x.shape).astype(np.float32))
            t0 = time.perf_counter()
            out = module.predict(x)              # FIRST call at this shape, as the board times it
            first_ms = (time.perf_counter() - t0) * 1e3 / n
            p = np.asarray(out["prediction"], np.float32)
            lo = np.asarray(out["lower"], np.float32)
            hi = np.asarray(out["upper"], np.float32)
            rows[n] = {
                "first_call_ms_per_sample": first_ms,
                "shape_ok": p.shape == (n, 20, 32, 64, 3),
                "finite": bool(np.isfinite(p).all() and np.isfinite(lo).all()
                               and np.isfinite(hi).all()),
                "pressure_zero": bool((p[..., 2] == 0).all()),
                "ordered": bool((lo <= hi).all()),
            }
        compiled = None
        fn = getattr(module, "_state", {}).get("fn")
        if fn is not None and hasattr(fn, "uses_compiled"):
            compiled = bool(fn.uses_compiled())
        # Whether the bundle ASKED for a compiled path, which is what decides whether
        # ``uses_compiled`` being False is correct or a silent failure. Without this the check
        # cannot tell an eager bundle (False is right) from a compiled one that died (False is
        # a defect), which is how FINDINGS 187 found this script printing PASS over both.
        requested = bool(getattr(module, "COMPILE_MODE", ""))
        queue.put({
            "ok": True, "import_s": import_s, "rows": rows,
            "uses_compiled": compiled, "compile_requested": requested,
            "graph_state": _graph_state(module),
        })
    except Exception as exc:  # noqa: BLE001 - the point is to report it, not to raise here
        import traceback

        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path, help="submission .zip or an unpacked directory")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument(
        "--platform-warmup",
        action="store_true",
        help="make one untimed synthetic call of the same shape before each timed call, as the "
        "board's harness does; for a bundle with no import-time warm-up, and for a "
        "t1_graph_bundle.py bundle, whose graph must then be captured at every size",
    )
    args = ap.parse_args()

    tmp = None
    bundle_dir = args.bundle
    if args.bundle.suffix == ".zip":
        tmp = tempfile.mkdtemp(prefix="t1daemon_")
        with zipfile.ZipFile(args.bundle) as z:
            z.extractall(tmp)
        bundle_dir = Path(tmp)
    if not (bundle_dir / "submission.py").is_file():
        ap.error(f"missing {bundle_dir / 'submission.py'}")

    ctx = mp.get_context("spawn")   # the parent must not have CUDA initialised in the child
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_run, args=(str(bundle_dir), queue, args.platform_warmup), daemon=True
    )
    proc.start()
    proc.join(args.timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise SystemExit(f"FAIL: daemon run exceeded {args.timeout} s")
    result = queue.get() if not queue.empty() else {
        "ok": False, "error": f"no result; exit code {proc.exitcode}"
    }
    if not result.get("ok"):
        print("FAIL:", result.get("error"))
        print(result.get("traceback", ""))
        return 1

    requested = result.get("compile_requested", False)
    live = result["uses_compiled"]
    print(f"daemon import + construction: {result['import_s']:.1f} s")
    print(f"compile requested by the bundle: {requested}")
    print(f"compiled path live at the end of the ladder: {live}")
    graph_state = result.get("graph_state", "n/a")
    print(f"whole-call graph state per size (t1_graph_bundle.py bundles only): {graph_state}")

    # Per-call totals, not the per-sample rate, because the rate legitimately rises as N falls
    # (fixed per-call cost over fewer samples) and would make every small N look pathological.
    # A call that takes seconds is pathological at ANY N -- FINDINGS 187 measured 21 s at N=105
    # inside the evaluation image while every neighbour sat under 0.6 s.
    totals = {n: r["first_call_ms_per_sample"] * n / 1000.0 for n, r in result["rows"].items()}
    ordered_totals = sorted(totals.values())
    median_total = ordered_totals[len(ordered_totals) // 2]
    # TWO rules, because a median-relative one alone has a blind spot that was measured rather
    # than imagined: `backend="cudagraphs", dynamic=False` recaptures the graph at EVERY new
    # shape, ~4.3 s each, so every call is slow, the median is 4.3 s, and a 10x-median cutoff
    # sits at 43 s and flags nothing. The absolute rule catches that; the relative one catches a
    # single outlier among fast calls (the 19 s at N=105 in FINDINGS 187a). A legitimate call is
    # 0.32 s at the largest N in the ladder, and the platform uses a SMALLER N than that, so a
    # 2 s floor has ample clearance.
    ABSOLUTE_CUT = 2.0
    relative_cut = 10.0 * median_total
    # With the platform warm-up the timed call is a pure replay whose cost scales with N (0.9
    # ms/sample at every size on the A10G, so 0.19 s at N=210 against a 0.011 s median across
    # the ladder), and the per-call relative rule flags the largest size for being large. The
    # outlier rule then compares the per-sample rate instead; the absolute cut stays.
    rates = {n: r["first_call_ms_per_sample"] for n, r in result["rows"].items()}
    ordered_rates = sorted(rates.values())
    median_rate = ordered_rates[len(ordered_rates) // 2]
    rate_cut = 10.0 * median_rate

    print(f"{'N':>5s} {'first-call ms/sample':>21s} {'call s':>8s}  shape finite zero-p ordered")
    bad = 0
    slow: list[int] = []
    for n, r in result["rows"].items():
        flags = (r["shape_ok"], r["finite"], r["pressure_zero"], r["ordered"])
        bad += sum(1 for f in flags if not f)
        mark = ""
        if args.platform_warmup:
            is_slow = totals[n] > ABSOLUTE_CUT or rates[n] > rate_cut
        else:
            is_slow = totals[n] > ABSOLUTE_CUT or totals[n] > relative_cut
        if is_slow:
            slow.append(n)
            mark = "  <-- SLOW"
        print(f"{n:>5d} {r['first_call_ms_per_sample']:>21.3f} {totals[n]:>8.3f}  "
              + "  ".join("ok " if f else "BAD" for f in flags) + mark)

    fail = []
    if bad:
        fail.append(f"{bad} contract violation(s)")
    # The check this script was missing, and the reason FINDINGS 187 caught it: for an eager
    # bundle `live is False` is correct, and for a compiled one it means the compiled path died
    # silently. Printing an unconditional PASS over both is how a bundle whose compiled path was
    # dead -- and which paid 21 s inside a call -- was reported as fine.
    if requested and not live:
        fail.append(
            "compile was REQUESTED but the compiled path is not live: it failed and demoted to "
            "eager, and the failed attempt is paid inside scored calls (FINDINGS 187)"
        )
    if slow and args.platform_warmup:
        fail.append(
            f"timed call at N={slow} exceeded {ABSOLUTE_CUT:.1f} s absolute or "
            f"{rate_cut:.2f} ms/sample (10x the {median_rate:.3f} ms/sample median) -- "
            f"graph capture or a slow path landing inside a scored call"
        )
    elif slow:
        fail.append(
            f"first call at N={slow} exceeded {ABSOLUTE_CUT:.1f} s absolute or "
            f"{relative_cut:.1f} s (10x the {median_total:.3f} s median) -- compilation or "
            f"graph capture landing inside a scored call"
        )
    if graph_state != "n/a":
        # A graph bundle that is not replaying is a silent failure: it serves the reference
        # path and pays the untimed capture attempt for nothing. With the platform warm-up
        # every size in the ladder must have been captured; without it the capture happens
        # inside the timed call and the sizes are expected to read "captured" too, but the
        # slow rule above is what judges that case.
        if graph_state in ("none", "not-built") or "fallback" in graph_state:
            fail.append(f"whole-call graph not replaying: {graph_state}")
        elif args.platform_warmup:
            captured = {int(s.split(":")[0]) for s in graph_state.split(",") if ":" in s}
            missing = sorted(set(LADDER) - captured)
            if missing:
                fail.append(f"whole-call graph not captured at N={missing}")
    if fail:
        for f in fail:
            print(f"FAIL: {f}")
        return 1
    state = "compiled path live" if requested else "eager as configured"
    print(f"PASS: daemonic import, {state}, contract clean and no slow call at any N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
