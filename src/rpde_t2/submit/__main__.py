"""``python -m rpde_t2.submit`` -- build and verify a Track-2 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .package import build_ttt_bundle, verify_ttt_bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="runs/<name>/ or an explicit last.pt")
    ap.add_argument("--name", default=None, help="bundle name (default t2_<run>)")
    ap.add_argument("--out-root", default=None, type=Path)
    ap.add_argument(
        "--precision",
        default="fp32",
        choices=("fp32", "bf16", "fp16"),
        help="fp32 by DEFAULT. The right answer is PATH-dependent, and both are measured. On "
        "the eager host path, autocast at batch 1 pays per-op cast kernels with nothing to "
        "amortise them: fp32 12.55 ms/step against fp16 13.40 on the day-20 residual bundle. "
        "On the --gpu-native compiled path the casts fuse into the one graph and the step is "
        "compute-bound on fp32 attention/conv/GEMM kernels, so fp16 cut the compiled forward "
        "4.40 -> 1.69 ms on the local T4 (2026-09-15). fp16 there is finiteness-guarded per "
        "step and sticky-demotes to eager fp32",
    )
    ap.add_argument(
        "--gain-strength",
        default=0.0,
        type=float,
        help="per-trajectory fluctuation-gain calibration (0.5 measured best)",
    )
    ap.add_argument(
        "--bias-strength",
        default=0.0,
        type=float,
        help="per-trajectory mean-residual calibration (0.5 measured best)",
    )
    ap.add_argument(
        "--tke-head",
        default=None,
        help="second-moment head checkpoint from `train tke-head` (the .pt, not the .json "
        "report). Pair it with `--tke-blend 0.0 --tke-shrink 0.7`, the shipped serving "
        "config: this CLI's blend default is 0.0, but leaving it unset elsewhere resolves to "
        "DEFAULT_BLEND=0.5, which is a different configuration",
    )
    ap.add_argument("--bound-scale", default=1.0, type=float)
    ap.add_argument("--bound-floor", default=0.0, type=float)
    ap.add_argument("--tke-blend", default=0.0, type=float)
    ap.add_argument("--tke-shrink", default=0.0, type=float)
    ap.add_argument(
        "--no-bounds",
        action="store_true",
        help="ship a point-only submission (the scorer then uses its default +/-5%% band)",
    )
    ap.add_argument(
        "--compile-mode",
        default="",
        choices=["", "default", "reduce-overhead", "max-autotune", "cudagraph"],
        help="torch.compile mode for the forward. A Track-2 win at batch 1 (fixed shape -> CUDA "
        "graph captured once in warm-up): reduce-overhead measured step 11.2->7.3 ms, all "
        "subscores bit-identical. Sticky-demotes to eager on any error. Default off. "
        "'cudagraph' is NOT a torch.compile mode: it captures the EAGER pipeline with a manual "
        "torch.cuda.CUDAGraph -- no C compiler / Triton / Inductor, so it actually engages in "
        "the offline eval image where torch.compile silently falls back to eager.",
    )
    ap.add_argument(
        "--gpu-native",
        action="store_true",
        help="keep the input on the device and run the whole tail as one compiled graph "
        "(FINDINGS 76b). Needs --compile-mode and the frozen-bounds config; falls back "
        "to the host path otherwise. Removes the batch-1 sync-bound host tail.",
    )
    ap.add_argument(
        "--bias-mode",
        default="meanflow",
        choices=["meanflow", "perframe"],
        help="meanflow: time-averaged first-moment bias, stacks on drop-path tke; "
        "perframe: old per-lead residual (moves tke).",
    )
    ap.add_argument(
        "--mean-bias-strength",
        default=0.0,
        type=float,
        help="per-trajectory mean-flow bias in the GPU-native path (FINDINGS 77): "
        "stacks on drop-path (mvpe/rel_l2, tke untouched). 0.75 measured best.",
    )
    ap.add_argument(
        "--bias-freeze-n",
        default=8,
        type=int,
        help="revealed residual windows used before freezing mean-flow bias",
    )
    ap.add_argument(
        "--mean-bias-smooth",
        default=0.0,
        type=float,
        help="blend weight on the twice-box-filtered 3x3 mean-flow bias",
    )
    ap.add_argument(
        "--history-fill",
        default="zeros",
        choices=["zeros", "repeat", "hold"],
        help="how to fill the absent earlier frames on a trajectory's FIRST window when the "
        "backbone reads more than 20 (inert at t_in=20; the stream has no history there)",
    )
    ap.add_argument(
        "--state-freeze-n",
        dest="state_freeze_n",
        type=int,
        default=0,
        help="stop re-estimating a stream-state backbone's trajectory statistics after this "
        "many reveals (0 = never). They are converged long before a stream ends, and freezing "
        "removes the per-step eager reductions -- the half of the state's cost that is not the "
        "stem. Training always sees the FULL history, so this is a train/serve mismatch and "
        "must be measured before it ships",
    )
    ap.add_argument(
        "--causal-v2",
        dest="causal_v2",
        action="store_true",
        help="use models/causal_history_v2.py (the module lifted from the v2 board artifact) "
        "instead of the one every recorded row ran. Required by the two knobs below",
    )
    ap.add_argument(
        "--causal-gain-floor",
        dest="causal_gain_floor",
        type=float,
        default=0.0,
        help="v2 only: PIV measurement-noise variance floor subtracted from the measured "
        "target variance before the causal gain ratio (the v2 artifact ships 2e-6)",
    )
    ap.add_argument(
        "--interval-rho-blend",
        dest="interval_rho_blend",
        type=float,
        default=0.0,
        help="v2 only: weight on the closed-form z*(rho) width built from the trajectory's own "
        "running revealed |error|, blended with the static width rule (v2 ships 0.6)",
    )
    ap.add_argument(
        "--interval-rho-ramp-n0",
        dest="interval_rho_ramp_n0",
        type=float,
        default=0.0,
        help="v2 only: evidence ramp b(n) = blend * n/(n + n0) on that weight (v2 ships 16)",
    )
    ap.add_argument(
        "--interval-rho-shape",
        dest="interval_rho_shape",
        type=float,
        default=1.0,
        help="v2 only: scale on the half-normal error-scale estimate (v2 ships 0.9)",
    )
    ap.add_argument(
        "--interval-recal",
        dest="interval_recal",
        default=None,
        help="learned interval calibrator checkpoint (models/interval_recal.py), bundled "
        "beside the weights; exclusive with --interval-autocal-n0s > 0",
    )
    ap.add_argument(
        "--interval-recal-fused",
        dest="interval_recal_fused",
        action="store_true",
        help="serve the calibrator through its jiterator-fused step (FINDINGS 208: +0.13 against "
        "+0.16 ms/step end to end on the A10G, widths within one fp16 ulp)",
    )
    ap.add_argument(
        "--interval-recal-merged",
        dest="interval_recal_merged",
        action="store_true",
        help="capture the calibrator's observe + apply inside the one pipeline graph "
        "(2026-09-25 night): no second graph launch, fast-path static input and in-graph "
        "finiteness flag kept",
    )
    ap.add_argument(
        "--interval-recal-fork",
        dest="interval_recal_fork",
        action="store_true",
        help="with --interval-recal-merged: the observe as a branch parallel to the backbone",
    )
    ap.add_argument(
        "--interval-autocal-n0s",
        dest="interval_autocal_n0s",
        type=float,
        default=0.0,
        help="self-calibrating intervals (FINDINGS 192): shrinkage n0 of the per-trajectory "
        "first-stage multiplier; 0 disables the calibrator",
    )
    ap.add_argument(
        "--interval-autocal-n0p",
        dest="interval_autocal_n0p",
        type=float,
        default=50.0,
        help="self-calibrating intervals: shrinkage n0 of the per-pixel second stage",
    )
    ap.add_argument(
        "--interval-autocal-stage1",
        dest="interval_autocal_stage1",
        choices=["scalar", "chanlead"],
        default="scalar",
        help="self-calibrating intervals: first-stage groups (one scalar, or channel x lead)",
    )
    ap.add_argument(
        "--interval-autocal-shift",
        dest="interval_autocal_shift",
        type=float,
        default=0.0,
        help="self-calibrating intervals: centre shift by beta x the pixel's running mean signed "
        "error (bounds only; 0 = off)",
    )
    ap.add_argument(
        "--interval-autocal-grid",
        dest="interval_autocal_grid",
        type=int,
        default=29,
        help="self-calibrating intervals: multipliers on the [1/2, 2] log grid (device cost)",
    )
    ap.add_argument(
        "--interval-autocal-lead-sub",
        dest="interval_autocal_lead_sub",
        type=int,
        default=1,
        help="self-calibrating intervals: credit tables from every Nth lead (device cost)",
    )
    ap.add_argument(
        "--interval-autocal-freeze-n",
        dest="interval_autocal_freeze_n",
        type=int,
        default=0,
        help="self-calibrating intervals: freeze the credit tables after N revealed windows "
        "(later steps skip the credit computation; 0 = never)",
    )
    ap.add_argument(
        "--pinned-io",
        dest="pinned_io",
        action="store_true",
        help="stage the batch-1 input/outputs through pinned host buffers (FINDINGS 194): same "
        "values, ~0.26 ms less per step on the T4",
    )
    ap.add_argument(
        "--tf32",
        action="store_true",
        help="set torch.set_float32_matmul_precision('high') before compile, so fp32 matmuls "
        "use the A800's TF32 tensor cores (10-bit mantissa, fp32 range, fp32 accumulate). "
        "cudnn.allow_tf32 is ALREADY on by default, so this only adds the matmul half. A "
        "hardware no-op below sm_80: a T4 build is bit-identical to a plain one, so only the "
        "board can measure it (FINDINGS 92)",
    )
    ap.add_argument(
        "--port-warmup",
        dest="port_warmup",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="drive synthetic trajectories through the PORT in get_ttt_model (untimed) so the "
        "first scored step does not pay the stream path's one-time build (FINDINGS 222c): "
        "0 = off, 1 = one trajectory of 3 steps, 2 = two trajectories of 40 + 12 steps",
    )
    ap.add_argument(
        "--step-log",
        dest="step_log",
        action="store_true",
        help="write `[t2]` lines to stderr: pid/load time at import, the port warm-up, every "
        "reset and every step (shape, dtype, our own ms). Never raises",
    )
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--causal-gain-strength", type=float, default=0.0)
    ap.add_argument("--interval-history-spatial", type=float, default=0.0)
    ap.add_argument("--interval-history-global", type=float, default=0.0)
    ap.add_argument("--causal-anom", type=float, default=0.0)
    args = ap.parse_args(argv)
    if (
        args.causal_gain_strength
        or args.interval_history_spatial
        or args.interval_history_global
        or args.causal_anom
    ) and not (args.gpu_native and args.compile_mode):
        ap.error("additional causal history requires --gpu-native and --compile-mode")

    # Since 2026-09-15 `make_gpu_predict` consumes `precision` (fp32/fp16: autocast inside the
    # compiled graph, finiteness-guarded, sticky-demote to eager fp32). Before that it was
    # fp32-only and a non-default `--precision` on this path was written into the bundle and
    # never read -- screening it returned a bit-identical null that read as "the model is
    # insensitive" when it meant "the flag did nothing" (the FINDINGS 102a failure class).
    # bf16 stays refused here: `torch.autocast` raises for it below sm_80, so a bf16 bundle
    # could never be validated on the local T4 before it shipped.
    if args.precision == "bf16" and args.gpu_native:
        ap.error(
            "--precision bf16 is not supported on the --gpu-native path: it cannot be "
            "validated on the local T4 (autocast raises below sm_80). Use fp16, which is "
            "measured end-to-end locally, or fp32."
        )

    bundle = build_ttt_bundle(
        args.run,
        name=args.name,
        out_root=args.out_root,
        tke_head=args.tke_head,
        return_bounds=not args.no_bounds,
        bound_scale=args.bound_scale,
        bound_floor=args.bound_floor,
        tke_blend=args.tke_blend,
        tke_shrink=args.tke_shrink,
        precision=args.precision,
        gain_strength=args.gain_strength,
        bias_strength=args.bias_strength,
        compile_mode=args.compile_mode,
        gpu_native=args.gpu_native,
        tf32=args.tf32,
        bias_mode=args.bias_mode,
        mean_bias_strength=args.mean_bias_strength,
        causal_gain_strength=args.causal_gain_strength,
        interval_history_spatial=args.interval_history_spatial,
        interval_history_global=args.interval_history_global,
        causal_anom=args.causal_anom,
        bias_freeze_n=args.bias_freeze_n,
        mean_bias_smooth=args.mean_bias_smooth,
        history_fill=args.history_fill,
        state_freeze_n=args.state_freeze_n,
        causal_v2=args.causal_v2,
        causal_gain_floor=args.causal_gain_floor,
        interval_rho_blend=args.interval_rho_blend,
        interval_rho_ramp_n0=args.interval_rho_ramp_n0,
        interval_rho_shape=args.interval_rho_shape,
        interval_recal=args.interval_recal,
        interval_recal_fused=args.interval_recal_fused,
        interval_recal_merged=args.interval_recal_merged,
        interval_recal_fork=args.interval_recal_fork,
        interval_autocal_n0s=args.interval_autocal_n0s,
        interval_autocal_n0p=args.interval_autocal_n0p,
        interval_autocal_stage1=args.interval_autocal_stage1,
        interval_autocal_shift=args.interval_autocal_shift,
        interval_autocal_grid=args.interval_autocal_grid,
        interval_autocal_lead_sub=args.interval_autocal_lead_sub,
        interval_autocal_freeze_n=args.interval_autocal_freeze_n,
        pinned_io=args.pinned_io,
        port_warmup=args.port_warmup,
        step_log=args.step_log,
    )
    print(
        f"{bundle.zip_path}  {bundle.members} members  "
        f"{bundle.extracted_mb:.1f} MiB extracted  {bundle.zip_mb:.1f} MiB zipped"
    )

    if args.skip_verify:
        return 0
    result = verify_ttt_bundle(bundle)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        print("VERIFY FAILED -- do not submit this bundle.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
