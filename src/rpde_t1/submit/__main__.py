"""``python -m rpde_t1.submit`` -- build, verify and report a submission zip."""

from __future__ import annotations

import argparse

from .package import (
    DEFAULT_BOUND_HALF_WIDTH,
    build_baseline_bundle,
    build_run_bundle,
    verify_bundle,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rpde_t1.submit", description=__doc__)
    p.add_argument(
        "--run",
        default=None,
        help="package one of OUR training runs (a name under runs/, or a path). Bounds "
        "follow the checkpoint's own config unless --bounds is given.",
    )
    p.add_argument(
        "--checkpoint",
        default="sim_real_cno.pth",
        help="baseline checkpoint filename or path (default: sim_real_cno.pth). "
        "With --run, the file to take from the run directory (default: last.pt).",
    )
    p.add_argument("--name", default=None, help="bundle name (default: derived from checkpoint)")
    p.add_argument(
        "--bounds",
        action="store_true",
        default=None,
        help="emit lower/upper interval bounds for the SPS subscore",
    )
    p.add_argument(
        "--bound-half-width",
        type=float,
        default=DEFAULT_BOUND_HALF_WIDTH,
        help=f"constant band half-width (default: {DEFAULT_BOUND_HALF_WIDTH}, SPS-optimal locally)",
    )
    p.add_argument(
        "--bound-floor",
        type=float,
        default=0.0,
        help="with --run, a constant physical half-width forced under the head's band",
    )
    p.add_argument(
        "--bound-scale",
        type=float,
        default=1.0,
        help="with --run, a global multiplier on the head's half-widths, from "
        "`train width-calib` fitted on one split and reported on another. Heads trained by "
        "this recipe come out ~40%% too wide and 0.60 is fitted independently on U-Net, FNO "
        "and CNO; worth +1.40 SPS held out on the U-Net. Moves `sps_score` only",
    )
    p.add_argument(
        "--width-rule",
        dest="width_rule",
        default=None,
        help="fitted width rule json (`scripts/sps/sps_select.py`), bundled beside the weights. "
        "Defaults to width_rule.json in the run directory when present. The per-bin "
        "generalisation of --bound-scale, worth +1.7 SPS over the best single multiplier on "
        "a log-error scale head; passing both is refused rather than composed",
    )
    p.add_argument(
        "--lead-anchor",
        dest="lead_anchor",
        default=None,
        help="fitted per-lead persistence anchor json (`scripts/sps/sps_anchor.py`), "
        "bundled beside the weights. Defaults to lead_anchor.json in the run directory "
        "when present. It moves "
        "the prediction, so --width-rule must be the rule refitted against the anchored error "
        "field; worth +0.087 sps and +0.063 tke for +0.19 ms/window",
    )
    p.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="forward precision under torch.autocast. The evaluation A800 runs bf16/fp16 at "
        "312 TFLOPS against 19.5 for fp32, and our T4->A800 factor of 2.91x (CNO's was 18.8x "
        "with fp16) says the 3D convs are not using the tensor cores. Changes the "
        "predictions, so re-score before shipping",
    )
    p.add_argument(
        "--pad-batches",
        action="store_true",
        help="repeat the last sample so every forward has exactly --batch rows. **Do not use "
        "for this platform**: it calls predict() with a small N repeatedly, so padding to "
        "--batch multiplies the compute per scored sample -- measured 20.96 ms/sample at N=37 "
        "against 4.80 unpadded, and it cost -8.5 time_score twice (FINDINGS 49)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=32,
        help="internal predict() batch size. Measured on a T4: CNO runs 135 ms/sample at "
        "32 against 180 at 16, worth ~3 time points; 64 gains a further ~0.7 but leaves "
        "less headroom on an unknown platform GPU (default: 32)",
    )
    p.add_argument(
        "--tke-head",
        dest="tke_head",
        default=None,
        help="second-moment head checkpoint from `train tke-head`, bundled alongside the "
        "backbone. NOT cut on time: 63m's +4.2 ms/sample was 98%% unoptimised host numpy and "
        "the on-device path is +0.62 (+0.32 without `tau`). At `--width 8 --depth 2`, "
        "`--tke-blend 0.0` and `--tke-shrink 0.7` it is worth **+0.109 final on the held-out "
        "hard corner** through `make_predict` -- rel_l2 -0.170, tke +1.279, sps +0.215 -- and "
        "+0.053 / +0.066 on the two in-distribution held-out splits. The shrink was selected "
        "on those two, never on the corner. Shrinking toward the closed-form BLEND is what "
        "63f measured as necessary on the FNO-era lineage and it is now negative at every "
        "exponent, which is why 0.0 is reachable",
    )
    p.add_argument(
        "--tke-blend",
        dest="tke_blend",
        type=float,
        default=None,
        help="post-hoc fluctuation-rescaling exponent. On its own this is the "
        "zero-training estimator; with --tke-head it is the blend the head shrinks toward, "
        "and omitting it takes the measured default. Pass 0.0 for a head shrunk toward the "
        "unrescaled field instead, which is what the current lineage measures as best",
    )
    p.add_argument(
        "--tke-shrink",
        dest="tke_shrink",
        type=float,
        default=None,
        help="weight on the head's log-ratio, against `--tke-blend`'s closed form (default: "
        "the measured 0.75). With `--tke-blend 0.0` it is a plain shrink toward the "
        "unrescaled field, and 0.7 is what the corner selects -- an unshrunk head over-"
        "corrects, because the rel_l2 cost of a wrong per-pixel gain is quadratic in it "
        "while the tke benefit is not",
    )
    p.add_argument(
        "--compile-mode",
        dest="compile_mode",
        default="",
        choices=["", "default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for the forward (default: off). With --compile-dynamic, "
        "`default` compiles ONE symbolic-shape graph that serves every N -- measured +25-29%% "
        "ms/sample on the batch ladder, no padding, no per-shape recompile (FINDINGS 148). "
        "`reduce-overhead` needs --pad-batches (its CUDA graph raises on a second shape). "
        "Sticky-demotes to eager on any failure, so the downside is the eager time",
    )
    p.add_argument(
        "--compile-dynamic",
        dest="compile_dynamic",
        action="store_true",
        help="compile with dynamic=True (one symbolic-shape graph for all N). Required for a "
        "variable-batch Track-1 bundle unless padding; N=1 is guarded to eager because Dynamo "
        "specialises size-1 dims into a ~29 s recompile (FINDINGS 148)",
    )
    p.add_argument(
        "--cond-device",
        dest="cond_device",
        action="store_true",
        help="estimate the operating point on the GPU instead of in host numpy "
        "(models/op_estimate.py). The largest host-side item in the timed region: 0.188 of a "
        "1.563 ms/window step at batch 210. make_predict REFUSES it for any configuration the "
        "device twin does not implement, so it cannot silently serve a different conditioning.",
    )
    p.add_argument(
        "--zero-p-device",
        dest="zero_p_device",
        action="store_true",
        help="zero the unscored pressure channel on the resident tensor instead of in three "
        "strided passes over the host output buffers. The last host-side step on a shipped "
        "Track-1 artefact: 11.68 ms against 1.43 ms per 210-window call on an A10G, because a "
        "2-byte element every 6 bytes touches every cache line. Score-neutral by construction; "
        "make_predict declines it when out_transform would map the zero elsewhere.",
    )
    p.add_argument(
        "--cudagraph",
        dest="cudagraph",
        default="",
        help="compiler-free CUDA-graph capture of the backbone: a comma-separated bucket "
        "ladder (e.g. '1,2,4,8,16,32,64,128,210'). Each scored call pads up to the smallest "
        "bucket >= N and replays the graph. Unlike --compile-mode this needs no C compiler, "
        "so it runs in the eval image (which has ptxas but no gcc). Captures at load on the "
        "eval GPU (untimed, arch-correct), bit-exact at an exact bucket, fp16-quantum when "
        "padded. Mutually exclusive with --compile-mode",
    )
    p.add_argument(
        "--re-corners",
        dest="re_corners",
        action="store_true",
        help="bake in the corners3 Reynolds estimator (AoA-free, three freestream corners) so the "
        "packaged submission serves it in place of §122's u_fs line. Halves §122's Reynolds "
        "extrapolation error; folds into --cond-device at ~zero extra time (op_estimate.py). "
        "Train the backbone with RPDE_RE_CORNERS=1 in the same run for train/serve match.",
    )
    p.add_argument(
        "--warmup-sizes",
        dest="warmup_sizes",
        default="",
        help="call sizes warmed at import, comma-separated with inclusive ranges "
        "(e.g. '1-32,64,105'); the bundle batch size is always warmed last "
        "(default: 1,2,4,8,16,32,64,105)",
    )
    p.add_argument(
        "--centre-shrink",
        dest="centre_shrink",
        type=float,
        default=1.0,
        help="centre the interval on the fluctuation-shrunk prediction m + c*(pred - m) "
        "(m the window's u/v time mean), the reported prediction unchanged (FINDINGS 196i: "
        "+0.025 sps held out at 0.95). The bundled head and rule must have been fitted on the "
        "same shrunk field (scripts/sps/shrink_cache.py, then sps_apply.py on the _c<c> tag). "
        "1.0 is off",
    )
    p.add_argument("--no-verify", action="store_true", help="skip the clean-room run")
    args = p.parse_args(argv)

    if args.run:
        bundle = build_run_bundle(
            args.run,
            name=args.name,
            checkpoint_file="last.pt" if args.checkpoint == "sim_real_cno.pth" else args.checkpoint,
            return_bounds=args.bounds,
            bound_floor=args.bound_floor,
            bound_scale=args.bound_scale,
            width_rule=args.width_rule,
            lead_anchor=args.lead_anchor,
            precision=args.precision,
            pad_batches=args.pad_batches,
            batch=args.batch,
            tke_head=args.tke_head,
            tke_blend=args.tke_blend,
            tke_shrink=args.tke_shrink,
            compile_mode=args.compile_mode,
            compile_dynamic=args.compile_dynamic,
            cond_device=args.cond_device,
            zero_p_device=args.zero_p_device,
            cudagraph=args.cudagraph,
            re_corners=args.re_corners,
            warmup_sizes=args.warmup_sizes,
            centre_shrink=args.centre_shrink,
        )
    else:
        bundle = build_baseline_bundle(
            args.checkpoint,
            name=args.name,
            return_bounds=bool(args.bounds),
            bound_half_width=args.bound_half_width,
            batch=args.batch,
        )

    print("BUNDLE")
    print(f"  zip            : {bundle.zip_path}")
    print(f"  staged dir     : {bundle.staged_dir}")
    print(f"  members        : {bundle.members}")
    print(f"  zip size       : {bundle.zip_mb:.1f} MB")
    print(f"  extracted size : {bundle.extracted_mb:.1f} MB  (cap 256 MB)")

    if args.no_verify:
        print("\nverification skipped (--no-verify)")
        return 0

    print("\nVERIFY (extract elsewhere, import in an isolated interpreter, run predict)")
    result = verify_bundle(bundle)
    for key in ("shape", "expected_shape", "shape_ok", "dtype", "finite", "p_channel_zero"):
        print(f"  {key:<16}: {result[key]}")
    if result["is_dict"]:
        print(f"  {'bound keys':<16}: {result['keys']}")
    print(f"  {'predict seconds':<16}: {result['seconds']:.4f}  (2 windows, cold start)")

    ok = result["shape_ok"] and result["finite"]
    print(f"\n{'READY TO SUBMIT' if ok else 'NOT READY -- see above'}: {bundle.zip_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
