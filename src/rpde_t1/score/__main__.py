"""``python -m rpde_t1.score`` -- build local eval sets and score submissions.

Two subcommands:

    eval-set   build data/local_eval/<name>/ from a list of HDF5 trajectories
    run        run a submission against an eval set and print the official scores
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .. import SUB_S
from ..paths import EXAMPLE_H5, LOCAL_EVAL_DIR, VENDOR_KIT
from .diagnostics import DEFAULT_BLEND_W
from .harness import (
    MIN_EVAL_STRIDE,
    NORMALIZATIONS,
    assert_leak_free_stride,
    load_eval_set,
    load_predict,
    run_official_scorer,
    run_submission,
    write_eval_set,
    write_predictions,
)
from .report import format_report


def _resolve_split(split: str) -> tuple[list[str], dict]:
    """Trajectory paths and provenance for a named split (see data/splits.py)."""
    from ..data.splits import (
        VAL_TIME_OFFSET_FRAC,
        all_cells,
        aoa_split_paths,
        assign_splits,
        parse_params,
        real_trajectories,
        val_time_cells,
    )

    splits = assign_splits()
    extra: dict = {"split": split}

    aoa_paths = aoa_split_paths(split)
    if aoa_paths is not None:
        # The angle folds are taken inside `train` (see splits.aoa_holdout), so they carry no
        # reserved tail and read whole trajectories -- the same treatment `train` gives the
        # cells that do not reserve one.
        paths = aoa_paths
    elif split == "val_time":
        # Same trajectories as training, but only their reserved tail frames.
        # Measures fit rather than generalization -- the optimistic number.
        # Only the stratified subset reserves a tail; taking all of splits["train"]
        # here would silently score windows the model was trained on.
        reserved = set(val_time_cells())
        paths = [
            p for p in splits["train"] if (parse_params(p).re, parse_params(p).aoa) in reserved
        ]
        # Resolved per trajectory: lengths vary (868 / 607 / 492 / 282), so an
        # absolute offset would silently drop the short ones.
        extra["t_offset_frac"] = VAL_TIME_OFFSET_FRAC
    elif split == "all":
        # Every real cell, whole trajectories. For the POST-HOC HEADS of a shipping rebuild:
        # the backbone they correct has itself trained on all 81, so fitting them on the 69
        # of `train` withholds the error maps of the 12 hardest cells -- every `hard5` cell
        # lives in those 12 -- and buys no honesty in exchange. A model must never be
        # trained on a set built from this; `splits.all_cells` says why.
        paths = all_cells()
    elif split in splits:
        paths = splits[split]
    else:
        raise SystemExit(f"Unknown split {split!r}; choose from {sorted([*splits, 'val_time'])}")

    if not paths:
        raise SystemExit(f"Split {split!r} contains no trajectories.")
    extra["trajectories"] = [parse_params(p).key for p in paths]
    extra["n_available_trajectories"] = len(real_trajectories())
    return [str(p) for p in paths], extra


def _build_sim_eval_set(args: argparse.Namespace) -> int:
    """Build one of the simulated eval sets straight from the training dataset.

    Deliberately *not* routed through :func:`build_window_set` like the real splits.
    A simulated window only becomes comparable to a real one after two
    transformations the dataset already owns -- the real PIV mask is imposed on it,
    and the pressure plane is zeroed -- and reproducing those here would be a second
    definition of the same thing, which is exactly how an eval set drifts from the
    distribution its model was trained on.

    Windows are written in **physical** units (``normalizer='none'``), as the real
    eval sets are: normalisation belongs to the inference path, not the data on disk.
    """
    from ..data.dataset import make_dataset

    ds = make_dataset(
        args.split,
        kind="sim",
        stride=args.stride,
        normalizer="none",
        normalize=False,
        zero_p=not args.keep_pressure,
        rescale_fluct=getattr(args, "rescale_fluct", False),
    )
    n = len(ds)
    if args.max_windows_per_traj is not None:
        keep = []
        seen: dict[int, int] = {}
        for i, w in enumerate(ds.windows):
            if seen.get(w.traj, 0) < args.max_windows_per_traj:
                seen[w.traj] = seen.get(w.traj, 0) + 1
                keep.append(i)
    else:
        keep = list(range(n))

    print(f"split '{args.split}': {len(ds.trajectories)} trajectories, {len(keep)} windows")
    items = [ds[i] for i in keep]
    inputs = np.stack([it["input"].numpy() for it in items])
    targets = np.stack([it["target"].numpy() for it in items])
    # Per-window operating point, because it cannot be recovered from a simulated
    # window: sim is already non-dimensionalised, so every cell has u_fs ~ 1.09 and
    # the real Reynolds estimator would return the same number for all of them.
    cells = [ds.trajectories[ds.windows[i].traj] for i in keep]

    out = write_eval_set(
        args.name,
        inputs,
        targets,
        metadata={
            "kind": "sim",
            "split": args.split,
            "stride": args.stride,
            "sub_s": args.sub_s,
            "zero_p": not args.keep_pressure,
            "rescale_fluct": getattr(args, "rescale_fluct", False),
            "masked_to_real_piv": True,
            "trajectories": sorted({f"{c.re}_{c.aoa}" for c in cells}),
            "window_re": [c.re for c in cells],
            "window_aoa": [c.aoa for c in cells],
            "origins": [[ds.windows[i].traj, ds.windows[i].start] for i in keep],
        },
    )
    print(f"  inputs  : {inputs.shape}")
    print(f"  targets : {targets.shape}")
    print(f"  written : {out}")
    return 0


def _cmd_eval_set(args: argparse.Namespace) -> int:
    from ..data.splits import is_sim_split
    from ..data.windows import build_window_set

    if args.split and is_sim_split(args.split):
        if not args.allow_leaky_stride:
            assert_leak_free_stride(args.stride)
        return _build_sim_eval_set(args)

    paths: list[str] = list(args.h5)
    extra: dict = {}
    t_offset_frac: float | None = None

    if args.split:
        paths, extra = _resolve_split(args.split)
        t_offset_frac = extra.get("t_offset_frac")
        print(f"split '{args.split}': {len(paths)} trajectories {extra['trajectories']}")
    elif not paths:
        if not EXAMPLE_H5.exists():
            print(
                f"No --h5 or --split given and the example trajectory is absent "
                f"({EXAMPLE_H5}).\nRun `make data` first, or pass explicit --h5 paths.",
                file=sys.stderr,
            )
            return 2
        paths = [str(EXAMPLE_H5)]

    if not args.allow_leaky_stride:
        assert_leak_free_stride(args.stride)
    print(f"Building eval set '{args.name}' from {len(paths)} trajectory file(s)")
    ws = build_window_set(
        paths,
        stride=args.stride,
        sub_s=args.sub_s,
        max_windows_per_traj=args.max_windows_per_traj,
        max_frames=args.max_frames,
        t_offset_frac=t_offset_frac,
    )
    out = write_eval_set(
        args.name,
        ws.inputs,
        ws.targets,
        metadata={
            "sources": list(ws.sources),
            "stride": args.stride,
            "sub_s": args.sub_s,
            "origins": ws.origins.tolist(),
            **extra,
        },
    )
    print(f"  windows : {len(ws)}")
    print(f"  inputs  : {ws.inputs.shape}")
    print(f"  targets : {ws.targets.shape}")
    print(f"  written : {out}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    inputs, targets, meta = load_eval_set(args.name)
    print(f"eval set '{args.name}': {inputs.shape[0]} windows, inputs {inputs.shape}")

    predict = load_predict(args.submission)
    print(f"submission: {args.submission}")

    prediction, lower, upper, timing = run_submission(predict, inputs, batch_size=args.batch_size)
    if prediction.shape != targets.shape:
        print(
            f"SHAPE MISMATCH: prediction {prediction.shape} vs target {targets.shape}. "
            "The real scorer zeroes every subscore for this.",
            file=sys.stderr,
        )

    mean_t = NORMALIZATIONS[args.time_normalization](timing)
    work_dir = LOCAL_EVAL_DIR / args.name
    write_predictions(work_dir, prediction, mean_t, lower, upper)

    scores = run_official_scorer(work_dir, work_dir / "scores")
    print(format_report(scores, timing, args.time_normalization, prediction, targets, lower, upper))

    (work_dir / "scores" / "timing.json").write_text(
        json.dumps(timing.as_dict(), indent=2), encoding="utf-8"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rpde_t1.score", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval-set", help="build a local eval set from HDF5 trajectories")
    e.add_argument("--name", default="example", help="eval set name under data/local_eval/")
    e.add_argument("--h5", nargs="*", default=[], help="trajectory files (default: data/example)")
    e.add_argument(
        "--split",
        default=None,
        help="named split from data/splits.py: val_extrap | val_interp | val_time | train, "
        "or one of the simulated splits sim_train | sim_val_time | sim_val_interp | "
        "sim_val_extrap",
    )
    e.add_argument(
        "--rescale-fluct",
        dest="rescale_fluct",
        action="store_true",
        help="simulated splits only: apply the per-cell fluctuation rescaling to the eval "
        "targets too. Gives a model trained on rescaled sim its OWN matched distribution; "
        "such a set is NOT comparable with one built without it.",
    )
    e.add_argument(
        "--keep-pressure",
        dest="keep_pressure",
        action="store_true",
        help="simulated splits only: keep the pressure plane instead of zeroing it. Off "
        "by default, because sim's p has std 12.6 against u's 0.44, so a target carrying "
        "it makes the official scorer measure three channels and rel_l2 becomes a "
        "pressure metric -- incomparable to anything measured on real data.",
    )
    e.add_argument(
        "--stride",
        type=int,
        default=MIN_EVAL_STRIDE,
        help=f"frames between window starts (default {MIN_EVAL_STRIDE}). Below "
        f"{MIN_EVAL_STRIDE} one window's target lands inside another's input.",
    )
    e.add_argument(
        "--allow-leaky-stride",
        action="store_true",
        help="build anyway with a stride below the horizon. Only for reproducing the "
        "original, flawed evaluation set.",
    )
    e.add_argument("--sub-s", type=int, default=SUB_S, help="spatial subsampling (2 -> 32x64)")
    e.add_argument("--max-windows-per-traj", type=int, default=None)
    e.add_argument("--max-frames", type=int, default=None, help="cap frames read per trajectory")
    e.set_defaults(func=_cmd_eval_set)

    r = sub.add_parser("run", help="score a submission against a local eval set")
    r.add_argument(
        "--submission",
        default=str(VENDOR_KIT / "submission_template.py"),
        help="path to a submission.py-style file (default: the kit's persistence template)",
    )
    r.add_argument("--name", default="example", help="eval set name under data/local_eval/")
    r.add_argument("--batch-size", type=int, default=None, help="chunk predict() calls")
    r.add_argument(
        "--time-normalization",
        choices=sorted(NORMALIZATIONS),
        default="per_window",
        help="how to turn wall clock into mean_t_neural_s "
        "(default: per_window, confirmed to match the platform)",
    )
    r.set_defaults(func=_cmd_run)

    dg = sub.add_parser("diagnostics", help="ceiling ladders: how much of each axis is reachable")
    dg.add_argument("--split", default="val_extrap", help="eval set with stored predictions")
    dg.add_argument(
        "--blend-w",
        dest="blend_w",
        type=float,
        default=DEFAULT_BLEND_W,
        help="exponent on the input variance map in the free geometric blend "
        f"(default {DEFAULT_BLEND_W}, chosen on val_time)",
    )
    dg.add_argument("--out", default=None, help="write the ladders as JSON")
    dg.set_defaults(func=_cmd_diagnostics)

    return p


def _cmd_diagnostics(args: argparse.Namespace) -> int:
    from .diagnostics import collect_diagnostics, format_diagnostics, write_diagnostics

    ladders = collect_diagnostics(args.split, blend_w=args.blend_w)
    print(format_diagnostics(ladders))
    if args.out:
        print(f"written: {write_diagnostics(ladders, Path(args.out))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    np.seterr(all="ignore")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
