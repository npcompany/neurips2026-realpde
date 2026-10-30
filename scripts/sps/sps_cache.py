"""Cache a bounded checkpoint's outputs on named eval sets, for offline interval work.

One forward pass per split, written to ``runs/sps_cache/<tag>__<split>.npz``. Everything
downstream (width rules, head refits, the oracle ladder) reads the cache, so a frozen
backbone is never run twice.

**This file is the whole chain's only source of run-to-run variation**, and it was a real
one. Everything downstream is bit-exact -- refitting a head on a fixed cache reproduces its
weights bit for bit and its SPS to four decimals -- but two runs of *this* script, on the
same weights and the same eval set, differed at 3e-7 relative on 41M of 45M elements. A
1500-step head fit amplifies that into weight differences of 0.39 (on weights up to 3.14)
and ~0.02 SPS per split, so an artefact was not reproducible from its own recipe.

The mechanism is algorithm *selection*, not arithmetic: ``cudnn.benchmark`` is off, but cuDNN
still picks a convolution algorithm against the workspace it can obtain, and this box shares
its GPU -- so the same forward pass takes a different path depending on what else is
resident. Measured 2026-08-12: the two caches were written with 14.9 GB and 7 GB free.

Pinning the choice costs caching time only. ``models.predictor.make_predict`` is deliberately
left alone, so nothing the platform times changes and no submission's score moves.

    uv run python scripts/sps/sps_cache.py runs/unet_bounds_frozen_cond/last.pt \
        --tag unet --splits train_probe val_time val_interp val_extrap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rpde_t1.models.predictor import estimate_freestream, infer_conditioning, make_predict
from rpde_t1.score.harness import load_eval_set
from rpde_t1.score.sps_lab import cache_path, save_cache
from rpde_t1.train.evaluate import load_checkpoint
from rpde_t1.train.loop import geom_names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--out-dir", default="runs/sps_cache")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--tke-head",
        dest="tke_head",
        default=None,
        help="apply a fitted second-moment head (the .pt beside a `train tke-head --out` JSON) "
        "to the cached predictions. Needed to ask whether the head's ceiling gain BANKS "
        "through the interval stack: FINDINGS 41 records ceiling gains failing to convert, so "
        "an accuracy number is not a banked one. The interval head is fitted on whatever is "
        "cached here, so caching the corrected field is what makes the two arms comparable",
    )
    ap.add_argument(
        "--tke-shrink",
        dest="tke_shrink",
        type=float,
        default=None,
        help="weight on the learned head (default DEFAULT_SHRINK). With `--tke-blend 0.0`, the "
        "configuration every ship script uses, the correction is exactly `shrink x head` and the "
        "shrink caps the rel_l2 cost of a head fitted on tke alone. 0.5 is what ships",
    )
    ap.add_argument(
        "--tke-blend",
        dest="tke_blend",
        type=float,
        default=0.0,
        help="closed-form blend exponent the head is shrunk toward. 0.0 -- the default here, and "
        "what every ship script passes -- means NO blend. Left unset this would take "
        "`make_predict`'s None default, which resolves to 0.5 whenever a head is present and "
        "serves a mixture no artifact ships (the correction appended to FINDINGS 134e)",
    )
    args = ap.parse_args()

    # Operating-point overrides (RPDE_RE_CORNERS / RPDE_AOA_NET): cache the corrected field under
    # the same conditioning the backbone was trained/served with. No-op unless the env is set.
    from rpde_t1.models.aoa_net import install_from_env  # noqa: PLC0415

    if install_from_env():
        print("[estimators] operating-point overrides active", flush=True)

    # See the module docstring: this is what makes a fitted head reproducible from its
    # recipe. Restricted to deterministic convolution algorithms and with autotuning
    # explicitly off, so the algorithm cannot be chosen by how much GPU memory happened to
    # be free. Verified by writing the cache twice and diffing, not by trusting the flag.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(Path(args.ckpt), device=device)
    out_dir = Path(args.out_dir)

    head = None
    head_tau = False
    if args.tke_head:
        from rpde_t1.models.tke_head import head_uses_tau, load_head  # noqa: PLC0415

        head = load_head(Path(args.tke_head), device=device)
        head_tau = head_uses_tau(Path(args.tke_head))
        print(
            f"second-moment head: {args.tke_head} "
            f"(shrink={args.tke_shrink}, blend={args.tke_blend})"
        )

    # A stream-state backbone (TrainConfig.stream_state) reads two causal pseudo-frames per
    # window; the eval sets store plain 20-frame windows, so the state is synthesised from
    # their recorded provenance. The CACHE keeps the eval set's own 20-frame inputs (the head
    # fit reads the measured window, same as serving via BoundedModel.state_frames).
    stream_state = str(getattr(cfg, "stream_state", "") or "")
    state_frames = 0
    if stream_state:
        from rpde_t1.models.factory import stream_state_frames

        state_frames = stream_state_frames(stream_state)
        print(f"stream-state backbone: synthesising '{stream_state}' pseudo-frames per window")

    for split in args.splits:
        inputs, target, meta = load_eval_set(split)
        if stream_state:
            from rpde_t1.score.harness import prepend_eval_state

            model_inputs = prepend_eval_state(inputs, meta, mode=stream_state)
        else:
            model_inputs = inputs
        predict = make_predict(
            model,
            device=device,
            batch_size=args.batch_size,
            return_bounds=True,
            state_frames=state_frames,
            normalizer=cfg.normalizer,
            domain="real",
            geom_channels=geom_names(cfg),
            cond_mode=cfg.cond_mode,
            tke_head=head,
            tke_shrink=args.tke_shrink,
            tke_blend=args.tke_blend,
            tke_head_uses_tau=head_tau,
        )
        out = predict(model_inputs, None)
        # A backbone-only checkpoint has no interval head. That is a legitimate input here --
        # the whole point of the log-error head is that it is fitted on cached outputs after
        # the fact -- so store degenerate bounds and record their absence, rather than
        # refusing. ``has_bounds`` exists so a zero half-width cannot be silently consumed as
        # if it were the incumbent head's prediction.
        bounded = isinstance(out, dict)
        pred = out["prediction"] if bounded else out
        path = cache_path(out_dir, split, args.tag)
        # Bounds are written only when they carry information. Without a head they equal
        # ``pred``, and storing them anyway tripled the dominant array -- 1.10 GB for
        # ``train_all`` where 0.40 GB says the same thing. ``load_cache`` reconstructs them
        # from ``pred`` when the keys are absent, so this is a storage change only.
        bounds = {"lower": out["lower"], "upper": out["upper"]} if bounded else {}
        save_cache(
            path,
            pred=pred,
            **bounds,
            has_bounds=np.array([float(bounded)]),
            u_fs=estimate_freestream(inputs),
            # ``cond_mode`` from the checkpoint, exactly as the forward above uses it:
            # ``scale_fit.build_head`` sizes the head's first layer and its FiLM
            # projection from THIS array's width, so a 3-wide vector cached for a
            # 4-wide model fits a head that cannot be assembled or served (caught by
            # sps_assemble's load; hit by the dt-conditioned recipe 2026-09-18, and
            # latent for ``operating_point_tau`` since the FiLM head landed).
            cond=infer_conditioning(inputs, None, domain="real", cond_mode=cfg.cond_mode),
        )
        print(
            f"{split:<12} n={inputs.shape[0]:<5} -> {path}  "
            f"mean|err|={np.abs(target[..., :2] - pred[..., :2]).mean():.5f}"
            f"{'' if bounded else '  (no bound head)'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
