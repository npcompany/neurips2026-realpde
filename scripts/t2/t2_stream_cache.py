#!/usr/bin/env python
"""Cache a Track-2 bundle's stream outputs through the SHIPPED serving path, for interval labs.

`t2_interval_cache.py` predates the stream-state backbone: it drives the host `make_predict` path,
which neither builds the state pseudo-frames nor matches what a bundle serves. This drives
`make_gpu_predict` behind the GPU-native `FrozenPort` -- the path every current Track-2 artefact
runs -- with the bundle's own width rule and second-moment head, and stores, per step, the measured
channels of the prediction, the target and the shipped bounds, plus the trajectory id, the window
index within the trajectory and the metric's per-window accuracy factor `pm` (which the interval
cannot move, so it is cached rather than recomputed).

    scripts/gpu.sh t2cache uv run python scripts/t2/t2_stream_cache.py \
        runs/t2_backbone/ctl_s0_bundle --splits val_interp val_extrap hard5 \
        --out-dir runs/t2_autocal/cache/ctl_s0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rpde_t1.models.factory import load_for_inference
from rpde_t1.models.predictor import make_gpu_predict
from rpde_t1.models.tke_head import head_uses_tau, load_head
from rpde_t1.models.width_rule import WidthRule
from rpde_t1.paths import REPO_ROOT
from rpde_t2.stream import build_stream, official_scoring, run_stream
from rpde_t2.ttt import FrozenPort

STREAM_DIR = REPO_ROOT / "data" / "local_eval"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle")
    ap.add_argument("--splits", nargs="+", default=["val_interp", "val_extrap", "hard5"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--compile-mode", default="cudagraph")
    ap.add_argument("--tke-shrink", type=float, default=0.5, help="second-moment head shrink")
    ap.add_argument("--no-tke-head", action="store_true", help="cache the head-free prediction")
    args = ap.parse_args(argv)

    run = Path(args.bundle)
    model, cfg = load_for_inference(run / "last.pt", kit_dir=None, device="cuda")
    mode = str(cfg.get("stream_state") or "")
    hp = run / "tke_head.pt"
    fn = make_gpu_predict(
        model,
        device="cuda",
        width_rule=WidthRule.from_json(run / "width_rule.json"),
        lead_anchor=None,
        cond_mode=cfg.get("cond_mode", "operating_point"),
        compile_mode=args.compile_mode,
        stream_state=mode,
        tke_head=None if args.no_tke_head else load_head(str(hp), device="cuda"),
        tke_shrink=args.tke_shrink,
        tke_blend=0.0,
        tke_head_uses_tau=False if args.no_tke_head else head_uses_tau(str(hp)),
    )
    official = official_scoring()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        steps = build_stream(sorted((STREAM_DIR / f"t2_{split}" / "test_real").glob("*.h5")))
        port = FrozenPort(fn, device="cuda", premapped=True, gpu_native=True, state_mode=mode)
        r = run_stream(port, steps, device="cuda")
        c = official.measured_channels(r.target)
        dm = official.rel_l2_per_sample(r.prediction, r.target, c)
        tke = official.tke_rel_l2_per_sample(r.prediction, r.target, c)
        mvpe = official.mvpe_rel_l2_per_sample(r.prediction, r.target)
        pm = 0.5 * (dm / (0.5 + dm)) + 0.3 * (tke / (0.5 + tke)) + 0.2 * (mvpe / (0.5 + mvpe))
        sims = sorted({s.sim_id for s in steps})
        np.savez(
            out_dir / f"{split}.npz",
            pred=r.prediction[..., :c].astype(np.float32),
            target=r.target[..., :c].astype(np.float32),
            lower=r.lower[..., :c].astype(np.float32),
            upper=r.upper[..., :c].astype(np.float32),
            traj=np.array([sims.index(s.sim_id) for s in steps], dtype=np.int64),
            win=np.array([s.time_id // 20 for s in steps], dtype=np.int64),
            pm=np.asarray(pm, dtype=np.float64),
            sims=np.array(sims),
        )
        sps, cov = official.aggregate_sps(r.prediction, r.target, c, lower=r.lower, upper=r.upper)
        print(
            f"{run.name} {split}: {len(steps)} steps / {len(sims)} traj, "
            f"sps_score={official.score_sps(sps):.4f} coverage={cov:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
