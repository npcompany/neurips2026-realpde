#!/usr/bin/env python
"""A LEARNED recurrent interval calibrator for the Track-2 stream (FINDINGS 208), offline lab.

FINDINGS 192's closed-form calibrator picks width multipliers by the SPS credit each trajectory's
revealed windows would have earned, and is worth ~+0.055 sps on ``hard5`` -- parked only because
its credit tables cost ~1.4 ms per step. FINDINGS 202 found that a learned memory of the SIGNED
error pays only at Re 3750, but the error's SIZE persists at every Reynolds number (192b: within-
trajectory sd 7.5% against 32-112% between trajectories). This learns the calibrator instead:

* per (pixel, channel), a GRU cell shared across pixels folds, window by window, features of the
  revealed error against the shipped interval -- per lead bucket, the mean log ``|e| / h`` and the
  coverage at the shipped width; the same features 3x3-pooled and pooled over the whole field;
* its state emits a log width multiplier per lead bucket (``tanh``-bounded to [1/2, 2], the
  closed-form grid's range), applied to the shipped half-width around the shipped centre;
* window k is served from windows 0..k-1 only (window 0 from the zero state).

Trained by maximising a smoothed SPS credit (the indicator replaced by a sigmoid whose temperature
anneals), on cached held-out stream outputs (``scripts/t2/t2_stream_cache.py`` through the SHIPPED
serving path of the train-split control stacks). CROSS-FITTED, like 192's selection: fit on the
three seeds' ``val_interp`` streams and report ``val_extrap``, and the reverse. ``hard5`` shares
its 16500 cells with ``val_interp`` and its 26700 cells with ``val_extrap``, so its clean read
scores each cell with the fold that never saw it. Every number is scored with the exact
(hard-indicator) credit of the vendored scorer, via the lab's own ``credit_total`` arithmetic.

Baselines on the same caches: the shipped intervals, and 192's closed-form pick (per pixel on the
channel x lead-bucket stage, n0s 5, n0p 50).

    scripts/gpu.sh t2recal uv run python scripts/t2/t2_recal_lab.py --out runs/t2_recal/lab.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_autocal_lab as lab  # noqa: E402

from rpde_t1.models.interval_recal import RecurrentCalibrator, save_calibrator  # noqa: E402


def load_split(cache: Path, split: str, device) -> dict:
    """One cached stream as per-trajectory tensors ``(n_traj, n_win, 20, H, W, C)``."""
    d = lab.load(cache / f"{split}.npz")
    trajs = np.unique(d["traj"])
    order = [np.where(d["traj"] == t)[0][np.argsort(d["win"][d["traj"] == t])] for t in trajs]
    n_win = {len(o) for o in order}
    if len(n_win) != 1:
        raise ValueError(f"{cache}/{split}: trajectories of unequal length {n_win}")
    idx = np.stack(order)  # (n_traj, n_win)

    def t(x, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(x[idx]), dtype=dtype, device=device)

    return {
        "ae": t(np.abs(d["e"])),
        "e": t(d["e"]),
        "h": t(d["h"]),
        "scored": t(d["scored"]),
        "w1": t(d["w1"]),
        "sims": [str(s) for s in d["sims"][trajs]],
        "np": d,
        "idx": idx,
    }


def credit(ae, h, scored, w1, mult, sigma, tau=None):
    """Per-trajectory (numerator, denominator) of the SPS credit, ``mult`` broadcast on ``h``.

    ``tau`` None is the exact indicator; a float is the sigmoid-smoothed training surrogate.
    ``ae`` may be a signed error minus a centre shift (``|e - shift|``): the credit only reads
    the distance to the interval centre.
    """
    k = mult * h
    inside = (ae <= k).float() if tau is None else torch.sigmoid((k - ae) / (tau * h + 1e-6))
    el = inside * torch.exp(-2.0 * k / sigma) * scored
    per_win = el.flatten(2).sum(-1) * w1  # (n_traj, n_win)
    return per_win.sum(1), scored.flatten(1).sum(1)


def fit(train: list[dict], sigma: float, args, device) -> RecurrentCalibrator:
    torch.manual_seed(args.seed)
    model = RecurrentCalibrator(
        args.hidden,
        arch=args.arch,
        context=not args.no_context,
        cov_grid=tuple(args.cov_grid),
        sigma=sigma,  # the width feature is always on since 2026-09-25 (sigma is required)
        observe_every=args.observe_every,
        signed=args.signed,
        shift_head=args.shift_head,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(args.steps):
        frac = step / max(1, args.steps - 1)
        tau = math.exp(math.log(args.tau0) + frac * (math.log(args.tau1) - math.log(args.tau0)))
        num = den = 0.0
        opt.zero_grad(set_to_none=True)
        for d in train:  # one backward per stream keeps peak memory at one split's graph
            m = model(d["ae"], d["h"], d["scored"], d["e"])
            ae = d["ae"]
            if model.shift_head:
                m, sh = m
                ae = (d["e"] - sh * d["h"]).abs()
            nu, de = credit(ae, d["h"], d["scored"], d["w1"], m, sigma, tau)
            tot = sum(float(x["scored"].sum()) for x in train)
            (-100.0 * nu.sum() / tot).backward()
            num += float(nu.sum())
            den += float(de.sum())
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(
                f"  step {step:5d} tau {tau:.3f} soft sps {100 * num / den:.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return model


def per_traj(d, mult, sigma, shift=None) -> dict[str, tuple[float, float]]:
    if isinstance(mult, tuple):  # a shift-head calibrator: (multiplier, shift in units of h)
        mult, sh = mult
        shift = sh * d["h"] if shift is None else shift + sh * d["h"]
    ae = d["ae"] if shift is None else (d["e"] - shift).abs()
    nu, de = credit(ae, d["h"], d["scored"], d["w1"], mult, sigma)
    return {s: (float(a), float(b)) for s, a, b in zip(d["sims"], nu, de, strict=True)}


def running_shift(d, beta: float, n0: float, per_bucket: bool = False) -> torch.Tensor:
    """FINDINGS 192's centre shift, causal: ``beta * r / (r + n0)`` times the per-pixel running
    mean signed error over the ``r`` revealed windows (leads pooled, or per lead bucket), as a
    shift of BOTH bounds, ``(n_traj, n_win, 20, H, W, C)``. Window 0 is unshifted."""
    e, sc = d["e"], d["scored"]
    n, nw, t, hh, ww, c = e.shape
    if per_bucket:
        per = t // 4
        num = (e * sc).view(n, nw, 4, per, hh, ww, c).sum(3)
        den = sc.view(n, nw, 4, per, hh, ww, c).sum(3).clamp_min(1.0)
        m = (num / den)[:, :, :, None].expand(n, nw, 4, per, hh, ww, c).reshape(n, nw, t, hh, ww, c)
    else:
        m = ((e * sc).sum(2) / sc.sum(2).clamp_min(1.0))[:, :, None].expand_as(e)
    cum = torch.cumsum(m, dim=1)
    prev = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)  # windows < k
    r = torch.arange(nw, device=e.device, dtype=e.dtype).view(1, nw, 1, 1, 1, 1)
    return beta * (r / (r + n0)) * prev / r.clamp_min(1.0)


def autocal_mult(d, sigma) -> torch.Tensor:
    """FINDINGS 192's pick on this stream, as a multiplier tensor aligned with ``d``'s layout."""
    lab.SIGMA = sigma
    nd = d["np"]
    lo, hi = lab.run_pix_on_scalar(nd, 5.0, 50.0, {}, base="chanlead")
    m = (0.5 * (hi - lo)) / np.where(nd["h"] > 0, nd["h"], 1.0)
    m = np.where(nd["h"] > 0, m, 1.0)
    return torch.as_tensor(m[d["idx"]], dtype=torch.float32, device=d["ae"].device)


def agg(rows: dict[str, tuple[float, float]], sims) -> float:
    num = sum(rows[s][0] for s in sims)
    den = sum(rows[s][1] for s in sims)
    return 100.0 * num / den


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--caches", nargs="+", default=[f"runs/t2_autocal/cache/ctl_s{s}" for s in range(3)]
    )
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--tau0", type=float, default=0.2)
    ap.add_argument("--tau1", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["gru", "mean"], default="gru")
    ap.add_argument("--no-context", action="store_true", help="drop the pooled/field features")
    ap.add_argument("--cov-grid", type=float, nargs="+", default=[1.0])
    ap.add_argument("--h-feature", action="store_true", help="add mean log(h/sigma) per bucket")
    ap.add_argument(
        "--observe-every", type=int, default=1, help="fold in every k-th revealed window only"
    )
    ap.add_argument(
        "--shift-betas",
        type=float,
        nargs="*",
        default=[],
        help="also score the closed-form running-signed-error centre shift at these betas, "
        "alone and on top of the learned multipliers (n0 5; leads pooled and per bucket)",
    )
    ap.add_argument("--load-dir", default="", help="skip fitting: load fit_<split>.pt from here")
    ap.add_argument("--signed", action="store_true", help="feed the mean signed error / h too")
    ap.add_argument(
        "--fit-all",
        action="store_true",
        help="SHIPPING fit: one calibrator on every cached stream of every seed (val_interp and "
        "val_extrap), saved as fit_all.pt in --save-dir; no held-out read (the cross-fitted "
        "rows are the evidence), so nothing is scored",
    )
    ap.add_argument("--shift-head", action="store_true", help="learn a centre shift as well")
    ap.add_argument("--no-autocal", action="store_true", help="skip the closed-form baseline")
    ap.add_argument("--save-dir", default="", help="save each fold's calibrator here")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from rpde_t2.stream import official_scoring

    sigma = float(official_scoring().SIGMA_GLOBAL)
    device = "cuda"
    data = {
        (Path(c).name, s): load_split(Path(c), s, device)
        for c in args.caches
        for s in ("val_interp", "val_extrap")
    }
    if args.fit_all:
        if not args.save_dir:
            raise SystemExit("--fit-all needs --save-dir")
        model = fit(list(data.values()), sigma, args, device)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        save_calibrator(model, Path(args.save_dir) / "fit_all.pt")
        print(f"wrote {Path(args.save_dir) / 'fit_all.pt'} (fit on {len(data)} streams; unscored)")
        return 0
    hard5 = ["16500_10", "16500_15", "16500_20", "26700_10", "26700_15"]
    report: dict = {"args": vars(args), "sigma": sigma, "folds": {}, "seeds": {}}
    rows: dict = {}  # (seed, split, method) -> per-trajectory credit
    for fit_split, rep_split in (("val_interp", "val_extrap"), ("val_extrap", "val_interp")):
        print(f"fold: fit {fit_split} x{len(args.caches)} seeds -> report {rep_split}", flush=True)
        if args.load_dir:
            from rpde_t1.models.interval_recal import load_calibrator

            model = load_calibrator(Path(args.load_dir) / f"fit_{fit_split}.pt", device)
        else:
            model = fit(
                [data[(Path(c).name, fit_split)] for c in args.caches], sigma, args, device
            )
        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            save_calibrator(model, Path(args.save_dir) / f"fit_{fit_split}.pt")
        report["folds"][f"{fit_split}->{rep_split}"] = {
            "n_params": sum(p.numel() for p in model.parameters())
        }
        with torch.no_grad():
            for c in args.caches:
                d = data[(Path(c).name, rep_split)]
                rows[(Path(c).name, rep_split, "learned")] = per_traj(
                    d, model(d["ae"], d["h"], d["scored"], d["e"]), sigma
                )
                rows[(Path(c).name, rep_split, "shipped")] = per_traj(
                    d, torch.ones_like(d["h"]), sigma
                )
                if args.shift_betas:
                    mult = model(d["ae"], d["h"], d["scored"], d["e"])
                    for beta in args.shift_betas:
                        for pb in (False, True):
                            sh = running_shift(d, beta, 5.0, per_bucket=pb)
                            tag = f"b{beta:g}{'pb' if pb else ''}"
                            rows[(Path(c).name, rep_split, f"shift_{tag}")] = per_traj(
                                d, torch.ones_like(d["h"]), sigma, shift=sh
                            )
                            rows[(Path(c).name, rep_split, f"learned+shift_{tag}")] = per_traj(
                                d, mult, sigma, shift=sh
                            )
                            del sh
                if not args.no_autocal:
                    rows[(Path(c).name, rep_split, "autocal")] = per_traj(
                        d, autocal_mult(d, sigma), sigma
                    )
        del model
        torch.cuda.empty_cache()

    methods = ["learned"] + ([] if args.no_autocal else ["autocal"])
    for beta in args.shift_betas:
        for pb in (False, True):
            tag = f"b{beta:g}{'pb' if pb else ''}"
            methods += [f"shift_{tag}", f"learned+shift_{tag}"]
    for c in args.caches:
        seed = Path(c).name
        out = {}
        for split in ("val_interp", "val_extrap"):
            base = rows[(seed, split, "shipped")]
            sims = list(base)
            out[split] = {"base_sps": agg(base, sims)}
            for m in methods:
                out[split][f"d_{m}"] = agg(rows[(seed, split, m)], sims) - agg(base, sims)
        # hard5, each cell read from the fold that never fitted on it
        merged = {m: {} for m in ["shipped", *methods]}
        for m in merged:
            for split in ("val_interp", "val_extrap"):
                merged[m].update(rows[(seed, split, m)])
        out["hard5_clean"] = {"base_sps": agg(merged["shipped"], hard5)}
        for m in methods:
            out["hard5_clean"][f"d_{m}"] = agg(merged[m], hard5) - agg(merged["shipped"], hard5)
        out["per_cell_d_learned"] = {
            s: 100 * (merged["learned"][s][0] - merged["shipped"][s][0]) / merged["shipped"][s][1]
            for s in merged["shipped"]
        }
        report["seeds"][seed] = out
        print(
            f"{seed}: "
            + "  ".join(
                f"{sp} "
                + " ".join(f"{k}={v:+.4f}" for k, v in out[sp].items() if k.startswith("d_"))
                for sp in ("val_interp", "val_extrap", "hard5_clean")
            ),
            flush=True,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
