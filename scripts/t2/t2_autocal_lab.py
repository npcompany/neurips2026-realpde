#!/usr/bin/env python
"""Self-calibrating intervals on the Track-2 stream: an offline lab over cached stream outputs.

The metric's credit for one scored element is ``(1 - pm_w) * 1{lower <= t <= upper} *
exp(-(upper - lower) / sigma)`` (vendored ``aggregate_sps``), so the optimal width of an element
depends on ITS error distribution against the global sigma, and the bounds need not be centred on
the prediction. At step k the stream has revealed every earlier window's target, so each
trajectory's own realised errors against the shipped intervals are observable -- the whole
question is which part of them is persistent enough to act on (FINDINGS 136: the per-window anomaly
is not; a per-trajectory scalar and pixel map are, a little).

Every calibrator here is CAUSAL: window k of a trajectory is adjusted using windows 0..k-1 only,
and window 0 is served as shipped. Each one DIRECTLY maximises the empirical credit it would have
earned on the revealed windows (a grid over multipliers / shifts), then shrinks toward the shipped
interval with an evidence weight ``n / (n + n0)``. Families:

``scalar``     one width multiplier per trajectory
``lead``       per trajectory x lead bucket (4 x 5 leads)
``chan``       per trajectory x channel
``chanlead``   per trajectory x channel x lead bucket
``pix``        per trajectory x pixel x channel (leads pooled), shrunk toward the ``scalar`` pick
``shift``      the ``scalar`` pick plus a per-pixel centre shift by the running signed error
``asym``       separate lower / upper multipliers per trajectory

Selection is CROSS: a hyper-parameter is picked on one held-out stream and reported on the other
(``val_interp`` <-> ``val_extrap``); ``hard5`` shares 3 of its 5 cells with ``val_interp`` and is
reported for reference only.

    uv run python scripts/t2/t2_autocal_lab.py --cache runs/t2_autocal/cache/ctl_s0 \
        --out runs/t2_autocal/lab_s0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIGMA = None  # read from the vendored scorer at runtime (never retyped: FINDINGS 136a)
KGRID = np.exp(np.linspace(np.log(0.5), np.log(2.0), 29))
N_LEAD_BUCKETS = 4
LEAD_SUB = 1  # credit tables from every LEAD_SUB-th lead (round 4: the device cost lever)
LEAD_ROT = False  # round 5: rotate which leads feed the tables, window to window


def load(path: Path) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["c"] = 0.5 * (d["lower"] + d["upper"])
    d["h"] = 0.5 * (d["upper"] - d["lower"])
    d["e"] = d["target"] - d["c"]  # signed error against the interval centre
    d["scored"] = d["target"] != 0.0
    d["w1"] = (1.0 - d["pm"]).astype(np.float64)
    return d


def credit_total(d: dict, lo: np.ndarray, hi: np.ndarray) -> float:
    t = d["target"]
    inside = (t >= lo) & (t <= hi) & d["scored"]
    el = np.exp(-(hi - lo) / SIGMA) * inside
    per = el.reshape(len(t), -1).sum(axis=1, dtype=np.float64)
    return float((per * d["w1"]).sum() / d["scored"].sum())


def score(d: dict, lo: np.ndarray, hi: np.ndarray) -> float:
    return 100.0 * credit_total(d, lo, hi)


def group_credit(d: dict, gid: np.ndarray, n_groups: int, mults: np.ndarray) -> np.ndarray:
    """Credit per (window, group, multiplier) for symmetric widths ``k * h`` around the centre.

    ``gid`` is broadcastable to the element shape ``(20, H, W, C)`` and labels each element's group.
    Returns ``(N, n_groups, K)``, already weighted by the window's accuracy factor.
    """
    n = len(d["e"])
    ae = np.abs(d["e"])
    h = d["h"]
    sc = d["scored"]
    if LEAD_SUB > 1 and LEAD_ROT:
        # ROTATING subsample: window i feeds the tables from leads (r_i mod sub)::sub, so every
        # lead contributes over a trajectory at the same per-step cost as a fixed subsample.
        off = (d["win"] % LEAD_SUB).astype(int)
        pick = np.stack([np.arange(o, 20, LEAD_SUB) for o in off])  # (N, 20/sub)
        rows = np.arange(n)[:, None]
        ae, h, sc = ae[rows, pick], h[rows, pick], sc[rows, pick]
        gfull = np.broadcast_to(gid, d["e"].shape[1:])
        gl = gfull[pick]  # (N, 20/sub, H, W, C)
        out = np.zeros((n, n_groups, len(mults)))
        for j, k in enumerate(mults):
            el = (ae <= k * h) * np.exp(-2.0 * k * h / SIGMA) * sc
            for i in range(n):
                out[i, :, j] = np.bincount(gl[i].reshape(-1), weights=el[i].reshape(-1),
                                           minlength=n_groups)
        return out * d["w1"][:, None, None]
    if LEAD_SUB > 1:  # tables from a lead subsample: the served width still covers every lead
        ae, h, sc = ae[:, ::LEAD_SUB], h[:, ::LEAD_SUB], sc[:, ::LEAD_SUB]
        gid = np.broadcast_to(gid, d["e"].shape[1:])[::LEAD_SUB]
    g = np.broadcast_to(gid, ae.shape[1:]).reshape(-1)
    out = np.zeros((n, n_groups, len(mults)))
    for j, k in enumerate(mults):
        el = (ae <= k * h) * np.exp(-2.0 * k * h / SIGMA) * sc
        el = el.reshape(n, -1)
        for i in range(n):
            out[i, :, j] = np.bincount(g, weights=el[i], minlength=n_groups)
    return out * d["w1"][:, None, None]


FREEZE_N = 0  # round 6: stop updating the tables (and r) after this many revealed windows


def online_pick(d: dict, gc: np.ndarray, n0: float, mults: np.ndarray) -> np.ndarray:
    """Causal multiplier per (window, group): argmax of the credit revealed so far, shrunk."""
    n, ng, _ = gc.shape
    logm = np.log(mults)
    pick = np.ones((n, ng))
    for tr in np.unique(d["traj"]):
        idx = np.where(d["traj"] == tr)[0]
        idx = idx[np.argsort(d["win"][idx])]
        cum = np.zeros((ng, len(mults)))
        for r, i in enumerate(idx):
            re = min(r, FREEZE_N) if FREEZE_N else r
            if r > 0:
                best = logm[np.argmax(cum, axis=1)]
                pick[i] = np.exp(best * (re / (re + n0)))
            if not FREEZE_N or r < FREEZE_N:
                cum += gc[i]
    return pick


def lead_bucket() -> np.ndarray:
    return (np.arange(20) * N_LEAD_BUCKETS // 20).reshape(20, 1, 1, 1)


def apply_group(d: dict, gid, pick: np.ndarray):
    """Widths ``k_g * h`` with the per-window group multipliers ``pick`` (N, n_groups)."""
    g = np.broadcast_to(gid, d["h"].shape[1:])
    k = pick[:, g]  # (N, 20, H, W, C)
    return d["c"] - k * d["h"], d["c"] + k * d["h"]


def family_groups(name: str, shape) -> tuple[np.ndarray, int]:
    _, h, w, c = shape
    if name == "scalar":
        return np.zeros((1, 1, 1, 1), dtype=np.int64), 1
    if name == "lead":
        return lead_bucket(), N_LEAD_BUCKETS
    if name == "chan":
        return np.arange(c).reshape(1, 1, 1, c), c
    if name == "chanlead":
        return lead_bucket() * c + np.arange(c).reshape(1, 1, 1, c), N_LEAD_BUCKETS * c
    if name == "pix":
        return np.arange(h * w * c).reshape(1, h, w, c), h * w * c
    raise KeyError(name)


def run_family(d: dict, name: str, n0: float, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    gid, ng = family_groups(name, d["h"].shape[1:])
    key = ("gc", name)
    if key not in cache:
        cache[key] = group_credit(d, gid, ng, KGRID)
    pick = online_pick(d, cache[key], n0, KGRID)
    return apply_group(d, gid, pick)


def run_pix_on_scalar(d: dict, n0_s: float, n0_p: float, cache: dict, base="scalar", smooth=0):
    """Per-pixel multiplier ON TOP of a coarser pick (``base``): pixel groups picked relative to
    it. ``smooth`` > 0 box-filters the per-pixel log multiplier over a (2s+1)^2 neighbourhood,
    pooling evidence across neighbours instead of trusting each pixel's own few samples."""
    lo_s, hi_s = run_family(d, base, n0_s, cache)
    ds = dict(d)
    ds["h"] = 0.5 * (hi_s - lo_s)
    key = ("gc_pix_on", base, n0_s)
    gid, ng = family_groups("pix", d["h"].shape[1:])
    if key not in cache:
        cache[key] = group_credit(ds, gid, ng, KGRID)
    pick = online_pick(ds, cache[key], n0_p, KGRID)
    if smooth:
        hh, ww, cc = d["h"].shape[2:]
        lp = np.log(pick).reshape(len(pick), hh, ww, cc)
        pad = np.pad(lp, ((0, 0), (smooth, smooth), (smooth, smooth), (0, 0)), mode="edge")
        acc = np.zeros_like(lp)
        for dy in range(2 * smooth + 1):
            for dx in range(2 * smooth + 1):
                acc += pad[:, dy : dy + hh, dx : dx + ww]
        pick = np.exp(acc / (2 * smooth + 1) ** 2).reshape(len(pick), -1)
    return apply_group(ds, gid, pick)


def running_signed(d: dict) -> np.ndarray:
    """Causal per-pixel mean signed error over revealed windows (leads pooled), (N, H, W, C)."""
    n = len(d["e"])
    out = np.zeros((n,) + d["e"].shape[2:], dtype=np.float32)
    for tr in np.unique(d["traj"]):
        idx = np.where(d["traj"] == tr)[0]
        idx = idx[np.argsort(d["win"][idx])]
        s = np.zeros(d["e"].shape[2:], dtype=np.float64)
        for r, i in enumerate(idx):
            if r > 0:
                out[i] = s / r
            s += (d["e"][i] * d["scored"][i]).mean(axis=0)
    return out


def run_shift(d: dict, n0: float, beta: float, cache: dict):
    lo, hi = run_family(d, "scalar", n0, cache)
    if "signed" not in cache:
        cache["signed"] = running_signed(d)
    r = np.array([min(d["win"][i], 1e9) for i in range(len(d["win"]))], dtype=np.float64)
    lam = (beta * r / (r + n0))[:, None, None, None, None]
    sh = lam * cache["signed"][:, None]
    return lo + sh, hi + sh


def run_asym(d: dict, n0: float, cache: dict):
    """Separate lower/upper multipliers per trajectory, each picked on its own half-credit."""
    if "asym" not in cache:
        n = len(d["e"])
        e, h, sc = d["e"], d["h"], d["scored"]
        m = KGRID
        gc = np.zeros((n, len(m), len(m)))
        for a, kl in enumerate(m):
            for b, ku in enumerate(m):
                inside = (e >= -kl * h) & (e <= ku * h) & sc
                el = inside * np.exp(-(kl + ku) * h / SIGMA)
                gc[:, a, b] = el.reshape(n, -1).sum(axis=1)
        cache["asym"] = gc * d["w1"][:, None, None]
    gc = cache["asym"]
    n = len(d["e"])
    kl_p, ku_p = np.ones(n), np.ones(n)
    lm = np.log(KGRID)
    for tr in np.unique(d["traj"]):
        idx = np.where(d["traj"] == tr)[0]
        idx = idx[np.argsort(d["win"][idx])]
        cum = np.zeros(gc.shape[1:])
        for r, i in enumerate(idx):
            if r > 0:
                a, b = np.unravel_index(np.argmax(cum), cum.shape)
                w = r / (r + n0)
                kl_p[i], ku_p[i] = np.exp(lm[a] * w), np.exp(lm[b] * w)
            cum += gc[i]
    kl_p = kl_p[:, None, None, None, None]
    ku_p = ku_p[:, None, None, None, None]
    return d["c"] - kl_p * d["h"], d["c"] + ku_p * d["h"]


N0S = (0.5, 2.0, 5.0, 10.0, 20.0)
BETAS = (0.25, 0.5, 1.0)


def arms_v2(d: dict, cache: dict):
    """The second round, around the first round's winner (per-pixel on the scalar pick)."""
    for n0p in (20.0, 50.0, 100.0, 200.0):
        yield "pix", f"n0s=5,n0p={n0p}", *run_pix_on_scalar(d, 5.0, n0p, cache)
    for n0p in (20.0, 50.0, 100.0):
        yield "pixcl", f"n0s=5,n0p={n0p}", *run_pix_on_scalar(
            d, 5.0, n0p, cache, base="chanlead"
        )
    for sm in (1, 2):
        for n0p in (5.0, 20.0, 50.0):
            yield f"pixsm{sm}", f"n0s=5,n0p={n0p}", *run_pix_on_scalar(
                d, 5.0, n0p, cache, smooth=sm
            )
    if "signed" not in cache:
        cache["signed"] = running_signed(d)
    r = d["win"].astype(np.float64)
    for n0p in (20.0, 50.0):
        lo, hi = run_pix_on_scalar(d, 5.0, n0p, cache)
        for beta in (0.125, 0.25):
            sh = (beta * r / (r + 5.0))[:, None, None, None, None] * cache["signed"][:, None]
            yield "pixshift", f"n0p={n0p},beta={beta}", lo + sh, hi + sh


def arms_v3(d: dict, cache: dict):
    """Round 3, around round 2's pick (per-pixel on the channel x lead stage, n0p 50)."""
    global KGRID
    for n0s in (2.0, 10.0):
        yield "pixcl", f"n0s={n0s},n0p=50.0", *run_pix_on_scalar(
            d, n0s, 50.0, cache, base="chanlead"
        )
    if "signed" not in cache:
        cache["signed"] = running_signed(d)
    r = d["win"].astype(np.float64)
    lo, hi = run_pix_on_scalar(d, 5.0, 50.0, cache, base="chanlead")
    for beta in (0.125, 0.25):
        sh = (beta * r / (r + 5.0))[:, None, None, None, None] * cache["signed"][:, None]
        yield "pixclshift", f"n0p=50.0,beta={beta}", lo + sh, hi + sh
    saved = KGRID
    try:
        KGRID = np.exp(np.linspace(np.log(1 / 3), np.log(3.0), 41))
        wide: dict = {}
        yield "pixclwide", "n0s=5,n0p=50.0", *run_pix_on_scalar(
            d, 5.0, 50.0, wide, base="chanlead"
        )
    finally:
        KGRID = saved


def arms_v4(d: dict, cache: dict):
    """Round 4: what the cheap device settings cost -- grid size x lead subsample, on the pick."""
    global KGRID, LEAD_SUB
    saved = (KGRID, LEAD_SUB)
    if "signed" not in cache:
        cache["signed"] = running_signed(d)
    r = d["win"].astype(np.float64)
    sh = (0.125 * r / (r + 5.0))[:, None, None, None, None] * cache["signed"][:, None]
    try:
        for k in (29, 15, 8):
            for sub in (1, 2, 4):
                KGRID = np.exp(np.linspace(np.log(0.5), np.log(2.0), k))
                LEAD_SUB = sub
                lo, hi = run_pix_on_scalar(d, 5.0, 50.0, {}, base="chanlead")
                yield "cost", f"K={k},leadsub={sub}", lo + sh, hi + sh
    finally:
        KGRID, LEAD_SUB = saved


def arms_v5(d: dict, cache: dict):
    """Round 5: a ROTATING lead subsample against the fixed one, at the same device cost."""
    global KGRID, LEAD_SUB, LEAD_ROT
    saved = (KGRID, LEAD_SUB, LEAD_ROT)
    if "signed" not in cache:
        cache["signed"] = running_signed(d)
    r = d["win"].astype(np.float64)
    sh = (0.125 * r / (r + 5.0))[:, None, None, None, None] * cache["signed"][:, None]
    try:
        for k, sub in ((29, 4), (15, 2), (15, 4), (29, 2)):
            KGRID = np.exp(np.linspace(np.log(0.5), np.log(2.0), k))
            LEAD_SUB, LEAD_ROT = sub, True
            lo, hi = run_pix_on_scalar(d, 5.0, 50.0, {}, base="chanlead")
            yield "rot", f"K={k},leadsub={sub}", lo + sh, hi + sh
    finally:
        KGRID, LEAD_SUB, LEAD_ROT = saved


def arms_v6(d: dict, cache: dict):
    """Round 6: freeze the tables after N revealed windows (the host can then skip the update)."""
    global FREEZE_N
    if "signed" not in cache:
        cache["signed"] = running_signed(d)
    r = d["win"].astype(np.float64)
    sh = (0.125 * r / (r + 5.0))[:, None, None, None, None] * cache["signed"][:, None]
    try:
        for fz in (8, 16, 24):
            FREEZE_N = fz
            lo, hi = run_pix_on_scalar(d, 5.0, 50.0, cache, base="chanlead")
            yield "freeze", f"N={fz}", lo + sh, hi + sh
    finally:
        FREEZE_N = 0


def arms(d: dict, cache: dict):
    """Yield (arm, hyper, lower, upper) for every configuration."""
    for fam in ("scalar", "lead", "chan", "chanlead"):
        for n0 in N0S:
            yield fam, f"n0={n0}", *run_family(d, fam, n0, cache)
    for n0p in (5.0, 20.0, 50.0):
        yield "pix", f"n0s=5,n0p={n0p}", *run_pix_on_scalar(d, 5.0, n0p, cache)
    for n0 in (5.0,):
        for beta in BETAS:
            yield "shift", f"n0={n0},beta={beta}", *run_shift(d, n0, beta, cache)
    for n0 in (2.0, 5.0, 20.0):
        yield "asym", f"n0={n0}", *run_asym(d, n0, cache)


def main(argv: list[str] | None = None) -> int:
    global SIGMA
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", nargs="+", default=["val_interp", "val_extrap", "hard5"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, default=1, help="1: family sweep, 2: around its winner")
    args = ap.parse_args(argv)
    from rpde_t2.stream import official_scoring

    SIGMA = float(official_scoring().SIGMA_GLOBAL)
    res: dict = {}
    for split in args.splits:
        d = load(Path(args.cache) / f"{split}.npz")
        base = score(d, d["lower"], d["upper"])
        cache: dict = {}
        rows = {}
        gen = {1: arms, 2: arms_v2, 3: arms_v3, 4: arms_v4, 5: arms_v5, 6: arms_v6}[args.round]
        for fam, hyper, lo, hi in gen(d, cache):
            key = f"{fam}|{hyper}"
            rows[key] = score(d, lo, hi) - base
            print(f"{split:10s} {fam:9s} {hyper:18s} dsps {rows[key]:+.4f}", flush=True)
        res[split] = {"base_sps": base, "delta": rows}
    # cross-selection between the two disjoint held-out streams
    cross = {}
    for sel, rep in (("val_interp", "val_extrap"), ("val_extrap", "val_interp")):
        if sel in res and rep in res:
            fams = sorted({k.split("|")[0] for k in res[sel]["delta"]})
            for fam in fams:
                cands = {k: v for k, v in res[sel]["delta"].items() if k.startswith(fam + "|")}
                best = max(cands, key=cands.get)
                cross[f"{fam}: {sel}->{rep}"] = {
                    "pick": best,
                    "select_dsps": cands[best],
                    "report_dsps": res[rep]["delta"][best],
                    "hard5_dsps": res.get("hard5", {}).get("delta", {}).get(best),
                }
                rd = res[rep]["delta"][best]
                print(f"CROSS {fam:9s} {sel}->{rep}: {best:28s} report {rd:+.4f}")
    res["cross"] = cross
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
