"""Ceiling ladders: how much of each axis is reachable, measured not guessed.

`ROADMAP.md` is prioritised by *ceilings* -- what the scorer would pay if one component
were perfect and everything else held fixed. A ceiling is only useful if it can be
recomputed against the current model, because the ranking moves as the model improves.
This module is that recomputation.

Three ladders, each answering a different question:

``tke_ladder``
    What is a second-moment head worth? Substitutes candidate estimates of the temporal
    variance map into the prediction and re-scores. The oracle rung is what a *perfect*
    variance head would earn: measured **tke 71.1 -> 98.7** on ``val_extrap`` for 0.20
    rel_l2 and *zero* mvpe. The `input` and `blend` rungs are what is available with no
    training at all.

``bound_ladder``
    What is left in the interval head? Four rungs that are easy to conflate and were
    conflated once, costing a wrong priority: the trained head (43.65), the closed-form
    width rule at a *realistic* scale estimate (~44-47), the same rule at an *oracle*
    scale (51.22), and a cheat bound that knows the realised error (58.1). Only the
    second is available.

``lead_skill``
    Where does the model actually beat the trivial baselines? Measured: it loses to
    persistence at lead 1 (0.0607 vs 0.0434) and crosses over at lead 2, while
    climatology overtakes persistence past lead ~10.

Every number comes from the **vendored** scorer's own functions -- ``kinetic_energy``,
``rel_l2_per_sample``, ``tke_rel_l2_per_sample``, ``mvpe_rel_l2_per_sample``,
``score_error`` -- so nothing here can drift from what the leaderboard computes. Numpy
only, no torch, matching the rest of ``score/``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..models.fluctuations import DEFAULT_MAX_GAIN, apply_fluctuation_gain
from ..models.interval import half_normal_scale, optimal_half_width
from ..paths import LOCAL_EVAL_DIR, OFFICIAL_SCORING, require
from .harness import load_eval_set
from .report import SPS_BRANCH_WEIGHTS

#: Exponent on the input-window variance map in the free geometric blend. Chosen on
#: ``val_time`` and reported on the other two splits, per the fit-here-report-there rule
#: that the width calibration violated.
DEFAULT_BLEND_W = 0.5

#: Log-normal noise levels applied to an oracle error scale, to show how good a scale
#: predictor has to be before the closed-form rule pays. Spearman 0.58-0.67 -- what four
#: independent estimators actually achieve -- corresponds to roughly 0.75-1.0 here.
SCALE_NOISE_LEVELS = (0.25, 0.5, 0.75, 1.0)

#: Multipliers for the global width sweep, which exists to show the *scale* is already at
#: its optimum so the remaining headroom is shape.
WIDTH_MULTIPLIERS = (0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0)

#: Leads printed in the text report. The full per-lead table is always in the JSON; this
#: is only about keeping the terminal output readable.
REPORTED_LEADS = (1, 2, 3, 5, 10, 15, 20)


def load_official_scoring() -> Any:
    """Import the vendored ``scoring.py`` as a module, to reuse its metric functions.

    The harness shells out to the same file as a subprocess for end-to-end scoring; here
    we want the individual metrics, so it is imported instead. Same file either way, which
    is the point -- there is one definition of every metric in this repo and it is theirs.
    """
    require(OFFICIAL_SCORING, "The vendored starting kit is missing from vendor/.")
    spec = importlib.util.spec_from_file_location("_official_scoring", OFFICIAL_SCORING)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot import {OFFICIAL_SCORING}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# TKE ladder
# --------------------------------------------------------------------------- #


def rescale_fluctuations(
    pred: np.ndarray,
    target_map: np.ndarray,
    pred_map: np.ndarray,
    clip: float = DEFAULT_MAX_GAIN,
    n_channels: int | None = None,
) -> np.ndarray:
    """Scale the prediction's fluctuations so its variance map matches ``target_map``.

    The time mean is preserved exactly, which is why ``mvpe`` -- computed on the
    time-averaged field -- is mathematically invariant under this operation, and measured
    invariant to every printed digit. Only ``tke`` and (slightly) ``rel_l2`` move.

    ``clip`` bounds the gain because ``pred_map`` can be near zero in the freestream,
    where the ratio is meaningless and would amplify numerical noise.

    ``n_channels`` defaults to every channel present, which is what the ladder wants: its
    energy maps come from the scorer over the ``measured_channels`` it found, so the
    rescale has to span the same width or it would correct two channels against a
    three-channel target. The deployed path passes the narrower default instead, since
    there the pressure plane is zeroed before this ever runs.
    """
    return apply_fluctuation_gain(
        pred,
        np.sqrt(np.maximum(target_map, 0.0) / np.maximum(pred_map, 1e-12)),
        max_gain=clip,
        n_channels=pred.shape[-1] if n_channels is None else n_channels,
    )


def _subscores(sc: Any, pred: np.ndarray, target: np.ndarray, c: int) -> dict[str, float]:
    return {
        "rel_l2_score": sc.score_error(float(np.mean(sc.rel_l2_per_sample(pred, target, c)))),
        "tke_score": sc.score_error(float(np.mean(sc.tke_rel_l2_per_sample(pred, target, c)))),
        "mvpe_score": sc.score_error(float(np.mean(sc.mvpe_rel_l2_per_sample(pred, target)))),
    }


def tke_ladder(
    pred: np.ndarray,
    target: np.ndarray,
    inputs: np.ndarray,
    blend_w: float = DEFAULT_BLEND_W,
) -> dict[str, Any]:
    """What a second-moment head is worth, rung by rung.

    The premise the ladder tests: an L2-trained model is unbiased in the mean and badly
    biased in the second moment, but the *information* is present in the input window.
    Measured on all three splits -- the prediction retains 0.36-0.44 of the target's
    fluctuation energy while the input window's total is unbiased to within 1-3%.
    """
    sc = load_official_scoring()
    c = sc.measured_channels(target)
    ke_p = sc.kinetic_energy(pred[..., :c])
    ke_t = sc.kinetic_energy(target[..., :c])
    ke_x = sc.kinetic_energy(inputs[..., :c])

    def map_rel_l2(est: np.ndarray) -> float:
        a = est.reshape(est.shape[0], -1)
        b = ke_t.reshape(ke_t.shape[0], -1)
        return float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1).clip(1e-8)))

    geo = np.exp(
        blend_w * np.log(np.maximum(ke_x, 1e-12))
        + (1.0 - blend_w) * np.log(np.maximum(ke_p, 1e-12))
    )
    estimators = {
        "backbone (baseline)": ke_p,
        "input window": ke_x,
        f"geometric blend w={blend_w:g}": geo,
        "ORACLE target map": ke_t,
    }

    rungs = []
    for name, est in estimators.items():
        q = pred if est is ke_p else rescale_fluctuations(pred, est, ke_p)
        rungs.append(
            {"estimator": name, "map_rel_l2": map_rel_l2(est), **_subscores(sc, q, target, c)}
        )

    scored = ke_t > 0
    return {
        "rungs": rungs,
        "energy_ratio_pred_over_target": float(ke_p[scored].sum() / ke_t[scored].sum()),
        "energy_ratio_input_over_target": float(ke_x[scored].sum() / ke_t[scored].sum()),
        "spatial_corr_pred_map": float(np.corrcoef(ke_p.ravel(), ke_t.ravel())[0, 1]),
        "spatial_corr_input_map": float(np.corrcoef(ke_x.ravel(), ke_t.ravel())[0, 1]),
    }


# --------------------------------------------------------------------------- #
# Bound ladder
# --------------------------------------------------------------------------- #


def _sps_from_interval(
    sc: Any, pred: np.ndarray, target: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float]:
    """SPS and coverage for a candidate interval, by the scorer's own arithmetic.

    Reimplemented here only in the sense of being *called* per candidate rather than once
    per submission: every term -- the branch weights, ``p = e/(0.5+e)``, the exponential
    width penalty, the ``target != 0`` scored mask -- comes from ``sc``.
    """
    c = sc.measured_channels(target)
    p, t = pred[..., :c], target[..., :c]
    lo, hi = lower[..., :c], upper[..., :c]
    inside = (t >= lo) & (t <= hi)
    nil = (hi - lo) / sc.SIGMA_GLOBAL
    mask = t != 0.0
    n = int(np.count_nonzero(mask))

    branch_err = {
        "rel_l2_score": sc.rel_l2_per_sample(pred, target, c),
        "tke_score": sc.tke_rel_l2_per_sample(pred, target, c),
        "mvpe_score": sc.mvpe_rel_l2_per_sample(pred, target),
    }
    total = 0.0
    for key, weight in SPS_BRANCH_WEIGHTS.items():
        e = branch_err[key]
        pm = (e / (0.5 + e)).reshape((-1,) + (1,) * (p.ndim - 1))
        elem = np.where(inside, (1.0 - pm) * np.exp(-nil), 0.0)
        total += weight * float(np.sum(elem, where=mask, dtype=np.float64) / n)
    coverage = float(np.count_nonzero(inside & mask) / n)
    return 100.0 * total, coverage


def bound_ladder(
    pred: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    u_fs: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """The four levels of interval quality, kept explicitly apart.

    The distinction that matters: an *oracle scale* is a quantity a head could in
    principle predict, while the *realised error* is not -- it depends on the particular
    draw. Sizing the closed-form rule at the oracle-scale rung, as an earlier draft of
    the roadmap did, overstates it by roughly 3x.
    """
    sc = load_official_scoring()
    c = sc.measured_channels(target)
    p, t = pred[..., :c], target[..., :c]
    err = np.abs(p - t)
    rng = np.random.default_rng(seed)

    def rung(name: str, lo: np.ndarray, hi: np.ndarray, note: str = "") -> dict[str, Any]:
        s, cov = _sps_from_interval(sc, pred, target, lo, hi)
        return {"level": name, "sps": s, "coverage": cov, "note": note}

    s_oracle = half_normal_scale(err)
    w_oracle = optimal_half_width(s_oracle, sigma=sc.SIGMA_GLOBAL, u_fs=u_fs)

    rungs = [
        rung("trained head", lower[..., :c], upper[..., :c], "what we ship"),
    ]
    for noise in SCALE_NOISE_LEVELS:
        s = s_oracle * np.exp(rng.standard_normal(s_oracle.shape) * noise)
        w = optimal_half_width(s, sigma=sc.SIGMA_GLOBAL, u_fs=u_fs)
        rungs.append(
            rung(
                f"closed form, scale log-noise {noise:g}",
                p - w,
                p + w,
                "realistic band" if noise >= 0.75 else "",
            )
        )
    rungs.append(rung("closed form, ORACLE scale", p - w_oracle, p + w_oracle, "not reachable"))
    # 1.001 so the target lies strictly inside rather than exactly on the boundary.
    rungs.append(rung("CHEAT: width = |error|", p - err * 1.001, p + err * 1.001, "knows the draw"))

    below, above = p - lower[..., :c], upper[..., :c] - p
    sweep = [
        {
            "k": k,
            **dict(
                zip(
                    ("sps", "coverage"),
                    _sps_from_interval(sc, pred, target, p - below * k, p + above * k),
                    strict=True,
                )
            ),
        }
        for k in WIDTH_MULTIPLIERS
    ]
    return {"rungs": rungs, "global_width_sweep": sweep}


# --------------------------------------------------------------------------- #
# Lead-time skill
# --------------------------------------------------------------------------- #


def lead_skill(pred: np.ndarray, target: np.ndarray, inputs: np.ndarray) -> list[dict[str, float]]:
    """Per-lead relative L2 against persistence and climatology.

    Persistence repeats the last observed frame; climatology repeats the input window's
    time mean. Both are free, and a model that loses to either at some lead has a
    structural gap there rather than a tuning problem.
    """
    sc = load_official_scoring()
    c = sc.measured_channels(target)
    persistence = np.repeat(inputs[:, -1:], target.shape[1], axis=1)
    climatology = np.repeat(inputs.mean(axis=1, keepdims=True), target.shape[1], axis=1)

    def at(x: np.ndarray, lead: int) -> float:
        a = x[:, lead, ..., :c].reshape(x.shape[0], -1)
        b = target[:, lead, ..., :c].reshape(x.shape[0], -1)
        return float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1).clip(1e-8)))

    return [
        {
            "lead": lead + 1,
            "model": at(pred, lead),
            "persistence": at(persistence, lead),
            "climatology": at(climatology, lead),
        }
        for lead in range(target.shape[1])
    ]


# --------------------------------------------------------------------------- #
# Reynolds ladder ends
# --------------------------------------------------------------------------- #


def window_cells(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """``(re, aoa)`` per window, from the eval set's own provenance. ``None`` if absent.

    Simulated sets persist the operating point directly, because it cannot be recovered
    from a non-dimensionalised window. Real sets persist ``origins`` + ``sources``, so the
    cell is read off the filename -- exact labels, not the ``u_fs`` proxy
    :func:`~rpde_t1.score.report.stratify` has to use when there is no provenance.
    """
    if "window_re" in meta and "window_aoa" in meta:
        return np.asarray(meta["window_re"]), np.asarray(meta["window_aoa"])
    if "origins" in meta and "sources" in meta:
        from ..data.splits import parse_params  # noqa: PLC0415

        cells = [parse_params(meta["sources"][t]) for t, _ in meta["origins"]]
        return (
            np.asarray([c.re for c in cells]),
            np.asarray([c.aoa for c in cells]),
        )
    return None


def _boot_ci(x: np.ndarray, n: int = 20000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and a 95% bootstrap interval over windows."""
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def _map_agreement(pred: np.ndarray, target: np.ndarray, c: int, sc: Any) -> tuple:
    """Per-window (spatial corr, energy ratio) between the two variance maps.

    Both are **denominator-free**, which is the point: ``tke_rel_l2`` divides by the
    target's own fluctuation energy, and that energy is ~57x smaller at the bottom of the
    Reynolds ladder than at the top. A relative error therefore looks far worse at low Re
    even when nothing structural has changed, so a ladder-end comparison that reports only
    the scored metric cannot separate "harder flow" from "smaller denominator".
    """
    kp, kt = sc.kinetic_energy(pred[..., :c]), sc.kinetic_energy(target[..., :c])
    n = kp.shape[0]
    a, b = kp.reshape(n, -1), kt.reshape(n, -1)
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    corr = (ac * bc).sum(1) / np.sqrt((ac**2).sum(1) * (bc**2).sum(1) + 1e-30)
    return corr, a.sum(1) / np.maximum(b.sum(1), 1e-30), kt.reshape(n, -1).mean(1)


def boundary_re_ladder(
    pred: np.ndarray, target: np.ndarray, meta: dict[str, Any]
) -> dict[str, Any] | None:
    """How the candidate behaves at the **lowest and highest Reynolds rows** of a split.

    The private test is decided on unseen `(Re, AoA)`, and the ladder ends are the only
    places a held-out set can say anything about that. On ``val_extrap`` the two ends are
    3750 and 26700 -- each exactly one 1275 rung outside the training range's 5025-25425,
    so they are matched in how far they extrapolate and differ only in direction.

    Two corrections make the comparison mean something, and without them it does not:

    * **Matched angles.** The rows do not hold the same AoA -- ``3750_15``, ``26700_5``
      and ``26700_20`` are sim-only cells -- so an unrestricted comparison confounds
      Reynolds with angle. Only the angles present at *both* ends are used.
    * **Denominator-free companions.** See :func:`_map_agreement`.

    Returns ``None`` when the split has a single Reynolds row (``val_interp``) or carries
    no provenance, since there is then no comparison to make.
    """
    cells = window_cells(meta)
    if cells is None:
        return None
    re, aoa = cells
    rows = sorted(set(re.tolist()))
    if len(rows) < 2:
        return None
    lo_re, hi_re = rows[0], rows[-1]

    shared = sorted(set(aoa[re == lo_re].tolist()) & set(aoa[re == hi_re].tolist()))
    if not shared:
        return None
    keep = np.isin(aoa, shared)

    sc = load_official_scoring()
    c = sc.measured_channels(target)
    per_window = {
        "rel_l2": np.asarray(sc.rel_l2_per_sample(pred, target, c)),
        "tke": np.asarray(sc.tke_rel_l2_per_sample(pred, target, c)),
        "mvpe": np.asarray(sc.mvpe_rel_l2_per_sample(pred, target)),
    }
    corr, ratio, energy = _map_agreement(pred, target, c, sc)
    per_window["map_corr"] = corr
    per_window["energy_captured"] = ratio

    ends: dict[str, Any] = {}
    for name, row in (("low", lo_re), ("high", hi_re)):
        mask = keep & (re == row)
        ends[name] = {
            "re": int(row),
            "n": int(mask.sum()),
            "target_energy": float(energy[mask].mean()),
            **{k: _boot_ci(v[mask]) for k, v in per_window.items()},
        }

    # The gap is bootstrapped as a difference of independent means rather than read off
    # the two intervals: overlapping CIs do not imply an insignificant difference.
    rng = np.random.default_rng(0)
    gaps = {}
    for key, v in per_window.items():
        a, b = v[keep & (re == lo_re)], v[keep & (re == hi_re)]
        d = a[rng.integers(0, len(a), (20000, len(a)))].mean(1) - b[
            rng.integers(0, len(b), (20000, len(b)))
        ].mean(1)
        p_lo, p_hi = np.percentile(d, [2.5, 97.5])
        gaps[key] = (float(a.mean() - b.mean()), float(p_lo), float(p_hi))

    return {
        "shared_aoa": [int(a) for a in shared],
        "ends": ends,
        "gap_low_minus_high": gaps,
        "target_energy_ratio_high_over_low": (
            ends["high"]["target_energy"] / max(ends["low"]["target_energy"], 1e-30)
        ),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def collect_diagnostics(split: str, blend_w: float = DEFAULT_BLEND_W) -> dict[str, Any]:
    """Run every ladder against the predictions already stored for ``split``.

    Reads ``data/local_eval/<split>/predictions.npz``, which is whatever was scored last, so
    the ladders always describe the current artefact rather than a remembered one.
    """
    inputs, target, meta = load_eval_set(split)
    pred_file = LOCAL_EVAL_DIR / split / "predictions.npz"
    require(pred_file, f"No predictions for {split!r}; run `score run` first.")
    with np.load(pred_file) as z:
        pred = z["prediction"].astype(np.float32)
        lower = z["lower"].astype(np.float32) if "lower" in z else None
        upper = z["upper"].astype(np.float32) if "upper" in z else None

    # Predictions and the eval set are two files that can drift apart: rebuilding a split
    # does not rewrite the predictions beside it. Found in practice -- `val_time` held 337
    # predictions against a 98-window set, having been rebuilt after they were written.
    # A count mismatch is caught by numpy downstream, but only as a broadcast error deep
    # inside the scorer; and had the rebuild preserved the count while changing the
    # windows, every ladder here would have silently scored mismatched pairs.
    if pred.shape[0] != target.shape[0]:
        raise ValueError(
            f"{pred_file} holds {pred.shape[0]} predictions but the '{split}' eval set has "
            f"{target.shape[0]} windows. The split was rebuilt after it was last scored; "
            f"re-run `score run --name {split} --submission <path>` before reading ladders."
        )

    out: dict[str, Any] = {
        "split": split,
        "n_windows": int(pred.shape[0]),
        "tke": tke_ladder(pred, target, inputs, blend_w=blend_w),
        "lead_skill": lead_skill(pred, target, inputs),
    }
    boundary = boundary_re_ladder(pred, target, meta)
    if boundary is not None:
        out["boundary_re"] = boundary
    if lower is not None and upper is not None:
        # No ``u_fs`` here, deliberately. ``predictions.npz`` holds what ``predict()``
        # returned, which ``make_predict`` has already multiplied back into physical
        # units -- the same units as ``SIGMA_GLOBAL``. Passing ``u_fs`` would scale by it
        # a second time and shrink every interval by up to 17x across the ladder. The
        # parameter exists for callers working in the model's non-dimensional space, such
        # as the training loop.
        out["bounds"] = bound_ladder(pred, target, lower, upper, u_fs=None)
    return out


def format_diagnostics(d: dict[str, Any]) -> str:
    """Human-readable ladders. One block per question."""
    lines = [f"diagnostics: {d['split']}  ({d['n_windows']} windows)", ""]

    t = d["tke"]
    lines += [
        "TKE ladder -- what a second-moment head is worth",
        f"  prediction keeps {t['energy_ratio_pred_over_target']:.3f} of the target's"
        f" fluctuation energy; the input window keeps {t['energy_ratio_input_over_target']:.3f}",
        f"  spatial corr of the variance map: prediction {t['spatial_corr_pred_map']:.3f},"
        f" input {t['spatial_corr_input_map']:.3f}",
        f"  {'estimator':<28}{'map relL2':>10}{'rel_l2':>9}{'tke':>8}{'mvpe':>8}",
    ]
    for r in t["rungs"]:
        lines.append(
            f"  {r['estimator']:<28}{r['map_rel_l2']:>10.4f}"
            f"{r['rel_l2_score']:>9.2f}{r['tke_score']:>8.2f}{r['mvpe_score']:>8.2f}"
        )

    if "bounds" in d:
        lines += ["", "Bound ladder -- what is left in the interval head"]
        lines.append(f"  {'level':<38}{'sps':>7}{'cover':>7}  note")
        for r in d["bounds"]["rungs"]:
            lines.append(f"  {r['level']:<38}{r['sps']:>7.2f}{r['coverage']:>7.3f}  {r['note']}")
        sweep = "  ".join(f"k={s['k']:g}:{s['sps']:.2f}" for s in d["bounds"]["global_width_sweep"])
        lines.append(f"  global width sweep: {sweep}")

    if "boundary_re" in d:
        b = d["boundary_re"]
        lo, hi = b["ends"]["low"], b["ends"]["high"]
        lines += [
            "",
            "Reynolds ladder ends -- behaviour at the lowest and highest Re rows",
            f"  matched on AoA {b['shared_aoa']}: "
            f"Re={lo['re']} (n={lo['n']}) vs Re={hi['re']} (n={hi['n']})",
            f"  {'measure':<20}{'low Re':>10}{'high Re':>10}{'low-high':>11}"
            f"{'95% CI on the gap':>24}",
        ]
        for key, label in (
            ("rel_l2", "rel_l2 err"),
            ("tke", "tke err"),
            ("mvpe", "mvpe err"),
            ("map_corr", "variance corr *"),
            ("energy_captured", "energy captured *"),
        ):
            g, g_lo, g_hi = b["gap_low_minus_high"][key]
            sig = " " if g_lo <= 0.0 <= g_hi else "!"
            lines.append(
                f"  {label:<20}{lo[key][0]:>10.4f}{hi[key][0]:>10.4f}{g:>+11.4f}"
                f"   [{g_lo:+.4f}, {g_hi:+.4f}] {sig}"
            )
        lines += [
            f"  * denominator-free. The target's fluctuation energy is "
            f"{b['target_energy_ratio_high_over_low']:.0f}x larger at the high end, so the",
            "    relative tke/mvpe errors above are not comparable between the ends on their",
            "    own -- read them as scoring exposure, and the starred rows as structure.",
            "  ! = the gap's 95% interval excludes zero.",
        ]

    lines += ["", "Lead skill -- relative L2 by lead (lower is better)"]
    lines.append(f"  {'lead':>5}{'model':>9}{'persist':>9}{'clim':>9}")
    for r in d["lead_skill"]:
        if r["lead"] in REPORTED_LEADS:
            cols = "".join(f"{r[k]:>9.4f}" for k in ("model", "persistence", "climatology"))
            lines.append(f"  {r['lead']:>5}{cols}")
    return "\n".join(lines)


def write_diagnostics(d: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return path
