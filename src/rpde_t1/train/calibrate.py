"""Post-hoc interval calibration against the official scorer.

The head learns the *shape* of the error field; a single multiplier then places the
overall width where the scorer actually pays best. Separating the two matters
because the training objective is a *relaxation* of SPS -- the hard
inside/outside indicator is replaced by sigmoids with temperature ``tau`` -- so the
width that optimises the relaxation is systematically slightly wrong for the real
metric. One scalar, fitted against the unmodified scorer, removes that bias.

The scan is fitted on one split and must be reported on another, otherwise the
multiplier is just another parameter tuned on the number being quoted.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from ..score.harness import (
    NORMALIZATIONS,
    load_eval_set,
    run_official_scorer,
    run_submission,
    write_predictions,
)

#: Multipliers to try. Spans a factor of ~30, which comfortably brackets the
#: optimum for any head that is even roughly calibrated.
DEFAULT_FACTORS = (0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.5, 5.0)


def scan_factor(
    predict,
    split: str,
    factors: tuple[float, ...] = DEFAULT_FACTORS,
    time_normalization: str = "per_window",
) -> tuple[list[dict], float]:
    """Score ``split`` at each width multiplier. Returns (rows, best factor).

    ``predict`` is run **once**; only the interval scaling changes between rows, so
    the comparison isolates the width and costs one forward pass rather than eleven.
    """
    inputs, targets, _ = load_eval_set(split)
    pred, lower, upper, timing = run_submission(predict, inputs, batch_size=None)
    if lower is None or upper is None:
        raise ValueError(
            f"The submission returned no bounds on {split!r}; there is nothing to "
            "calibrate. Build it with bounds enabled."
        )
    mean_t = NORMALIZATIONS[time_normalization](timing)

    below, above = pred - lower, upper - pred
    tmp = Path(tempfile.mkdtemp())
    (tmp / "ref").mkdir()
    np.savez_compressed(tmp / "ref" / "targets.npz", target=targets)

    rows: list[dict] = []
    for f in factors:
        lo, hi = pred - below * f, pred + above * f
        scored = targets[..., :2] != 0.0
        inside = (targets[..., :2] >= lo[..., :2]) & (targets[..., :2] <= hi[..., :2])
        write_predictions(tmp, pred, mean_t, lo, hi)
        s = run_official_scorer(tmp, tmp / "out")
        rows.append(
            {
                "factor": f,
                "sps_score": s["sps_score"],
                # Coverage over the elements v9 actually scores, which is the
                # quantity the metric responds to -- not coverage over the full field.
                "coverage_scored": float(inside[scored].mean()),
                "mean_half_width": float(((hi - lo)[..., :2] / 2)[scored].mean()),
                **{k: s[k] for k in ("rel_l2_score", "tke_score", "mvpe_score", "time_score")},
            }
        )
    best = max(rows, key=lambda r: r["sps_score"])["factor"]
    return rows, best


#: Blend exponents to try for the post-hoc fluctuation rescaling. ``0`` is the
#: identity (the control) and ``1`` trusts the input window's variance map entirely.
DEFAULT_BLENDS = (0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)


def scan_tke_blend(
    predict,
    split: str,
    blends: tuple[float, ...] = DEFAULT_BLENDS,
) -> tuple[list[dict], float]:
    """Score ``split`` at each fluctuation-blend exponent. Returns (rows, best blend).

    Same shape as :func:`scan_factor` and for the same reason: ``predict`` runs once and
    only a post-hoc numpy transform changes between rows, so eight exponents cost one
    forward pass. The exponent must be *fitted here and reported elsewhere* -- the global
    interval-width multiplier looked worth +5.2 on its own split and delivered +0.26 held
    out, which is the precedent this discipline exists for.

    Scored through the vendored ``scoring.py`` imported in process rather than shelled
    out: it is the same module the harness executes, and a subprocess per exponent would
    dominate the runtime.
    """
    from ..models.predictor import rescale_fluctuations
    from ..score.diagnostics import _subscores, load_official_scoring

    inputs, targets, _ = load_eval_set(split)
    pred, _, _, _ = run_submission(predict, inputs, batch_size=None)

    sc = load_official_scoring()
    c = sc.measured_channels(targets)

    rows: list[dict] = []
    for b in blends:
        q = rescale_fluctuations(pred, inputs, blend=b)
        rows.append({"blend": b, **_subscores(sc, q, targets, c)})
    best = max(rows, key=lambda r: r["tke_score"])["blend"]
    return rows, best


def format_blend_scan(rows: list[dict], split: str) -> str:
    base = next((r for r in rows if r["blend"] == 0.0), None)
    head = f"{'blend':>7}{'rel_l2':>9}{'tke':>8}{'mvpe':>8}{'d_rel_l2':>10}{'d_tke':>8}"
    out = [f"fluctuation-blend scan on {split}", head, "-" * len(head)]
    best = max(r["tke_score"] for r in rows)
    for r in rows:
        d_rel = r["rel_l2_score"] - base["rel_l2_score"] if base else float("nan")
        d_tke = r["tke_score"] - base["tke_score"] if base else float("nan")
        mark = "  <- best" if r["tke_score"] == best else ""
        out.append(
            f"{r['blend']:>7.2f}{r['rel_l2_score']:>9.2f}{r['tke_score']:>8.2f}"
            f"{r['mvpe_score']:>8.2f}{d_rel:>+10.2f}{d_tke:>+8.2f}{mark}"
        )
    out.append("")
    out.append("mvpe is invariant by construction (the gain is constant in time); if it")
    out.append("moves by more than rounding, the rescaling is wired up wrong.")
    return "\n".join(out)


def format_scan(rows: list[dict], split: str) -> str:
    head = (
        f"{'factor':>8}{'coverage':>10}{'half-width':>12}{'sps':>8}"
        f"{'rel_l2':>9}{'tke':>8}{'mvpe':>8}"
    )
    out = [f"width calibration on {split} (n={len(rows)} factors)", head, "-" * len(head)]
    best = max(r["sps_score"] for r in rows)
    for r in rows:
        mark = "  <- best" if r["sps_score"] == best else ""
        out.append(
            f"{r['factor']:>8.2f}{r['coverage_scored']:>10.3f}{r['mean_half_width']:>12.5f}"
            f"{r['sps_score']:>8.2f}{r['rel_l2_score']:>9.2f}{r['tke_score']:>8.2f}"
            f"{r['mvpe_score']:>8.2f}{mark}"
        )
    return "\n".join(out)


def save_scan(rows: list[dict], best: float, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"best_factor": best, "rows": rows}, indent=2), encoding="utf-8")
    return path
