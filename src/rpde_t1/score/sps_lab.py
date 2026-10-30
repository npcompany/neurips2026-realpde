"""In-process SPS arithmetic, so interval experiments cost seconds instead of a forward pass.

Why this exists. Every interval-head experiment re-runs the *same* frozen backbone over
the *same* windows and then differs only in the numbers it puts in ``lower``/``upper``.
Routing each one through ``make_predict`` plus a subprocess scorer costs ~30 s and hides
the structure of the metric behind a single scalar. This module caches the backbone's
outputs once per split and reimplements the scorer's SPS arithmetic elementwise, which
buys three things the harness cannot give:

* **the per-element credit field**, so the loss can be attributed to coverage or to width;
* **upper bounds** -- what an oracle scale, or an oracle *interval*, would score on the
  same predictions, which is what says whether a head is worth improving at all;
* **iteration in numpy**, so a width rule can be fitted on one split and reported on
  another in one process.

The reimplementation is checked against the vendored scorer to 1e-9 by
``tests/test_sps_lab.py``; that test is the only reason to trust anything here.

One asymmetry in the scorer drives most of what follows and is easy to miss: the width
penalty ``exp(-(upper - lower)/sigma)`` is applied **only where the target is covered**.
An element that falls outside its interval scores zero whatever the width, so there is no
penalty for being wide and wrong -- which makes the optimal interval a *per-element*
problem with no global budget, and the objective for one element

    maximise over (l, u):  P(target in [l, u]) * exp(-(u - l) / sigma)

Nothing in the scorer requires ``lower <= prediction <= upper``, either, so the interval's
centre is a free parameter that the reported prediction does not constrain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import N_MEASURED, SIGMA_GLOBAL, mvpe_probe_indices

#: Accuracy-axis weights inside the SPS aggregate, straight from ``aggregate_sps``.
SPS_WEIGHTS = {"dm": 0.5, "tke": 0.3, "mvpe": 0.2}

#: ``normalize_factor`` in the scorer: ``pm = err / (NORMALIZE + err)``.
NORMALIZE = 0.5


# --------------------------------------------------------------------------- #
# The scorer's own quantities, elementwise
# --------------------------------------------------------------------------- #


def _rel_l2(pred: np.ndarray, target: np.ndarray, c: int = N_MEASURED) -> np.ndarray:
    p = pred[..., :c].reshape(pred.shape[0], -1).astype(np.float64)
    t = target[..., :c].reshape(target.shape[0], -1).astype(np.float64)
    return np.linalg.norm(p - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-8)


def _ke(x: np.ndarray) -> np.ndarray:
    u, v = x[..., 0].astype(np.float64), x[..., 1].astype(np.float64)
    up = np.mean((u - u.mean(axis=1, keepdims=True)) ** 2, axis=1)
    vp = np.mean((v - v.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 0.5 * (up + vp)


def _tke_rel_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = _ke(pred).reshape(pred.shape[0], -1)
    t = _ke(target).reshape(target.shape[0], -1)
    return np.linalg.norm(p - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-8)


def _mvpe_rel_l2(pred: np.ndarray, target: np.ndarray, sub_s_real: int = 2) -> np.ndarray:
    n, _, h, w, _ = pred.shape
    probe_y, probe_xs = mvpe_probe_indices(h, w, sub_s_real)
    if not probe_y:
        return np.zeros((n,), dtype=np.float64)
    errors = []
    for probe_x in probe_xs:
        pp = pred[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(n, -1).astype(np.float64)
        tt = target[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(n, -1).astype(np.float64)
        errors.append(
            np.linalg.norm(pp - tt, axis=1) / np.maximum(np.linalg.norm(tt, axis=1), 1e-8)
        )
    if not errors:
        return np.zeros((n,), dtype=np.float64)
    return np.mean(np.stack(errors, axis=0), axis=0)


def quality_weights(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample ``1 - pm`` for each accuracy axis, and their SPS-weighted sum.

    This is the whole of the accuracy channel into SPS. The three branches of
    ``aggregate_sps`` share one interval term and differ only by this scalar, so

        sps = mean_over_scored_elements( credit(element) * q(its sample) )

    with ``q = 0.5*(1-pm_dm) + 0.3*(1-pm_tke) + 0.2*(1-pm_mvpe)``. ``100 * mean_e q`` is
    the *ceiling*: what SPS would be with zero-width always-covering intervals.
    """
    out = {}
    for key, err in (
        ("dm", _rel_l2(pred, target)),
        ("tke", _tke_rel_l2(pred, target)),
        ("mvpe", _mvpe_rel_l2(pred, target)),
    ):
        out[key] = NORMALIZE / (NORMALIZE + err)
    out["q"] = sum(SPS_WEIGHTS[k] * out[k] for k in SPS_WEIGHTS)
    return out


def scored_mask(target: np.ndarray, c: int = N_MEASURED) -> np.ndarray:
    """Elements SPS averages over: non-zero target, measured channels only."""
    return target[..., :c] != 0.0


@dataclass
class SpsBreakdown:
    """SPS and the pieces it factorises into."""

    sps: float
    ceiling: float
    efficiency: float
    coverage: float
    mean_width_penalty: float
    mean_half_width: float

    def as_dict(self) -> dict[str, float]:
        return {
            "sps": self.sps,
            "ceiling": self.ceiling,
            "efficiency": self.efficiency,
            "coverage": self.coverage,
            "mean_width_penalty": self.mean_width_penalty,
            "mean_half_width": self.mean_half_width,
        }


def sps_breakdown(
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    q: np.ndarray | None = None,
) -> SpsBreakdown:
    """``sps_score`` plus its factors, matching ``scoring.aggregate_sps`` exactly.

    ``q`` may be passed to reuse the accuracy weights across many interval variants on a
    fixed prediction -- the common case, and the expensive part of the computation.
    """
    c = N_MEASURED
    t = target[..., :c]
    lo, hi = lower[..., :c], upper[..., :c]
    sc = t != 0.0
    n = int(np.count_nonzero(sc))
    if n == 0:
        raise ValueError("no scored element")
    if q is None:
        q = quality_weights(pred, target)["q"]
    qs = q.reshape((-1,) + (1,) * (t.ndim - 1))

    inside = (t >= lo) & (t <= hi)
    penalty = np.exp(-(hi - lo).astype(np.float64) / SIGMA_GLOBAL)
    credit = np.where(inside, penalty, 0.0)

    sps = float(np.sum(credit * qs, where=sc, dtype=np.float64) / n)
    ceiling = float(np.sum(np.broadcast_to(qs, t.shape), where=sc, dtype=np.float64) / n)
    return SpsBreakdown(
        sps=100.0 * sps,
        ceiling=100.0 * ceiling,
        efficiency=sps / ceiling if ceiling else 0.0,
        coverage=float(np.count_nonzero(inside & sc) / n),
        mean_width_penalty=float(np.mean(penalty[sc])),
        mean_half_width=float(np.mean(((hi - lo) / 2)[sc])),
    )


# --------------------------------------------------------------------------- #
# The per-element optimum, under an arbitrary standardised error law
# --------------------------------------------------------------------------- #


def credit_curve(
    z: np.ndarray, rho: np.ndarray, coverage_of_z: np.ndarray
) -> np.ndarray:  # pragma: no cover - trivial
    """``coverage(z) * exp(-2 z rho)`` on the outer product of ``z`` and ``rho``."""
    return coverage_of_z[:, None] * np.exp(-2.0 * z[:, None] * rho[None, :])


def optimal_z_table(
    standardised: np.ndarray,
    rho_grid: np.ndarray,
    z_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``z*(rho)`` under the *empirical* law of a standardised error, plus its credit.

    ``models.interval`` tabulates the same function under a Gaussian assumption. That
    assumption is the one thing in the width rule that is not derived from the scorer,
    and it is wrong in the direction that matters: the per-element error is heavy tailed,
    so the coverage bought by widening arrives more slowly than ``2*Phi(z)-1`` promises
    and every Gaussian width comes out too large. Feeding the measured law instead turns
    the fitted global multiplier of FINDINGS 10 into a derived, rho-dependent curve.

    ``standardised`` is ``(target - centre) / scale`` over a fitting sample; only its
    distribution is used, so it may be subsampled freely.
    """
    z_grid = np.linspace(1e-3, 8.0, 2048) if z_grid is None else z_grid
    a = np.sort(np.abs(np.asarray(standardised, dtype=np.float64).ravel()))
    # Empirical P(|e| <= z), evaluated by counting -- no density estimate to tune.
    coverage = np.searchsorted(a, z_grid, side="right") / max(1, a.size)
    credit = coverage[:, None] * np.exp(-2.0 * z_grid[:, None] * rho_grid[None, :])
    best = np.argmax(credit, axis=0)
    return z_grid[best], credit[best, np.arange(rho_grid.size)]


def oracle_interval_credit(
    err: np.ndarray, sigma: float = SIGMA_GLOBAL
) -> np.ndarray:  # pragma: no cover - trivial
    """Credit of the best *realisation-aware* interval: ``exp(-2|err|/sigma)``.

    Not reachable -- it needs the realised error -- but it is the correct ceiling for any
    interval method, and it separates "the head cannot rank" from "the metric cannot pay".
    """
    return np.exp(-2.0 * np.abs(err) / sigma)


# --------------------------------------------------------------------------- #
# Cached backbone outputs
# --------------------------------------------------------------------------- #


@dataclass
class Cached:
    """One split's frozen-backbone outputs, plus everything a width rule needs."""

    split: str
    inputs: np.ndarray
    target: np.ndarray
    pred: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    u_fs: np.ndarray
    cond: np.ndarray
    #: False for a backbone-only checkpoint, whose ``lower``/``upper`` are pinned to the
    #: prediction. Guards :attr:`half` so a degenerate zero width cannot be consumed as if
    #: it were a trained head's output.
    has_bounds: bool = True

    @property
    def measured_shape(self) -> tuple[int, ...]:
        """Shape of a per-element measured-channel field, ``(N, T, H, W, N_MEASURED)``.

        Separate from :attr:`half` because a caller that only needs to *allocate* such a field
        must not be forced to have bounds: a backbone-only cache has none, and the guard on
        ``half`` correctly refuses it.
        """
        return (*self.pred.shape[:-1], N_MEASURED)

    @property
    def half(self) -> np.ndarray:
        """The head's own half-widths about the prediction, ``(N,T,H,W,2)``, symmetrised."""
        if not self.has_bounds:
            raise ValueError(
                f"cache for {self.split!r} was written from a checkpoint with no interval "
                "head, so it has no half-widths. Fit a scale head on it instead."
            )
        return 0.5 * (self.upper - self.lower)[..., :N_MEASURED]

    @property
    def err(self) -> np.ndarray:
        """``target - prediction`` on the measured channels, physical units."""
        return (self.target[..., :N_MEASURED] - self.pred[..., :N_MEASURED]).astype(np.float64)

    @property
    def scored(self) -> np.ndarray:
        return scored_mask(self.target)

    def q(self, pred: np.ndarray | None = None) -> np.ndarray:
        return quality_weights(self.pred if pred is None else pred, self.target)["q"]


@dataclass
class ScoredFlat:
    """One split's scored measured elements, flattened -- a configuration sweep's working set.

    :func:`sps_breakdown` stays the authority on the metric and is what step 3 of
    ``sps_select.py`` reports through. But a sweep reads a single scalar off it per call, while
    the composition it scores -- ``rule.apply`` then ``sps_breakdown`` -- rebuilds six full
    ``(N, T, H, W, 3)`` arrays and a float64 penalty field over all 23M of them, of which the
    8M scored measured ones are the only ones the metric averages. Holding those flat turns
    each variant into six passes over 8M.

    The saving is in what is skipped, not in different arithmetic: :func:`ruled_sps` performs
    the same float32 operations in the same order, so it agrees **elementwise and exactly**.
    Only the final reduction differs -- a float64 sum over a flat array rather than a masked
    5-D one, so its pairwise accumulation order differs, worth ~1e-11 on a 0-100 scale.
    ``tests/test_sps_lab.py`` pins it against the composition; that test is the only reason to
    trust a second path here at all, exactly as for :meth:`WidthRule.apply_torch`.
    """

    pred: np.ndarray
    target: np.ndarray
    q: np.ndarray
    ch: np.ndarray
    #: The metric's denominator: scored elements, which is what ``sps_breakdown`` divides by.
    n: int


def flatten_scored(c: Cached, q: np.ndarray | None = None) -> ScoredFlat:
    """Gather one split's scored measured elements, with each one's accuracy weight."""
    sc = c.scored
    shape = c.measured_shape
    qq = c.q() if q is None else q
    return ScoredFlat(
        pred=c.pred[..., :N_MEASURED][sc],
        target=c.target[..., :N_MEASURED][sc],
        # Per *sample* in the metric, so broadcast before masking -- the same reshape
        # ``sps_breakdown`` applies, and float64 there, so float64 here.
        q=np.broadcast_to(qq.reshape((-1,) + (1,) * (len(shape) - 1)), shape)[sc].astype(
            np.float64
        ),
        ch=np.broadcast_to(np.arange(N_MEASURED, dtype=np.int32).reshape(1, 1, 1, 1, -1), shape)[
            sc
        ],
        n=int(np.count_nonzero(sc)),
    )


def ruled_sps(rule, flat: ScoredFlat, lower: np.ndarray, upper: np.ndarray) -> float:
    """``sps_breakdown(pred, *rule.apply(pred, lower, upper), target).sps``, on flat elements.

    ``lower`` / ``upper`` are the head's own bounds at ``flat``'s elements, i.e. the same
    arrays :meth:`WidthRule.apply` would be handed, flattened the same way. They are passed
    rather than a half-width because ``apply`` recovers the half-width from them as
    ``0.5 * ((p - lower) + (upper - p))``, and in float32 that round trip is *not* the identity
    -- reproducing it is what makes this path exact rather than merely close.

    Raises for a rule whose lead axis is not degenerate, which this path cannot index; that is
    :meth:`WidthRule.apply_torch`'s contract too, and for the same reason.
    """
    c = N_MEASURED
    p = flat.pred
    below = p - lower
    above = upper - p
    half = 0.5 * (below + above)

    i_rho = rule.rho_index(half)
    m_lo = rule._gather(rule.a_lo, i_rho, c, flat.ch)
    m_up = rule._gather(rule.a_up, i_rho, c, flat.ch)
    if m_lo is None or m_up is None:
        raise ValueError(
            "ruled_sps needs a degenerate lead axis; score through sps_breakdown instead"
        )

    new_lo = p - below * m_lo
    new_hi = p + above * m_up
    lo = np.minimum(new_lo, new_hi)
    hi = np.maximum(new_lo, new_hi)

    inside = (flat.target >= lo) & (flat.target <= hi)
    penalty = np.exp(-(hi - lo).astype(np.float64) / SIGMA_GLOBAL)
    credit = np.where(inside, penalty, 0.0)
    return 100.0 * float(np.sum(credit * flat.q, dtype=np.float64) / flat.n)


def cache_path(root: Path, split: str, tag: str) -> Path:
    return root / f"{tag}__{split}.npz"


def save_cache(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
    return path


def load_cache(path: Path, split: str) -> Cached:
    """Read a cache written by :func:`save_cache`, or by ``scripts/sps/sps_cache.py``.

    ``lower`` / ``upper`` are optional. A backbone-only checkpoint has no interval head, so
    the writer stores no bounds at all rather than three identical copies of ``pred`` -- which
    on the ``train_all`` fit set is 700 MB of duplication per backbone, and the reason a
    campaign could stack two arms instead of six. Absent bounds are reconstructed as
    ``pred``, exactly what the writer used to store, so ``has_bounds=False`` behaves
    identically and every cache written before this change still loads.
    """
    from .harness import load_eval_set

    inputs, target, _ = load_eval_set(split, mmap_inputs=True)
    with np.load(path) as z:
        pred = z["pred"]
        return Cached(
            split=split,
            inputs=inputs,
            target=target,
            pred=pred,
            lower=z["lower"] if "lower" in z else pred,
            upper=z["upper"] if "upper" in z else pred,
            u_fs=z["u_fs"],
            cond=z["cond"],
            has_bounds=bool(z["has_bounds"][0]) if "has_bounds" in z else True,
        )


__all__ = [
    "Cached",
    "SPS_WEIGHTS",
    "ScoredFlat",
    "SpsBreakdown",
    "cache_path",
    "flatten_scored",
    "load_cache",
    "ruled_sps",
    "optimal_z_table",
    "oracle_interval_credit",
    "quality_weights",
    "save_cache",
    "scored_mask",
    "sps_breakdown",
]
