"""Human-readable rendering of the official scores.

The scorer emits five subscores and their mean. On its own that hides where the
headroom is, because the five axes have very different reachable ranges:

* the three error subscores use ``100/(1+0.5*err)``, which is steep near
  ``err=0`` and very flat past ``err~1``;
* ``time_score`` uses ``100/(1+sqrt(r))`` and spans roughly 50..97 in practice;
* ``sps_score`` is ``100/(1+exp(-v))`` with ``v`` in ``[0,1]``, so it is
  structurally confined to **50.0 .. 73.1** -- half the axis is unreachable.

So we print, next to each subscore, the final-score points still available on
that axis. That is the number worth optimizing against.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .. import SUBSCORES
from .harness import Timing

#: Used only to express headroom in comparable units. With the combination
#: unpublished this is an assumption, not the rule -- flagged wherever it is used.
N_SUBSCORES = 5

#: Structural bounds of each axis.
#:
#: ``sps_score`` changed at the August 2026 restart: it now maps its aggregate to
#: 0-100 **linearly** (``sps_score = 100 * weighted``), replacing the logistic that
#: confined it to [50, 73.1]. The whole axis is reachable, which turns SPS from the
#: narrowest opportunity into one of the widest.
AXIS_RANGE = {
    "rel_l2_score": (0.0, 100.0),
    "tke_score": (0.0, 100.0),
    "mvpe_score": (0.0, 100.0),
    "time_score": (0.0, 100.0),
    "sps_score": (0.0, 100.0),
}

#: Since the restart the organizers do not publish how the five subscores combine
#: into ``final_score``, and v9 of the scorer no longer emits one at all. We report
#: the five subscores plus our own equal-weight mean, labelled as ours.
COMPOSITE_IS_PUBLISHED = False

#: Board aggregation, **fitted** on every submitted row in ``ledger_t1.json``
#: (FINDINGS 6c-ter, refit in 30a, 43, and again on rows 15-18). Four axes enter
#: linearly; ``sps`` enters
#: concavely, which is the only reading that reconciles the day-1 row and which shows up
#: directly as a local slope falling 0.296 -> 0.252 as ``sps`` climbs 22 -> 38.
#:
#: **These are not published constants and not equal weights.** The equal-weight mean this
#: module used to assume overstates ``tke`` headroom fourfold and understates ``rel_l2``
#: twofold, which is how a dead axis stayed on the roadmap.
#:
#: The fit is **overdetermined** — thirty-two rows against six parameters, twenty-six degrees
#: of freedom — so reproducing the board is not available by construction and the agreement is
#: evidence. It reproduces every row to 1.1e-02 (rms 2.8e-03); the one row at that size is
#: ``hoextrap_ft12dp02_ema999_tkehead_b210`` and every other row is inside 5.0e-03. Two things
#: say the *form* is right rather than merely flexible:
#: refitting with the ``sps`` term forced linear, or with any other single axis carrying the
#: curvature instead, is ~30x worse (max residual 0.095-0.106); and the residuals correlate
#: with no axis (|r| <= 0.055), so there is no missing term to find.
#:
#: Refit with :func:`fit_board_weights` after **every** submission and paste the result here.
#: ``tests/test_aggregation.py`` fails when these stop being the current fit — which is how
#: days 9-12 were caught having landed against constants nobody had refitted.
BOARD_WEIGHTS = {
    "rel_l2_score": 0.4193941015134065,
    "tke_score": 0.09842381663761396,
    "mvpe_score": 0.11057764457153887,
    "time_score": 0.09695531639575386,
}
SPS_WEIGHT = 1.000364666528556
SPS_EXPONENT = 0.7145503003949999


def board_final(scores: dict) -> float:
    """Predicted ``final_score`` from the five subscores, by :data:`BOARD_WEIGHTS`.

    Fitted on *board* rows, so feeding it *local* subscores answers "what would this score
    if the board agreed with our splits", which it does not -- local validation runs
    1.8-6.0 points optimistic per axis (FINDINGS 7). Use it on board numbers.
    """
    total = sum(w * float(scores[k]) for k, w in BOARD_WEIGHTS.items())
    return total + SPS_WEIGHT * float(scores["sps_score"]) ** SPS_EXPONENT


def board_marginals(scores: dict) -> dict[str, float]:
    """final-score points per subscore point, at this operating point.

    Linear for four axes; for ``sps`` the derivative of the concave term, which *falls* as
    ``sps`` rises -- 0.251 at 38, 0.238 at 45, 0.218 at 60. This is the number to prioritise
    with, and it excludes the indirect path where an accuracy gain also lifts the SPS
    ceiling (see ``sps_ceiling``): count that separately.
    """
    out = dict(BOARD_WEIGHTS)
    sps = max(float(scores["sps_score"]), 1e-9)
    out["sps_score"] = SPS_WEIGHT * SPS_EXPONENT * sps ** (SPS_EXPONENT - 1.0)
    return out


@dataclass(frozen=True)
class BoardFit:
    """One least-squares refit of the board aggregation on submitted rows.

    Same functional form as the shipped constants — four linear axes plus
    ``sps_weight * sps ** sps_exponent``, no intercept — so the fields are directly
    comparable to :data:`BOARD_WEIGHTS`, :data:`SPS_WEIGHT` and :data:`SPS_EXPONENT`.
    """

    weights: dict[str, float]
    sps_weight: float
    sps_exponent: float
    residuals: dict[str, float]
    """``predicted - actual`` ``final_score`` per row, keyed by row name."""

    @property
    def max_abs_residual(self) -> float:
        """The worst row — the number a tolerance should be set against."""
        return max((abs(v) for v in self.residuals.values()), default=0.0)

    @property
    def rms_residual(self) -> float:
        """Root-mean-square residual over the fitted rows."""
        values = list(self.residuals.values())
        return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def fit_board_weights(
    boards: Mapping[str, Mapping[str, float]],
    exponent_range: tuple[float, float] = (0.3, 1.4),
) -> BoardFit:
    """Refit all six aggregation parameters on the board rows given.

    ``boards`` maps a row name to its five subscores plus the observed ``final_score``,
    i.e. the ``board`` block of every ``ledger_t1.json`` entry that has one. Once the
    exponent is fixed the model is linear in the five coefficients, so this is a 1-D
    search over the exponent with an exact least-squares solve inside. The search is a
    deterministic coarse-to-fine bracket rather than an iterative optimiser, so the same
    ledger always yields the same constants to machine precision — which is what lets a
    test demand that the shipped constants *equal* the fit.

    Pin ``exponent_range`` to a single point to fit a restricted form: ``(1.0, 1.0)``
    gives the all-linear rule that FINDINGS 6c-ter eliminated, and is how the test
    keeps that elimination executable.

    Nothing in the package calls this. The shipped constants stay literals so that a
    report never silently depends on the ledger; the test is what forces them current.
    """
    names = list(boards)
    n_params = len(BOARD_WEIGHTS) + 2
    if len(names) <= n_params:
        raise ValueError(
            f"the fit needs more rows than its {n_params} parameters to say anything, "
            f"got {len(names)}"
        )
    axes = list(BOARD_WEIGHTS)
    linear = np.array([[float(boards[n][k]) for k in axes] for n in names])
    sps = np.array([float(boards[n]["sps_score"]) for n in names])
    final = np.array([float(boards[n]["final_score"]) for n in names])

    def solve(exponent: float) -> tuple[np.ndarray, np.ndarray]:
        design = np.hstack([linear, (sps**exponent)[:, None]])
        coef, *_ = np.linalg.lstsq(design, final, rcond=None)
        return coef, design @ coef - final

    lo, hi = exponent_range
    for _ in range(6):  # each pass narrows the bracket 100x, so six reach machine precision
        grid = np.linspace(lo, hi, 201)
        sse = [float(r @ r) for _, r in (solve(float(p)) for p in grid)]
        i = int(np.argmin(sse))
        lo, hi = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, grid.size - 1)])
    exponent = 0.5 * (lo + hi)
    coef, residuals = solve(exponent)
    return BoardFit(
        weights={k: float(v) for k, v in zip(axes, coef[:-1], strict=True)},
        sps_weight=float(coef[-1]),
        sps_exponent=exponent,
        residuals={n: float(v) for n, v in zip(names, residuals, strict=True)},
    )


#: Which raw error each error-subscore inverts back to.
ERROR_AXES = {"rel_l2_score": "rel-L2", "tke_score": "TKE rel-L2", "mvpe_score": "MVPE rel-L2"}


def invert_score_error(score: float, scale: float = 0.5) -> float:
    """Recover ``err`` from ``100/(1+scale*err)``. Inf when the score is zero."""
    if score <= 0:
        return float("inf")
    return (100.0 / score - 1.0) / scale


def format_report(
    scores: dict,
    timing: Timing,
    time_normalization: str,
    prediction: np.ndarray,
    targets: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("")
    add("=" * 76)
    add("OFFICIAL SCORES  (vendor/realpde_t1_starting_kit_v9/scoring.py)")
    add("=" * 76)

    if "error" in scores:
        add(f"  !! scorer reported an error: {scores['error']}")

    add(f"  {'subscore':<16}{'value':>9}{'raw err':>11}{'ceiling':>10}{'headroom':>11}")
    add("  " + "-" * 55)
    marginals = board_marginals(scores)
    for key in SUBSCORES:
        value = float(scores.get(key, 0.0))
        lo, hi = AXIS_RANGE[key]
        # Weighted by the fitted board marginal, not by 1/5: an equal-weight mean overstates
        # `tke` headroom fourfold and understates `rel_l2` twofold (FINDINGS 6c-ter).
        head = (hi - value) * marginals[key]
        raw = ""
        if key in ERROR_AXES:
            err = invert_score_error(value)
            raw = "inf" if err == float("inf") else f"{err:.4f}"
        add(f"  {key:<16}{value:>9.3f}{raw:>11}{hi:>10.1f}{head:>+11.2f}")
    add("  " + "-" * 55)
    subs = [float(scores.get(k, 0.0)) for k in AXIS_RANGE]
    add(f"  {'mean (ours)':<16}{sum(subs) / len(subs):>9.3f}")
    if not COMPOSITE_IS_PUBLISHED:
        add("  NOTE: the leaderboard's combination rule is not published and the v9")
        add("        scorer emits no final_score. The mean above is ours, useful only")
        add("        for ranking our own runs. Compare subscores, not totals.")

    add("")
    add("TIMING")
    add(f"  predict() wall clock total : {timing.total_s:.4f} s over {timing.n_windows} windows")
    add(f"  first call (incl. warm-up) : {timing.first_call_s:.4f} s")
    add(f"  per predicted timestep     : {timing.per_step_s:.6f} s")
    add(f"  per window                 : {timing.per_window_s:.6f} s")
    add(f"  reported as mean_t_neural_s: {time_normalization}")
    if time_normalization != "per_window":
        add("  WARNING: the platform normalizes per SAMPLE (per_window), confirmed against")
        add("           a real submission. Any other choice mis-scores the time axis.")
    add("  NOTE: this is YOUR hardware. A800 measured ~18x faster per window than a T4")
    add("        on the CNO baseline, so treat the local time axis as a pessimistic bound.")

    add("")
    add("SPS DIAGNOSTICS")
    c = 2  # the scorer restricts to measured channels; p is all-zero in real data
    if lower is None or upper is None:
        p = prediction[..., :c]
        lo_arr, hi_arr = p - 0.05 * np.abs(p), p + 0.05 * np.abs(p)
        add("  bounds: NOT SUPPLIED -> scorer default band +-0.05*|pred|")
    else:
        lo_arr, hi_arr = lower[..., :c], upper[..., :c]
        add("  bounds: supplied by the submission")
    t = targets[..., :c]
    inside = (t >= lo_arr) & (t <= hi_arr)
    width = hi_arr - lo_arr
    sigma = 0.0563870259  # SIGMA_GLOBAL, frozen for the season
    add(f"  coverage (fraction inside) : {float(np.mean(inside)):.4f}")
    add(f"  mean interval width        : {float(np.mean(width)):.6f}")
    add(f"  mean width / sigma_global  : {float(np.mean(width) / sigma):.4f}")
    add("  (SPS rewards coverage*exp(-width/sigma): widen until coverage stops paying.)")

    add("")
    add("WHERE THE POINTS ARE")
    marg = board_marginals(scores)
    ranked = sorted(
        ((marg[k] * (AXIS_RANGE[k][1] - float(scores.get(k, 0.0))), k) for k in AXIS_RANGE),
        reverse=True,
    )
    add(f"  {'axis':<16}{'to ceiling':>11}{'final/point':>13}{'final available':>17}")
    for head, key in ranked:
        gap = AXIS_RANGE[key][1] - float(scores.get(key, 0.0))
        add(f"  {key:<16}{gap:>11.2f}{marg[key]:>13.4f}{head:>17.2f}")
    add("  (weights fitted on the board rows, not published -- see BOARD_WEIGHTS.)")
    add("=" * 76)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Stratified scoring
# --------------------------------------------------------------------------- #
#
# Why this exists. Every subscore depends strongly on the *composition* of the
# evaluation set, not only on the model. Measured on 589 pooled validation windows
# with one fixed model:
#
#   by freestream quartile   sps 54.5 -> 32.5,  coverage flat at 0.90-0.93
#   by angle of attack       sps 47.5 -> 36.0,  mvpe 98.2 -> 94.3
#
# So a single internal number cannot be compared against a leaderboard number, or
# against another internal number computed on a differently-composed set. Reporting
# per stratum makes model-vs-model comparisons valid whatever the mix, and makes the
# mix itself visible instead of implicit.

#: Strata boundaries. Freestream speed is a proxy for Reynolds (they are
#: proportional); AoA is estimated from the window, so this needs no labels and works
#: on any eval set.
UFS_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
AOA_EDGES = (0.0, 5.0, 10.0, 15.0, 21.0)


def stratify(inputs: np.ndarray, by: str = "u_fs") -> list[tuple[str, np.ndarray]]:
    """Boolean masks over windows, as ``[(label, mask), ...]``.

    ``by`` is ``"u_fs"`` (quartiles of freestream speed) or ``"aoa"`` (fixed bins of
    the estimated angle of attack).
    """
    from ..features import FREESTREAM_ROWS, estimate_batch

    if by == "u_fs":
        v = np.abs(inputs[:, :, :FREESTREAM_ROWS, :, 0]).mean(axis=(1, 2, 3))
        edges = np.quantile(v, UFS_QUANTILES)
    elif by == "aoa":
        v = estimate_batch(inputs)[:, 1]
        edges = np.asarray(AOA_EDGES)
    else:
        raise ValueError(f"by must be 'u_fs' or 'aoa', got {by!r}")

    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (v >= lo) & (v <= hi) if hi == edges[-1] else (v >= lo) & (v < hi)
        if mask.any():
            out.append((f"{lo:.3g}-{hi:.3g}", mask))
    return out


# --------------------------------------------------------------------------- #
# SPS decomposition
# --------------------------------------------------------------------------- #
#
# A raw SPS number does not transfer between evaluation sets, which is why our local
# 41.94 came back as 29.99 on the leaderboard. SPS factorises exactly:
#
#     SPS = ceiling(accuracy) x efficiency(intervals)
#
# The ceiling is fixed by the *other* subscores -- it is the SPS a model would get with
# zero-width intervals that always cover -- and the efficiency is what the interval head
# actually achieves against it. The ceiling moves with how hard the data is; the
# efficiency is a property of the head. Reporting them separately is what makes the
# profile comparable across arms, and gives a quantity worth calibrating against the
# leaderboard.

#: Weights the scorer uses to combine the three accuracy branches inside SPS.
SPS_BRANCH_WEIGHTS = {"rel_l2_score": 0.5, "tke_score": 0.3, "mvpe_score": 0.2}

#: The scorer's ``normalize_factor``, which maps a branch error to ``p = e/(nf+e)``.
SPS_NORMALIZE_FACTOR = 0.5


def sps_ceiling(
    subscores: dict[str, float], normalize_factor: float = SPS_NORMALIZE_FACTOR
) -> float:
    """Highest SPS reachable at this accuracy, with perfect intervals.

    Each branch contributes ``w * (1 - p)`` where ``p = err/(nf+err)``, so the ceiling
    is ``100 * sum(w * nf/(nf+err))``. Measured example: at rel_l2 94.08, tke 73.76,
    mvpe 91.37 the ceiling is **66.85** -- so SPS could not have exceeded 67 however
    good the bounds were, and the only route past it is a more accurate model.
    """
    total = 0.0
    for key, weight in SPS_BRANCH_WEIGHTS.items():
        err = invert_score_error(subscores[key])
        total += weight * normalize_factor / (normalize_factor + err)
    return 100.0 * total


def sps_efficiency(subscores: dict[str, float]) -> float:
    """``sps_score / sps_ceiling`` -- how much of the reachable credit the head takes.

    This is the number to track across evaluation sets and against the leaderboard,
    because it factors out the accuracy dependence that makes raw SPS incomparable.
    Measured: 0.609 on ``val_extrap`` against 0.449 on the leaderboard, a ratio of
    0.736 -- one datapoint, so a hypothesis rather than a calibration.
    """
    ceiling = sps_ceiling(subscores)
    return subscores["sps_score"] / ceiling if ceiling > 0 else 0.0


def latest_board_row(path=None) -> dict[str, float] | None:
    """The newest ``ledger_t1.json`` entry that carries board subscores, or ``None``.

    Lazy import of :mod:`.profile` because that module imports this one; keeping the
    dependency inside the function is what avoids the cycle.
    """
    # The submission ledger (score/profile.py) is not included in this package.
    return None


def board_accuracy_weights(
    operating_point: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Final-score points per point of ``rel_l2``/``tke``/``mvpe``: direct **plus** via SPS.

    :data:`BOARD_WEIGHTS` alone understates an accuracy axis, because a subscore also
    lifts ``sps`` through the ceiling. With ``sps = ceiling * efficiency`` and the ceiling
    being the scorer's own :data:`SPS_BRANCH_WEIGHTS` blend, one point of an axis adds
    ``SPS_BRANCH_WEIGHTS[axis] * efficiency`` points of ``sps``, so

        w[axis] = BOARD_WEIGHTS[axis] + SPS_BRANCH_WEIGHTS[axis] * efficiency * dfinal/dsps

    Efficiency is held fixed, which is the right assumption for a *point-prediction*
    change: the interval head is fitted separately and its efficiency is not a function of
    the accuracy it is fitted against. Both efficiency and the ``sps`` marginal depend on
    where we currently sit, so the operating point is an argument rather than a frozen
    constant; ``None`` reads the newest board row in the ledger. At the 2026-08-25 row this
    returns ``rel_l2 0.4872``, ``tke 0.1427``, ``mvpe 0.1458`` -- a ``rel_l2``-to-``tke``
    ratio of **3.42** against the **1.67** the SPS ceiling's own 0.5/0.3 implies. Note the
    efficiency here is the scorer's own ``sps_ceiling`` (error-space), 0.613 at this row,
    not a score-space weighted average of the three axes.

    Raises ``LookupError`` when no operating point is given and the ledger has no board row,
    rather than silently falling back to a guess.
    """
    op = dict(operating_point) if operating_point is not None else latest_board_row()
    if op is None:
        raise LookupError(
            "board_accuracy_weights needs an operating point: the ledger carries no row "
            "with all five board subscores. Pass `operating_point=` explicitly."
        )
    marginal_sps = board_marginals(op)["sps_score"]
    efficiency = sps_efficiency(op)
    return {
        axis: BOARD_WEIGHTS[axis] + branch * efficiency * marginal_sps
        for axis, branch in SPS_BRANCH_WEIGHTS.items()
    }
