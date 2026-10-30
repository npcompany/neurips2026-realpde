"""The SPS-optimal interval width, in closed form.

The scorer credits a scored element ``exp(-(upper - lower) / sigma)`` where the target
falls inside the interval and zero where it does not. For a symmetric interval of
half-width ``w`` around a prediction whose error has scale ``s``, the expected credit is

    (2 * Phi(w / s) - 1) * exp(-2 * w / sigma)

Maximising over ``z = w / s`` gives a factor that depends on **only** the dimensionless
ratio ``rho = s / sigma``:

    ====== ==== ==== ==== ====
    rho    0.05  0.1  0.5  2.0
    z*     2.06 1.72 0.80 0.25
    ====== ==== ==== ==== ====

Two things follow, and both matter more than the arithmetic.

*The optimal coverage is not a free parameter.* It falls from about 96% to about 20% as
the error grows relative to ``sigma``. Any method that targets a fixed nominal coverage
-- conformal prediction included -- is therefore optimising the wrong thing for this
metric.

*It explains the Reynolds dependence we measured empirically.* ``sigma`` is a fixed
physical constant while the physical error grows with freestream speed, so ``rho`` grows
with Reynolds and ``z*`` falls. The per-quartile width multipliers fitted in
``FINDINGS.md`` §6b (0.85 / 0.70 / 0.50 / 0.50, monotonically tighter as Reynolds rises)
are this function, discovered by search. Four fitted constants become one derived rule,
which matters because the organizers retrain the shortlisted pipelines and a fitted
constant is a liability there.

**What this is not.** Feeding the rule an *oracle* error scale reaches SPS 51.22 against
a trained head's 43.65, but that gap is not available: four independent uncertainty
estimators all saturate at Spearman 0.58-0.67 against the true per-element error
(``PROTOCOL.md``), and at that quality the rule lands at 44-47. The rule is worth
adopting for its lack of fitted constants, not for a large score gain. See
``score.diagnostics.bound_ladder``, which measures all four levels side by side so the
distinction cannot be lost again.

Numpy only, so this module is safe inside a submission payload.
"""

from __future__ import annotations

import numpy as np

# Re-exported: callers import it from here as often as from the package root.
from .. import SIGMA_GLOBAL

#: Grid on which ``z*`` is tabulated. Geometric because ``rho`` spans decades: the
#: freestream ladder alone covers 7.5x, and ``|error|`` spans far more than that.
_RHO_GRID = np.geomspace(1e-4, 1e2, 512)

#: Candidate ``z`` values. The upper end is generous -- at very small ``rho`` the
#: optimum drifts past 3 -- and the resolution sets the accuracy of the table.
_Z_GRID = np.linspace(1e-3, 8.0, 4096)


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    """``Phi(z)``, without scipy -- which the eval container does not ship.

    ``math.erf`` is scalar, but it is only called once per grid point at import, so a
    comprehension is cheaper than an approximation that would need its own accuracy
    argument.
    """
    from math import erf, sqrt

    return np.array([0.5 * (1.0 + erf(float(v) / sqrt(2.0))) for v in np.ravel(z)]).reshape(
        np.shape(z)
    )


def _tabulate() -> np.ndarray:
    """``z*`` for each entry of :data:`_RHO_GRID`, by direct maximisation."""
    coverage = 2.0 * _standard_normal_cdf(_Z_GRID) - 1.0
    # (n_z, n_rho): credit for every (z, rho) pair.
    credit = coverage[:, None] * np.exp(-2.0 * _Z_GRID[:, None] * _RHO_GRID[None, :])
    return _Z_GRID[np.argmax(credit, axis=0)]


#: Tabulated once at import. Construction happens outside the scorer's timed region,
#: so this costs nothing at inference.
_Z_STAR = _tabulate()


def optimal_half_width_factor(rho: np.ndarray | float) -> np.ndarray:
    """``z*`` such that ``w = z* * s`` maximises the SPS credit, given ``rho = s/sigma``.

    Interpolated from a table rather than solved per element: the function is smooth and
    monotone decreasing, and an eval set has tens of millions of elements.
    """
    return np.interp(np.asarray(rho, dtype=np.float64), _RHO_GRID, _Z_STAR)


def optimal_half_width(
    scale: np.ndarray, sigma: float = SIGMA_GLOBAL, u_fs: np.ndarray | None = None
) -> np.ndarray:
    """SPS-optimal half-width for a predicted error ``scale``.

    ``u_fs`` converts a **non-dimensional** scale to physical units before comparing it
    with ``sigma``, which is physical. Omitting that conversion mis-scales ``rho`` by up
    to 17x across the Reynolds ladder and silently undoes the whole point of the rule --
    the same trap ``train.losses.sps_loss`` documents.

    Returns a half-width in the same units as ``scale``.
    """
    s = np.asarray(scale, dtype=np.float64)
    if u_fs is not None:
        physical = s * np.asarray(u_fs, dtype=np.float64).reshape(-1, *([1] * (s.ndim - 1)))
    else:
        physical = s
    return optimal_half_width_factor(physical / sigma) * s


def half_normal_scale(abs_error: np.ndarray) -> np.ndarray:
    """Convert ``|error|`` to the sd of the zero-mean normal it came from.

    ``E|X| = sigma * sqrt(2/pi)`` for ``X ~ N(0, sigma^2)``, so the sd is
    ``|error| * sqrt(pi/2)``. Used only to turn an oracle error field into an oracle
    *scale*, which is the quantity a head could in principle predict -- unlike the
    realised error, which no predictor can know.
    """
    return np.asarray(abs_error, dtype=np.float64) * np.sqrt(np.pi / 2.0)


def torch_optimal_half_width_factor(rho):  # noqa: ANN001, ANN201
    """Differentiable ``z*(rho)``, interpolated from the *same* table as the numpy path.

    Exists so a head can be *trained* to predict an error scale and have its width derived
    from the closed form, rather than predicting the width directly. Sharing ``_Z_STAR``
    with :func:`optimal_half_width_factor` means the trained model and the submitted
    numpy inference path cannot disagree about the rule.
    """
    import torch

    grid = torch.as_tensor(_RHO_GRID, dtype=rho.dtype, device=rho.device)
    table = torch.as_tensor(_Z_STAR, dtype=rho.dtype, device=rho.device)
    r = rho.clamp(float(grid[0]), float(grid[-1]))
    i = torch.searchsorted(grid, r.contiguous()).clamp(1, len(grid) - 1)
    lo, hi = grid[i - 1], grid[i]
    t = ((r - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
    return table[i - 1] + t * (table[i] - table[i - 1])


__all__ = [
    "SIGMA_GLOBAL",
    "half_normal_scale",
    "optimal_half_width",
    "optimal_half_width_factor",
    "torch_optimal_half_width_factor",
]
