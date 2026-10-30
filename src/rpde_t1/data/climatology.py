"""Per-cell climatology of the mean flow, and its interpolation to unseen operating points.

The mean flow is the part of this problem that *is* smooth in the operating point. FINDINGS 5f
measures the second moment's map correlating 0.935 between adjacent Reynolds rungs and 0.735
between adjacent angles; the first moment is smoother still, because it is the time average of
exactly the quantity those correlations are computed on. That is what licenses interpolating it
between cells -- and it is the reason the generator
(:mod:`rpde_t1.train.gen_fit`) is handed a mean field rather than asked to invent one.

Everything here is in **freestream-normalised** units, each cell divided by its own cached
``u_fs``, so a climatology at Re=3750 and one at Re=26700 are the same object at 1% in the mean
(L1) rather than 18.9x apart.

The interpolation is bilinear in ``(log Re, AoA)`` over whichever cells are supplied, which for
every use here is the **training** split alone. Two properties matter and both are deliberate:

* At an existing cell it returns that cell's own climatology exactly, so "generate at a
  training cell" is not secretly a different object from "generate off-grid".
* Off the grid it is a *blend*, and a blend of the 0-degree and 5-degree mean flows is not the
  2.5-degree mean flow -- the wake shifts nonlinearly with incidence, and past 20 degrees the
  flow enters deep stall. So angle interpolation is an approximation whose error grows with the
  gap, which is why ``gen_sample`` keeps the Reynolds-only schedule separate from the angle one:
  Re is the smooth axis and AoA is the rough one, measured.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from .. import N_MEASURED, T_IN
from .cache import CACHE_DIR, load_index, load_masks

#: Where the computed climatologies are memoised. One float32 map per cell is 16 kB, so the
#: whole real grid is ~1.3 MB -- cheap enough that the cache exists for the ~20 s of memmap
#: reads it saves, not for the space.
CLIMATOLOGY_PATH = CACHE_DIR / "climatology_{kind}.npz"

#: Per-cell fluctuation amplitude, memoised per block length because the quantity depends on it.
ENERGY_PATH = CACHE_DIR / "block_energy_{kind}_{span}.json"


def build_climatology(
    kind: str = "real", overwrite: bool = False
) -> dict[tuple[int, int], np.ndarray]:
    """``{(Re, AoA): (H, W, 2)}`` time-mean velocity of every cached trajectory, normalised.

    Averaged over **every** frame of the trajectory, including the ``val_time`` tail: this is a
    property of the cell, and the tail is statistically the same flow (splits.py measures the
    last 25% differing from the first 75% by -0.02% +/- 0.41% in freestream speed). Callers that
    must not see a cell at all exclude the *cell*, which is what :func:`interpolator` does.
    """
    path = CACHE_DIR / CLIMATOLOGY_PATH.name.format(kind=kind)
    if path.exists() and not overwrite:
        with np.load(path) as z:
            return {tuple(int(v) for v in k.split("_")): z[k] for k in z.files}  # type: ignore[misc]

    out: dict[tuple[int, int], np.ndarray] = {}
    masks = load_masks(kind)
    for c in load_index(kind):
        arr = c.open()
        mean = np.asarray(arr[..., :N_MEASURED], dtype=np.float64).mean(axis=0) / max(c.u_fs, 1e-9)
        mean[masks[(c.re, c.aoa)]] = 0.0
        out[(c.re, c.aoa)] = mean.astype(np.float32)
    np.savez_compressed(path, **{f"{r}_{a}": m for (r, a), m in out.items()})
    return out


def _axis_weights(x: float, grid: np.ndarray) -> list[tuple[int, float]]:
    """Linear-interpolation weights of ``x`` on a sorted 1-D ``grid``, clamped at both ends.

    Clamped rather than extrapolated: a linear extrapolation of a mean flow past the end of the
    ladder invents a flow regime, and 5f's asymmetry says the honest move at the edge is the
    nearest rung. An angle beyond 20 degrees is deep stall and no interpolation reaches it.
    """
    if len(grid) == 1 or x <= grid[0]:
        return [(0, 1.0)]
    if x >= grid[-1]:
        return [(len(grid) - 1, 1.0)]
    j = int(np.searchsorted(grid, x))
    lo, hi = grid[j - 1], grid[j]
    w = (x - lo) / (hi - lo)
    return [(j - 1, 1.0 - w), (j, w)]



#: Per-cell temporal standard deviation of the velocity, memoised like the mean climatology.
CLIMATOLOGY_STD_PATH = CACHE_DIR / "climatology_std_{kind}.npz"


def build_climatology_std(
    kind: str = "real", overwrite: bool = False
) -> dict[tuple[int, int], np.ndarray]:
    """``{(Re, AoA): (H, W, 2)}`` temporal std of ``u``/``v`` per pixel, freestream-normalised.

    The second-moment twin of :func:`build_climatology` -- the map FINDINGS 5f correlates 0.935
    between adjacent Reynolds rungs -- over every frame of the trajectory, zero in the mask.
    Used as a neighbour prior (``models.wrappers.NbrPriorWrapper``, FINDINGS 224), never for the
    cell itself.
    """
    path = CACHE_DIR / CLIMATOLOGY_STD_PATH.name.format(kind=kind)
    if path.exists() and not overwrite:
        with np.load(path) as z:
            return {tuple(int(v) for v in k.split("_")): z[k] for k in z.files}  # type: ignore[misc]

    out: dict[tuple[int, int], np.ndarray] = {}
    masks = load_masks(kind)
    for c in load_index(kind):
        arr = np.asarray(c.open()[..., :N_MEASURED], dtype=np.float64) / max(c.u_fs, 1e-9)
        std = arr.std(axis=0)
        std[masks[(c.re, c.aoa)]] = 0.0
        out[(c.re, c.aoa)] = std.astype(np.float32)
    np.savez_compressed(path, **{f"{r}_{a}": m for (r, a), m in out.items()})
    return out

def build_block_energy(
    kind: str = "real", span: int = 2 * T_IN, overwrite: bool = False
) -> dict[tuple[int, int], float]:
    """``{(Re, AoA): mean fluctuation energy per element}`` over non-overlapping blocks.

    The **amplitude** half of the operating-point mapping, and the target a generated block's
    fluctuation is calibrated to (:func:`rpde_t1.train.gen_sample.write_synthetic_set`). Measured
    over blocks of the same length the generator emits, because a fluctuation about a *finite*
    window's own mean is smaller than one about the trajectory mean and the two must not be mixed.

    Why an amplitude table exists at all: 6f measures per-cell amplitude recalibration as the only
    intervention that closes any of the sim-real second-moment gap (76% of it in error terms, 85%
    of the closable distance in score terms) while blur closed 0.001 and noise 0.004. A generator
    that gets the *shape* approximately right and the *level* wrong is the same situation, and it
    has the same fix.
    """
    path = CACHE_DIR / ENERGY_PATH.name.format(kind=kind, span=span)
    if path.exists() and not overwrite:
        return {
            tuple(int(v) for v in k.split("_")): float(x)  # type: ignore[misc]
            for k, x in json.loads(path.read_text(encoding="utf-8")).items()
        }

    out: dict[tuple[int, int], float] = {}
    for c in load_index(kind):
        arr = np.asarray(c.open()[..., :N_MEASURED], dtype=np.float32) / max(c.u_fs, 1e-9)
        n = arr.shape[0] // span
        blocks = arr[: n * span].reshape(n, span, *arr.shape[1:])
        out[(c.re, c.aoa)] = float(((blocks - blocks.mean(axis=1, keepdims=True)) ** 2).mean())
    path.write_text(json.dumps({f"{r}_{a}": v for (r, a), v in out.items()}), encoding="utf-8")
    return out


def interpolator(cells: list[tuple[int, int]], kind: str = "real", table: dict | None = None):
    """Return ``f(re, aoa)``, bilinear in ``(log Re, AoA)`` over ``cells``.

    ``table`` defaults to the mean-field climatology, giving ``f(re, aoa) -> (H, W, 2)``; pass
    :func:`build_block_energy` for the scalar amplitude instead. One interpolator for both, so the
    first and second moment cannot end up interpolated by different rules -- and so the whitelist
    argument below is enforced once rather than twice.

    ``cells`` is the whitelist -- pass the training split's cells and no held-out operating
    point can reach the output, which is the property that keeps a synthetic set from carrying
    the answer to the measurement it is meant to improve.

    **Separable, with the Reynolds grid looked up per angle.** The obvious implementation -- one 2x2
    stencil on the outer product of the two axes' values -- is wrong on this release, and it failed
    61 cells into a sampling run rather than silently: real is **81 distinct trajectories on an 18x5
    grid**, so 9 cells simply do not exist, and holding out whole Reynolds rows punches more. At
    ``Re=23512.5, AoA=5`` all four corners were missing cells at once and there was nothing to
    interpolate. Interpolating along Reynolds *within* each bracketing angle, using the rungs that
    exist **at that angle**, uses the data that is actually there; the fallback is never a
    fabricated corner.

    Two consequences worth stating: at an existing cell it still returns that cell exactly, and a
    request whose angle sits outside the available angles clamps rather than extrapolating.
    """
    clim = build_climatology(kind) if table is None else table
    have = {c: clim[c] for c in cells if c in clim}
    if not have:
        raise ValueError(f"None of the {len(cells)} requested cells exist in the {kind} cache.")
    aoa_grid = np.array(sorted({c[1] for c in have}), dtype=np.float64)
    #: Rungs available *at each angle*, which is what makes the grid's holes harmless.
    rungs_at = {
        a: np.array(sorted({c[0] for c in have if c[1] == a}), dtype=np.float64) for a in aoa_grid
    }

    def f(re: float, aoa: float):
        total, weight = None, 0.0
        for j, wa in _axis_weights(float(aoa), aoa_grid):
            angle = aoa_grid[j]
            rungs = rungs_at[angle]
            if not len(rungs):
                continue
            for i, wr in _axis_weights(float(np.log(re)), np.log(rungs)):
                key = (int(rungs[i]), int(angle))
                w = wr * wa
                total = have[key] * w if total is None else total + have[key] * w
                weight += w
        if total is None or weight <= 0:
            raise ValueError(f"No training cell within the interpolation stencil of {re=}, {aoa=}")
        # ``np.float32`` for a field, a plain float for a scalar table: the caller of the scalar
        # form divides by it, and a float32 there would quietly halve the precision of a gain.
        blended = total / weight
        return blended.astype(np.float32) if isinstance(blended, np.ndarray) else float(blended)

    return f


@lru_cache(maxsize=4)
def _masks(kind: str) -> dict:
    """``load_masks`` memoised. It decompresses an npz, and ``borrow_mask`` is called per cell.

    Read-only by contract: every caller here indexes and negates, none mutates. A mutation would
    be shared across calls, which is why this is private and the contract is written down.
    """
    return load_masks(kind)


def borrow_mask(re: float, aoa: float, cells: list[tuple[int, int]], kind: str = "real"):
    """The measurement mask of the ``cells`` entry nearest ``(re, aoa)``, angle first.

    A PIV mask is not the airfoil alone -- it is the body, its laser shadow and the rejected
    vectors -- so it cannot be drawn from geometry at an arbitrary angle the way
    :func:`rpde_t1.geometry.airfoil_sdf` can. Incidence dominates where it lands, hence angle
    before Reynolds in the ordering (the same rule ``cache.real_mask_for_sim`` uses in the other
    direction). Returned as ``valid``, i.e. 1 where a measurement exists.
    """
    masks = _masks(kind)
    have = [c for c in cells if c in masks]
    if not have:
        raise ValueError(f"No mask available among the {len(cells)} requested cells.")
    key = min(have, key=lambda c: (abs(c[1] - aoa), abs(np.log(c[0]) - np.log(re))))
    return (~masks[key]).astype(np.float32), key
