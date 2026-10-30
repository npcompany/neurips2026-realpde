"""Masks over the input window, for masked spatio-temporal modelling.

Three geometries, because they ask the model for three different things and the study is
about which one the forecaster actually benefits from:

``tube``
    Spatial patches hidden in **every** input frame, so no frame carries the answer.
    VideoMAE's tube mask. The model must inpaint spatially and can only use the temporal
    axis through advection -- content blows into the hole from upstream. This is the one
    that resembles the real defect: PIV loses a *fixed* region (airfoil, laser shadow) for
    the whole trajectory, which is exactly a tube.

``frame``
    Whole frames hidden. Pure temporal interpolation; spatial structure is untouched and
    the neighbouring frames make it easy. The cheap end of the difficulty range, and the
    control that says whether difficulty is what matters.

``tail``
    A contiguous suffix of frames hidden. Causal temporal state estimation: there is no
    observation after the gap, so the model must extrapolate the observed dynamics.

``cube``
    Patches hidden over a limited span of frames -- tubelets. Between the two: the hole
    closes in time, so the model can carry structure across the gap in either direction.

Granularity is a patch, not a pixel, and deliberately: at ``PATCH=4`` on a ``32x64`` grid a
hole is 4 cells across, wider than the ~2-cell correlation length of the smallest resolved
structures, so filling it needs the flow rather than the neighbouring pixel. Pixel-wise
masking at any ratio is a denoising task the first convolution solves, which is what
``piv_noise`` already covers on better-calibrated terms.

The mask says which cells are **hidden**. Zero-filling them is not self-describing -- the
PIV measurement mask is also zero -- so the dataset appends a visibility plane to the
input; see :data:`MASK_PLANES`.
"""

from __future__ import annotations

import numpy as np

from .. import MASK_PLANES, n_mask_planes

__all__ = ["MASK_MODES", "MASK_PLANES", "PATCH", "TUBELET_T", "n_mask_planes", "sample_input_mask"]

#: Mask geometries. ``""`` disables masking entirely.
MASK_MODES = ("tube", "frame", "tail", "cube")

#: Spatial patch edge, in cells of the scored grid.
PATCH = 4

#: Temporal extent of a ``cube`` tubelet, in frames.
TUBELET_T = 4


def sample_input_mask(
    rng: np.random.Generator,
    mode: str,
    ratio: float,
    t: int,
    h: int,
    w: int,
    patch: int = PATCH,
    tubelet_t: int = TUBELET_T,
) -> np.ndarray:
    """A ``(t, h, w)`` boolean mask, ``True`` where the input is hidden.

    ``ratio`` is the fraction of *patches* hidden, which equals the fraction of cells
    hidden whenever ``patch`` divides ``h`` and ``w`` (it does at 32x64 for ``PATCH=4``).
    At least one patch is always taken when ``ratio > 0``, so a small ratio degrades to a
    single hole rather than silently to no masking at all.
    """
    if not mode:
        return np.zeros((t, h, w), dtype=bool)
    if mode not in MASK_MODES:
        raise ValueError(f"mask mode must be one of {MASK_MODES} or '', got {mode!r}")
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mask ratio must lie in (0, 1), got {ratio}")

    if mode == "frame":
        n = max(1, int(round(ratio * t)))
        frames = rng.choice(t, size=min(n, t - 1), replace=False)
        mask = np.zeros((t, h, w), dtype=bool)
        mask[frames] = True
        return mask

    if mode == "tail":
        n = max(1, min(int(round(ratio * t)), t - 1))
        mask = np.zeros((t, h, w), dtype=bool)
        mask[-n:] = True
        return mask

    ph, pw = h // patch, w // patch
    n_patches = ph * pw
    if mode == "tube":
        n = max(1, int(round(ratio * n_patches)))
        take = rng.choice(n_patches, size=min(n, n_patches - 1), replace=False)
        flat = np.zeros(n_patches, dtype=bool)
        flat[take] = True
        plane = np.repeat(np.repeat(flat.reshape(ph, pw), patch, 0), patch, 1)
        return np.broadcast_to(plane, (t, h, w)).copy()

    # cube: independent patch choices per temporal chunk.
    n_chunks = max(1, t // tubelet_t)
    n_cells = n_chunks * n_patches
    n = max(1, int(round(ratio * n_cells)))
    take = rng.choice(n_cells, size=min(n, n_cells - 1), replace=False)
    flat = np.zeros(n_cells, dtype=bool)
    flat[take] = True
    grid = flat.reshape(n_chunks, ph, pw)
    plane = np.repeat(np.repeat(grid, patch, 1), patch, 2)  # (n_chunks, h, w)
    # The trailing frames of a horizon that does not divide by tubelet_t join the last
    # chunk, so every frame is covered exactly once.
    reps = np.full(n_chunks, tubelet_t)
    reps[-1] += t - reps.sum()
    return np.repeat(plane, reps, axis=0)
