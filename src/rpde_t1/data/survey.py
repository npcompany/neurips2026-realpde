"""Cheap dataset-wide surveys, shared by the exploration notebooks.

These read only HDF5 shapes and small slices, so a full survey of both datasets
costs seconds rather than the ~16 GB a naive pass would move. Keeping them here
rather than inline in notebooks means they are importable, testable, and identical
across notebooks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..features import invalid_mask
from ..paths import DATA_DIR
from .h5io import frame_count, read_trajectory
from .splits import Params, parse_params

# ``invalid_mask`` is re-exported: it now lives in ``rpde_t1.features``, which imports
# only numpy so a submission can use it without dragging this module's h5py
# dependency into the evaluation container.
__all__ = [
    "CellStats",
    "PHASE_KINDS",
    "phase_kinds",
    "base_kind",
    "cells",
    "coverage_matrix",
    "dataset_root",
    "frame_counts",
    "grid_matrix",
    "invalid_mask",
    "phase_offset",
    "survey",
    "survey_cell",
    "temporal_tke",
]


@dataclass
class CellStats:
    """Summary of one ``(Re, AoA)`` trajectory, from a bounded frame sample."""

    params: Params
    n_frames: int
    has_pressure: bool
    u_mean: float
    u_std: float
    v_mean: float
    v_std: float
    tke_mean: float
    extra: dict = field(default_factory=dict)


#: A real cache built at one of the four ``[::2, ::2]`` phases of the recorded 64x128
#: grid, named ``real_p<row><col>``. ``real`` itself is phase (0, 0) -- the one the
#: competition scores -- so only the other three are ever built.
_PHASE = re.compile(r"^(?P<base>real|sim)_p(?P<oh>[01])(?P<ow>[01])$")

#: The three additional phases of the real grid, kept as a name because it predates the
#: simulated variants. Prefer :func:`phase_kinds`, which serves either dataset.
PHASE_KINDS: tuple[str, ...] = ("real_p01", "real_p10", "real_p11")


def phase_kinds(base: str = "real") -> tuple[str, ...]:
    """The three unscored phases of ``base``'s grid. Both grids are 64x128."""
    if base not in ("real", "sim"):
        raise ValueError(f"phase caches exist for 'real' and 'sim', not {base!r}")
    return tuple(f"{base}_p{o}" for o in ("01", "10", "11"))


def phase_offset(kind: str) -> tuple[int, int]:
    """``(row, col)`` offset into the 64x128 grid that a cache kind samples at.

    Every non-phase kind is (0, 0), so this is safe to call on any kind.
    """
    m = _PHASE.match(kind)
    return (int(m["oh"]), int(m["ow"])) if m else (0, 0)


def base_kind(kind: str) -> str:
    """The delivered dataset a (possibly phase-shifted) cache kind is built from.

    Every behaviour that depends on *which dataset* a sample came from -- masking sim to
    the real pattern, the is_sim conditioning bit, the fluctuation-rescaling table, the
    sim-only augmentations -- must branch on this rather than on the raw kind, or a
    ``sim_p01`` sample is silently treated as real.
    """
    m = _PHASE.match(kind)
    return m["base"] if m else kind


def dataset_root(kind: str) -> Path:
    """``'real'`` -> ``data/train_real``, ``'sim'`` -> ``data/train_sim``."""
    kind = base_kind(kind)
    if kind not in ("real", "sim"):
        raise ValueError(f"kind must be 'real' or 'sim', got {kind!r}")
    return DATA_DIR / ("train_real" if kind == "real" else "train_sim")


def cells(kind: str) -> dict[tuple[int, int], Path]:
    """Map ``(Re, AoA)`` to the trajectory file for one dataset."""
    root = dataset_root(kind)
    paths = sorted(root.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 under {root}. Run `make data` then `make extract`.")
    return {(parse_params(p).re, parse_params(p).aoa): p for p in paths}


def frame_counts(kind: str) -> dict[tuple[int, int], int]:
    """Frame count per cell, read from dataset shapes only (no bulk I/O)."""
    return {k: frame_count(v) for k, v in cells(kind).items()}


def temporal_tke(fields: np.ndarray) -> np.ndarray:
    """Per-pixel temporal fluctuation energy -- the quantity ``tke_score`` scores.

    Mirrors the official scorer's ``kinetic_energy``: variance about the *time*
    mean of u and v, halved and summed. Defined here for visualization; the scored
    value always comes from the vendored scorer.
    """
    u, v = fields[..., 0], fields[..., 1]
    return 0.5 * (
        ((u - u.mean(axis=0, keepdims=True)) ** 2).mean(axis=0)
        + ((v - v.mean(axis=0, keepdims=True)) ** 2).mean(axis=0)
    )


def survey_cell(path: Path, n_frames: int = 200, sub_s: int = 2) -> CellStats:
    """Summarize one trajectory from its leading ``n_frames`` frames."""
    traj = read_trajectory(path, sub_s=sub_s, t_slice=slice(0, n_frames))
    u, v = traj.fields[..., 0], traj.fields[..., 1]
    return CellStats(
        params=parse_params(path),
        n_frames=frame_count(path),
        has_pressure=traj.has_pressure,
        u_mean=float(u.mean()),
        u_std=float(u.std()),
        v_mean=float(v.mean()),
        v_std=float(v.std()),
        tke_mean=float(temporal_tke(traj.fields).mean()),
    )


def survey(kind: str, n_frames: int = 200, sub_s: int = 2) -> list[CellStats]:
    """Survey every cell of one dataset. ~1 s per 10 trajectories."""
    return [survey_cell(p, n_frames=n_frames, sub_s=sub_s) for p in sorted(cells(kind).values())]


def grid_matrix(stats: list[CellStats], attr: str) -> tuple[np.ndarray, list[int], list[int]]:
    """Lay one attribute out on the (Re x AoA) grid for plotting.

    Returns ``(matrix, re_values, aoa_values)`` with NaN where a cell is absent --
    which is itself the interesting signal, since the missing real cells are where
    the organizers withheld data.
    """
    res = sorted({s.params.re for s in stats})
    aoas = sorted({s.params.aoa for s in stats})
    matrix = np.full((len(res), len(aoas)), np.nan)
    index = {(s.params.re, s.params.aoa): s for s in stats}
    for i, r in enumerate(res):
        for j, a in enumerate(aoas):
            s = index.get((r, a))
            if s is not None:
                matrix[i, j] = getattr(s, attr)
    return matrix, res, aoas


def coverage_matrix() -> tuple[np.ndarray, list[int], list[int]]:
    """Which cells exist in which dataset: 0 neither, 1 sim only, 2 real only, 3 both.

    Real is a strict subset of sim, so in practice only 1 and 3 occur -- the 1s mark
    the parameter cells where real data was withheld.
    """
    real, sim = set(cells("real")), set(cells("sim"))
    res = sorted({r for r, _ in real | sim})
    aoas = sorted({a for _, a in real | sim})
    matrix = np.zeros((len(res), len(aoas)))
    for i, r in enumerate(res):
        for j, a in enumerate(aoas):
            matrix[i, j] = ((r, a) in sim) + 2 * ((r, a) in real)
    return matrix, res, aoas
