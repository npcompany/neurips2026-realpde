"""Cutting trajectories into the (T_in -> T_out) windows the competition scores.

Track 1 is a fixed block map: 20 observed frames in, the next 20 frames out. A
"window" is one such consecutive pair, so a trajectory of ``T`` frames yields
``T - (T_in + T_out) + 1`` overlapping windows, or fewer with a stride.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import SUB_S, T_IN, T_OUT
from .h5io import Trajectory, frame_count, read_trajectory


@dataclass(frozen=True)
class WindowSet:
    """A batch of windows, ready for ``predict()`` and for the official scorer.

    ``inputs`` / ``targets`` are ``(N, 20, 32, 64, 3)``. ``origins`` records the
    ``(trajectory index, start frame)`` of each window so a bad score can be
    traced back to a specific place in a specific trajectory.
    """

    inputs: np.ndarray
    targets: np.ndarray
    origins: np.ndarray
    sources: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.inputs.shape[0])


def window_starts(
    n_frames: int, t_in: int = T_IN, t_out: int = T_OUT, stride: int = 1
) -> list[int]:
    """Valid window start indices for a trajectory of ``n_frames`` frames."""
    span = t_in + t_out
    if n_frames < span:
        return []
    return list(range(0, n_frames - span + 1, stride))


def windows_from_trajectory(
    traj: Trajectory,
    stride: int = 1,
    t_in: int = T_IN,
    t_out: int = T_OUT,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Split one already-loaded trajectory into (inputs, targets, starts)."""
    starts = window_starts(traj.n_frames, t_in, t_out, stride)
    if max_windows is not None:
        starts = starts[:max_windows]
    if not starts:
        empty = np.zeros((0, t_in, *traj.fields.shape[1:]), dtype=np.float32)
        return empty, empty, []

    inputs = np.stack([traj.fields[s : s + t_in] for s in starts]).astype(np.float32)
    targets = np.stack([traj.fields[s + t_in : s + t_in + t_out] for s in starts]).astype(
        np.float32
    )
    return inputs, targets, starts


def build_window_set(
    h5_paths: list[str | Path],
    stride: int = 20,
    sub_s: int = SUB_S,
    t_in: int = T_IN,
    t_out: int = T_OUT,
    max_windows_per_traj: int | None = None,
    max_frames: int | None = None,
    t_offset: int = 0,
    t_offset_frac: float | None = None,
) -> WindowSet:
    """Build an evaluation set from a list of trajectory files.

    ``stride`` defaults to 20 (non-overlapping targets) so windows are not
    near-duplicates of one another -- overlapping windows would make a local
    score look more precise than it is.

    ``max_frames`` caps the frames read per trajectory, which keeps a quick
    smoke-scoring pass cheap.

    Skipping leading frames (how ``val_time`` reads only the tail withheld from
    training) comes in two forms. ``t_offset_frac`` is the one to use: it resolves
    per trajectory against that file's own length. ``t_offset`` is an absolute
    frame index and is only safe when every trajectory is known to be long enough
    -- real trajectory lengths are **not** uniform (79 have 868 frames, three have
    282 / 492 / 607), so an absolute offset silently drops the short ones.

    Recorded ``origins`` are shifted back to absolute frame indices, so a window is
    always traceable to its place in the original trajectory.
    """
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    sources: list[str] = []
    dropped: list[str] = []

    for traj_idx, path in enumerate(h5_paths):
        offset = t_offset
        if t_offset_frac is not None:
            offset = int(round(frame_count(path) * t_offset_frac))
        stop = offset + max_frames if max_frames else None
        t_slice = slice(offset, stop) if (offset or max_frames) else None
        traj = read_trajectory(path, sub_s=sub_s, t_slice=t_slice)
        inputs, targets, starts = windows_from_trajectory(
            traj, stride=stride, t_in=t_in, t_out=t_out, max_windows=max_windows_per_traj
        )
        if len(starts) == 0:
            # Too few frames after the offset to form even one window. Recorded
            # rather than passed over, so a shrinking eval set is never silent.
            dropped.append(Path(path).name)
            continue
        all_inputs.append(inputs)
        all_targets.append(targets)
        origins.extend((traj_idx, offset + s) for s in starts)
        sources.append(str(path))

    if dropped:
        warnings.warn(
            f"{len(dropped)} trajectory(ies) produced no windows and were excluded: "
            f"{dropped}. Each needs {t_in + t_out} frames after the offset.",
            stacklevel=2,
        )

    if not all_inputs:
        raise ValueError(
            f"No windows produced from {len(h5_paths)} file(s); "
            f"each trajectory needs at least {t_in + t_out} frames."
        )

    return WindowSet(
        inputs=np.concatenate(all_inputs, axis=0),
        targets=np.concatenate(all_targets, axis=0),
        origins=np.asarray(origins, dtype=np.int64),
        sources=tuple(sources),
    )
