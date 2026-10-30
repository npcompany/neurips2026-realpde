"""Reading RealPDE airfoil trajectories out of HDF5.

There are two layouts in the wild and both must be supported:

* ``data/example/3750_0.h5``  -- ``u``, ``v`` at the top level.
* ``train_real.tar.gz``       -- ``u``, ``v`` under ``measured_data/``.

The simulated tarball additionally carries a pressure field. Rather than
hard-code either layout we search a small set of candidate paths per channel,
which also keeps us honest if the organizers reshuffle a future release.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

#: Candidate dataset paths per channel, in priority order.
_CANDIDATES: dict[str, tuple[str, ...]] = {
    "u": ("u", "measured_data/u", "data/u", "fields/u"),
    "v": ("v", "measured_data/v", "data/v", "fields/v"),
    "p": ("p", "measured_data/p", "numerical_data/p", "data/p", "fields/p"),
}

#: Candidate paths for the scalar operating parameters.
_SCALARS: dict[str, tuple[str, ...]] = {
    "aoa": ("aoa", "AoA", "alpha", "measured_data/aoa", "attrs/aoa"),
    "re": ("re", "Re", "reynolds", "measured_data/re", "attrs/re"),
}


def _find(f: h5py.File, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in f and isinstance(f[key], h5py.Dataset):
            return key
    return None


def _read_scalar(f: h5py.File, candidates: tuple[str, ...]) -> float | None:
    for key in candidates:
        if key in f and isinstance(f[key], h5py.Dataset):
            return float(np.asarray(f[key]).reshape(-1)[0])
    for key in candidates:
        # Some releases store the operating point as an attribute instead.
        if key in f.attrs:
            return float(np.asarray(f.attrs[key]).reshape(-1)[0])
    return None


@dataclass(frozen=True)
class Trajectory:
    """One airfoil trajectory: fields plus its operating point.

    ``fields`` has shape ``(T, H, W, 3)`` in channel order ``[u, v, p]``, with
    the ``p`` plane left at zero when the file has no pressure (all real PIV).
    """

    fields: np.ndarray
    aoa: float | None
    re: float | None
    source: Path
    has_pressure: bool

    @property
    def n_frames(self) -> int:
        return int(self.fields.shape[0])


def read_trajectory(
    path: str | Path,
    sub_s: int = 1,
    t_slice: slice | None = None,
    offset: tuple[int, int] = (0, 0),
) -> Trajectory:
    """Load a trajectory, subsampling space by ``sub_s`` (2 for the scored 32x64).

    ``t_slice`` restricts the frames read from disk, which matters: a full
    trajectory at 64x128 float32 is ~100 MB per channel.

    ``offset`` shifts where the subsampling grid starts. At ``sub_s=2`` the four offsets
    are four equally valid 32x64 recordings of the same flow, and the competition scores
    only ``(0, 0)`` -- which is what makes the other three an augmentation rather than a
    different task (``survey.PHASE_KINDS``). Ignored when ``sub_s=1``, where there is only
    one grid.
    """
    path = Path(path)
    sel = t_slice if t_slice is not None else slice(None)
    oh, ow = (0, 0) if sub_s == 1 else offset
    if not (0 <= oh < sub_s and 0 <= ow < sub_s):
        raise ValueError(f"offset {offset} is outside the sub_s={sub_s} grid")
    with h5py.File(path, "r") as f:
        u_key = _find(f, _CANDIDATES["u"])
        v_key = _find(f, _CANDIDATES["v"])
        if u_key is None or v_key is None:
            raise KeyError(
                f"{path}: could not locate u/v datasets. "
                f"Top-level keys present: {sorted(f.keys())[:20]}"
            )
        u = np.asarray(f[u_key][sel, oh::sub_s, ow::sub_s], dtype=np.float32)
        v = np.asarray(f[v_key][sel, oh::sub_s, ow::sub_s], dtype=np.float32)

        p_key = _find(f, _CANDIDATES["p"])
        if p_key is not None:
            p = np.asarray(f[p_key][sel, oh::sub_s, ow::sub_s], dtype=np.float32)
        else:
            p = np.zeros_like(u)

        aoa = _read_scalar(f, _SCALARS["aoa"])
        re = _read_scalar(f, _SCALARS["re"])

    return Trajectory(
        fields=np.stack([u, v, p], axis=-1),
        aoa=aoa,
        re=re,
        source=path,
        has_pressure=p_key is not None,
    )


def frame_count(path: str | Path) -> int:
    """Number of frames in a trajectory, read from the dataset shape only.

    Trajectory lengths are **not** uniform: of the 82 delivered real
    trajectories, 79 have 868 frames and three have 282 / 492 / 607. Any code
    that slices by absolute frame index must consult this rather than assume.
    """
    with h5py.File(Path(path), "r") as f:
        key = _find(f, _CANDIDATES["u"])
        if key is None:
            raise KeyError(f"{path}: no u dataset to size.")
        return int(f[key].shape[0])


def describe(path: str | Path) -> dict[str, object]:
    """Cheap structural summary of an HDF5 file -- no bulk reads.

    Used by ``make inspect`` to work out what a freshly extracted tarball
    actually contains before we commit to a loader.
    """
    path = Path(path)
    info: dict[str, object] = {"path": str(path), "datasets": {}, "attrs": {}}
    with h5py.File(path, "r") as f:

        def visit(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                info["datasets"][name] = {  # type: ignore[index]
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "compression": obj.compression,
                }

        f.visititems(visit)
        info["attrs"] = {k: str(np.asarray(v).reshape(-1)[:4]) for k, v in f.attrs.items()}
    return info
