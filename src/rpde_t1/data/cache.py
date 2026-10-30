"""Pre-resolved float32 caches of every trajectory at the scored resolution.

Reading windows straight from the HDF5 files during training is a bad trade: the
real trajectories are internally gzip-compressed, so every random window read
decompresses whole chunks. Resolving each trajectory once to a ``.npy`` memmap at
32x64 float32 turns that into a pointer arithmetic problem.

The cache is small enough to be free: 82 real trajectories at 868x32x64x3 float32
is ~1.7 GB and 100 sim trajectories at 1000x32x64x3 is ~2.5 GB.

Alongside the arrays we persist, per trajectory:

* ``u_fs`` -- the freestream speed, which is what makes the datasets comparable.
  Real data is in physical m/s and scales linearly with Re (measured r = +1.0000
  across the ladder); sim is already non-dimensionalised by freestream (u ~ 1.09
  everywhere). Dividing each sample by its own freestream speed collapses a 6.5x
  amplitude range onto one scale, and -- more importantly -- means a model never has
  to *extrapolate an amplitude* at an unseen Reynolds number.
* ``invalid`` -- the measurement mask. Real PIV loses ~5-11% of the field to the
  airfoil, its laser shadow and rejected vectors; CFD loses only the ~0.2% solid
  section. Sim samples must be masked to the real pattern before being used as a
  pretraining target, or we teach the model to fill a region every real target
  leaves empty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import SUB_S
from ..features import FREESTREAM_ROWS, freestream_speed, invalid_mask
from ..paths import DATA_DIR
from .h5io import read_trajectory
from .survey import cells, phase_offset

#: Re-exported so existing callers keep working; the definitions live in
#: ``rpde_t1.features``, which a submission can import without dragging in h5py.
__all__ = [
    "FREESTREAM_ROWS",
    "freestream_speed",
    "CACHE_DIR",
    "build_cache",
    "build_native_cache",
    "load_index",
    "load_masks",
    "load_native_paths",
    "native_dir",
    "real_mask_for_sim",
    "CachedTrajectory",
    "cache_paths",
]

CACHE_DIR = DATA_DIR / "cache"


@dataclass(frozen=True)
class CachedTrajectory:
    """One cached trajectory: a memmapped array plus its scalars."""

    path: Path
    re: int
    aoa: int
    n_frames: int
    u_fs: float
    kind: str

    def open(self) -> np.ndarray:
        """Memmap the ``(T, 32, 64, 3)`` float32 array."""
        return np.load(self.path, mmap_mode="r")


def cache_paths(kind: str) -> dict[tuple[int, int], Path]:
    return {k: CACHE_DIR / kind / f"{k[0]}_{k[1]}.npy" for k in cells(kind)}


def build_cache(kind: str, sub_s: int = SUB_S, overwrite: bool = False) -> list[CachedTrajectory]:
    """Resolve every trajectory of one dataset to a float32 ``.npy`` at 32x64.

    A ``real_p<row><col>`` kind samples the recorded 64x128 grid at that phase instead of
    at the scored ``(0, 0)`` -- three more views of the same 82 trajectories, which is what
    ``phase_aug`` trains on. Everything downstream is per-kind already, so the measurement
    mask is rebuilt from the shifted array rather than borrowed from ``real``: at a shifted
    phase the airfoil and its laser shadow fall on different pixels.
    """
    offset = phase_offset(kind)
    out_dir = CACHE_DIR / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[CachedTrajectory] = []

    for (re_, aoa), src in sorted(cells(kind).items()):
        dst = out_dir / f"{re_}_{aoa}.npy"
        if dst.exists() and not overwrite:
            arr = np.load(dst, mmap_mode="r")
            u_fs = freestream_speed(np.asarray(arr[: min(64, len(arr))]))
            index.append(CachedTrajectory(dst, re_, aoa, int(arr.shape[0]), u_fs, kind))
            continue

        traj = read_trajectory(src, sub_s=sub_s, offset=offset)
        fields = np.ascontiguousarray(traj.fields, dtype=np.float32)
        np.save(dst, fields)
        index.append(
            CachedTrajectory(dst, re_, aoa, int(fields.shape[0]), freestream_speed(fields), kind)
        )

    _write_index(kind, index)
    return index


def native_dir(kind: str) -> Path:
    """Where the un-subsampled ``64x128`` mirror of ``kind``'s cache lives."""
    return CACHE_DIR / f"{kind}_native"


def build_native_cache(kind: str, overwrite: bool = False) -> dict[tuple[int, int], Path]:
    """Mirror every trajectory of ``kind`` at the **recording** resolution, ``64x128``.

    Supervision target for the fine-resolution auxiliary head (``w_fine``) and the
    sub-grid variance head (``w_subgrid``). It is a *target-side* cache only: the model's
    input stays ``32x64``, because no fine input exists at submission time.

    Stored float16, which is not a corner cut. The quantisation step at O(1) field values
    is ~5e-4, so its variance contribution is ~2e-8 against a within-window fluctuation
    variance of ~1e-3 -- 0.002% of the quantity the fine head is meant to teach. Halving
    the footprint matters more: the 100 simulated trajectories are 4.9 GB at float16 and
    9.8 GB at float32, against 22 GB of free disk.

    Frame indexing is identical to :func:`build_cache`'s, so a window ``[s : s+40]``
    addresses the same physical frames in both, and ``native[..., ::2, ::2]`` is the
    coarse array exactly -- both come from the same HDF5 read path with a different
    ``sub_s``. :func:`tests.test_native_cache` pins that.
    """
    out_dir = native_dir(kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[int, int], Path] = {}

    for (re_, aoa), src in sorted(cells(kind).items()):
        dst = out_dir / f"{re_}_{aoa}.npy"
        if not dst.exists() or overwrite:
            traj = read_trajectory(src, sub_s=1)
            np.save(dst, np.ascontiguousarray(traj.fields, dtype=np.float16))
        paths[(re_, aoa)] = dst
    return paths


def load_native_paths(kind: str) -> dict[tuple[int, int], Path]:
    """Existing native-resolution arrays for ``kind``, keyed by cell.

    Missing files are omitted rather than raised on: the native cache is optional and
    only the arms that ask for fine supervision need it, so the caller decides whether an
    absence is fatal (:class:`WindowDataset` does, with a message naming the builder).
    """
    d = native_dir(kind)
    return {k: p for k, p in ((k, d / f"{k[0]}_{k[1]}.npy") for k in cells(kind)) if p.exists()}


def _index_path(kind: str) -> Path:
    return CACHE_DIR / f"index_{kind}.json"


def _write_index(kind: str, index: list[CachedTrajectory]) -> None:
    masks = _build_masks(kind, index)
    payload = {
        "kind": kind,
        "freestream_rows": FREESTREAM_ROWS,
        "trajectories": [
            {
                "re": c.re,
                "aoa": c.aoa,
                "n_frames": c.n_frames,
                "u_fs": c.u_fs,
                "path": c.path.name,
                "invalid_fraction": float(masks[(c.re, c.aoa)].mean()),
            }
            for c in index
        ],
    }
    _index_path(kind).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        CACHE_DIR / f"masks_{kind}.npz",
        **{f"{r}_{a}": m for (r, a), m in masks.items()},
    )


def _build_masks(kind: str, index: list[CachedTrajectory]) -> dict[tuple[int, int], np.ndarray]:
    """Per-trajectory measurement mask, from the full cached trajectory."""
    out: dict[tuple[int, int], np.ndarray] = {}
    for c in index:
        out[(c.re, c.aoa)] = invalid_mask(np.asarray(c.open()))
    return out


def load_index(kind: str) -> list[CachedTrajectory]:
    """Read a cache index written by :func:`build_cache`."""
    path = _index_path(kind)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `make cache`.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        CachedTrajectory(
            CACHE_DIR / kind / t["path"], t["re"], t["aoa"], t["n_frames"], t["u_fs"], kind
        )
        for t in payload["trajectories"]
    ]


def load_masks(kind: str) -> dict[tuple[int, int], np.ndarray]:
    path = CACHE_DIR / f"masks_{kind}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `make cache`.")
    with np.load(path) as z:
        return {tuple(int(v) for v in k.split("_")): z[k] for k in z.files}  # type: ignore[misc]


def real_mask_for_sim(sim_key: tuple[int, int], real_masks: dict[tuple[int, int], np.ndarray]):
    """The real measurement mask to impose on a simulated cell.

    18 simulated cells have no real counterpart (they are where real data was
    withheld), so we borrow the mask from the nearest Reynolds number at the *same*
    angle of attack -- the mask geometry is set by the airfoil's incidence and its
    shadow, which depends on AoA far more than on Re.
    """
    re_, aoa = sim_key
    if sim_key in real_masks:
        return real_masks[sim_key]
    same_aoa = [k for k in real_masks if k[1] == aoa]
    if not same_aoa:
        raise KeyError(f"No real mask at AoA={aoa} to borrow for {sim_key}.")
    return real_masks[min(same_aoa, key=lambda k: abs(k[0] - re_))]
