"""RealPDE Track 1 (Sim2Real): airfoil PIV flow prediction."""

from __future__ import annotations

#: Scored geometry, fixed by the competition. See ``vendor/.../README.md``:
#: the raw PIV fields are 64x128 but Track 1 evaluates at 32x64 (``SUB_S = 2``).
T_IN = 20
T_OUT = 20
H = 32
W = 64
C = 3
SUB_S = 2

#: Channel order of the ``(N, T, H, W, C)`` arrays. ``p`` is unmeasured in real
#: PIV data (all zeros) and is never scored -- the official scorer restricts every
#: metric to the non-zero target channels, which is ``[u, v]``.
CHANNELS = ("u", "v", "p")

#: How many of those channels the scorer actually measures on real data, i.e.
#: ``len(CHANNELS) - 1`` for the unscored pressure plane. Kept as a name because
#: several modules slice ``[..., :N_MEASURED]`` and a bare ``2`` there reads as a
#: magic number.
N_MEASURED = 2

INPUT_SHAPE = (T_IN, H, W, C)
OUTPUT_SHAPE = (T_OUT, H, W, C)

#: The five subscores the official scorer emits, in the order every report prints
#: them. Fixed by the competition, like the geometry above; the final score is *not*
#: their mean and is not computed anywhere in this package (see ``score/report.py``).
SUBSCORES = ("rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score")

#: Nominal measurement noise, frozen for the season in the official scorer's SPS
#: term. **Physical units**, so anything comparing against it must be in physical
#: units too -- a model working in freestream-normalised space has to multiply back
#: first. Defined here rather than beside either user because the interval head
#: (``models.interval``) and the training objective (``train.losses``) both need it
#: and neither can import the other.
SIGMA_GLOBAL = 0.0563870259

#: Input planes a masked-modelling run appends: one visibility channel, 1 where the cell is
#: shown to the model and 0 where it is hidden. Needed because zero-fill alone is ambiguous
#: -- the PIV measurement hole is also zero -- so the model would have to guess which zeros
#: it is being asked about.
#:
#: Here for the same reason as ``SIGMA_GLOBAL``: the layer that builds the tensor
#: (``data.masking``) and the layer that sizes the first convolution (``models.factory``)
#: both need it, and ``models`` ships in a submission bundle while ``data`` does not, so
#: neither can import the other.
MASK_PLANES = 1


def n_mask_planes(mode: str, enabled: bool = True) -> int:
    """Extra input channels the masking scheme ``mode`` adds. Zero when it is empty."""
    return MASK_PLANES if mode and enabled else 0


#: Probe geometry of ``mvpe_score``, in **native 64x128 pixels**, straight from
#: ``mvpe_rel_l2_per_sample`` in the vendored ``scoring.py``: probe spacing ``d``, the
#: station origin ``(center_x, center_y)``, and how many rows each column samples.
MVPE_D, MVPE_CENTER_X, MVPE_CENTER_Y, MVPE_N_PROBE = 16, 10, 32, 9
MVPE_N_COLUMNS = 4


def mvpe_probe_indices(h: int = H, w: int = W, sub_s: int = SUB_S) -> tuple[list[int], list[int]]:
    """``(rows, columns)`` the ``mvpe`` metric samples, for a ``(h, w)`` grid.

    A transcription of the vendored scorer's index arithmetic and nothing more --
    including its integer truncation and its out-of-range filtering, which are what
    decide the stations at any resolution other than the scored one. At the scored
    ``32x64`` it gives rows ``8..24`` step 2 and columns ``13, 21, 29, 37``.

    Here rather than beside either caller because there are now two: the numpy mirror
    in ``score.sps_lab`` and the differentiable term in ``train.losses``. Those must
    sample identical points or the objective is not the metric, and a second
    transcription is how they would drift. The package root is the only place both can
    import from -- ``score`` must keep working without torch installed.
    """
    center = int(MVPE_CENTER_Y / sub_s)
    step = min(2, int(h / (MVPE_N_PROBE + 1)))
    lo = -(MVPE_N_PROBE - 1) // 2
    rows = [center + step * j for j in range(lo, MVPE_N_PROBE + lo)]
    rows = [y for y in rows if 0 <= y < h]

    wide = int((2 * MVPE_D + MVPE_CENTER_X) / sub_s) < w
    cols = []
    for i in range(MVPE_N_COLUMNS):
        x = (
            int(((i + 1) * MVPE_D + MVPE_CENTER_X) / sub_s)
            if wide
            else int((0.5 * (i + 2) * MVPE_D + MVPE_CENTER_X) / sub_s)
        )
        if 0 <= x < w:
            cols.append(x)
    return rows, cols


__all__ = [
    "C",
    "CHANNELS",
    "MASK_PLANES",
    "MVPE_CENTER_X",
    "MVPE_CENTER_Y",
    "MVPE_D",
    "MVPE_N_COLUMNS",
    "MVPE_N_PROBE",
    "H",
    "INPUT_SHAPE",
    "N_MEASURED",
    "OUTPUT_SHAPE",
    "SIGMA_GLOBAL",
    "SUBSCORES",
    "SUB_S",
    "T_IN",
    "T_OUT",
    "W",
    "mvpe_probe_indices",
    "n_mask_planes",
]
