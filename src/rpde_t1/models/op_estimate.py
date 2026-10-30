"""The operating-point estimator, on the device.

``features.estimate_batch`` is the definition and stays the definition. This mirrors it
operation for operation for a window that is **already resident on the GPU**, which is the
deployment case: ``predict()`` uploads the window anyway, and the estimator is the largest
host-side item left inside the timed region.

It lives under ``models/`` rather than beside its numpy twin because ``features.py`` is
deliberately torch-free -- ``tests/test_features.py`` parses its AST and fails on any torch
import, including a deferred one, so that ``make score`` runs without torch installed. The
dependency direction is one-way: this imports ``features``, never the reverse.

Why it is worth having, measured on the shipped bundle at batch 210 (a 103 MB window),
idle T4 under the GPU lock:

    | numpy ``estimate_batch`` (host, shipped) | 54.8 ms | 0.261 ms/sample |
    |   of which the hole-mask reduction       | 31.7 ms | 58% of it       |
    |   of which the wake profile              |  9.2 ms | 17%             |
    | this, on the device                      |  2.30 ms| **23.9x**       |
    | this, inside ``torch.compile``           |  1.17 ms| **46.9x**       |

The hole mask dominates because it is two full passes of ``abs().sum(axis=1)`` over two
thirds of the array to recover ONE number per window -- the topmost row containing a hole.
On the device those passes are bandwidth rather than ~10 GB/s of host memory.

This matters more *after* ``torch.compile`` than before: the estimator is 8.0% of the eager
path but **11.6% of the compiled one**. Speeding the GPU up makes the host half a larger
share -- Amdahl, not a surprise, but it moves this from a rounding error to the biggest
remaining host item.

**float32, where the original accumulates in float64, deliberately.** A faithful-dtype port
is *slower than the host version it replaces*: a T4 runs fp64 at 1/32 rate and the window is
103 MB (measured, +37% end to end). The agreement this costs is measured rather than assumed
-- on 1166 real windows across five held-out splits, every one of which contains a
measurement hole, the worst disagreement is **9.7e-05 degrees of AoA and 4.2e-05 % of Re**,
against an estimator whose own held-out error is 1.15-1.77 degrees and 1.2% (FINDINGS 122).

On gaussian noise it reads 2.4e-07 -- **400x better than the truth**, which is exactly why
``estimate_batch``'s own test insists on the eval set: the mask and the wake are structural
and random arrays exercise neither.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import features  # for the mutable override globals + CORNERS3_COEF, read at call time
from ..features import (
    AOA_CLIP_DEG,
    AOA_COEF_FLUCT_EXTENT,
    AOA_COEF_MASK_YTOP,
    AOA_COEF_WAKE_DEPTH,
    AOA_INTERCEPT,
    AOA_MAX,
    FLUCT_EXTENT_COLS,
    FLUCT_EXTENT_SUB_T,
    FREESTREAM_ROWS,
    N_MEASURED_CHANNELS,
    RE_INTERCEPT,
    RE_MAX,
    RE_MIN,
    RE_PER_U_FS,
    RE_PER_U_FS_AOA,
    SIM_DOMAINS,
    TOP_ROW_ARM_PX,
    TOP_ROW_PHI_DEG,
    TOP_ROW_R0,
    Domain,
)


def corners3_reynolds_torch(x: torch.Tensor) -> torch.Tensor:
    """Device twin of :func:`rpde_t1.features.corners3_reynolds` (three freestream corners).

    Reads ``features.CORNERS3_COEF`` at call time so a train-only refit is honoured. AoA-free,
    so it needs no angle estimate. Accumulates in fp32 like the rest of this module; agrees with
    the numpy original to the fp32 quantum (``tests/test_features.py``).
    """
    u = x[..., 0].abs().float()
    tr = u[:, :, 0:1, 56:64].mean(dim=(1, 2, 3))
    br = u[:, :, -1:, 56:64].mean(dim=(1, 2, 3))
    tl = u[:, :, 0:1, 0:8].mean(dim=(1, 2, 3))
    a, b, cc, d = features.CORNERS3_COEF
    return (a * tr + b * br + cc * tl + d).clamp_min(1.0)


def estimate_batch_torch(x: torch.Tensor) -> torch.Tensor:
    """``(N, 2)`` ``(Re, AoA)`` for a resident ``(N, T, H, W, C)`` window in physical units.

    Device twin of :func:`rpde_t1.features.estimate_batch`. The reductions are written over
    the same axes in the same order as the numpy original, so the two stay comparable line by
    line; ``tests/test_features.py`` asserts the agreement on real eval windows.
    """
    xd = x.float()
    u_fs = xd[:, :, :FREESTREAM_ROWS, :, 0].abs().mean(dim=(1, 2, 3))

    # Two channels reduced separately, as in the numpy original and for its reason: the
    # strided ``[..., :2]`` slice measured 11x slower there. A sum of non-negatives is zero
    # exactly when every term is, so the predicate is unchanged rather than merely close.
    mask = (xd[..., 0].abs().sum(dim=1) + xd[..., 1].abs().sum(dim=1)) == 0  # (N, H, W)
    row_has_hole = mask.any(dim=2)  # (N, H)
    hrows = mask.shape[1]
    # ``argmax`` on a boolean returns 0 both for "first row" and for "no row at all", so the
    # empty case is selected explicitly rather than left to that collision -- same as numpy.
    first = row_has_hole.float().argmax(dim=1)
    top = torch.where(row_has_hole.any(dim=1), first, torch.full_like(first, hrows)).float()
    ratio = ((TOP_ROW_R0 - top) / TOP_ROW_ARM_PX).clamp(-1.0, 1.0)
    top_angle = torch.rad2deg(torch.arcsin(ratio)) - TOP_ROW_PHI_DEG

    profile = xd[..., 0].mean(dim=1).mean(dim=2)  # (N, H)
    peak = profile.max(dim=1).values
    depth = (peak - profile.min(dim=1).values) / peak.clamp_min(1e-9)

    c0, c1 = FLUCT_EXTENT_COLS
    band = xd[:, ::FLUCT_EXTENT_SUB_T, :, c0:c1, :]  # (N, T', H, B, 3)
    fmean = band.mean(dim=1)
    fsq = (band * band).sum(dim=1) / band.shape[1]
    fvar = (fsq - fmean * fmean)[..., :N_MEASURED_CHANNELS]
    fprof = 0.5 * fvar.sum(dim=3).mean(dim=2)  # (N, H)
    fsum = fprof.sum(dim=1)
    # ``1e-300`` in the original is a float64 floor: it sits below fp32's smallest normal
    # (~1e-38), so here it would clamp to zero AND, inside a compiled region, promote the
    # accumulator to fp64 and fail Triton codegen with a loop-carried dtype mismatch. The
    # ``where`` below is what actually handles the degenerate case, so any small positive
    # floor is equivalent.
    ftot = fsum.clamp_min(1e-30)
    rows_ = torch.arange(fprof.shape[1], dtype=torch.float32, device=x.device)
    centre = (fprof * rows_).sum(dim=1) / ftot
    extent = torch.sqrt(((rows_[None] - centre[:, None]) ** 2 * fprof).sum(dim=1) / ftot)
    extent = torch.where(fsum > 0.0, extent, torch.zeros_like(extent))

    aoa = (
        AOA_COEF_MASK_YTOP * top_angle
        + AOA_COEF_WAKE_DEPTH * depth
        + AOA_COEF_FLUCT_EXTENT * extent
        + AOA_INTERCEPT
    ).clamp(*AOA_CLIP_DEG)
    # A neural-anchored AoA override (RPDE_AOA_NET) has no device twin. Refuse loudly rather
    # than silently serve §122's AoA -- the same "cannot silently serve a different
    # conditioning" contract make_predict enforces for cond_device at construction.
    if features._aoa_override is not None:
        raise NotImplementedError(
            "cond_device: a neural AoA override (RPDE_AOA_NET) has no device twin; "
            "use the host numpy path for that configuration"
        )
    # Angle first, as in ``estimate_batch``: the §122 Reynolds slope depends on it. corners3
    # (RPDE_RE_CORNERS) is AoA-free and replaces the slope entirely; any other Re override is
    # unimplemented and refused for the same reason as the AoA one.
    if features._re_override is None:
        re = ((RE_PER_U_FS + RE_PER_U_FS_AOA * aoa) * u_fs + RE_INTERCEPT).clamp_min(1.0)
    elif features._re_override is features.corners3_reynolds:
        re = corners3_reynolds_torch(x)
    else:
        raise NotImplementedError(
            "cond_device: this Re override has no device twin; use the host numpy path"
        )
    return torch.stack([re, aoa], dim=1)


def conditioning_features_batch_torch(
    re_aoa: torch.Tensor, domain: Domain = "real"
) -> torch.Tensor:
    """``(N, COND_DIM)`` conditioning vector from :func:`estimate_batch_torch`'s output.

    Device twin of :func:`rpde_t1.features.conditioning_features_batch` for the shipped
    ``operating_point`` mode. ``tau`` and ``noise`` are deliberately absent: both are refuted
    as conditioning (FINDINGS 78, 88) and neither appears in any shipped recipe, so there is
    no device path to keep in step with them. A mode that needs them must use the numpy path.
    """
    re, aoa = re_aoa[:, 0], re_aoa[:, 1]
    log_re = (torch.log(re) - float(np.log(RE_MIN))) / float(np.log(RE_MAX) - np.log(RE_MIN))
    is_sim = torch.full_like(aoa, 1.0 if domain in SIM_DOMAINS else 0.0)
    return torch.stack([log_re, aoa / AOA_MAX, is_sim], dim=1)
