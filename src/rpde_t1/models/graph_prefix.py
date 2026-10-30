"""The device prefix of the graphed Track-1 call -- the operating-point estimate and its
conditioning vector, the freestream scale, the normalised window and the second-moment head's
constant planes -- in about half the kernels.

Same values as ``op_estimate.estimate_batch_torch``, ``conditioning_features_batch_torch`` and
the graph's plane builder, element for element: every reduction is the same torch op over the
same axes in the same order, and the scalar tails between the reductions (about forty tiny
element-wise ops on vectors of one value per window) are three fused kernels that perform the
same fp32 operations in the same order. Two details of torch's CUDA element-wise kernels are
mirrored, not approximated: a division by a Python scalar multiplies by the fp32 reciprocal, and
``a - b`` is ``a + (-1) * b``. The freestream reduction, which the estimate and the scale both
need, runs once. The definitions stay in ``op_estimate``; this is a serving twin, checked
against them by the graph's own first-call comparison and by ``tests/test_graph_prefix.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .. import T_IN
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
    TOP_ROW_ARM_PX,
    TOP_ROW_PHI_DEG,
    TOP_ROW_R0,
)

#: torch.rad2deg multiplies by this double, cast to fp32 inside the kernel
_DEG_PER_RAD = 57.295779513082320876798154814105


def _inv32(c: float) -> float:
    """``1 / c`` as torch's scalar-division kernel computes it: in fp32."""
    return float(np.float32(1.0) / np.float32(c))


_SRC = {
    # fsq / T' - fmean * fmean  (the band's per-element temporal variance)
    "fvar": (
        "T fsq, T fmean, T inv_t",
        1,
        "return T(__fsub_rn(__fmul_rn(float(fsq), float(inv_t)),"
        " __fmul_rn(float(fmean), float(fmean))));",
    ),
    # the first scalar tail: top-row angle, wake depth, the fluctuation floor and centre
    "tail1": (
        "T top, T peak, T pmin, T fsum, T s1, T r0, T inv_arm, T deg, T phi,"
        " T& top_angle, T& depth, T& centre, T& ftot",
        4,
        """
    float ratio = fminf(fmaxf(__fmul_rn(__fsub_rn(float(r0), float(top)), float(inv_arm)),
                              -1.0f), 1.0f);
    top_angle = T(__fsub_rn(__fmul_rn(asinf(ratio), float(deg)), float(phi)));
    depth = T(__fdiv_rn(__fsub_rn(float(peak), float(pmin)), fmaxf(float(peak), 1e-9f)));
    float ft = fmaxf(float(fsum), 1e-30f);
    ftot = T(ft);
    centre = T(__fdiv_rn(float(s1), ft));
""",
    ),
    # (rows - centre) ** 2 * fprof
    "spread": (
        "T rows, T centre, T fprof",
        1,
        "float d = __fsub_rn(float(rows), float(centre));"
        " return T(__fmul_rn(__fmul_rn(d, d), float(fprof)));",
    ),
    # the second scalar tail: extent, the angle, the Reynolds number, the conditioning columns
    "tail2": (
        "T s2, T ftot, T fsum, T top_angle, T depth, T u_fs, T ca, T cb, T cc, T cd,"
        " T alo, T ahi, T ru, T ra, T ri, T lmin, T inv_lrange, T inv_amax,"
        " T& re, T& aoa, T& log_re, T& aoa_n",
        4,
        """
    float ext = sqrtf(__fdiv_rn(float(s2), float(ftot)));
    ext = float(fsum) > 0.0f ? ext : 0.0f;
    float a = __fadd_rn(__fmul_rn(float(ca), float(top_angle)), __fmul_rn(float(cb), float(depth)));
    a = __fadd_rn(__fadd_rn(a, __fmul_rn(float(cc), ext)), float(cd));
    a = fminf(fmaxf(a, float(alo)), float(ahi));
    float r = __fadd_rn(float(ru), __fmul_rn(float(ra), a));
    r = fmaxf(__fadd_rn(__fmul_rn(r, float(u_fs)), float(ri)), 1.0f);
    re = T(r); aoa = T(a);
    log_re = T(__fmul_rn(__fsub_rn(logf(r), float(lmin)), float(inv_lrange)));
    aoa_n = T(__fmul_rn(a, float(inv_amax)));
""",
    ),
}
_JIT: dict[Any, Any] = {}


def _kernel(name: str, **scalars: float):
    """Kernel ``name`` with ``scalars`` as its extra (double) arguments, compiled once."""
    key = (name, tuple(sorted(scalars.items())))
    fn = _JIT.get(key)
    if fn is None:
        from torch.cuda.jiterator import (  # noqa: PLC0415
            _create_jit_fn,
            _create_multi_output_jit_fn,
        )

        sig, n_out, body = _SRC[name]
        if n_out == 1:
            src = f"template <typename T> T {name}({sig}) {{\n{body}\n}}"
            fn = _create_jit_fn(src, **scalars)
        else:
            src = f"template <typename T> void {name}({sig}) {{\n{body}\n}}"
            fn = _create_multi_output_jit_fn(src, n_out, **scalars)
        _JIT[key] = fn
    return fn


_K1 = {
    "r0": float(TOP_ROW_R0),
    "inv_arm": _inv32(TOP_ROW_ARM_PX),
    "deg": _DEG_PER_RAD,
    "phi": float(TOP_ROW_PHI_DEG),
}
_K2 = {
    "ca": float(AOA_COEF_MASK_YTOP),
    "cb": float(AOA_COEF_WAKE_DEPTH),
    "cc": float(AOA_COEF_FLUCT_EXTENT),
    "cd": float(AOA_INTERCEPT),
    "alo": float(AOA_CLIP_DEG[0]),
    "ahi": float(AOA_CLIP_DEG[1]),
    "ru": float(RE_PER_U_FS),
    "ra": float(RE_PER_U_FS_AOA),
    "ri": float(RE_INTERCEPT),
    "lmin": float(np.log(RE_MIN)),
    "inv_lrange": _inv32(float(np.log(RE_MAX) - np.log(RE_MIN))),
    "inv_amax": _inv32(AOA_MAX),
}


@torch.no_grad()
def device_prefix(x: torch.Tensor, angles: torch.Tensor, stack: torch.Tensor):
    """``(cond, ub, xb, sp)`` for a resident ``(N, T, H, W, 3)`` window in physical units:
    the conditioning vector ``(N, 3)``, the freestream scale ``(N,)``, the window divided by
    it, and the ``(N, 4, H, W)`` constant planes of the second-moment head (``angles`` and
    ``stack`` are the tabulated signed-distance fields, float64 angles and float32 fields)."""
    n, _, hh, ww, _ = x.shape
    dev = x.device
    r = x[:, -T_IN:]  # the estimate and the scale read the last T_IN frames
    # freestream: one reduction for the estimate and for the scale (the same op, so the same
    # value)
    u_fs = r[:, :, :FREESTREAM_ROWS, :, 0].abs().mean(dim=(1, 2, 3))
    ub = u_fs.clamp_min(1e-06)
    xb = x / ub.view(-1, 1, 1, 1, 1)
    # the hole mask: a pixel whose measured components are zero on every frame. One reduction
    # over frames and components -- a sum of non-negatives is zero exactly when every term is.
    mask = r[..., :N_MEASURED_CHANNELS].abs().sum(dim=(1, -1)) == 0  # (N, H, W)
    row_has_hole = mask.any(dim=2)
    first = row_has_hole.float().argmax(dim=1)
    top = torch.where(row_has_hole.any(dim=1), first, torch.full_like(first, hh)).float()
    profile = r[..., 0].mean(dim=1).mean(dim=2)  # (N, H)
    peak = profile.max(dim=1).values
    pmin = profile.min(dim=1).values
    c0, c1 = FLUCT_EXTENT_COLS
    band = r[:, ::FLUCT_EXTENT_SUB_T, :, c0:c1, :]  # (N, T', H, B, 3)
    fmean = band.mean(dim=1)
    fsq = (band * band).sum(dim=1)
    fvar = _kernel("fvar", inv_t=_inv32(band.shape[1]))(fsq, fmean)[..., :N_MEASURED_CHANNELS]
    fprof = 0.5 * fvar.sum(dim=3).mean(dim=2)  # (N, H)
    fsum = fprof.sum(dim=1)
    rows = torch.arange(hh, dtype=torch.float32, device=dev)
    s1 = (fprof * rows).sum(dim=1)
    top_angle, depth, centre, ftot = _kernel("tail1", **_K1)(top, peak, pmin, fsum, s1)
    s2 = _kernel("spread")(rows[None], centre[:, None], fprof).sum(dim=1)
    re, aoa, log_re, aoa_n = _kernel("tail2", **_K2)(s2, ftot, fsum, top_angle, depth, u_fs)
    cond = torch.stack([log_re, aoa_n, torch.zeros_like(aoa_n)], dim=1)
    # the head's constant planes: the nearest tabulated signed-distance field (float64
    # comparison, as the host lookup) and the two conditioning columns broadcast
    v = cond[:, 1].double() * 20.0
    idx = (angles[None, :] - v[:, None]).abs().argmin(dim=1)
    sp = torch.zeros((n, 4, hh, ww), dtype=torch.float32, device=dev)
    sp[:, 0] = stack[idx]
    sp[:, 1:3] = cond[:, :2, None, None]
    return cond, ub, xb, sp
