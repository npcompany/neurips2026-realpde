"""Feature computations shared by training and inference. **numpy only.**

Why this module exists as a separate file, rather than these functions living beside
the code that also uses them:

A submission runs inside ``pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime``, which
ships only torch and numpy. Our *training* code legitimately needs ``h5py`` to read
the 182 HDF5 trajectories; our *inference* code never opens a file, because the
platform hands ``predict()`` a numpy array directly. But Python executes every
import in a chain, so a submission that reached for one small numpy function in
``data/operating_point.py`` used to pull in ``data/survey.py`` -> ``data/h5io.py``
-> ``import h5py`` and die at import time, before any of our code ran.

So the rule for this file: **it may import numpy and nothing else from the
scientific stack, and it may not import from ``rpde_t1.data``.** Everything a
submission needs at inference lives here; the training modules import *from* here
rather than the other way round. ``tests/test_features.py`` enforces that.

Contents:

* the measurement mask (which pixels carry no PIV measurement);
* the freestream-speed estimator that non-dimensionalises every window;
* the operating-point estimators, needed because ``metadata`` is an empty dict on
  scored calls, so ``(Re, AoA)`` must be recovered from the 20 observed frames;
* the conditioning-feature encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Domain = Literal["real", "sim", "sim_resampled", "sim_resampled2", "sim_units"]

#: Domains that are simulated, and therefore share every sim-only behaviour: the real
#: PIV mask is imposed on them, ``p`` exists, the organizers' dataloader augmentations
#: apply, and the conditioning vector's domain bit is 1. ``sim_resampled`` is ``sim``
#: with its time axis rescaled per cell (FINDINGS 3, ``scripts/gen/build_resampled_sim.py``);
#: nothing about a *frame* differs, so it must condition identically or a model
#: pretrained on one could not be fine-tuned against the other. ``sim_resampled2`` is
#: ``sim_resampled`` built from two start offsets (0 and half a step) so decimation keeps
#: every source frame; same clock, twice the series. ``sim_units`` is ``sim`` resampled on
#: the units conversion instead of the measured timescale ratio: one sim frame is 0.02002
#: chord-over-free-stream units and one real frame is 0.02 s, so the step is a straight line
#: in Reynolds number (``build_resampled_sim.py --step-mode units``), with as many start
#: offsets per cell as the rounded step so every source frame is used about once.
SIM_DOMAINS: frozenset[str] = frozenset({"sim", "sim_resampled", "sim_resampled2", "sim_units"})

# --------------------------------------------------------------------------- #
# Measurement mask
# --------------------------------------------------------------------------- #

#: ``u`` and ``v`` are the two measured channels of ``[u, v, p]``; ``p`` is unmeasured in real
#: PIV and zero in the inputs, and no estimator here may read it.
N_MEASURED_CHANNELS = 2

#: Rows used to estimate the freestream speed. The measurement mask never reaches
#: the top of the frame (its bounding box starts at y=22 of 64, i.e. y=11 of 32), so
#: these rows are undisturbed flow at every angle of attack.
FREESTREAM_ROWS = 4


def invalid_mask(fields: np.ndarray) -> np.ndarray:
    """Pixels carrying no measurement: ``u`` and ``v`` identically zero in every frame.

    Returns a ``(H, W)`` boolean array.

    In real PIV this is a substantial region (~5-11% of the field at native
    resolution): the opaque airfoil occludes the camera, the model casts a
    laser-sheet shadow in which no seeding particles are illuminated, and vectors
    failing the correlation-quality check are written as zero. In the CFD it is only
    ~0.2% -- the solid airfoil section itself, a subset of the real mask.

    Computable from an input window alone, so it is available at inference.
    """
    return np.abs(fields[..., :2]).sum(axis=(0, -1)) == 0


def freestream_speed(fields: np.ndarray, rows: int = FREESTREAM_ROWS) -> float:
    """Mean streamwise velocity in the undisturbed rows of ``(T, H, W, C)`` fields.

    Estimated from the sample itself rather than from the Reynolds label, so the
    identical procedure works at training and at inference. Measured deviation
    between a 20-frame window estimate and the whole-trajectory value is 0.6% mean,
    1.7% max.
    """
    return float(np.abs(fields[:, :rows, :, 0]).mean())


def batch_freestream_speed(x: np.ndarray, rows: int = FREESTREAM_ROWS) -> np.ndarray:
    """Per-sample freestream speed for a batch of ``(N, T, H, W, C)`` windows."""
    return np.abs(x[:, :, :rows, :, 0]).mean(axis=(1, 2, 3)).clip(min=1e-6)


# --------------------------------------------------------------------------- #
# Operating point, recovered from the input window
# --------------------------------------------------------------------------- #

#: ``Re = RE_PER_U_FS * u_fs + RE_PER_U_FS_AOA * u_fs * AoA_hat + RE_INTERCEPT``, least
#: squares over the **81 valid** real trajectories (20-frame windows at stride 40), with
#: ``AoA_hat`` the *estimated* angle -- the quantity inference has. The second term is the
#: blockage correction FINDINGS 4 left unimplemented: within one Reynolds row the freestream
#: reading rises with incidence (a more inclined body squeezes the flow past the top rows),
#: so a single line reads Re high at 20 degrees and low at 0. Fitting the slope as a function
#: of the estimated angle takes the mean error from 1.70% to 1.22% and the worst cell from
#: 4.8% to 3.3% held out by Reynolds row, and helps on an unseen angle too (FINDINGS 122).
#: The zero-incidence slope still equals c/nu for a 69.7 mm chord in water, the independent
#: check that the estimator measures what it claims to.
#:
#: Refit history: after finding that ``7575_0`` is a duplicate of ``6300_0`` carrying the
#: wrong label (excluding it took the max error from 17.44% to 4.62%), and again with the
#: blockage term. Refit with ``scripts/eval/fit_operating_point.py``.
RE_PER_U_FS = 71941.3157
RE_PER_U_FS_AOA = -171.9889
RE_INTERCEPT = -269.7475

#: ``AoA = A*mask_top_angle + B*wake_depth + D*wake_fluct_extent + C``, fitted on the ``train``
#: split only. The top row enters through :func:`mask_top_angle`, an arcsin map that makes it
#: homogeneous to an angle (FINDINGS 122d). Selection rule for any change here: it must not
#: lose on the unseen-angle test (leave one angle out) or at either end of the Reynolds ladder,
#: whatever it gains in range.
#: Two candidates failed it on 2026-09-09 (FINDINGS 122): the hole fraction (+0.8 deg on an
#: unseen 0 degrees) and every extent of the *mean* wake deficit (width, depth, displacement
#: thickness), which drift with Reynolds at 15-20 degrees and so trade one end of the ladder for
#: the other. The extent of the *fluctuating* band (:func:`wake_fluct_extent`) passed it: the
#: fluctuation energy, unlike the mean deficit, does not drift with Reynolds at high incidence,
#: and the feature is within 0.02 deg of the two-feature plane on every unseen angle while
#: taking 0.2 deg off both ends of the Reynolds ladder and 0.15 off the hard corner.
AOA_COEF_MASK_YTOP = 0.4505
AOA_COEF_WAKE_DEPTH = 33.0626
AOA_COEF_FLUCT_EXTENT = -0.3050
AOA_INTERCEPT = -16.8176

#: :func:`mask_top_angle` reads the topmost masked row as ``r0 - L sin(AoA + phi)``: the point
#: that sets the hole's upper edge swings on an arm of length ``L`` about the pivot row ``r0``,
#: offset by ``phi`` from the chord. ``L`` is *not* fitted: it is the trailing-edge arm from the
#: known geometry, ``(1 - PIVOT_FRAC) * chord`` in pixels of the 64-column grid
#: (``geometry.CHORD_M`` over the 3.449 mm pixel of ``geometry.GRID_X_M``; a test pins the
#: number to the geometry module). Left free it fits to 10.3 px, so the data agree with the
#: geometry. ``r0`` and ``phi`` are fitted on the train split's per-angle mean top rows
#: (``scripts/eval/fit_operating_point.py``). Over the five rows the release exercises the map is
#: indistinguishable from a line; it exists so that a hole top past row 11 is read on a physical
#: curve rather than a line.
TOP_ROW_ARM_PX = 11.114
TOP_ROW_R0 = 7.6963
TOP_ROW_PHI_DEG = -41.4

#: Range the served angle estimate is clipped to, in degrees. Wider than the five labelled
#: angles (0-20) on purpose: the plane's raw output can be trusted a little past the fitted
#: range (fitted on 0-15 it reads 23 at a true 20), and a test set beyond 20 degrees should be
#: reported as beyond 20 rather than pinned to the edge. The bounds are the range the airfoil
#: geometry table covers (``geometry.SDF_TABLE_RANGE``, asserted equal in the tests), so every
#: consumer of the angle -- the second-moment head's distance field included -- is defined on
#: the whole interval; the clip still stops a bad window from running away.
AOA_CLIP_DEG = (-5.0, 30.0)

#: Column band of :func:`wake_fluct_extent`: the near wake, downstream of every PIV hole (the
#: hole never reaches column 21) and before the far wake starts to meander -- the outlet band's
#: window-to-window noise is four times this one's, since 20 frames are one or two shedding
#: periods (FINDINGS 122a).
FLUCT_EXTENT_COLS = (24, 40)

#: Frame stride of :func:`wake_fluct_extent`. The extent is a ratio of moments of a variance
#: profile and barely depends on how many frames feed it: from every 4th frame it correlates
#: 0.997 with the 20-frame value and every held-out angle column is identical to 0.01 deg
#: (FINDINGS 122c). It runs inside the timed region on Track 1, and the gather of the band is
#: the whole cost, so five frames instead of twenty is a 4x saving for nothing.
FLUCT_EXTENT_SUB_T = 4

#: Measured accuracy of the fits above, so a caller can reason about the noise it is
#: feeding into a conditioning pathway. Re: mean over all valid cells; AoA: held-out cells.
RE_MEAN_REL_ERROR = 0.0122
RE_MAX_REL_ERROR = 0.0331
AOA_MAE_DEG = 1.65


def wake_depth(window: np.ndarray) -> float:
    """Relative depth of the velocity deficit in the row-averaged mean flow.

    The wake deepens and broadens with incidence, which is what makes this track the
    angle of attack (measured correlation +0.933).
    """
    profile = window[..., 0].mean(axis=0).mean(axis=1)
    peak = float(profile.max())
    return float(peak - profile.min()) / max(1e-9, peak)


def wake_fluct_extent(
    window: np.ndarray, cols: tuple[int, int] = FLUCT_EXTENT_COLS, sub_t: int = FLUCT_EXTENT_SUB_T
) -> float:
    """Vertical extent of the fluctuating band in the near wake, in rows.

    The moment width (standard deviation over rows) of the row profile of the window's
    temporal fluctuation energy ``0.5 (var_t u + var_t v)`` over every ``sub_t``-th frame,
    averaged over the columns in ``cols``. A ratio of moments, so it needs no freestream
    normalisation and is the same in physical and normalised units. The separated region
    behind the body fattens as incidence rises, and its fluctuation energy -- unlike the mean
    deficit -- keeps the same footprint across the Reynolds ladder, which is what makes this
    usable as an angle feature.
    """
    x = window[::sub_t, :, cols[0] : cols[1], :2].astype(np.float64)
    profile = 0.5 * (x[..., 0].var(axis=0) + x[..., 1].var(axis=0)).mean(axis=1)
    total = float(profile.sum())
    if total <= 0.0:
        return 0.0
    rows = np.arange(profile.shape[0], dtype=np.float64)
    centre = float((rows * profile).sum()) / total
    return float(np.sqrt(((rows - centre) ** 2 * profile).sum() / total))


def top_row_to_angle(top: np.ndarray | float) -> np.ndarray | float:
    """The arcsin map behind :func:`mask_top_angle`, on a row index or an array of them."""
    ratio = np.clip((TOP_ROW_R0 - np.asarray(top, dtype=np.float64)) / TOP_ROW_ARM_PX, -1.0, 1.0)
    return np.degrees(np.arcsin(ratio)) - TOP_ROW_PHI_DEG


def mask_top_angle(window: np.ndarray) -> float:
    """Topmost masked row read as an angle, degrees: ``arcsin((r0 - top) / L) - phi``.

    Same information as :func:`mask_top_row`, in units homogeneous to the angle it estimates,
    so the plane's coefficient on it is dimensionless and its extrapolation past the observed
    rows follows the geometry's sine rather than a straight line.
    """
    return float(top_row_to_angle(mask_top_row(window)))


def mask_top_row(window: np.ndarray) -> float:
    """Topmost row containing a measurement hole.

    The airfoil and its shadow move up the frame as incidence increases, so this
    reads geometry directly rather than the flow (correlation -0.934 with AoA).
    """
    mask = invalid_mask(window)
    rows = np.where(mask.any(axis=1))[0]
    return float(rows.min()) if len(rows) else float(mask.shape[0])


#: Optional operating-point overrides for the improved estimators (docs: estimator study). Both are
#: ``None`` by default -- so this file stays numpy-only (tests/test_features.py) and the default
#: estimate path is byte-identical -- and are installed out-of-module by
#: ``models.aoa_net.install_from_env`` (which owns the torch/os imports) when the ``RPDE_AOA_NET`` /
#: ``RPDE_RE_CORNERS`` env vars are set, so a run can condition on the better estimators in both
#: training and inference.
_aoa_override = None  # neural-anchored AoA residual over §122, when RPDE_AOA_NET is set
_re_override = None   # corners3 Reynolds (:func:`corners3_reynolds`), when RPDE_RE_CORNERS is set

#: `corners3` Re: three single-row freestream corners fit on the real cells. Halves §122's Reynolds
#: extrapolation error (LORO worst 1.39% vs 2.97%) and its advantage grows out of range; AoA-free
#: (the corner asymmetry encodes the blockage), pure numpy, cheaper than §122's top-4-rows mean.
CORNERS3_COEF = (35542.67, 24855.70, 10891.34, -270.80)


def corners3_reynolds(x: np.ndarray) -> np.ndarray:
    """Re from three freestream corners for a batch ``(N,T,H,W,C)`` -> ``(N,)``.

    TR = top-right, BR = bottom-right (downstream, above/below the wake), TL = top-left (upstream).
    """
    u = np.abs(np.asarray(x)[..., 0].astype(np.float64))  # (N, T, H, W)
    tr = u[:, :, 0:1, 56:64].mean(axis=(1, 2, 3))
    br = u[:, :, -1:, 56:64].mean(axis=(1, 2, 3))
    tl = u[:, :, 0:1, 0:8].mean(axis=(1, 2, 3))
    a, b, cc, d = CORNERS3_COEF
    return np.maximum(1.0, a * tr + b * br + cc * tl + d)


def _aoa_net_override(windows: np.ndarray, aoa: np.ndarray) -> np.ndarray:
    """Apply the installed neural-AoA override, or return §122's AoA unchanged."""
    return aoa if _aoa_override is None else _aoa_override(windows, aoa)


def estimate_reynolds(
    window: np.ndarray, rows: int = FREESTREAM_ROWS, aoa: float | None = None
) -> float:
    """Estimate Re from the freestream speed of a **real** input window.

    Real PIV is in physical m/s and the freestream speed scales linearly with Re
    (measured r = +0.9985), so ``Re = u_fs * c / nu``, with the slope corrected for
    blockage by the estimated angle (``aoa``; estimated here when not given). Accuracy
    1.22% mean relative error. This does **not** work on simulated data, which is already
    non-dimensionalised (u ~ 1.09 regardless of Re); use the label there.
    """
    u_fs = float(np.abs(window[:, :rows, :, 0]).mean())
    if aoa is None:
        aoa = estimate_aoa(window)
    return max(1.0, (RE_PER_U_FS + RE_PER_U_FS_AOA * aoa) * u_fs + RE_INTERCEPT)


def estimate_aoa(window: np.ndarray, clip: tuple[float, float] = AOA_CLIP_DEG) -> float:
    """Estimate the angle of attack in degrees from a real input window.

    1.65 deg MAE on held-out cells, served clip included. Deliberately *not* snapped to
    the five known angles, and clipped to :data:`AOA_CLIP_DEG` rather than to the labelled
    range: the private test is expected at unseen operating points.
    """
    value = (
        AOA_COEF_MASK_YTOP * mask_top_angle(window)
        + AOA_COEF_WAKE_DEPTH * wake_depth(window)
        + AOA_COEF_FLUCT_EXTENT * wake_fluct_extent(window)
        + AOA_INTERCEPT
    )
    return float(np.clip(value, *clip))


def estimate_operating_point(window: np.ndarray) -> tuple[float, float]:
    """``(Re, AoA)`` for one ``(T, H, W, C)`` real input window.

    The angle first: the Reynolds fit uses it for its blockage term.
    """
    win = np.asarray(window)
    aoa = float(_aoa_net_override(win[None], np.array([estimate_aoa(window)]))[0])
    re = (float(_re_override(win[None])[0]) if _re_override is not None
          else estimate_reynolds(window, aoa=aoa))
    return re, aoa


def estimate_batch(x: np.ndarray) -> np.ndarray:
    """``(N, 2)`` array of ``(Re, AoA)`` for a batch of ``(N, T, H, W, C)`` windows.

    Vectorised over the batch because it runs **inside the timed region**: profiled on the
    shipped submission it cost **1.20 ms/window against a 21.5 ms budget -- 5.6% of
    `time_score`** to recover two numbers per window, since the per-sample path pays Python
    dispatch plus a mask, a wake profile and three reductions per call. Third instance of the
    same shape, after the width rule and the lead anchor.

    Every reduction is written over the *same axes in the same order* as the per-sample
    functions above, which is what makes the two agree on real windows to ~1e-6 degrees
    and, through the blockage term, ~1e-4 in Re (``tests/test_features.py`` asserts it on the
    eval set rather than on random noise, since the masks and the wake are structural and
    random arrays would not exercise them).
    """
    u_fs = np.abs(x[:, :, :FREESTREAM_ROWS, :, 0]).mean(axis=(1, 2, 3)).astype(np.float64)

    # ``mask_top_row``: first row holding a measurement hole, else H. ``argmax`` on a boolean
    # returns 0 both for "first row" and for "no row at all", so the empty case is selected
    # explicitly rather than left to that collision.
    #
    # The two channels are reduced *separately* rather than as ``abs(x[..., :2]).sum((1, -1))``.
    # That slice is strided over a 3-channel array and measured 0.907 ms/window against 0.083
    # for this form -- 11x, and the whole cost of this function. The predicate is unchanged, not
    # merely close: both sides sum non-negative terms, and a sum of non-negatives is zero exactly
    # when every term is, so no rounding can turn a hole into a measurement or back.
    mask = (np.abs(x[..., 0]).sum(axis=1) + np.abs(x[..., 1]).sum(axis=1)) == 0  # (N, H, W)
    row_has_hole = mask.any(axis=2)  # (N, H)
    top = np.where(row_has_hole.any(axis=1), row_has_hole.argmax(axis=1), mask.shape[1])

    # ``wake_depth``: mean over time, then over columns, exactly as the scalar path.
    profile = x[..., 0].mean(axis=1).mean(axis=2)  # (N, H)
    peak = profile.max(axis=1)
    depth = (peak - profile.min(axis=1)) / np.maximum(1e-9, peak)

    # ``wake_fluct_extent``: variance over time of the near-wake band, row profile, moment width.
    #
    # The band is gathered with all three channels, then the pressure channel is dropped *after*
    # the reduction: ``x[..., c0:c1, :2]`` copies two floats out of every three and measured 28 ms
    # per 210 windows against 8.6 ms for the contiguous 48-float runs of ``x[..., c0:c1, :]`` --
    # the gather is the whole cost of this feature, the arithmetic is 3 ms. The variance is
    # ``E[x^2] - E[x]^2`` with float64 accumulation over float32 input, which agrees with the
    # per-window ``var`` to 1e-13 rows.
    c0, c1 = FLUCT_EXTENT_COLS
    band = np.ascontiguousarray(x[:, ::FLUCT_EXTENT_SUB_T, :, c0:c1, :])  # (N, T', H, B, 3)
    fmean = band.mean(axis=1, dtype=np.float64)
    fsq = np.einsum("nthbc,nthbc->nhbc", band, band, dtype=np.float64) / band.shape[1]
    fvar = (fsq - fmean * fmean)[..., :N_MEASURED_CHANNELS]  # (N, H, B, 2)
    fprof = 0.5 * fvar.sum(axis=3).mean(axis=2)  # (N, H)
    ftot = np.maximum(fprof.sum(axis=1), 1e-300)
    rows_ = np.arange(fprof.shape[1], dtype=np.float64)
    centre = (fprof * rows_).sum(axis=1) / ftot
    extent = np.sqrt(((rows_[None, :] - centre[:, None]) ** 2 * fprof).sum(axis=1) / ftot)
    extent = np.where(fprof.sum(axis=1) > 0.0, extent, 0.0)

    aoa = np.clip(
        AOA_COEF_MASK_YTOP * top_row_to_angle(top)
        + AOA_COEF_WAKE_DEPTH * depth.astype(np.float64)
        + AOA_COEF_FLUCT_EXTENT * extent
        + AOA_INTERCEPT,
        *AOA_CLIP_DEG,
    )
    aoa = _aoa_net_override(x, aoa)  # neural residual when RPDE_AOA_NET is set, else no-op
    # Angle first, as in ``estimate_operating_point``: the Reynolds slope depends on it.
    if _re_override is not None:  # corners3 Re (RPDE_RE_CORNERS), AoA-independent
        re = _re_override(x)
    else:
        re = np.maximum(1.0, (RE_PER_U_FS + RE_PER_U_FS_AOA * aoa) * u_fs + RE_INTERCEPT)
    return np.stack([re, aoa], axis=1).astype(np.float64)


# --------------------------------------------------------------------------- #
# Decorrelation time
# --------------------------------------------------------------------------- #

#: Spatial stride used when correlating frames. The e-folding time is a *global* statistic
#: of the fluctuation field, so it survives subsampling: at stride 2 it reproduces the full
#: computation at r = 0.997-0.998 on all three validation sets for 3.5x less work, and this
#: runs inside the timed region. Raising it to 1 costs ~0.8 ms/window and changes nothing.
TAU_SUB_S = 2


def decorrelation_time_batch(x: np.ndarray, sub_s: int = TAU_SUB_S) -> np.ndarray:
    """``(N,)`` fluctuation e-folding time in frames, from the input window alone.

    How long the flow remembers itself: the fluctuation field about the window's own time
    mean is correlated at every lag -- averaged over all frame pairs at that lag, not read
    off a single reference frame -- and the result is the first lag where the correlation
    falls below ``1/e``, linearly interpolated.

    **Why this is a conditioning feature and not a diagnostic.** It is the shedding period
    as sampled by the fixed PIV frame rate, and it is what decides whether a 20-frame
    horizon is forecastable: measured on ``st_condconv_real``, it predicts the model's own
    realised phase skill at the last predicted frame at r = 0.69-0.82, and it separates
    the two failure corners (high-Re/high-AoA on ``rel_l2``, high-Re/low-AoA on ``tke``)
    that ``(Re, AoA)`` only reaches through a nonlinear physical map the model would
    otherwise have to learn -- and, being a map it must *extrapolate* in AoA, does not.

    Estimated from the 20 observed frames, so training and inference see the same
    quantity, the same rule as the freestream speed and the geometry planes. Masked PIV
    cells are exactly zero in the fluctuation and so drop out of every inner product
    without needing to be excluded.
    """
    f = np.asarray(x[:, :, ::sub_s, ::sub_s, :2], dtype=np.float32)
    f = f - f.mean(axis=1, keepdims=True)
    n, t = f.shape[0], f.shape[1]
    flat = f.reshape(n, t, -1)
    # One batched GEMM rather than a lag loop: (N,T,T) of every frame pair's inner product.
    gram = flat @ flat.transpose(0, 2, 1)
    norm = np.sqrt(np.einsum("ntt->nt", gram)).clip(min=1e-12)
    corr = gram / (norm[:, :, None] * norm[:, None, :])

    lags = np.arange(t)
    rho = np.stack([corr[:, lags[: t - k], lags[k:]].mean(axis=1) for k in range(t)], axis=1)
    below = rho < 1.0 / np.e
    # ``argmax`` on a boolean returns 0 both for "crosses at lag 0" and for "never crosses";
    # the second case is selected explicitly, as in ``estimate_batch`` above.
    first = np.where(below.any(axis=1), below.argmax(axis=1), t - 1)
    prev = np.take_along_axis(rho, np.maximum(first - 1, 0)[:, None], 1)[:, 0]
    cur = np.take_along_axis(rho, first[:, None], 1)[:, 0]
    frac = (prev - 1.0 / np.e) / np.maximum(prev - cur, 1e-12)
    return np.where(first == 0, 0.0, first - 1 + frac).astype(np.float64)


def decorrelation_time(window: np.ndarray, sub_s: int = TAU_SUB_S) -> float:
    """E-folding time in frames for one ``(T, H, W, C)`` input window."""
    return float(decorrelation_time_batch(window[None], sub_s)[0])


# --------------------------------------------------------------------------- #
# Conditioning features
# --------------------------------------------------------------------------- #

#: Reference values from the simulated grid, which is the widest available. Fixed
#: constants rather than dataset statistics, so a checkpoint stays valid if the
#: split changes.
RE_MIN, RE_MAX = 3750.0, 27975.0
AOA_MAX = 20.0

#: Length of the vector :func:`conditioning_features` returns. Anything sizing a layer
#: from it must read this rather than hardcode 3, or a change here silently produces a
#: checkpoint that no longer loads.
COND_DIM = 3

#: Divisor putting :func:`decorrelation_time_batch` on the same [0, 1] footing as the
#: other conditioning components. It is the input length because that is the longest
#: lag the estimator can resolve, so the feature saturates at 1 by construction.
TAU_MAX_FRAMES = 20.0

#: The conditioning vocabularies. ``operating_point_tau`` appends the window's own
#: decorrelation time to ``(log-Re, AoA, is_sim)``; ``operating_point_noise`` appends the
#: realized synthetic-noise dose instead (``data/dataset.py``'s ``real_noise_scale``, a
#: DPOT-style denoising conditioning: the model is told how corrupted its input is rather
#: than being handed one fixed, unconditioned dose). Each is a separate mode rather than an
#: extra element on the existing one because the vector's length sizes a layer: widening
#: ``operating_point`` in place would leave every checkpoint in ``runs/`` unloadable.
COND_MODES = (
    "none",
    "operating_point",
    "operating_point_tau",
    "operating_point_noise",
    "operating_point_dt",
)

#: Modes that append one extra scalar to ``(log-Re, AoA, is_sim)``, so ``cond_dim`` and
#: ``conditioning_features[_batch]`` widen by exactly one column. ``operating_point_dt``
#: appends the window's frame interval relative to a real frame at its Reynolds number
#: (:func:`dt_cond_value`; 0 for every scored window, since real frames ARE the reference).
_COND_MODES_WITH_EXTRA = ("operating_point_tau", "operating_point_noise", "operating_point_dt")

# --------------------------------------------------------------------------- #
# The real frame interval, in simulated frames
# --------------------------------------------------------------------------- #
#
# One simulated frame is 0.02002 chord-over-freestream units and one real frame is 0.02 s,
# so a real frame at Reynolds number Re spans ``0.02 s * U / (chord * 0.02002)`` simulated
# frames with ``U = Re * nu / chord``: a straight line in Re, 0.77 at Re 3750 and 5.5 at
# 26700 (``scripts/gen/build_resampled_sim.py --step-mode units``, FINDINGS 125). The
# time-step augmentation (``data/dataset.py``, ``dt_ratio_aug``) resamples simulated
# windows at a multiple of this step and tells the model the multiple, so the per-frame
# displacement of the flow stops being locked to Re during training.

#: Real PIV frame interval, seconds (50 Hz).
DT_REAL_S = 0.02
#: Simulated frame interval, chord-over-freestream units (read off the sim ``t`` vectors).
DT_SIM_CONVECTIVE = 0.02002
#: Kinematic viscosity of water, m^2/s (the real release is a water tunnel).
NU_WATER = 1.0e-6
#: Chord of the airfoil in metres, ``geometry.CHORD_M`` (repeated here because this module
#: imports nothing from the package, so a submission can take it alone).
CHORD_M = 0.0697


def real_step_sim_frames(re: float) -> float:
    """How many simulated frames one real frame spans at Reynolds number ``re``."""
    return DT_REAL_S * float(re) * NU_WATER / (CHORD_M**2 * DT_SIM_CONVECTIVE)


def dt_cond_value(ratio: float) -> float:
    """The ``operating_point_dt`` slot for a window whose frame interval is ``ratio`` real frames.

    ``log2(ratio) / 2``: 0 at the real clock, +0.5 at twice the real step, -0.5 at half. Log
    because a doubling and a halving should sit at the same distance from the reference;
    the divisor keeps the augmentation range (a factor of 4 either way) inside about
    ``[-1, 1]``, the scale of the other components.
    """
    return float(np.log2(ratio) / 2.0)


def cond_dim(cond_mode: str) -> int:
    """Width of the conditioning vector a mode emits.

    ``none`` keeps ``COND_DIM``: the pathway is replaced by a learned constant, but the
    ablation still sizes ``cond_content`` planes from this, so the two arms of an on/off
    A/B stay one variable apart.
    """
    if cond_mode not in COND_MODES:
        raise ValueError(f"cond_mode must be one of {COND_MODES}, got {cond_mode!r}")
    return COND_DIM + 1 if cond_mode in _COND_MODES_WITH_EXTRA else COND_DIM


def conditioning_features(
    re: float,
    aoa: float,
    domain: Domain,
    tau: float | None = None,
    noise: float | None = None,
    dt: float | None = None,
) -> np.ndarray:
    """``(log-Re in [0,1], AoA in [0,1], is_sim)`` float32, plus ``tau``/``noise``/``dt`` if given.

    Re is taken in log space: the ladder spans a factor of 7.5 and the flow physics
    tracks its order of magnitude rather than its absolute value.

    ``tau`` is the window's decorrelation time in frames (:func:`decorrelation_time`),
    scaled by :data:`TAU_MAX_FRAMES`. Unlike the first two components it is a property of
    the *window* rather than of the cell, and it is the one component that is not a
    relabelling of the operating point: real's timescale scales as 1/U and simulated data's
    does not, so at a matched ``(Re, AoA)`` the two domains disagree by up to 1.7x.

    ``noise`` is the fraction of the run's configured noise ceiling actually injected into
    this window (0 = clean), already normalized to ``[0, 1]`` by the caller -- unlike
    ``tau`` it is not a property of the physical flow, so there is nothing here to further
    scale it by. ``tau`` and ``noise`` are mutually exclusive in every mode ``cond_dim``
    declares, but both may be passed at once without erroring.
    """
    log_re = (np.log(re) - np.log(RE_MIN)) / (np.log(RE_MAX) - np.log(RE_MIN))
    is_sim = 1.0 if domain in SIM_DOMAINS else 0.0
    base = [log_re, aoa / AOA_MAX, is_sim]
    if tau is not None:
        base.append(min(tau / TAU_MAX_FRAMES, 1.0))
    if noise is not None:
        base.append(noise)
    if dt is not None:
        # Already the slot's value (:func:`dt_cond_value`), not a ratio.
        base.append(dt)
    return np.array(base, dtype=np.float32)


def conditioning_features_batch(
    re: np.ndarray,
    aoa: np.ndarray,
    domain: Domain,
    tau: np.ndarray | None = None,
    noise: np.ndarray | None = None,
    dt: np.ndarray | None = None,
) -> np.ndarray:
    """``(N, cond_dim)`` for a whole batch, without a Python loop.

    Same numbers as :func:`conditioning_features`, and bit-identical to stacking it
    per sample (asserted in ``tests/test_features.py``): the arithmetic is float64 either
    way and only the final store rounds to float32.

    It exists because the loop is *inside the timed region*. Profiled on the shipped
    submission at batch 32, ``infer_conditioning`` cost **1.27 ms/window against a 46.3 ms
    budget** -- 2.7% of `time_score` spent building 96 floats, because the per-sample call
    pays Python dispatch and a three-element ``np.array`` allocation per window. The same
    L11 shape as the width rule and the lead anchor before it.
    """
    extra = int(tau is not None) + int(noise is not None) + int(dt is not None)
    out = np.empty((np.shape(re)[0], COND_DIM + extra), dtype=np.float32)
    out[:, 0] = (np.log(re) - np.log(RE_MIN)) / (np.log(RE_MAX) - np.log(RE_MIN))
    out[:, 1] = np.asarray(aoa) / AOA_MAX
    out[:, 2] = 1.0 if domain in SIM_DOMAINS else 0.0
    col = COND_DIM
    if tau is not None:
        out[:, col] = np.minimum(np.asarray(tau) / TAU_MAX_FRAMES, 1.0)
        col += 1
    if noise is not None:
        out[:, col] = np.asarray(noise)
        col += 1
    if dt is not None:
        out[:, col] = np.asarray(dt)
    return out


# --------------------------------------------------------------------------- #
# Normalization schemes
# --------------------------------------------------------------------------- #
#
# Moved here from ``data/normalizers.py`` for the same reason as everything else in
# this file: a gaussian-normalized submission needs ``gaussian_stats`` at inference,
# and importing anything under ``rpde_t1.data`` runs that package's ``__init__``,
# which imports h5py.

#: Published by the organizers, extracted from their cached ``mean_std_real.pt``
#: (see ``submission_example_fno.py``). Used verbatim rather than refitted, so a
#: reproduction run matches the released checkpoints' normalization exactly.
#:
#: The pressure std is forced to 1 because ``p`` is unmeasured in real data and its
#: fitted std is 0 -- ``GaussianNormalizer.__init__`` does the same substitution.
REAL_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
REAL_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
REAL_MEAN_TARGET = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
REAL_STD_TARGET = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

#: Fitted on our simulated cache the way ``GaussianNormalizer`` fits, since the
#: organizers publish statistics only for the real split.
SIM_MEAN = np.array([0.858670962, -0.004991819, 1.016372625], dtype=np.float32)
SIM_STD = np.array([0.443817632, 0.214760456, 12.578333177], dtype=np.float32)


@dataclass(frozen=True)
class GaussianStats:
    """Per-channel z-score statistics for one domain."""

    mean_in: np.ndarray
    std_in: np.ndarray
    mean_target: np.ndarray
    std_target: np.ndarray

    def encode(self, inputs: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (inputs - self.mean_in) / self.std_in, (target - self.mean_target) / self.std_target

    def decode_target(self, target: np.ndarray) -> np.ndarray:
        return target * self.std_target + self.mean_target


def gaussian_stats(kind: str) -> GaussianStats:
    """Statistics for ``'real'`` (published) or a simulated domain (fitted here).

    ``sim_resampled`` reuses ``sim``'s statistics deliberately: resampling changes only
    which instants are kept, not the distribution of values at an instant, so refitting
    would reproduce the same numbers with less data behind them.
    """
    if kind == "real":
        return GaussianStats(REAL_MEAN_IN, REAL_STD_IN, REAL_MEAN_TARGET, REAL_STD_TARGET)
    if kind in SIM_DOMAINS:
        return GaussianStats(SIM_MEAN, SIM_STD, SIM_MEAN, SIM_STD)
    raise ValueError(f"kind must be 'real' or one of {sorted(SIM_DOMAINS)}, got {kind!r}")


def normalized_mse(pred: np.ndarray, target: np.ndarray, kind: str = "real") -> float:
    """The organizers' validation metric: MSE in z-scored space on ``u, v``.

    Confirmed against ``realpdebench/train.py``, which computes
    ``mse_loss(pred[..., :c], target[..., :c])`` *after* ``data_normalizer.preprocess``
    and logs it as ``normalized_mse``. Their released ``sim_real_cno`` checkpoint
    records a best value of 0.05104 on their own validation split.

    Inputs are in **physical** units; the z-scoring happens here, so this can be
    applied to any model's output regardless of how that model was trained.
    """
    stats = gaussian_stats(kind)
    p = (np.asarray(pred, dtype=np.float64)[..., :2] - stats.mean_target[:2]) / stats.std_target[:2]
    t = (np.asarray(target, dtype=np.float64)[..., :2] - stats.mean_target[:2]) / stats.std_target[
        :2
    ]
    return float(((p - t) ** 2).mean())


# --------------------------------------------------------------------------- #
# Derived flow planes as extra model input channels
# --------------------------------------------------------------------------- #
#
# The geometry planes (``geometry.GEOMETRY_CHANNELS``) tell the model where the body is.
# These tell it what the *observed window* is doing, in quantities the first layer cannot
# reach for itself -- and the distinction between the two kinds below is the whole
# experiment, because only one of them adds anything an untouched stem could not compute:
#
# * ``vort`` is **linear** in the input. A 3x3 convolution represents the discrete curl
#   exactly, so the stem can already build this plane and the channel adds no information.
#   What it can still change is *conditioning*: vorticity's rms is 0.110 against the
#   velocity's 0.857, so as an internal feature it is a small difference of large numbers,
#   and handing it over pre-scaled is a different optimisation problem even though it is
#   the same function class. Read a null here as "the stem was already doing it", which is
#   what FINDINGS 43 found for the geometry planes.
# * ``enstrophy`` and ``tke`` are **quadratic**, and no linear stem of any width produces
#   them. They are genuinely new to the first layer. ``tke`` is also the exact quantity
#   the post-hoc fluctuation blend exploits -- the input window's own fluctuation energy is
#   unbiased for the target's where the model's prediction is not (FINDINGS 5c / N3) -- so
#   this channel is that correction offered to the trunk as a feature instead of applied
#   afterwards.
#
# Every one is computable from the 20 observed frames alone, which is the binding
# constraint on any input feature here: ``metadata`` is an empty dict on scored calls.

#: Per-plane divisors, so each channel enters the stem at unit rms -- the same footing as
#: the freestream-normalised velocities, whose rms is 0.857. **Measured, not chosen**:
#: ``scripts/eval/vorticity_diagnostic.py`` over real ``train``, and stable across the three
#: validation splits (vorticity 0.109-0.112, tke 0.048-0.053, enstrophy 0.033-0.039), which
#: is what licenses one constant for a Reynolds ladder spanning 57x in fluctuation energy.
FLOW_SCALES: dict[str, float] = {"vort": 0.1102, "enstrophy": 0.0366, "tke": 0.0529}

#: Channel groups :func:`flow_channels` understands, with their widths.
FLOW_CHANNELS: dict[str, int] = {
    #: Vorticity ``dv/dx - du/dy`` of each observed frame, in grid units.
    "vort": 1,
    #: ``sqrt(var_t(vorticity))`` over the window, one plane broadcast over time.
    "enstrophy": 1,
    #: ``sqrt(0.5 (var_t(u) + var_t(v)))`` over the window -- the input window's own
    #: fluctuation velocity, broadcast over time.
    "tke": 1,
}


def n_flow_channels(names: tuple[str, ...]) -> int:
    """Total width of the requested flow-channel groups."""
    unknown = set(names) - set(FLOW_CHANNELS)
    if unknown:
        raise ValueError(f"unknown flow channel(s) {sorted(unknown)}")
    return sum(FLOW_CHANNELS[n] for n in names)


def parse_flow_channels(spec: str | None) -> tuple[str, ...]:
    """``"vort,tke"`` -> ``("vort", "tke")``; empty, ``None`` or trailing commas -> ``()``.

    One parser shared by the dataset, the model builder and the submission, for the same
    reason :func:`~rpde_t1.geometry.parse_geom_channels` is: three copies of this split is
    how the trained input and the served input end up different widths.
    """
    return tuple(n for n in str(spec or "").split(",") if n)


def window_vorticity(window: np.ndarray) -> np.ndarray:
    """``(T, H, W)`` vorticity of a ``(T, H, W, C)`` window, in grid units.

    Central differences with **edge padding**, so the plane is the same shape as the frame
    it describes. That differs from :func:`rpde_t1.train.losses.vorticity`, which drops the
    boundary ring rather than invent a value for it -- and the difference is deliberate: a
    loss may discard cells, an input plane may not, because the stem's shape is fixed. The
    edge convention halves the difference at the boundary; it is applied identically at
    training and at inference, which is the property that matters for an input feature.

    The grid is square (``geometry.GRID_X_M`` / ``GRID_Y_M`` agree to 5e-5), so one spacing
    divides both derivatives and is absorbed into :data:`FLOW_SCALES`.
    """
    u = np.pad(window[..., 0], ((0, 0), (1, 1), (0, 0)), mode="edge")
    v = np.pad(window[..., 1], ((0, 0), (0, 0), (1, 1)), mode="edge")
    return (0.5 * (v[..., 2:] - v[..., :-2]) - 0.5 * (u[:, 2:] - u[:, :-2])).astype(np.float32)


def flow_channels(window: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """``(T, H, W, n)`` derived planes for a ``(T, H, W, C)`` **normalised** window.

    Normalised, because :data:`FLOW_SCALES` was measured in freestream-normalised space and
    a plane computed before the division would carry the 6.7x Reynolds amplitude spread the
    normaliser exists to remove.

    The static groups are broadcast over time rather than returned as ``(H, W)``: the input
    tensor has a time axis and every channel must fill it, and broadcasting here keeps the
    concatenation in one place instead of at each of the two call sites.
    """
    if not names:
        return np.zeros((*window.shape[:3], 0), dtype=np.float32)
    n_flow_channels(names)  # reject unknown names before any work

    planes = []
    w = window_vorticity(window) if {"vort", "enstrophy"} & set(names) else None
    for name in names:
        if name == "vort":
            planes.append(w / FLOW_SCALES["vort"])
        elif name == "enstrophy":
            fluct = np.sqrt(w.var(axis=0)) / FLOW_SCALES["enstrophy"]
            planes.append(np.broadcast_to(fluct, w.shape))
        else:
            ke = 0.5 * (window[..., 0].var(axis=0) + window[..., 1].var(axis=0))
            planes.append(np.broadcast_to(np.sqrt(ke) / FLOW_SCALES["tke"], window.shape[:3]))
    return np.stack(planes, axis=-1).astype(np.float32)
