"""Torch datasets over cached windows, in non-dimensional space.

Design decisions worth stating, because each was measured rather than assumed:

**Non-dimensionalisation.** Every sample is divided by its own freestream speed,
estimated from the *input window* (not the Re label, and not the whole trajectory)
so training and inference use the identical procedure. Measured deviation between
a 20-frame window estimate and the trajectory value is 0.6% mean / 1.7% max. This
collapses real data's 6.6x amplitude range onto sim's scale and removes the need to
extrapolate an amplitude at an unseen Reynolds number.

**Sim masking.** Simulated samples are zeroed on the real measurement mask, so the
model is never asked to fill a region that every real target leaves empty. Cells
with no real counterpart borrow the mask from the nearest Re at the same AoA.

**Conditioning.** ``(log Re, AoA, domain)`` are provided as a small feature vector
for adaLN-style conditioning. The simulated grid covers Re=15225 and 27975, which
the real data withholds, so the conditioning pathway can be trained at exactly the
operating points the private test is expected to use.

**No spatial augmentation.** The flow is not symmetric -- the sign of the angle of
attack matters and time has a direction -- so flips and reversals would be
physically wrong.

**Two simulated grids.** ``sim`` is the release; ``sim_resampled`` is the same
trajectories with the time axis rescaled per cell, because the two domains keep
different clocks -- a real frame is 0.02 s while a simulated frame is a fixed fraction
of a convective time, so 20 frames is 1.1-9.0 integral timescales in real against
1.4-3.3 in sim (FINDINGS 3). Both are in ``SIM_DOMAINS`` and behave identically in
every other respect, conditioning vector included, so a model pretrained on one can be
fine-tuned against real exactly as the other would be.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import N_MEASURED, T_IN, T_OUT, C
from ..features import (
    AOA_CLIP_DEG,
    COND_MODES,
    SIM_DOMAINS,
    Domain,
    conditioning_features,
    decorrelation_time,
    dt_cond_value,
    estimate_aoa,
    estimate_operating_point,
    flow_channels,
    n_flow_channels,
    real_step_sim_frames,
)
from ..geometry import geometry_channels, n_geometry_channels
from ..models.factory import (
    STREAM_STATE_FAST_WINDOWS,
    STREAM_STATE_FRAMES,
    covariance_stats_from_sums,
    phase_planes_np,
)
from .cache import (
    CachedTrajectory,
    freestream_speed,
    load_index,
    load_masks,
    load_native_paths,
    real_mask_for_sim,
)
from .climatology import build_climatology
from .masking import sample_input_mask
from .normalizers import gaussian_stats
from .piv_noise import add_piv_noise, piv_sigma_rel
from .splits import (
    DEFECTIVE_PRESSURE_CELLS,
    SIM_ALL_BUT_REAL_VAL,
    SIM_HIFI,
    SIM_ONLY_ROWS,
    SIM_SPLITS,
    VAL_EXTRAP_RE,
    VAL_TIME_TAIL,
    all_cells,
    aoa_split_paths,
    assign_sim_splits,
    assign_splits,
    parse_params,
    sim_all_but_real_val_paths,
    sim_all_but_sim_only_rows_paths,
    sim_hifi_paths,
    sim_val_time_cells,
    val_time_cells,
)
from .survey import base_kind

#: Ceiling on the effective step of a ``dt_ratio_aug`` draw, in source frames per frame. Real's
#: fastest clock is 5.8 frames (Re 27975), so 8 lets every cell be shown at every speed real
#: has, with margin, and stops the top rows (real step 5.5) from being drawn at 4x real speed,
#: where a 20-frame horizon is a different and unscored forecasting problem.
DT_RATIO_MAX_STEP = 8.0

#: ``stream_state`` modes, from ``models.factory``'s frame table so the two cannot disagree.
STREAM_STATE_MODES = tuple(STREAM_STATE_FRAMES)

#: Re-exported; the one definition is in ``models.factory`` (the serving port needs it too).
FAST_WINDOWS = STREAM_STATE_FAST_WINDOWS


def fractional_block(arr: np.ndarray, start: int, step: float, span: int) -> np.ndarray:
    """``span`` frames of ``arr``, every ``step`` source frames from ``start``, linearly blended.

    Frame ``i`` sits at source position ``start + i * step``; a non-integer position is the
    linear blend of the two frames around it, exactly ``build_resampled_sim.resample`` at
    offset 0, so a ratio-1 draw reproduces the units cache. An integer ``step`` reads the
    frames themselves (the blend weight is exactly 0). The last position must lie inside the
    array; the window list and ``_dt_ratio`` guarantee it, and this asserts it.
    """
    pos = start + step * np.arange(span, dtype=np.float64)
    i0 = np.floor(pos).astype(np.int64)
    frac = (pos - i0).astype(np.float32)
    if i0[-1] + (frac[-1] > 0) >= arr.shape[0]:
        raise IndexError(f"window reaches frame {pos[-1]:.2f} of {arr.shape[0]}")
    i1 = np.minimum(i0 + 1, arr.shape[0] - 1)
    a = np.asarray(arr[i0], dtype=np.float32)
    if not frac.any():
        return a
    b = np.asarray(arr[i1], dtype=np.float32)
    wgt = frac.reshape(-1, 1, 1, 1)
    return a * (1.0 - wgt) + b * wgt


@dataclass(frozen=True)
class WindowIndex:
    """Where one training window comes from."""

    traj: int
    start: int


def covariance_stats(hist: np.ndarray) -> dict[str, np.ndarray]:
    """The ``meanvarcov`` statistics of a revealed history ``(n, H, W, 2)``, physical units.

    One home for the arithmetic, so the dataset and the host port cannot drift: the port calls
    :func:`covariance_stats_from_sums` on running sums, and this is that function applied to the
    sums of a whole history. ``n >= 1``; with one frame there is no lag pair and ``rho1`` is 0.
    """
    h = np.asarray(hist, dtype=np.float64)
    n = h.shape[0]
    omega = np.gradient(h[..., 1], axis=2) - np.gradient(h[..., 0], axis=1)
    sums = {
        "n": n,
        "s": h.sum(axis=0),
        "sq": (h * h).sum(axis=0),
        "uv": (h[..., 0] * h[..., 1]).sum(axis=0),
        "w2": (omega * omega).sum(axis=0),
        "lag": (h[1:] * h[:-1]).sum(axis=0) if n > 1 else np.zeros_like(h[0]),
    }
    return covariance_stats_from_sums(sums)


#: The real Reynolds ladder's ends, which every operating-point-dependent dose is anchored to.
REAL_RE_LO, REAL_RE_HI = VAL_EXTRAP_RE


def re_dose(re: float, lo: float, hi: float) -> float:
    """A dose linear in ``log Re`` from ``lo`` at Re 3750 to ``hi`` at Re 26700, clamped outside.

    Log because the ladder is geometric in what the model sees (displacement per frame is
    linear in Re, its effect on the window is not), and clamped because a simulated cell above
    26700 should not extrapolate a dose nobody chose.
    """
    if lo == hi:
        return float(lo)
    t = (np.log(re) - np.log(REAL_RE_LO)) / (np.log(REAL_RE_HI) - np.log(REAL_RE_LO))
    return float(lo + (hi - lo) * min(max(t, 0.0), 1.0))


def boost_target_fluctuation(block: np.ndarray, t_in: int, r: float) -> np.ndarray:
    """Scale the fluctuation of frames ``t_in:`` about THEIR time mean by ``r``; input untouched."""
    tail = block[t_in:]
    m = tail.mean(axis=0, keepdims=True)
    return np.concatenate([block[:t_in], m + (tail - m) * np.float32(r)], axis=0)


def mean_aug_deltas(
    cells: list[tuple[int, int]],
    clim: dict[tuple[int, int], np.ndarray],
    invalid: list[np.ndarray],
) -> list[list[np.ndarray]]:
    """Per cell, the normalised mean-flow differences to its adjacent-Re cells at the same angle.

    Adjacent among ``cells`` -- the nearest rung above and the nearest below, whatever lies
    between them in the release -- so a cell at the end of the ladder has one donor and a cell
    whose angle has no other rung has none. The difference is zeroed wherever either cell has
    no measurement, and one pixel around it: the climatology is 0 in a hole, a hole's
    "difference" is not a flow, and the vectors bordering a PIV hole are the unreliable ones
    (measured: over the 69 training cells the median |delta| is 0.013 freestream, p99 0.15,
    and every value above 0.4 -- up to 0.95 -- sits two pixels from the airfoil hole). The
    magnitude is also clipped at ``MEAN_AUG_CLIP`` so no single pixel is shifted by more than
    the p99.9 of the honest differences.
    """
    by_aoa: dict[int, list[tuple[int, int]]] = {}
    for i, (re, aoa) in enumerate(cells):
        by_aoa.setdefault(aoa, []).append((re, i))
    out: list[list[np.ndarray]] = []
    for i, (re, aoa) in enumerate(cells):
        rungs = sorted(by_aoa[aoa])
        k = [r for r, _ in rungs].index(re)
        donors = [rungs[j][1] for j in (k - 1, k + 1) if 0 <= j < len(rungs)]
        deltas = []
        for j in donors:
            delta = (clim[cells[j]] - clim[cells[i]]).astype(np.float32)
            mag = np.sqrt((delta**2).sum(-1, keepdims=True))
            delta = delta * np.minimum(1.0, MEAN_AUG_CLIP / np.maximum(mag, 1e-12))
            delta[_dilate(invalid[i] | invalid[j])] = 0.0
            deltas.append(delta.astype(np.float32))
        out.append(deltas)
    return out


#: Largest mean-flow shift ``mean_aug`` may apply at one pixel, in freestream units (the p99.9
#: of the adjacent-rung climatology differences over the training cells, 2026-09-23).
MEAN_AUG_CLIP = 0.3


def _dilate(hole: np.ndarray) -> np.ndarray:
    """``hole`` grown by one pixel in every direction (8-neighbourhood), edges clamped."""
    p = np.pad(hole, 1, mode="edge")
    out = np.zeros_like(hole)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            out |= p[di : di + hole.shape[0], dj : dj + hole.shape[1]]
    return out


class WindowDataset(Dataset):
    """Windows of ``T_IN`` observed frames -> ``T_OUT`` future frames.

    Each item is a dict of tensors:

    ``input``   ``(T_IN, H, W, 3)``  non-dimensional, channel order ``[u, v, p]``
    ``target``  ``(T_OUT, H, W, 3)`` non-dimensional
    ``cond``    ``(3,)``             ``(log-Re, AoA, is_sim)``
    ``valid``   ``(H, W)``           1 where the measurement exists, 0 in the mask
    ``u_fs``    scalar               the divisor, for returning to physical units
    ``has_p``   scalar               1 for simulated samples, 0 for real
    """

    def __init__(
        self,
        kind: Domain,
        trajectories: list[tuple[int, int]] | None = None,
        stride: int = 1,
        t_in: int = T_IN,
        t_out: int = T_OUT,
        frame_range: tuple[float, float] | dict[tuple[int, int], tuple[float, float]] = (0.0, 1.0),
        mask_sim: bool = True,
        normalize: bool = True,
        normalizer: str = "freestream",
        mask_prob: float = 0.0,
        mask_swap_prob: float = 0.0,
        noise_scale: float = 0.0,
        real_noise_scale: float = 0.0,
        fluct_aug: float = 0.0,
        aoa_jitter_deg: float = 0.0,
        fluct_aug_hi: float | None = None,
        tgt_amp_aug: float = 0.0,
        tgt_amp_aug_lo: float = 0.0,
        tgt_fluct_aug: float = 0.0,
        tgt_fluct_aug_hi: float | None = None,
        mean_aug: float = 0.0,
        zero_p: bool = False,
        rescale_fluct: bool = False,
        geom_channels: tuple[str, ...] = (),
        flow_channels: tuple[str, ...] = (),
        cond_mode: str = "none",
        cond_source: str = "label",
        piv_noise: float = 0.0,
        piv_noise_target: float = 0.0,
        aux_mask_mode: str = "",
        aux_mask_ratio: float = 0.0,
        aux_mask_plane: bool = True,
        aux_p: bool = False,
        p_target: str = "",
        fine: bool = False,
        dt_aug: tuple[int, ...] = (),
        dt_ratio_aug: tuple[float, ...] = (),
        dt_ratio_p1: float = 0.5,
        history_dropout: float = 0.0,
        recent_dropout: float = 0.0,
        recent_dropout_max: int = 3,
        stream_state: str = "",
        state_dropout: float = 0.0,
        state_dropout_max: int = 4,
        seed: int = 0,
    ) -> None:
        self.kind = kind
        # Which *dataset* this is, with any phase suffix stripped. Every sim-vs-real
        # behaviour below branches on this: a ``sim_p01`` sample is simulated data and
        # must be masked, conditioned and rescaled as such (see survey.base_kind).
        self.base = base_kind(kind)
        self.t_in, self.t_out = t_in, t_out
        # Probability that a window's EARLIER frames (everything before the most recent
        # T_IN) are zeroed. Only meaningful when ``t_in > T_IN``. Track 2's stream has no
        # history on a trajectory's first window, so a 40-frame model served there is handed
        # zeros it has never seen unless this teaches it the case during the fine-tune.
        # The conditioning is untouched: every estimate reads ``inp[-T_IN:]``.
        self.history_dropout = float(history_dropout)
        # Probability that a window's MOST RECENT frames are replaced by its own time mean,
        # and how deep that goes. The opposite end from ``history_dropout`` and a different
        # question: the shipped backbone's effective input memory is about two frames
        # (``scripts/eval/input_memory_probe.py``), and this hides exactly those frames so the
        # model has to use the rest of the window or lose. The mean, not zeros, because zeros
        # are a state the flow never takes and the probe this is matched against uses the mean.
        self.recent_dropout = float(recent_dropout)
        self.recent_dropout_max = int(recent_dropout_max)
        # Causal trajectory-state pseudo-frames (Track-2 stream arm; TrainConfig.stream_state).
        # "meanvar" prepends running mean / temporal-std frames over everything revealed
        # through the current input. History is quantised to the stream's 20-frame reveals so
        # per-trajectory prefix statistics stay small enough to cache (~1.4 MB/trajectory);
        # on stride-20-aligned starts the quantisation is exact.
        if stream_state not in STREAM_STATE_MODES:
            raise ValueError(
                f"stream_state must be one of {STREAM_STATE_MODES}, got {stream_state!r}"
            )
        if stream_state and (dt_aug or dt_ratio_aug):
            raise ValueError("stream_state assumes the scored clock; drop dt_aug/dt_ratio_aug")
        if stream_state and rescale_fluct:
            raise ValueError(
                "stream_state reads raw trajectory frames; rescale_fluct would "
                "desynchronise the state from the window"
            )
        if stream_state and not (normalizer == "freestream" and normalize or normalizer == "none"):
            raise ValueError(
                "stream_state needs the freestream normalizer (state divided with the window) "
                f"or normalizer='none' (the make_predict path, which divides downstream); got "
                f"normalizer={normalizer!r} normalize={normalize}"
            )
        if stream_state and t_in != T_IN:
            raise ValueError(
                "stream_state and a longer raw context (t_in > 20) are two "
                "spellings of the same history; combine them deliberately or not "
                "at all (runs/t2_backbone/PLAN.md measures them apart)"
            )
        if stream_state and (geom_channels or flow_channels or aux_mask_mode):
            raise ValueError(
                "stream_state pseudo-frames carry the 3 data channels only; "
                "extra input planes would need matching state planes"
            )
        self.stream_state = stream_state
        # Cold-state dropout: with probability ``state_dropout`` a training window's state
        # pseudo-frames are rebuilt from only its first ``k`` reveals (k uniform in
        # ``1..state_dropout_max``, clamped to the window's true reveal count), so the backbone
        # meets the noisy early-stream state paired with the small count plane that indexes it.
        # A train-only augmentation (the val loaders never pass it), and the state arm's answer
        # to ``history_dropout``, which ``stream_state`` cannot use (it fixes ``t_in == T_IN``).
        self.state_dropout = float(state_dropout)
        self.state_dropout_max = int(state_dropout_max)
        if self.state_dropout > 0 and not stream_state:
            raise ValueError("state_dropout needs stream_state (it perturbs the state frames)")
        self._state_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.normalize = normalize
        self.mask_sim = mask_sim and self.base in SIM_DOMAINS
        # Pressure exists in sim and never in real. Zeroing it makes a simulated
        # sample structurally identical to a real one, which is what lets an
        # architecture ranking measured on sim be read against the real arms -- and
        # what keeps the official scorer at two measured channels, since sim's ``p``
        # has std 12.6 against ``u``'s 0.44 and would otherwise dominate rel_l2.
        self.zero_p = zero_p
        # Per-cell fluctuation gain toward real's turbulence intensity. The freestream
        # division fixes the mean-flow scale but leaves u'/U untouched -- it is already
        # dimensionless -- and sim over-fluctuates by 1.8x to 20x in energy at every cell.
        self.fluct_scale: dict[tuple[int, int], float] | None = None
        if rescale_fluct:
            raise ValueError("rescale_fluct (data/fluctuation_scale.py) is not included in this package")
        # Geometry planes appended to the input. Built from the angle *estimated from the
        # input window*, never the label, so training sees the same 1.77 deg-MAE estimate
        # that inference does -- the same rule as the freestream speed.
        self.geom_channels = tuple(geom_channels)
        if self.geom_channels:
            n_geometry_channels(self.geom_channels)  # validate names now, not per item
        # Planes derived from the observed window itself (vorticity and the two second
        # moments). Unlike the geometry planes these depend on nothing outside the input, so
        # they are computed after normalisation from exactly what the model is handed.
        self.flow_channels = tuple(flow_channels)
        if self.flow_channels:
            n_flow_channels(self.flow_channels)
        # Which conditioning vector to emit. Only ``operating_point_tau`` changes what is
        # computed here -- it needs the window, which the label-based components do not.
        if cond_mode not in COND_MODES:
            raise ValueError(f"cond_mode must be one of {COND_MODES}, got {cond_mode!r}")
        self.cond_mode = cond_mode
        # ``estimate`` conditions real windows on the operating point read off the raw window,
        # as inference does; simulated windows keep their labels whatever is asked, because
        # the freestream estimator cannot read a Reynolds number from non-dimensional data.
        if cond_source not in ("label", "estimate"):
            raise ValueError(f"cond_source must be 'label' or 'estimate', got {cond_source!r}")
        self.cond_source = cond_source
        self.estimate_cond = cond_source == "estimate" and self.base not in SIM_DOMAINS
        if normalizer not in ("freestream", "gaussian", "none"):
            raise ValueError(f"normalizer must be freestream|gaussian|none, got {normalizer!r}")
        self.normalizer = normalizer
        self.stats = gaussian_stats(kind) if normalizer == "gaussian" else None
        # Both augmentations are numerical-data-only in the reference loader
        # (realpdebench/data/fluid_dataset.py), so they are gated on kind here too.
        self.mask_prob = mask_prob if self.base in SIM_DOMAINS else 0.0
        # Unlike the two above this one is *not* sim-gated: its whole purpose is to vary
        # the real measurement geometry, which only real data has.
        self.mask_swap_prob = mask_swap_prob
        self.noise_scale = noise_scale if self.base in SIM_DOMAINS else 0.0
        # The mirror image: synthetic measurement noise on *real* inputs. Multiplicative,
        # like the organizers' sim noise, which has a property that matters here -- it
        # leaves exact zeros at zero, so the PIV holes stay holes instead of being filled
        # with noise the model would then have to learn to ignore. Under
        # ``cond_mode="operating_point_noise"`` this is reinterpreted as a per-window dose
        # *ceiling* rather than a fixed dose -- see the draw in ``__getitem__``.
        self.real_noise_scale = real_noise_scale if self.base not in SIM_DOMAINS else 0.0
        # Fluctuation-amplitude augmentation of real windows (``TrainConfig.fluct_aug``).
        self.fluct_aug = fluct_aug if self.base not in SIM_DOMAINS else 0.0
        # AoA-conditioning jitter of real windows (``TrainConfig.aoa_jitter_deg``). Sim windows
        # already span the five angles and are never jittered; eval datasets pass 0.
        self.aoa_jitter_deg = aoa_jitter_deg if self.base not in SIM_DOMAINS else 0.0
        # Operating-point-dependent doses (2026-09-23, FINDINGS 184i/195: the flat fluct-aug's
        # gain sits at Re 3750 and its cost at the top of the ladder). Every dose below is read
        # off the cell LABEL through ``re_dose`` -- a property of the training schedule, like
        # ``re_tilt``'s sampler weights, not of the model's input. ``fluct_aug_hi`` None keeps
        # the flat sigma, which is the pre-existing behaviour bit for bit.
        real = self.base not in SIM_DOMAINS
        self.fluct_aug_hi = float(fluct_aug_hi) if fluct_aug_hi is not None and real else None
        # One-sided TARGET-only fluctuation boost (``TrainConfig.tgt_amp_aug``): a bias, not an
        # invariance -- it changes the input-to-target amplitude relation on purpose.
        self.tgt_amp_aug = float(tgt_amp_aug) if real else 0.0
        self.tgt_amp_aug_lo = float(tgt_amp_aug_lo) if real else 0.0
        # Symmetric TARGET-only log-normal (``TrainConfig.tgt_fluct_aug``): the target half of
        # ``fluct_aug`` on its own, a mechanism probe rather than a bias.
        self.tgt_fluct_aug = float(tgt_fluct_aug) if real else 0.0
        self.tgt_fluct_aug_hi = (
            float(tgt_fluct_aug_hi) if tgt_fluct_aug_hi is not None and real else None
        )
        # Mean-flow perturbation toward an adjacent-Reynolds training cell
        # (``TrainConfig.mean_aug``); the donor table is built once the trajectories are known.
        self.mean_aug = float(mean_aug) if real else 0.0
        # Synthetic PIV noise. Multipliers on the Reynolds-dependent measured level, so
        # 1.0 is "as noisy as the delivered real data" and the number is interpretable.
        # Simulated only: real data already carries its own sensor noise and adding more
        # would double-count it.
        self.piv_noise = piv_noise if self.base in SIM_DOMAINS else 0.0
        self.piv_noise_target = piv_noise_target if self.base in SIM_DOMAINS else 0.0
        # Masked spatio-temporal modelling of the input window. Applies to both domains:
        # it is a representation-learning device, not a sim2real bridge.
        self.aux_mask_mode = aux_mask_mode
        self.aux_mask_ratio = aux_mask_ratio
        self.aux_mask_plane = aux_mask_plane
        # All-to-all sampling: the temporal stride *within* a window, drawn per item from
        # this multiset (repeat a value to weight it). Distinct from the ``stride``
        # argument, which is the spacing *between* window starts and changes only how many
        # windows exist. A stride-s window observes frames (t, t+s, ... t+19s) and is
        # supervised on (t+20s, ... t+39s), so the pair keeps one clock and the horizon
        # scales with s: the task stays 20->20 and stays well-posed, because the input's
        # own decorrelation rate tells the model which s it was given. That is the whole
        # design -- no conditioning slot is added, so nothing about inference changes.
        # The motivation is physical: dt* = dt.U/c grows with Re at fixed dt, so an unseen
        # high Re is partly an unseen *horizon*, and s>1 windows of a training cell supply
        # exactly that horizon from data we already have.
        if any(s < 1 for s in dt_aug):
            raise ValueError(f"dt_aug strides must be >= 1, got {dt_aug!r}")
        self.dt_aug = tuple(int(s) for s in dt_aug)
        # Time-step augmentation on the REAL clock (``TrainConfig.dt_ratio_aug``): a simulated
        # window is resampled at ``ratio * real_step_sim_frames(Re)`` source frames per frame,
        # the units cache's own step times a log-uniform ratio, by the cache's own linear
        # blend of neighbouring frames. Simulated only -- a real window's interval is the PIV
        # rate -- and from the NATIVE cache only: resampling an already-resampled cache would
        # put the ratio on the wrong reference and blend twice.
        if dt_ratio_aug and self.base not in SIM_DOMAINS:
            dt_ratio_aug = ()
        if dt_ratio_aug:
            if len(dt_ratio_aug) != 2 or not 0 < dt_ratio_aug[0] <= dt_ratio_aug[1]:
                raise ValueError(
                    f"dt_ratio_aug must be (lo, hi) with 0 < lo <= hi, got {dt_ratio_aug!r}"
                )
            if self.base != "sim":
                raise ValueError(
                    f"dt_ratio_aug needs the native 'sim' cache, got kind={kind!r}: the ratio is "
                    "relative to the real step and the blend must be applied once"
                )
            if fine:
                raise ValueError("dt_ratio_aug and fine=True are not supported together")
            if not 0.0 <= dt_ratio_p1 <= 1.0:
                raise ValueError(f"dt_ratio_p1 must be in [0, 1], got {dt_ratio_p1}")
        self.dt_ratio_aug = tuple(float(r) for r in dt_ratio_aug)
        self.dt_ratio_p1 = float(dt_ratio_p1)
        # Pressure as an auxiliary target while ``target`` stays p-free. Simulated only:
        # real PIV has no pressure at all, so there is nothing to supervise against.
        self.aux_p = aux_p and self.base in SIM_DOMAINS
        # Pressure on the model's OWN third output channel -- the one the field loss already
        # has a term for and inference already discards -- rather than on an extra head.
        # ``"gauge"`` is the only mode, and it names the three corrections that make the
        # channel supervisable at all (see the block in ``__getitem__``): the per-frame
        # spatial gauge is removed, the divisor is ``u_fs**2`` because pressure scales as
        # ``rho U**2``, and the measurement mask is honoured. Simulated only, like ``aux_p``.
        if p_target not in ("", "gauge"):
            raise ValueError(f"p_target must be '' or 'gauge', got {p_target!r}")
        if p_target and normalizer != "freestream":
            # The gaussian statistics z-score ``p`` with a fitted std of 12.58 and the
            # freestream divisor is what ``u_fs**2`` refers to, so the correction has no
            # meaning under any other scheme. Refuse rather than apply it to the wrong scale.
            raise ValueError(f"p_target='gauge' needs normalizer='freestream', got {normalizer!r}")
        self.p_target = p_target if self.base in SIM_DOMAINS else ""
        self._rng = np.random.default_rng(seed)

        index = load_index(kind)
        if trajectories is not None:
            wanted = set(trajectories)
            index = [c for c in index if (c.re, c.aoa) in wanted]
            if not index:
                raise ValueError(f"None of the {len(wanted)} requested cells exist in {kind}.")
        self.trajectories: list[CachedTrajectory] = index

        # A phase-shifted real cache has its own measurement mask -- the airfoil and its
        # shadow land on different pixels at a shifted sampling grid -- so the mask follows
        # the kind. Simulated kinds still borrow real's, which is the point of that branch.
        real_masks = load_masks("real" if self.base in SIM_DOMAINS else kind)
        self.masks: list[np.ndarray] = [
            real_mask_for_sim((c.re, c.aoa), real_masks)
            if self.base in SIM_DOMAINS
            else real_masks[(c.re, c.aoa)]
            for c in self.trajectories
        ]
        # ``mean_aug`` donors: per trajectory, the normalised mean-flow differences to its
        # adjacent-Reynolds TRAINING cells at the same angle. Built from ``self.trajectories``
        # alone, so a withheld operating point never reaches a training window through its
        # climatology; a withheld row between two training rungs is simply skipped over, and
        # a fraction of that longer step is what "an unseen operating point" looks like here.
        self._mean_aug_delta: list[list[np.ndarray]] = []
        if self.mean_aug > 0:
            clim = build_climatology(kind)
            self._mean_aug_delta = mean_aug_deltas(
                [(int(c.re), int(c.aoa)) for c in self.trajectories], clim, self.masks
            )

        # A release defect, not a modelling choice: one simulated cell stores ``p`` on
        # an absolute gauge 268x the dataset median spread, and supervising it carried
        # 99.9% of the sim-pretraining loss mass. Its ``u``/``v`` are in family, so the
        # trajectory stays and only the channel goes. See splits.DEFECTIVE_PRESSURE_CELLS.
        self._p_defective: list[bool] = [
            (c.re, c.aoa) in DEFECTIVE_PRESSURE_CELLS for c in self.trajectories
        ]

        # Native-resolution targets for the fine and sub-grid auxiliary heads. The input
        # is never touched: no 64x128 field exists at submission time, so a trunk that
        # consumed one would be untrainable against the actual contract.
        self.fine = fine
        self._native_paths: list[Path] = []
        self._native: list[np.ndarray | None] = []
        if fine:
            available = load_native_paths(kind if kind in SIM_DOMAINS else kind)
            missing = [(c.re, c.aoa) for c in self.trajectories if (c.re, c.aoa) not in available]
            if missing:
                raise FileNotFoundError(
                    f"fine=True needs the {kind} native cache; {len(missing)} cells absent "
                    f"(first: {missing[0]}). Build it with "
                    f"`python -c 'from rpde_t1.data.cache import build_native_cache;"
                    f' build_native_cache("{kind}")\'`.'
                )
            self._native_paths = [available[(c.re, c.aoa)] for c in self.trajectories]
            self._native = [None] * len(self.trajectories)

        self._arrays: list[np.ndarray | None] = [None] * len(self.trajectories)
        self.windows: list[WindowIndex] = []
        # Last frame index each trajectory may be read up to. ``dt_aug`` needs it: a
        # strided window reaches (span-1)*s frames past its start rather than span, and
        # over-running a reserved ``val_time`` tail would train on validation frames.
        self._frame_limit: list[int] = [0] * len(self.trajectories)
        span = t_in + t_out
        for i, c in enumerate(self.trajectories):
            # A per-cell mapping lets only a subset of trajectories reserve a temporal
            # tail. Reserving it on all 69 costs 26% of training windows; on a
            # stratified 20 it costs 7.6% for the same diagnostic.
            lo_f, hi_f = (
                frame_range[(c.re, c.aoa)] if isinstance(frame_range, dict) else frame_range
            )
            lo = int(round(c.n_frames * lo_f))
            hi = int(round(c.n_frames * hi_f))
            # Under ``dt_ratio_aug`` every start must host the ratio-1 window (the real clock,
            # ``(span-1) * step(Re)`` source frames), so the reference clock is never squeezed
            # out near the end of a trajectory; faster draws are then limited by the room left.
            need = self._base_span(c.re) if self.dt_ratio_aug else span
            if hi - lo < need:
                # Not an error: three real trajectories are short (282/492/607 frames),
                # so a narrow frame_range legitimately excludes them.
                continue
            self._frame_limit[i] = hi
            self.windows.extend(WindowIndex(i, s) for s in range(lo, hi - need + 1, stride))
        if not self.windows:
            raise ValueError(
                f"No windows for kind={kind} over frame_range={frame_range}; "
                f"each trajectory needs {span} frames in range."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def _state_frames(
        self,
        w: WindowIndex,
        scale: float,
        n_mul: float,
        channels: int,
        n_rev: int | None = None,
    ) -> np.ndarray:
        """The ``stream_state`` pseudo-frames for one window, ``(F, H, W, channels)``.

        History is every frame the stream has revealed through the current input,
        ``arr[: start + T_IN]``, quantised DOWN to the stream's 20-frame reveals -- the serving
        port accumulates whole windows, so on stride-20-aligned starts the quantisation is
        exact and elsewhere it under-reports by at most one reveal. Per-(trajectory, reveal)
        statistics are memoised.

        The modes, each a strict superset of the one before in what it tells the model:

        ``meanvar`` (2 frames)
            ``[running mean u, running mean v, n_revealed/820]`` and
            ``[running temporal std u, running temporal std v, 0]``. The level of the
            trajectory's mean flow and of its fluctuation.
        ``meanvarfast`` (4 frames)
            the two above, plus the same two statistics over the RECENT
            ``FAST_WINDOWS`` reveals only. Two timescales let the model read a TREND -- a
            trajectory whose recent mean differs from its lifetime mean is drifting, which a
            single running mean cannot express.
        ``meanvarperr`` (3 frames)
            ``meanvar`` plus ``[mean |x_t - x_{t-20}| over revealed t, ..., 0]`` -- the
            20-lag PERSISTENCE error map. It is the error field of the zero-parameter
            forecaster on this trajectory's own history, so it marks where prediction is hard
            (wake, shear layers) using only revealed frames and no model. Chosen because the
            causally reachable part of the error field is its PERSISTENT structure (FINDINGS
            136c: the per-window anomaly decorrelates at lag 1, the per-trajectory part does
            not), and this is exactly that part, available at train and serve time alike.

        ``meanvarcov`` (4 frames)
            ``meanvar`` plus ``[signed sqrt of cov(u, v), rms vorticity, 0]`` and
            ``[lag-1 temporal autocorrelation of u, of v, 0]`` (FINDINGS 192). The three
            per-pixel quantities the level statistics cannot express: the Reynolds shear
            stress (a cross-channel second moment), the enstrophy, and how fast each pixel
            DECORRELATES -- the one-frame predictability that should govern how far a forecast
            damps toward the mean. The autocorrelation is dimensionless and is not divided
            by ``scale``; pixels with no temporal variance (the body, invalid PIV) read 0.

        Everything is divided by the window's own freestream ``scale`` as the real frames are
        (``perr`` is a velocity difference, so it carries the same units). On the
        ``normalizer='none'`` path the caller passes ``scale=1`` and the dimensionless count
        plane is pre-multiplied by ``n_mul``.

        ``n_rev`` overrides the revealed-frame count (cold-state dropout): a value below the
        window's true count rebuilds the state as it stood after fewer reveals, and the count
        plane, cache key and statistics all follow it. ``None`` uses the true count; any value
        is clamped into ``[T_IN, true count]`` so it can only ever look further back, never
        ahead of the stream.
        """
        true_rev = max(T_IN, (w.start + T_IN) // T_IN * T_IN)
        n_rev = true_rev if n_rev is None else max(T_IN, min(int(n_rev), true_rev))
        key = (w.traj, n_rev)
        got = self._state_cache.get(key)
        if got is None:
            hist = np.asarray(self._array(w.traj)[:n_rev, ..., :2], dtype=np.float32)
            stats = {"mean": hist.mean(axis=0), "sd": hist.std(axis=0)}
            if self.stream_state == "meanvarfast":
                lo = max(0, n_rev - T_IN * FAST_WINDOWS)
                stats["fmean"] = hist[lo:].mean(axis=0)
                stats["fsd"] = hist[lo:].std(axis=0)
            elif self.stream_state == "meanvarcov":
                stats.update(covariance_stats(hist))
            elif self.stream_state == "meanvarperr":
                # |x_t - x_{t-20}|: the 20-lead persistence error, over revealed frames only.
                stats["perr"] = (
                    np.abs(hist[T_IN:] - hist[:-T_IN]).mean(axis=0)
                    if n_rev > T_IN
                    else np.zeros_like(hist[0])
                )
            elif self.stream_state == "meanvarphase":
                # The single-frequency Fourier mode at the ONLINE-estimated shedding frequency,
                # normalised, per (pixel, channel). The frequency is estimated from these same
                # revealed frames, so ``(traj, n_rev)`` still keys the cache; ``phase_planes_np``
                # is the one arithmetic the serving port mirrors.
                stats["phase"] = phase_planes_np(hist)
            got = self._state_cache[key] = stats
        h, wid = got["mean"].shape[:2]
        n_frames = STREAM_STATE_FRAMES[self.stream_state]
        out = np.zeros((n_frames, h, wid, channels), dtype=np.float32)
        out[0, ..., :2] = got["mean"] / scale
        out[0, ..., 2] = min(n_rev / 820.0, 1.0) * n_mul
        out[1, ..., :2] = got["sd"] / scale
        if self.stream_state == "meanvarfast":
            out[2, ..., :2] = got["fmean"] / scale
            out[3, ..., :2] = got["fsd"] / scale
        elif self.stream_state == "meanvarperr":
            out[2, ..., :2] = got["perr"] / scale
        elif self.stream_state == "meanvarcov":
            out[2, ..., 0] = got["cov_ssqrt"] / scale
            out[2, ..., 1] = got["vort_rms"] / scale
            out[3, ..., :2] = got["rho1"]
        elif self.stream_state == "meanvarphase":
            # Dimensionless (phase and coherence), so NOT divided by ``scale``.
            out[2, ..., :2] = got["phase"][0]
            out[3, ..., :2] = got["phase"][1]
        return out

    def window_re(self) -> np.ndarray:
        """``(N,)`` Reynolds number of the cell each window came from.

        Exposed so a sampler can weight windows by operating point without reaching into
        ``windows``/``trajectories``, which are implementation detail. Read from the cell
        *label*, not estimated: this decides how often a window is drawn, which is a
        property of the training schedule rather than of the model's input.
        """
        return np.array([self.trajectories[w.traj].re for w in self.windows], dtype=np.float64)

    def _array(self, traj: int) -> np.ndarray:
        # Memmaps are opened lazily and per worker process, so a DataLoader with
        # workers does not share (and cannot corrupt) file handles.
        if self._arrays[traj] is None:
            self._arrays[traj] = self.trajectories[traj].open()
        return self._arrays[traj]  # type: ignore[return-value]

    def _native_array(self, traj: int) -> np.ndarray:
        if self._native[traj] is None:
            self._native[traj] = np.load(self._native_paths[traj], mmap_mode="r")
        return self._native[traj]  # type: ignore[return-value]

    def _base_span(self, re: float) -> int:
        """Source frames a ratio-1 window of a cell at ``re`` spans (never under ``span``)."""
        span = self.t_in + self.t_out
        return int(np.ceil((span - 1) * max(real_step_sim_frames(re), 1.0))) + 1

    def _dt_ratio(self, w: WindowIndex) -> tuple[float, float]:
        """``(source frames per frame, ratio to the real step)`` for one simulated window.

        With probability ``dt_ratio_p1`` the window is on the real clock (ratio 1, the units
        cache's own step). Otherwise the ratio is log-uniform on ``[lo, hi]`` narrowed by two
        guards: the effective step stays at or above one source frame -- no frame is invented
        between two frames; a cell whose real step is already below 1 only ever goes faster --
        and at or below ``DT_RATIO_MAX_STEP``, and the window stays inside the frames this
        trajectory may read (a reserved tail is never entered). When the guards leave nothing,
        the draw falls back to the real clock.
        """
        step1 = real_step_sim_frames(self.trajectories[w.traj].re)
        span = self.t_in + self.t_out
        if self._rng.random() < self.dt_ratio_p1:
            return step1, 1.0
        room = self._frame_limit[w.traj] - 1 - w.start
        lo = max(self.dt_ratio_aug[0], 1.0 / step1)
        hi = min(self.dt_ratio_aug[1], DT_RATIO_MAX_STEP / step1, room / ((span - 1) * step1))
        if hi <= lo:
            return step1, 1.0
        ratio = float(np.exp(self._rng.uniform(np.log(lo), np.log(hi))))
        return ratio * step1, ratio

    def _dt_step(self, w: WindowIndex) -> int:
        """Temporal stride for one window, drawn from ``dt_aug`` among those that fit.

        Strides too long for this start are dropped rather than clipped to the end of the
        trajectory: clipping would pile every long-horizon window onto the same few final
        frames, and near a reserved tail it would read the frames the tail withholds.
        ``1`` always fits -- the window list is built for it -- so the fallback is the
        scored task, not an error.
        """
        if not self.dt_aug:
            return 1
        room = self._frame_limit[w.traj] - 1 - w.start
        fits = [s for s in self.dt_aug if (self.t_in + self.t_out - 1) * s <= room]
        if not fits:
            return 1
        return int(fits[int(self._rng.integers(len(fits)))])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        w = self.windows[i]
        c = self.trajectories[w.traj]
        arr = self._array(w.traj)
        span = self.t_in + self.t_out
        if self.dt_ratio_aug:
            # Fractional step on the real clock (or a multiple of it), the units cache's blend.
            eff, ratio = self._dt_ratio(w)
            block = fractional_block(arr, w.start, eff, span)
            step = 0  # the integer path is not in use; ``fine`` is refused with this option
        else:
            step = self._dt_step(w)
            ratio = float(step)  # a real window's reference interval IS one frame
            block = np.asarray(
                arr[w.start : w.start + (span - 1) * step + 1 : step]
                if step > 1
                else arr[w.start : w.start + span],
                dtype=np.float32,
            )

        invalid = self.masks[w.traj]
        if self.mask_sim:
            block = block.copy()
            block[:, invalid, :] = 0.0

        if self.mask_swap_prob > 0 and self._rng.random() < self.mask_swap_prob:
            # Borrow another cell's measurement mask and take the UNION, never a
            # replacement. Replacing would mark this cell's own holes "valid" while the
            # cached array still holds zeros there, teaching the model that a valid pixel
            # can read exactly zero -- the opposite of what the mask means. A union only
            # ever adds holes, so every pixel it declares invalid really is.
            donor = self.masks[int(self._rng.integers(len(self.masks)))]
            invalid = invalid | donor
            block = block.copy()
            block[:, invalid, :] = 0.0

        # The 64x128 mirror of the same frames, put through the identical treatment so the
        # fine target is the coarse target's own refinement and not a second, differently
        # conditioned field. ``fine[..., ::2, ::2] == coarse`` holds on the raw arrays, and
        # every step below is applied to both, so it keeps holding.
        fine_block: np.ndarray | None = None
        if self.fine:
            # The identical frames, strided identically: a fine target on a different clock
            # from its coarse twin would supervise a refinement of a different window.
            fine_block = np.asarray(
                self._native_array(w.traj)[w.start : w.start + (span - 1) * step + 1 : step],
                dtype=np.float32,
            )
            if self.mask_sim:
                fine_block[:, np.repeat(np.repeat(invalid, 2, 0), 2, 1), :] = 0.0

        if self.fluct_scale is not None:
            # About the block's own time mean, matching how ``kinetic_energy`` (and hence
            # ``tke_score``) defines the fluctuation. Applied to the whole 40-frame block
            # before it is split, so input and target stay one consistent trajectory.
            r = self.fluct_scale.get((c.re, c.aoa))
            if r is not None:
                m = block.mean(axis=0, keepdims=True)
                block = m + (block - m) * np.float32(r)
                if fine_block is not None:
                    fm = fine_block.mean(axis=0, keepdims=True)
                    fine_block = fm + (fine_block - fm) * np.float32(r)

        if self.mean_aug > 0 and self._mean_aug_delta[w.traj]:
            # Shift the whole block's MEAN flow part of the way toward an adjacent-Reynolds
            # training cell's climatology at the same angle, fluctuations untouched. The
            # climatology is in freestream units, the block is raw, hence ``c.u_fs``. The
            # operating-point estimate below reads the shifted window, so the conditioning
            # follows the mean flow -- the window is presented as a nearby operating point,
            # not as a noisy copy of its own. Only the measured channels move.
            deltas = self._mean_aug_delta[w.traj]
            delta = deltas[int(self._rng.integers(len(deltas)))]
            alpha = np.float32(self._rng.uniform(0.0, self.mean_aug)) * np.float32(c.u_fs)
            block = block.copy()
            block[..., :N_MEASURED] += delta * alpha
            if fine_block is not None:
                fine_block = fine_block.copy()
                fine_block[..., :N_MEASURED] += np.repeat(np.repeat(delta, 2, 0), 2, 1) * alpha

        sigma = (
            self.fluct_aug
            if self.fluct_aug_hi is None
            else re_dose(c.re, self.fluct_aug, self.fluct_aug_hi)
        )
        if sigma > 0:
            # One log-normal factor per window on the whole block's fluctuation about its own
            # time mean (input and target together), as ``fluct_scale`` above does per cell.
            r = np.float32(np.exp(self._rng.normal(0.0, sigma)))
            m = block.mean(axis=0, keepdims=True)
            block = m + (block - m) * r
            if fine_block is not None:
                fm = fine_block.mean(axis=0, keepdims=True)
                fine_block = fm + (fine_block - fm) * r

        tsig = (
            self.tgt_fluct_aug
            if self.tgt_fluct_aug_hi is None
            else re_dose(c.re, self.tgt_fluct_aug, self.tgt_fluct_aug_hi)
        )
        if tsig > 0:
            # The same log-normal factor ``fluct_aug`` would draw, applied to the TARGET frames'
            # fluctuation only (about their own time mean), input untouched. Zero-mean in log, so
            # unlike ``tgt_amp_aug`` it injects no bias: it asks whether ``fluct_aug``'s gain is
            # label noise on the target amplitude (this half) or invariance to the input's.
            r = np.float32(np.exp(self._rng.normal(0.0, tsig)))
            block = boost_target_fluctuation(block, self.t_in, r)
            if fine_block is not None:
                fine_block = boost_target_fluctuation(fine_block, self.t_in, r)

        boost = re_dose(c.re, self.tgt_amp_aug_lo, self.tgt_amp_aug)
        if boost > 0:
            # One-sided: the TARGET frames' fluctuation about their own time mean is scaled by
            # ``exp(U(0, boost))`` >= 1 and the input is left alone. Deliberately inconsistent
            # -- it teaches the model to emit more fluctuation energy than the input's own
            # continuation carries, dosed against FINDINGS 183b's one-sided deficit.
            r = np.float32(np.exp(self._rng.uniform(0.0, boost)))
            block = boost_target_fluctuation(block, self.t_in, r)
            if fine_block is not None:
                fine_block = boost_target_fluctuation(fine_block, self.t_in, r)

        inp, tgt = block[: self.t_in].copy(), block[self.t_in :].copy()

        # Estimated before any normalisation or augmentation touches the window, so it is
        # the same quantity inference computes from the raw array it is handed.
        # Every estimate reads the LAST ``T_IN`` frames: with a longer context (``t_in`` 40)
        # that keeps the conditioning, geometry and normalisation identical to the 20-frame
        # model's and to what inference computes from the most recent frames.
        recent = inp[-T_IN:]
        if self.estimate_cond:
            re_cond, aoa_cond = estimate_operating_point(recent)
            aoa_est = aoa_cond
        else:
            re_cond, aoa_cond = float(c.re), float(c.aoa)
            # The geometry plane's angle. On real windows it is estimated, as inference must.
            # On simulated windows it is the LABEL: the estimator reads the PIV hole and the
            # wake depth, and simulation has no hole, so on sim it came back biased +3.9 deg
            # (mean abs 3.9-4.4 deg, max 10.5, against 1.5 deg on real; unet2d campaign,
            # 2026-09-17) and drew the foil too steep through the whole pretrain.
            if self.base in SIM_DOMAINS:
                aoa_est = aoa_cond
            else:
                aoa_est = estimate_aoa(recent) if self.geom_channels else 0.0
        # AoA-conditioning jitter (``TrainConfig.aoa_jitter_deg``): perturb the conditioning and
        # geometry-plane angle by ``U(-J, J)`` so the model sees a CONTINUOUS AoA axis and
        # interpolates to unseen angles rather than memorising the five labels (§52d). Applied to
        # ``aoa_cond`` and ``aoa_est`` together (same draw) so the conditioning and any geometry
        # plane agree; the raw pixels and the target are untouched. Real windows only (guarded in
        # ``__init__``); off (0.0) on every eval dataset.
        if self.aoa_jitter_deg > 0.0:
            delta = float(self._rng.uniform(-self.aoa_jitter_deg, self.aoa_jitter_deg))
            aoa_cond = float(np.clip(aoa_cond + delta, *AOA_CLIP_DEG))
            aoa_est = float(np.clip(aoa_est + delta, *AOA_CLIP_DEG))
        # Same rule, same line of the pipeline: the decorrelation time is a property of
        # the raw window, and a model trained on augmented windows would otherwise be
        # conditioned on a timescale that inference never reproduces.
        tau = decorrelation_time(recent) if self.cond_mode == "operating_point_tau" else None

        # Kept before ``zero_p`` blanks it. Pressure is unscored and absent from real PIV,
        # so it never belongs in ``target``; as a *separate* auxiliary target it is free
        # physics -- ``lap p = -rho div(u . grad u)`` makes it a well-posed function of the
        # velocity field the model already has to predict. The per-frame spatial mean is
        # removed because p is defined only up to a gauge, and supervising the gauge would
        # spend capacity on an arbitrary constant.
        raw_p = None
        if self.aux_p and not self._p_defective[w.traj]:
            raw_p = tgt[..., 2:3] - tgt[..., 2:3].mean(axis=(1, 2), keepdims=True)

        # ``p_target`` supervises the same quantity on the backbone's own third output
        # channel, and differs from ``aux_p`` in the three ways that decide whether the
        # channel is learnable at all. Measured on the release, per frame:
        #
        # * **Gauge.** Incompressible pressure is defined up to ``p + f(t)``, and the mean is
        #   taken over the *valid* cells only -- the masked holes hold exact zeros, so
        #   including them would bias the datum by the hole fraction. This alone repairs the
        #   cell ``splits.DEFECTIVE_PRESSURE_CELLS`` names: ``13950_15``'s raw ``p`` has a
        #   spatial mean of 120.5 and a per-frame-mean spread of 38.3, but its gauge-removed
        #   spread is 0.150 against an AoA=15 family median of 0.149. So the defect is a
        #   drifting datum and not a broken field, and the cell is kept here rather than
        #   dropped.
        # * **Scale.** ``p ~ rho U**2``, so the freestream divisor is squared. Confirmed
        #   independently: regressing ``lap p`` on the 2D Poisson source
        #   ``-(u_x**2 + 2 u_y v_x + v_y**2)`` over the release returns a slope of 0.86-1.03,
        #   i.e. ``p`` is already non-dimensionalised in the same ``U**2`` units as ``u, v``.
        # * **Mask.** The airfoil's solid section reaches into the window at *every* incidence
        #   -- 110 to 119 of the 8192 raw cells hold ``u = v = 0`` exactly -- and at four of
        #   the five it stores an unremarkable interior pressure, with less variance than the
        #   flow around it (ratio 0.3-0.8). **At AoA=10 it does not**: in-body pressure
        #   variance is **177x** the flow's and reaches ``-6.0``, so 1.5% of the cells carry
        #   **76% of the gauge-removed squared-pressure mass** on all 20 AoA=10 cells -- a
        #   fifth of the simulated grid, four times louder than the rest. So it is those
        #   values that are anomalous, not the geometry. ``mask_sim`` already zeroes the body
        #   here, and ``losses.field_loss`` now excludes it from the term's normaliser too.
        #
        # The *input* plane is blanked whatever ``zero_p`` says. A simulated window that
        # carries ``p`` in its input is a different task from the real one -- the model would
        # learn to read a plane inference never supplies, and the transfer measurement would
        # be of that gap rather than of the objective.
        gauge_p = None
        if self.p_target:
            v = ~invalid
            m = tgt[..., 2][:, v].mean(axis=1, dtype=np.float64)[:, None, None]
            gauge_p = np.where(v, tgt[..., 2] - m, 0.0).astype(np.float32)

        # A zeroed pressure plane is not a pressure measurement: leaving ``has_p`` at 1 where
        # the target has been blanked supervises the field loss's pressure term toward zero.
        # Tracked as one flag because three separate branches blank it.
        p_supervised = self.base in SIM_DOMAINS and (bool(self.p_target) or not self.zero_p)
        p_defect = self._p_defective[w.traj] and not self.p_target
        if self.zero_p or self.p_target or p_defect:
            inp[..., 2] = 0.0
        if self.zero_p or p_defect:
            tgt[..., 2] = 0.0
            p_supervised = p_supervised and not p_defect

        # The organizers' pressure dropout: with probability mask_prob the simulated
        # p channel is zeroed, so the model sees the real data's missing-pressure
        # case during pretraining. A sim2real bridge, not a regulariser.
        if self.mask_prob > 0 and self._rng.random() < self.mask_prob:
            inp[..., 2] = 0.0
            tgt[..., 2] = 0.0
            gauge_p, p_supervised = None, False

        # Track-2 cold-start augmentation: present the model the no-history case it will meet
        # on the first window of every stream trajectory. Applied after the operating point is
        # estimated (which reads the recent frames only), so it changes the input the stem
        # sees and nothing else.
        if self.history_dropout > 0 and self.t_in > T_IN:
            if self._rng.random() < self.history_dropout:
                inp[: self.t_in - T_IN] = 0.0

        # The recent-frame counterpart, and it sits here for the same reason: every operating
        # point estimate has already been taken off the clean recent frames, so this changes
        # what the stem sees and nothing else -- which is what makes it a clean test of the
        # stem's memory rather than of the estimator.
        if self.recent_dropout > 0 and self._rng.random() < self.recent_dropout:
            k = int(self._rng.integers(1, self.recent_dropout_max + 1))
            k = min(k, self.t_in - 1)  # never blank the whole window
            if k > 0:
                inp[self.t_in - k :] = inp.mean(axis=0, keepdims=True)

        # Their multiplicative noise, mimicking PIV measurement noise on clean CFD.
        if self.noise_scale > 0:
            inp = (
                inp
                + inp * self._rng.standard_normal(inp.shape).astype(np.float32) * self.noise_scale
            )
            tgt = (
                tgt
                + tgt * self._rng.standard_normal(tgt.shape).astype(np.float32) * self.noise_scale
            )

        if self.normalizer == "gaussian":
            assert self.stats is not None
            inp, tgt = self.stats.encode(inp, tgt)
            scale = 1.0
        elif self.normalizer == "freestream" and self.normalize:
            # Estimated from the input window alone -- exactly what is available at
            # inference time.
            scale = max(float(freestream_speed(inp[-T_IN:])), 1e-6)
            inp, tgt = inp / scale, tgt / scale
        else:
            scale = 1.0

        if gauge_p is not None:
            # ``tgt`` has already been divided by ``scale`` once with the velocities, so this
            # overwrite -- not a second division -- is what makes the exponent 2.
            tgt[..., 2] = gauge_p / np.float32(scale) ** 2

        if fine_block is not None:
            fine_block = fine_block[self.t_in :] / np.float32(scale)
            if self.zero_p or self._p_defective[w.traj]:
                fine_block[..., 2] = 0.0

        # Applied *after* normalisation, and that placement is the whole correctness of the
        # "input only" claim: the freestream divisor is estimated from the input window, so
        # noising before this point perturbs u_fs and rescales the target as well. Measured
        # -- a first version did exactly that and the target came back changed.
        #
        # Multiplicative, so exact zeros stay zero and the PIV holes are not filled with
        # noise the model would then have to learn to ignore.
        #
        # ``noise_level`` -- the fraction of ``real_noise_scale`` actually injected -- is
        # computed unconditionally (0 = clean) so every sample carries a defined value on
        # the ``operating_point_noise`` conditioning slot, corrupted or not. Under every
        # other mode the dose stays fixed at ``real_noise_scale``, unchanged from before this
        # mode existed: a constant dose gives a conditioning feature nothing to vary over, so
        # this branch is the one place ``real_noise_scale`` stops being a plain regulariser
        # and starts being a DPOT-style "tell the model how noisy its input is" dose.
        noise_level = 0.0
        if self.real_noise_scale > 0:
            dose = (
                self._rng.uniform(0.0, self.real_noise_scale)
                if self.cond_mode == "operating_point_noise"
                else self.real_noise_scale
            )
            inp = inp + inp * self._rng.standard_normal(inp.shape).astype(np.float32) * dose
            if self.cond_mode == "operating_point_noise":
                noise_level = dose / self.real_noise_scale

        # PIV noise. Applied to a normalised field, where the level is a fraction of a
        # freestream of 1 -- see data/piv_noise.py for why that is the same number either
        # side of the division. Input and target carry independent draws and independent
        # multipliers: (noise, clean) is a denoising objective and asks the model to invert
        # the measurement, (noise, noise) is the test-time task. The *clean* input is kept
        # for the reconstruction target below, so a masked-modelling arm reconstructs the
        # true field rather than the corruption.
        sigma = piv_sigma_rel(c.re)
        clean_inp = inp
        if self.piv_noise > 0:
            inp = add_piv_noise(inp, self._rng, sigma * self.piv_noise)
        if self.piv_noise_target > 0:
            tgt = add_piv_noise(tgt, self._rng, sigma * self.piv_noise_target)

        # Masked spatio-temporal modelling. Zero-fill plus a visibility plane: a hidden
        # cell and a PIV-masked cell are both zero, so without the plane the model cannot
        # tell "this is unknown to you" from "there is nothing here", and hedges over both.
        aux_mask = np.zeros(inp.shape[:3], dtype=np.float32)
        if self.aux_mask_mode:
            if self.aux_mask_ratio > 0:
                hidden = sample_input_mask(
                    self._rng, self.aux_mask_mode, self.aux_mask_ratio, *inp.shape[:3]
                )
                inp = inp.copy()
                inp[hidden] = 0.0
                aux_mask = hidden.astype(np.float32)
            # Appended whether or not anything was hidden, so a masked-modelling model can
            # be validated -- and shipped -- on unmasked input without a shape mismatch.
            # ``ratio=0`` is therefore both the evaluation setting and the ablation that
            # isolates the extra channel from the masking it exists to describe.
            if self.aux_mask_plane:
                inp = np.concatenate([inp, (1.0 - aux_mask)[..., None]], axis=-1)

        if self.geom_channels:
            # ``aoa_est`` came from the *raw* window, above: ``wake_depth`` is a ratio and
            # so survives freestream scaling, but the gaussian normalizer is affine and
            # would shift it. The planes are appended *after* normalisation because they
            # are dimensionless geometry -- dividing an SDF by a freestream speed would be
            # meaningless.
            planes = geometry_channels(aoa_est, self.geom_channels, inp.shape[1], inp.shape[2])
            inp = np.concatenate([inp, np.broadcast_to(planes, (inp.shape[0], *planes.shape))], -1)

        if self.flow_channels:
            # LAST, and after every other plane, because that is the order
            # ``models.predictor`` rebuilds at inference and the order the stem-widening
            # surgery assumes. Computed from the data channels only (``inp[..., :C]``), so
            # appending a visibility or geometry plane before this cannot change it.
            inp = np.concatenate([inp, flow_channels(inp[..., :C], self.flow_channels)], axis=-1)

        if self.stream_state:
            # LAST: the guards in __init__ ensure no extra channel planes exist, every
            # estimate above read the raw recent frames, and ``scale`` is already the
            # window's freestream divisor. Prepended on the frame axis, so the model's
            # stem reads t_in + 2 frames and ``widen_stem_history`` places these on the
            # zero-initialised older slots.
            if self.normalizer == "freestream" and self.normalize:
                div, n_mul = scale, 1.0
            else:
                # normalizer "none": physical windows for make_predict, which divides every
                # frame by the freestream it estimates from the LAST 20 -- so the state rides
                # raw and the n-plane pre-multiplied (see _state_frames).
                div, n_mul = 1.0, max(float(freestream_speed(inp[-T_IN:])), 1e-6)
            # Cold-state dropout: sometimes present the state as it stood after only the first
            # k reveals, so the model meets the noisy early-stream state and its small count
            # plane. Drawn here (not in _state_frames) so the memoised statistics stay keyed on
            # the reveal count and a cold state is shared with the real window at that count.
            n_rev_state = None
            if self.state_dropout > 0 and self._rng.random() < self.state_dropout:
                true_k = max(T_IN, (w.start + T_IN) // T_IN * T_IN) // T_IN
                hi = min(self.state_dropout_max, true_k)
                n_rev_state = int(self._rng.integers(1, hi + 1)) * T_IN
            inp = np.concatenate(
                [self._state_frames(w, div, n_mul, inp.shape[-1], n_rev=n_rev_state), inp], axis=0
            )

        item = {
            "input": torch.from_numpy(np.ascontiguousarray(inp)),
            "target": torch.from_numpy(np.ascontiguousarray(tgt)),
            "cond": torch.from_numpy(
                conditioning_features(
                    re_cond,
                    aoa_cond,
                    self.base,
                    tau,
                    noise_level if self.cond_mode == "operating_point_noise" else None,
                    dt_cond_value(ratio) if self.cond_mode == "operating_point_dt" else None,
                )
            ),
            "valid": torch.from_numpy(~invalid).to(torch.float32),
            "u_fs": torch.tensor(scale, dtype=torch.float32),
            "has_p": torch.tensor(1.0 if p_supervised else 0.0, dtype=torch.float32),
            "traj": torch.tensor(w.traj, dtype=torch.long),
            # The cell's LABEL Reynolds number, for training-side doses only (never conditioning).
            "re": torch.tensor(float(c.re), dtype=torch.float32),
            # Its label angle: ``nbr_prior`` excludes the cell itself from its prior in training.
            "aoa": torch.tensor(float(c.aoa), dtype=torch.float32),
        }
        if self.aux_mask_mode:
            item["aux_mask"] = torch.from_numpy(aux_mask)
            item["recon_target"] = torch.from_numpy(np.ascontiguousarray(clean_inp))
        if fine_block is not None:
            item["target_fine"] = torch.from_numpy(np.ascontiguousarray(fine_block))
        if self.aux_p:
            item["target_p"] = torch.from_numpy(
                np.ascontiguousarray(
                    (raw_p / np.float32(scale))
                    if raw_p is not None
                    else np.zeros_like(tgt[..., :1])
                )
            )
            item["has_p_aux"] = torch.tensor(0.0 if raw_p is None else 1.0, dtype=torch.float32)
        return item


def _sim_dataset(split: str, stride: int = 1, kind: Domain = "sim", **kwargs) -> WindowDataset:
    """Dataset for one of the simulated splits (see ``splits.SIM_SPLITS``).

    Mirrors the real branch of :func:`make_dataset` exactly -- whole Reynolds rows held
    out, a temporal tail reserved on a stratified subset of the training cells -- so a
    number measured here answers the same question its real counterpart does.

    ``kind`` selects which simulated grid: ``sim`` or the time-resampled ``sim_resampled``.
    The cell lists are identical, so the two are directly comparable.
    """
    splits = assign_sim_splits()
    cells = [(parse_params(p).re, parse_params(p).aoa) for p in splits[SIM_SPLITS[split]]]
    reserved = set(sim_val_time_cells())
    head, tail = (0.0, 1.0 - VAL_TIME_TAIL), (1.0 - VAL_TIME_TAIL, 1.0)

    if split == "sim_train":
        frame_range: tuple[float, float] | dict[tuple[int, int], tuple[float, float]] = {
            c: head if c in reserved else (0.0, 1.0) for c in cells
        }
    elif split == "sim_val_time":
        cells = [c for c in cells if c in reserved]
        frame_range = tail
    else:
        frame_range = (0.0, 1.0)
    return WindowDataset(kind, trajectories=cells, stride=stride, frame_range=frame_range, **kwargs)


def split_cells(split: str) -> list[tuple[int, int]]:
    """``(Re, AoA)`` cells belonging to a named split (see ``data/splits.py``)."""
    splits = assign_splits()
    key = "train" if split == "val_time" else split
    if key not in splits:
        raise KeyError(f"Unknown split {split!r}; have {sorted([*splits, 'val_time'])}")
    return [(parse_params(p).re, parse_params(p).aoa) for p in splits[key]]


def make_dataset(
    split: str,
    kind: Domain = "real",
    stride: int = 1,
    **kwargs,
) -> WindowDataset:
    """Build the dataset for a named split.

    ``train`` uses the leading frames of the training trajectories and ``val_time``
    the withheld tail, so the two never share a frame. ``val_interp`` / ``val_extrap``
    are whole held-out Re rows and use every frame.

    For ``kind='sim'`` the split selects *matching cells*; cells with no real
    counterpart are included only for ``split='sim_all'``, which is what pretraining
    should use since the simulated grid covers the withheld operating points.

    The ``sim_*`` splits (``splits.SIM_SPLITS``) are the sim grid's own held-out
    protocol and imply a simulated ``kind``: passing ``kind='real'`` with one of them
    selects plain ``sim``, while ``kind='sim_resampled'`` is honoured, so a pretrain and
    its validation stay on the same clock.
    """
    sim_kind: Domain = kind if base_kind(kind) in SIM_DOMAINS else "sim"

    if split == "sim_all":
        return WindowDataset(sim_kind, trajectories=None, stride=stride, **kwargs)

    if split == SIM_ALL_BUT_REAL_VAL:
        # The grid minus real's validation Reynolds rows, every frame: the "never seen the
        # operating point in either domain" arm of the sim-coverage question (splits.py).
        cells = [(parse_params(p).re, parse_params(p).aoa) for p in sim_all_but_real_val_paths()]
        return WindowDataset(sim_kind, trajectories=cells, stride=stride, **kwargs)

    if split == SIM_HIFI:
        # The grid minus the low-incidence, high-Reynolds corner where the measured
        # sim-to-real fluctuation ratio is worst (splits.SIM_HIFI): 68 cells of 100.
        cells = [(parse_params(p).re, parse_params(p).aoa) for p in sim_hifi_paths()]
        return WindowDataset(sim_kind, trajectories=cells, stride=stride, **kwargs)

    if split == SIM_ONLY_ROWS:
        # The grid minus the Reynolds rows real does not have: what the bracketing rungs at
        # 15225 and 27975 are worth, which FINDINGS 140 kept in every arm (splits.py).
        cells = [
            (parse_params(p).re, parse_params(p).aoa) for p in sim_all_but_sim_only_rows_paths()
        ]
        return WindowDataset(sim_kind, trajectories=cells, stride=stride, **kwargs)

    if split in SIM_SPLITS:
        return _sim_dataset(split, stride=stride, kind=sim_kind, **kwargs)

    if split == "all":
        # Reproduction only: no trajectory is held out (see splits.all_cells).
        cells = [(parse_params(p).re, parse_params(p).aoa) for p in all_cells()]
        return WindowDataset(kind, trajectories=cells, stride=stride, **kwargs)

    if split.startswith("all_but_"):
        # Every cell except one named split's, which stays genuinely unseen. This exists
        # because FINDINGS 64c makes it mandatory for a whole class of comparison: an
        # ablation run on ``all`` has no honest local column, and a contaminated delta
        # inverts the sign of any change that interacts with in-sample fit -- capacity,
        # data volume, regularisation, or a correction that closes an out-of-sample
        # deficit. ``train`` (69 cells) is the alternative, but it gives up 12 of 81
        # cells and so cannot speak for a chain fitted on all of them. ``all_but_extrap``
        # gives up 7 and answers the same question at 91% of the shipped data volume.
        held = set(split_cells(split[len("all_but_") :]))
        cells = [
            c
            for c in ((parse_params(p).re, parse_params(p).aoa) for p in all_cells())
            if c not in held
        ]
        return WindowDataset(kind, trajectories=cells, stride=stride, **kwargs)

    aoa_paths = aoa_split_paths(split)
    if aoa_paths is not None:
        # Name parsed by `splits.aoa_split_paths`, which the eval-set builder also calls, so a
        # fold cannot mean one thing in training and another in scoring.
        cells = [(parse_params(p).re, parse_params(p).aoa) for p in aoa_paths]
        return WindowDataset(kind, trajectories=cells, stride=stride, **kwargs)

    reserved = set(val_time_cells())
    head, tail = (0.0, 1.0 - VAL_TIME_TAIL), (1.0 - VAL_TIME_TAIL, 1.0)

    if split == "train":
        # Only the reserved cells give up their tail; the rest train on full
        # trajectories. Holding it on all 69 cost 26% of training windows.
        cells = split_cells(split)
        frame_range = {c: head if c in reserved else (0.0, 1.0) for c in cells}
    elif split == "all_train_frames":
        # Every cell, with the reserved tails still withheld: what ``train`` is to the 69
        # training cells, this is to all 81.
        #
        # It exists for the POST-HOC HEADS of a shipping rebuild. There the backbone has
        # already trained on all 81, so withholding 12 cells from a head's fit buys no
        # honesty -- it only costs the head the error maps of the hardest operating points
        # (every ``hard5`` cell lives in those 12). But fitting a head on ``all`` instead
        # would swallow ``val_time``'s reserved frames, and ``train tke-head`` chooses its
        # stopping step on that split: the stopping point would then be picked on data the
        # head had just minimised on. This split gives the head every cell while keeping
        # those frames genuinely unseen, so the early stopping stays honest.
        cells = [(parse_params(p).re, parse_params(p).aoa) for p in all_cells()]
        frame_range = {c: head if c in reserved else (0.0, 1.0) for c in cells}
    elif split == "val_time":
        cells = [c for c in split_cells(split) if c in reserved]
        frame_range = tail
    else:
        cells = split_cells(split)
        frame_range = (0.0, 1.0)
    return WindowDataset(kind, trajectories=cells, stride=stride, frame_range=frame_range, **kwargs)
