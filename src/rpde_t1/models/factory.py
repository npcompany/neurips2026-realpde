"""Construct a model from its config -- the single builder, for training and inference.

``train.loop`` cannot own this: it imports ``rpde_t1.data``, whose package ``__init__``
pulls in ``h5py``, which the evaluation container does not ship. So construction lives
here, on the submission-safe side of that line, and ``train.loop.build_model`` calls in.
It used to be a second implementation that "mirrored" this one, and the mirror had
already cracked: only the training copy widened the input for the geometry planes, so a
``geom_channels`` checkpoint could be trained but not rebuilt.

This module may import **torch, numpy, the vendored starting kit, and the parts of
``rpde_t1`` that ship in a bundle** (``submit.package.PACKAGE_PAYLOAD``: ``features``,
``geometry``, ``models``) — nothing else. ``tests/test_packaging.py`` enforces the
container package set over the whole inference path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as _np
import torch
from torch import nn

from .. import N_MEASURED, n_mask_planes
from ..features import cond_dim as cond_vec_dim
from ..features import n_flow_channels, parse_flow_channels
from ..geometry import n_geometry_channels, parse_geom_channels
from .aux import AuxModel, aux_weights
from .bounds import (
    BoundedModel,
    BoundHead,
    head_in_channels,
)
from .wrappers import (
    NBR_PRIOR_MODES,
    BaselineAdapter,
    DampedResidualWrapper,
    NbrPriorWrapper,
    ResidualWrapper,
    StateAnchorWrapper,
    StateNormWrapper,
    StateRefineWrapper,
)

BASELINES = ("cno", "fno", "transolver")

#: Architectures implemented in this package. ``train.loop`` re-exports both tuples
#: rather than declaring its own, because a model that trains but cannot be rebuilt for
#: inference is a checkpoint nobody can score.
OURS = (
    "transformer",
    "modpct",
    "mocpt",
    "stformer",
    "stformer3d",
)


def parse_arch(spec: str) -> dict[str, Any]:
    """``"dim=512,depth=10,patch=20x2x2"`` -> keyword arguments for a model constructor.

    One config field instead of one per hyper-parameter, because the alternative is a
    ``TrainConfig`` that grows a column every time an architecture gains a knob -- and
    every one of those columns then has to be defaulted correctly forever so old
    checkpoints still rebuild (``evaluate.POST_HOC_DEFAULTS`` is the standing cost of
    getting that wrong). The spec is stored verbatim in ``config.json``, so a checkpoint
    still rebuilds from its own directory alone.

    ``int`` / ``float`` / ``bool`` / ``AxBxC`` tuples are recognised; anything else stays
    a string. An unknown key raises from the constructor rather than being ignored,
    which is the failure mode that matters -- a silently dropped ``depth=12`` trains a
    different model than the one the sweep believes it ran.
    """
    kwargs: dict[str, Any] = {}
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        key, _, raw = item.partition("=")
        kwargs[key.strip()] = _coerce(raw.strip())
    return kwargs


def _coerce(raw: str) -> Any:
    if "x" in raw and all(p.lstrip("-").isdigit() for p in raw.split("x")):
        return tuple(int(p) for p in raw.split("x"))
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


#: Scored geometry. Kept local rather than imported from ``rpde_t1`` so this module
#: stands alone if the package is ever trimmed further for a submission.
IN_SHAPE = (20, 32, 64, 3)

#: Pseudo-frames each ``TrainConfig.stream_state`` mode prepends to the input window
#: (runs/t2_backbone/PLAN.md). ONE home for the count: the training loop's model build, the
#: warm-start stem surgery and ``build_for_inference`` all read it, so a checkpoint carrying
#: ``stream_state`` rebuilds at the stem width it was trained at on every path.
STREAM_STATE_FRAMES = {
    "": 0,
    "meanvar": 2,
    "meanvarfast": 4,
    "meanvarperr": 3,
    "meanvarcov": 4,
    # ``meanvar`` plus the causal single-frequency temporal Fourier mode of the revealed
    # fluctuation at the cell's shedding frequency (``shedding_omega``), as a per-(pixel,
    # channel) UNIT PHASOR ``[cos phase, sin phase]``. The two extra planes carry the one thing
    # the level statistics are blind to: the SPATIAL PHASE of the von-Karman wake -- where each
    # pixel sits in the shedding cycle and, through the phase gradient, which way and how fast
    # vortices convect. Mean/std collapse the oscillation into an envelope; this keeps its
    # phase. Scale-free by construction (the per-channel freestream divisor cancels in the unit
    # phasor), so it is NOT divided by ``scale`` like the level planes. FINDINGS 193a/210 closed
    # the level-statistic ladder; this is the untested phase axis (STATUS clause 47's reopen
    # bar: "a mechanism that reaches something the level statistics cannot express").
    "meanvarphase": 4,
}

#: Reveals the ``meanvarfast`` "recent" statistics average over. 5 windows = 100 frames, which
#: is 11-90 integral timescales (FINDINGS 3) -- long enough to be a statistic rather than a
#: realisation, short enough to differ from the lifetime mean when a trajectory drifts. Lives
#: here, not in ``data.dataset``, because the serving port needs it and cannot import h5py.
STREAM_STATE_FAST_WINDOWS = 5

#: Config flags whose wrappers add ONE nesting level on the input side, outside the residual
#: anchor. ``train.loop`` counts these when re-keying an ``--init-from`` checkpoint: each one
#: the new run adds over the source needs another ``backbone.`` prefix, or every tensor reports
#: missing and the fine-tune silently trains from scratch. Declared here, in one place, because
#: hard-coding the list at the use site reproduced that bug on each of the first three wrappers.
STATE_WRAPPER_FLAGS = ("state_norm", "state_anchor", "state_planes", "nbr_prior")


def covariance_stats_from_sums(sums: dict) -> dict[str, _np.ndarray]:
    """``meanvarcov`` statistics from running sums: frame count ``n`` (lag pairs are ``n - 1``,
    the history being contiguous), ``s``/``sq``/``lag`` of shape ``(H, W, 2)`` and ``uv``/``w2``
    of shape ``(H, W)``."""
    n = float(sums["n"])
    m = sums["s"] / n
    var = _np.maximum(sums["sq"] / n - m * m, 0.0)
    cov = sums["uv"] / n - m[..., 0] * m[..., 1]
    if n > 1:
        rho = (sums["lag"] / (n - 1.0) - m * m) / _np.where(var > STATE_VAR_FLOOR, var, 1.0)
        rho = _np.where(var > STATE_VAR_FLOOR, _np.clip(rho, -1.0, 1.0), 0.0)
    else:
        rho = _np.zeros_like(m)
    return {
        "cov_ssqrt": _np.sign(cov) * _np.sqrt(_np.abs(cov)),
        "vort_rms": _np.sqrt(_np.maximum(sums["w2"] / n, 0.0)),
        "rho1": rho,
    }


#: Temporal variance (physical units^2) below which a pixel is treated as having none -- the
#: airfoil body and invalid-PIV pixels are exactly 0 in every frame.
STATE_VAR_FLOOR = 1e-10

# -- ``meanvarphase`` online shedding-mode estimator ------------------------------------------
#
# The calibration (scripts/gen/shedding_calib.py, kept for provenance) established that the wake
# has a SHARP shedding tone per trajectory (peak/median spectral power ~1e3-1e4) but that its
# frequency is NOT a function of Re alone -- it depends on (Re, AoA) partly erratically -- so a
# fixed per-cell table cannot serve the HELD-OUT Track-2 operating points. The frequency is
# therefore ESTIMATED ONLINE per trajectory from the revealed history, identically at train and
# serve. Because the tone is sharp, the estimator is unambiguous even from a few hundred frames.

#: Fewest revealed frames before a shedding mode is estimated at all; under this the two planes
#: are zero (a stream's first window cannot resolve a period). Two 20-frame reveals.
PHASE_MIN_FRAMES = 40
#: Lowest resolvable cycle count -- ignore periodogram bins below this many full cycles over the
#: revealed span, where a "peak" is drift or leakage rather than shedding.
PHASE_MIN_CYCLES = 2
#: Serving-side cap on the retained revealed frames (the online recompute keeps the raw history).
#: Longer than any real trajectory (~868 frames); a stream past this freezes its history.
PHASE_MAX_FRAMES = 1024


def _peak_omega_np(hist: _np.ndarray) -> float:
    """Dominant temporal angular frequency (rad/frame) of a history ``(n, H, W, 2)``.

    Estimated from the TRANSVERSE channel (``v``) only, energy-weighted over the per-pixel power
    spectra (positive, so the antisymmetric von-Karman street does NOT cancel as a spatial mean
    would), peak bin refined by parabolic interpolation. Single channel on purpose: its one
    freestream scale then cancels top-and-bottom, so the estimate is identical in evaluator and
    physical space -- the property the serving parity relies on. Identical to ``_peak_omega_torch``.
    """
    v = _np.asarray(hist[..., 1], dtype=_np.float64)  # (n, H, W) transverse velocity
    n = v.shape[0]
    v = v - v.mean(axis=0, keepdims=True)
    weight = v.var(axis=0)
    wsum = float(weight.sum())
    if wsum <= 0.0:
        return 0.0
    spec = _np.abs(_np.fft.rfft(v, axis=0)) ** 2  # (F, H, W)
    power = (spec * weight[None]).reshape(spec.shape[0], -1).sum(axis=1) / wsum
    power[: max(1, PHASE_MIN_CYCLES)] = 0.0
    # The exact DFT bin, no sub-bin interpolation: over the length-n span the mode is then the
    # k-th DFT coefficient (maximally coherent, zero leakage), and the estimate is a pure argmax
    # -- identical on the serving side's torch FFT, which sub-bin refinement is too ill-
    # conditioned to guarantee at the broad low-frame-count peaks.
    return float(2.0 * _np.pi * int(_np.argmax(power)) / n)


def phase_planes_np(hist: _np.ndarray) -> _np.ndarray:
    """The two ``meanvarphase`` planes ``(2, H, W, 2)`` from a revealed history ``(n, H, W, 2)``.

    The demeaned single-frequency temporal Fourier mode at the estimated shedding frequency,
    normalised by the pixel's fluctuation energy ``n * sd`` (bounded to [-1, 1] by
    Cauchy-Schwarz): the re/im ratio is the shedding PHASE, the magnitude its COHERENCE. Zero
    below ``PHASE_MIN_FRAMES`` and at pixels with no temporal variance. Dimensionless and
    scale-free, so it equals the serving port's evaluator-space value.
    """
    h = _np.asarray(hist, dtype=_np.float64)
    n = h.shape[0]
    hh, ww = h.shape[1:3]
    if n < PHASE_MIN_FRAMES:
        return _np.zeros((2, hh, ww, 2), dtype=_np.float32)
    om = _peak_omega_np(h)
    t = _np.arange(n, dtype=_np.float64)
    cos, sin = _np.cos(om * t), _np.sin(om * t)
    m = h.mean(axis=0)
    phi_re = _np.tensordot(cos, h, axes=(0, 0)) - m * cos.sum()
    phi_im = _np.tensordot(sin, h, axes=(0, 0)) - m * sin.sum()
    sd = h.std(axis=0)
    # Gate strictly on ``sd > 0``: a dead pixel (body, invalid PIV) is EXACTLY zero in every
    # frame, so its std is exactly 0 in both physical and evaluator space; every live pixel is
    # strictly positive in both. A nonzero threshold would sit in the range where the per-channel
    # freestream scale flips borderline pixels between the two spaces (train/serve skew). The
    # normalised value itself is exactly scale-invariant, so a passed pixel agrees to precision.
    ok = sd > 0.0
    inv = _np.where(ok, 1.0 / (n * _np.where(ok, sd, 1.0)), 0.0)
    return _np.stack([phi_re * inv, phi_im * inv]).astype(_np.float32)


def _peak_omega_torch(hist):  # hist: (n, H, W, 2) float64 tensor -> python float
    import torch as _t  # noqa: PLC0415

    v = hist[..., 1]  # (n, H, W) transverse channel only -- scale-invariant peak (see np twin)
    v = v - v.mean(dim=0, keepdim=True)
    weight = v.var(dim=0, unbiased=False)  # population variance, to match numpy's ddof=0
    wsum = float(weight.sum())
    if wsum <= 0.0:
        return 0.0
    spec = _t.fft.rfft(v, dim=0).abs() ** 2  # (F, H, W)
    power = (spec * weight[None]).reshape(spec.shape[0], -1).sum(dim=1) / wsum
    power[: max(1, PHASE_MIN_CYCLES)] = 0.0
    k = int(_t.argmax(power).item())  # exact DFT bin, no sub-bin refinement (see the np twin)
    import math as _m  # noqa: PLC0415

    return float(2.0 * _m.pi * k / hist.shape[0])


def phase_planes_torch(hist):
    """Torch mirror of :func:`phase_planes_np` for the GPU-native serving path.

    ``hist`` is the revealed history ``(n, H, W, 2)`` in evaluator space; the result is scale-
    free, so it matches the dataset's physical-space planes. Returns a ``(2, H, W, 2)`` tensor.
    """
    import torch as _t  # noqa: PLC0415

    h = hist.double()
    n, hh, ww = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    if n < PHASE_MIN_FRAMES:
        return _t.zeros((2, hh, ww, 2), device=hist.device, dtype=hist.dtype)
    om = _peak_omega_torch(h)
    t = _t.arange(n, device=h.device, dtype=_t.float64)
    cos, sin = _t.cos(om * t), _t.sin(om * t)
    cv = cos.view(n, 1, 1, 1)
    sv = sin.view(n, 1, 1, 1)
    m = h.mean(dim=0)
    phi_re = (cv * h).sum(0) - m * cos.sum()
    phi_im = (sv * h).sum(0) - m * sin.sum()
    sd = h.std(dim=0, unbiased=False)  # population std (ddof=0), to match numpy
    ok = sd > 0.0  # strictly -- see the np twin: the only scale-consistent gate is exact zero
    inv = _t.where(ok, 1.0 / (n * _t.where(ok, sd, _t.ones_like(sd))), _t.zeros_like(sd))
    return _t.stack([phi_re * inv, phi_im * inv]).to(hist.dtype)


def stream_state_frames(value: str | None) -> int:
    """Extra input frames ``stream_state`` adds; 0 for unset/older checkpoints."""
    return STREAM_STATE_FRAMES[value or ""]


def backbone_state_frames(config: dict[str, Any]) -> int:
    """State frames the BACKBONE's stem must accept.

    Zero under ``state_norm``: there the state is consumed by ``StateNormWrapper`` as the
    normaliser and never reaches the stem, so the backbone stays a plain 20-frame model.
    """
    if config.get("state_norm", False) or config.get("state_planes", False):
        return 0
    return stream_state_frames(config.get("stream_state"))


OUT_SHAPE = (20, 32, 64, 3)


def _kit_on_path(kit_dir: str | Path | None) -> None:
    """Put the vendored kit on ``sys.path`` so ``load_baseline`` resolves.

    Inside a submission the kit sits beside ``submission.py`` at the archive root, so
    the default of "this file's directory, four levels up" is wrong there and the
    caller passes an explicit directory.
    """
    if kit_dir is None:
        return
    d = str(Path(kit_dir).resolve())
    if d not in sys.path:
        sys.path.insert(0, d)


def geom_channels_of(config: dict[str, Any]) -> tuple[str, ...]:
    """Geometry channel groups ``config`` asks for."""
    return parse_geom_channels(config.get("geom_channels", ""))


def n_cond_planes(config: dict[str, Any]) -> int:
    """Width the conditioning vector adds to the input, when it is fed as planes.

    Non-zero only for the vendored baselines with ``cond_planes`` on: they have no
    conditioning slot and ``vendor/`` is never edited, so the operating point reaches them
    as constant input channels instead (see :class:`~..models.wrappers.BaselineAdapter`).
    ``cond_mode="none"`` contributes nothing, because there is no operating point to feed.

    Our own models are refused rather than widened. They already take ``cond`` directly and
    nothing appends these planes to their *data*, so enabling it there would size a stem for
    channels the dataset never produces -- a shape error at the first batch if you are lucky
    and, with a wrapper in the way, a confusing one.
    """
    if not config.get("cond_planes", False):
        return 0
    if config.get("model", "cno") not in BASELINES:
        raise ValueError(
            f"cond_planes is for the vendored baselines {BASELINES}, which have no "
            f"conditioning input; {config.get('model')!r} takes cond directly."
        )
    cond_mode = config.get("cond_mode", "none")
    return 0 if cond_mode == "none" else cond_vec_dim(cond_mode)


def flow_channels_of(config: dict[str, Any]) -> tuple[str, ...]:
    """Derived flow-plane groups ``config`` asks for (``features.FLOW_CHANNELS``)."""
    return parse_flow_channels(config.get("flow_channels", ""))


def input_shape_with_geometry(
    config: dict[str, Any], in_shape: tuple[int, ...] = IN_SHAPE
) -> tuple[int, ...]:
    """``in_shape`` widened by every plane ``config`` appends to the input.

    Three sources: the geometry planes, the visibility plane a masked-modelling run adds
    (``data/masking.py``), and the flow planes derived from the window itself
    (``features.FLOW_CHANNELS``). Only the input widens -- the target is still the three
    scored channels, so ``out_shape`` never moves.

    All three are counted here rather than at each call site because this is the number that
    sizes the first convolution, and a checkpoint built against a different count does not
    load. That is the failure this function exists to make impossible.
    """
    n_extra = (
        2 * stream_state_frames(config.get("stream_state"))
        * int(bool(config.get("state_planes", False)))
        + n_geometry_channels(geom_channels_of(config))
        + n_mask_planes(config.get("aux_mask", ""), config.get("aux_mask_plane", True))
        + n_flow_channels(flow_channels_of(config))
        + n_cond_planes(config)
        + (
            NBR_PRIOR_MODES[config.get("nbr_prior_mode", "all")]
            if config.get("nbr_prior", False)
            else 0
        )
    )
    return (*in_shape[:-1], in_shape[-1] + n_extra) if n_extra else tuple(in_shape)


def build_backbone(
    config: dict[str, Any],
    kit_dir: str | Path | None = None,
    in_shape: tuple[int, ...] = IN_SHAPE,
    out_shape: tuple[int, ...] = OUT_SHAPE,
) -> nn.Module:
    """The bare architecture named by ``config``, before residual and bound wrapping.

    Separate from :func:`wrap_model` only so the training loop can inject dropout hooks
    into the backbone alone; every caller that just wants a model wants
    :func:`build_for_inference`.

    The baselines come from the vendored starting kit, untrained, at exactly the
    hyper-parameters read off the released checkpoints -- so a retrained baseline is
    architecturally identical to the published one and the only difference is our
    training pipeline and split.
    """
    name = config.get("model", "cno")
    in_shape = input_shape_with_geometry(config, in_shape)

    if name in BASELINES:
        _kit_on_path(kit_dir)
        from load_baseline import build_model as build_baseline  # noqa: PLC0415

        core = build_baseline(name, device="cpu", in_shape=in_shape, out_shape=out_shape)
        # ``in_shape`` is already widened by ``n_cond_planes`` above, so the baseline's first
        # layer is sized for the planes the adapter will concatenate.
        return BaselineAdapter(core, name=name, cond_planes=n_cond_planes(config))
    if name == "stformer3d":
        from .stformer import STFormer3D  # noqa: PLC0415

        return STFormer3D(
            in_shape=in_shape,
            out_shape=out_shape,
            cond_mode=config.get("cond_mode", "none"),
            **parse_arch(config.get("arch", "")),
        )
    if name == "stformer":
        from .stformer import STFormer  # noqa: PLC0415

        return STFormer(
            in_shape=in_shape,
            out_shape=out_shape,
            cond_mode=config.get("cond_mode", "none"),
            **parse_arch(config.get("arch", "")),
        )
    if name in ("transformer", "mocpt", "modpct"):
        from .transformer import MOCPT, ModPCT, PatchTransformer  # noqa: PLC0415

        cond_mode = config.get("cond_mode", "none")
        if name == "transformer":
            return PatchTransformer(in_shape=in_shape, out_shape=out_shape)
        if name == "modpct":
            return ModPCT(in_shape=in_shape, out_shape=out_shape, cond_mode=cond_mode)
        # Superseded by ``modpct``; kept so its checkpoints still load. See
        # models/transformer.py for why its query decoder was a mistake.
        return MOCPT(in_shape=in_shape, out_shape=out_shape, cond_mode=cond_mode)
    raise ValueError(f"Unknown model {name!r}; expected one of {BASELINES} or {OURS}.")


def wrap_model(model: nn.Module, config: dict[str, Any]) -> nn.Module:
    """Apply the residual anchor, then the auxiliary heads, then the bound head.

    The order is load-bearing: ``state_dict`` keys are prefixed by the nesting, so a
    checkpoint saved under one order cannot be loaded under the other.

    The auxiliary heads sit *outside* the residual anchor and *inside* the bound head. The
    first because they read a trunk activation, which a hook finds at any depth, so nesting
    buys nothing and unwrapping stays a one-line prefix strip
    (``models.aux.strip_aux``). The second because the bound head is trained against a
    frozen backbone and must see exactly the prediction a submission sees.
    """
    if config.get("state_norm", False):
        # OUTERMOST of the input-side wrappers: it must see the state frames and hand the
        # backbone (and every wrapper inside it, the residual anchor included) the
        # standardised 20-frame window, then restore before anything downstream reads the
        # prediction.
        state_norm_pending = True
    if config.get("nbr_prior", False):
        # Innermost: the planes go to the backbone's stem only; the residual anchor and every
        # wrapper outside it see the plain window. FINDINGS 224.
        model = NbrPriorWrapper(
            model,
            n_cells=len(config.get("nbr_cells") or []),
            mode=config.get("nbr_prior_mode", "all"),
        )
    if config.get("residual", False):
        # ``residual_mode`` is absent from every config written before 2026-09-08 and reads
        # back as ``last``, the only behaviour that existed, so old checkpoints rebuild as is.
        mode = config.get("residual_mode", "last")
        if mode == "last":
            model = ResidualWrapper(model, enabled=True)
        elif mode == "damped":
            model = DampedResidualWrapper(model, tau=float(config.get("residual_tau", 3.0)))
        else:
            raise ValueError(f"residual_mode must be 'last' or 'damped', got {mode!r}")

    if config.get("state_planes", False):
        model = StateRefineWrapper(
            model, n_state=stream_state_frames(config.get("stream_state"))
        )

    if config.get("state_anchor", False):
        # Inside the state-frame split and outside the residual anchor it corrects: it reads
        # the state from the full input and shifts the prediction the anchor produced.
        model = StateAnchorWrapper(
            model, n_state=stream_state_frames(config.get("stream_state"))
        )

    if locals().get("state_norm_pending"):
        model = StateNormWrapper(
            model,
            n_state=stream_state_frames(config.get("stream_state")),
            floor=float(config.get("state_norm_floor", 0.05)),
            mode=str(config.get("state_norm_mode", "meanstd")),
        )

    if aux_weights(config):
        model = AuxModel(model, config)

    if config.get("bounds", False):
        # Absent from configs written before the conditioning channels existed, so the
        # default must be False or those checkpoints would build a wider head and fail
        # to load.
        bound_cond = config.get("bound_cond", False)
        # ``bound_logs``/``bound_coords`` default False so every checkpoint written before
        # they existed still builds the exact input width it was trained with.
        bound_logs = config.get("bound_logs", False)
        bound_coords = config.get("bound_coords", False)
        bound_noise = config.get("bound_noise", False)
        bound_grad = config.get("bound_grad", False)
        bound_pool = config.get("bound_pool", False)
        # Default off/0 so every checkpoint written before these existed builds the exact input
        # width and shape it was trained with, as for `bound_logs`/`bound_coords` above.
        bound_fluc = config.get("bound_fluc", False)
        bound_lstd = config.get("bound_lstd", 0)
        bound_film = config.get("bound_film", 0)
        model = BoundedModel(
            model,
            head=BoundHead(
                in_channels=head_in_channels(
                    bound_cond,
                    bound_noise,
                    bound_logs,
                    bound_coords,
                    bound_grad,
                    bound_pool,
                    cond_vec_dim(config.get("cond_mode", "operating_point")),
                    bound_fluc,
                    bound_lstd,
                ),
                width=config.get("bound_width", 16),
                depth=config.get("bound_depth", 3),
                mode=config.get("bound_mode", "offsets"),
                film=bound_film,
                kt=config.get("bound_kt", 3),
                cond_dim=cond_vec_dim(config.get("cond_mode", "operating_point")),
            ),
            freeze_backbone=config.get("freeze_backbone", True),
            state_frames=stream_state_frames(config.get("stream_state")),
            use_cond=bound_cond,
            fluc=bound_fluc,
            lstd=bound_lstd,
            noise=bound_noise,
            logs=bound_logs,
            coords=bound_coords,
            grad=bound_grad,
            pool=bound_pool,
        )
    return model


def build_for_inference(
    config: dict[str, Any],
    kit_dir: str | Path | None = None,
    in_shape: tuple[int, ...] = IN_SHAPE,
    out_shape: tuple[int, ...] = OUT_SHAPE,
) -> nn.Module:
    """Reconstruct the architecture described by a saved training config.

    The input context comes from the config, not from ``in_shape``'s default. A checkpoint
    trained with ``--t-in 40`` carries a stem twice as wide (``loop.widen_stem_history``), and
    before this read every inference path built a 20-frame model and failed to load it -- so a
    long-context checkpoint could be trained and scored through the training loop but never
    served. Only the time axis is taken from the config; a caller that passes a different
    spatial shape or channel count keeps it.
    """
    t_in = int(config.get("t_in", in_shape[0]) or in_shape[0])
    t_in += backbone_state_frames(config)
    if t_in != in_shape[0]:
        in_shape = (t_in, *in_shape[1:])
    if config.get("moment_of"):
        # A moment network (loop.moment_loss): log sigma^2 for u and v, no residual (the training
        # loop forced ``residual`` off in the saved config, so wrap_model adds none).
        out_shape = (*out_shape[:-1], N_MEASURED)
    elif config.get("moment_joint") and out_shape[-1] == IN_SHAPE[-1]:
        # Frames plus log sigma^2 for u and v (loop.compute_losses); the residual anchor is zero
        # on the extra channels and the predictor keeps the frames (``pred[..., :C]``).
        out_shape = (*out_shape[:-1], out_shape[-1] + N_MEASURED)
    return wrap_model(build_backbone(config, kit_dir, in_shape, out_shape), config)


def unpack_fp16(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Restore an fp32 ``state_dict`` from the half-precision packing.

    Mirrors the kit's ``pack_ckpt_fp16.py`` load side. Complex tensors — FNO's
    spectral weights — are stored as real/imag pairs rather than as complex-half,
    which does not serialise reliably across torch versions. Inference still runs in
    fp32; this only shrinks the file so FNO fits the 256 MB cap.
    """
    complex_keys = set(payload.get("complex_keys", ()))
    state: dict[str, torch.Tensor] = {}
    for key, value in payload["state_fp16"].items():
        if key in complex_keys:
            state[key] = torch.view_as_complex(value.float())
        elif torch.is_tensor(value) and value.dtype == torch.float16:
            state[key] = value.float()
        else:
            state[key] = value
    return state


def load_for_inference(
    checkpoint: str | Path,
    kit_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a checkpoint written by the training loop and return ``(model, config)``.

    Accepts both the plain training checkpoint and the fp16-packed form the packager
    produces for oversized models.

    ``weights_only=False`` because our checkpoints carry the config dict alongside the
    tensors. The file comes from our own training run inside the submission zip, not
    from an untrusted source.
    """
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    config = dict(payload.get("config", {}))
    state = unpack_fp16(payload) if payload.get("fp16_packed") else payload["model"]
    model = build_for_inference(config, kit_dir=kit_dir)
    model.load_state_dict(state)
    return model.to(device).eval(), config
