"""Per-element prediction intervals for the SPS subscore.

Why this is the highest-value component right now: with v9's linear mapping,
``sps_score`` spans the full 0-100 and our retrained baselines measure ~20, so this
axis carries roughly +15.9 final points of headroom against TKE's +5.2.

Why a *per-element* interval rather than a tuned constant. The SPS credit is
``P(target inside) * exp(-width / sigma)``, so the optimal width is set by the local
error magnitude -- and that magnitude is wildly non-uniform. Measured on
``val_extrap``: per-element ``|error|`` has mean 0.0059 but standard deviation
0.0136, and the error is concentrated in the wake while the freestream is nearly
exact. A single width is necessarily too wide in the freestream (paying the
exponential penalty for nothing) and too narrow in the wake (scoring zero because
the target falls outside). Sweeping the best constant on ``val_extrap`` confirmed it
saturates: the optimum was flat between 0.005 and 0.007 and worth only +0.04.

The head therefore predicts, per element and per channel, how far the prediction is
likely to be wrong -- with lead time as an explicit input, because predictability
decays across the horizon (from ~86% of the variance explained at lead 1 to ~33% at
lead 20).

Bounds are asymmetric (a separate offset above and below), since nothing in the
scorer requires symmetry and the error distribution is skewed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .. import N_MEASURED
from ..features import COND_DIM
from .interval import SIGMA_GLOBAL, torch_optimal_half_width_factor
from .wrappers import split_output

#: Clamp on the predicted log half-width, in non-dimensional units. The lower end
#: stops the head collapsing an interval to zero -- which the scorer punishes
#: absolutely, since a target outside the interval earns no credit at all, however
#: accurate the prediction was.
LOG_WIDTH_MIN, LOG_WIDTH_MAX = -7.0, 1.0

#: Initial half-width, chosen so the head starts near the measured error scale
#: rather than at an arbitrary point.
INIT_HALF_WIDTH = 0.05

#: Clamp on the *signed* lower offset of a ``quantile`` head, in non-dimensional units. The
#: fields are divided by the freestream speed, so 2 is far outside any real residual; the clamp
#: is a guard against a diverging fit, not a prior.
QUANTILE_SHIFT_MAX = 2.0

#: Clamp on the predicted log *spread* of the log-error, for ``logerr_tau`` heads.
#: Wider than the measured range on both sides: the point of the second output is to let the
#: spread vary per element, so the clamp is a guard against a diverging fit and not a prior.
LOG_TAU_MIN, LOG_TAU_MAX = -3.0, 1.5

#: Initial ``tau``, and the only number in this file taken from a measurement of the *residual*
#: rather than of the error itself. A predictor that knew the conditional spread exactly would
#: still leave ``sd(log(|err|/s)) = pi/sqrt(8) = 1.1107``; the shipped head measures 1.1564 on
#: ``val_extrap`` (FINDINGS 35). If the excess is scale mixing, the per-element log-scale spread
#: is ``sqrt(1.1564**2 - 1.1107**2) = 0.32`` -- which is both the sensible starting point and,
#: since only its predictable part can be earned, the whole budget of a ``tau`` axis.
INIT_TAU = 0.32


#: Channels built from the prediction and input window alone, before ``cond``.
BASE_FEATURE_CHANNELS = 10

#: Extra channels when ``logs=True``: the log-magnitudes of the three physical scale
#: features (departure from persistence, predicted spread, observed spread), per measured
#: channel.
#:
#: Why a log of something the head already has. The structure that predicts the local error
#: is *multiplicative* -- a five-feature least squares on log-magnitudes reaches SPS 49.61
#: on ``val_extrap`` against the trained 14k-parameter head's 48.71, with six parameters
#: (FINDINGS 14). A ReLU/GELU conv stack on linear magnitudes has to approximate a
#: log-additive function over three decades of dynamic range, which is the one thing it is
#: badly suited to; handing it the logs directly removes the approximation rather than
#: adding information. The features are ratios of non-dimensional quantities, so they carry
#: no freestream scale -- which matters, because a fitted coefficient on ``log u_fs``
#: extrapolates the *level* of the width to unseen Reynolds numbers and cost 11 SPS when
#: it was tried (FINDINGS 14).
LOG_FEATURE_CHANNELS = 3 * N_MEASURED

#: Floor inside the log features, in non-dimensional units. Well below the smallest
#: meaningful spread, so it only regularises exact zeros (the PIV measurement hole).
LOG_EPS = 1e-6

#: Extra channels when ``coords=True``: normalised row and column.
#:
#: The head is a stack of padded 3x3x3 convolutions, so it has no access to absolute
#: position -- it can only infer "this is the wake" from the flow features in a 7x7x7
#: neighbourhood. But the error field is strongly *positional*: a fitted climatology of
#: ``|err|`` per (lead, row, col, channel) ranks the error at Spearman 0.63 on
#: ``val_extrap`` knowing nothing about the individual window. Two coordinate planes let
#: the head represent that itself, and unlike a fitted climatology table they carry no
#: parameters and cannot go stale against a retrained backbone -- which matters because the
#: organizers retrain the shortlisted pipelines.
COORD_FEATURE_CHANNELS = 2

#: Spatial-derivative channels: ``log|grad pred|^2`` and ``log|grad x_last|^2``.
#:
#: Added 2026-08-23 (FINDINGS 72g). The head's other features are all *levels* -- the
#: prediction, its departure from persistence, two temporal spreads -- and none of them says
#: where the field is STEEP. Error concentrates at shear layers and the wake edge, which is a
#: spatial-derivative property, and a depth-2 3x3x3 conv has to spend hidden channels
#: reconstructing a gradient it can be handed for four input channels. That matters because
#: 72c measured this head to be capacity-bound: width 32 -> 64 was worth +0.272 sps on its own.
#:
#: Two channels and not four, and in log-squared form, because the head is in the inference
#: critical path: the raw-plus-log four-channel version measured +0.112 sps for +0.240
#: ms/window, which is -0.015 final against +0.015 gained -- an exact wash. Dropping the raw
#: pair and the ``sqrt`` keeps the information (the other scale features are the ones where
#: ``logs`` matters most: -0.29 sps without them at width 64) and halves the added cost.
GRAD_FEATURE_CHANNELS = 2

#: Pooled-context channels: the log of four window-level reductions of the two scale features,
#: broadcast back to every element.
#:
#: Added 2026-08-23 (FINDINGS 72h). ``scripts/eval/g_eff_ceiling.py`` measures the exact efficiency
#: bound for this backbone's own errors at **0.8763** against the fitted stack's 0.7330, with the
#: best possible *constant* sizer at 0.6678 -- so the head has taken 31% of the reachable range
#: and 0.143 of efficiency is still there, worth ~+10.7 `sps`. The head is two 3x3x3 convs, i.e.
#: a **5x5x5 receptive field** on a 20x32x64 volume, while the magnitude of the error is set by
#: flow structures much larger than that. 72c already found capacity saturated past width 64, so
#: what is missing is not capacity but CONTEXT, and a pooled reduction is an infinite receptive
#: field for the price of a mean.
POOL_FEATURE_CHANNELS = 4

#: Extra channels when ``noise=True``: a temporal and a spatial estimate of the PIV
#: measurement-noise level (see :func:`piv_noise_features`).
NOISE_FEATURE_CHANNELS = 2

#: Extra channels when ``fluc=True``: the prediction's SIGNED departure from its own temporal
#: mean, ``pred - mean_t(pred)``.
#:
#: Added 2026-09-16 (Bruno). The head already has ``departure`` -- distance from the last
#: OBSERVED frame -- and ``pred_sd``, the MAGNITUDE of the temporal fluctuation pooled over the
#: horizon. It does not have the signed fluctuation itself, and these are not the same quantity:
#: persistence is the input's final state while the temporal mean is the model's own. Where the
#: model thinks the flow has swung far from its own average is a candidate difficulty signal, and
#: unlike ``pred_sd`` it varies per lead.
FLUC_FEATURE_CHANNELS = N_MEASURED

#: Extra channels when ``lstd > 0``: the temporal standard deviation of the prediction inside a
#: centred window of ``lstd`` frames -- per element and PER LEAD.
#:
#: Added 2026-09-16 (Bruno). The sharper half of the same observation. ``pred_sd`` and ``obs_sd``
#: are both computed over all 20 frames and then broadcast identically to every lead, so the only
#: per-lead signals the head has are the prediction, its departure from persistence and a scalar
#: lead index -- while the error itself grows 3.5x from lead 1 to lead 20 on ``val_extrap``. A
#: windowed local spread says how fast the field is moving NEAR THAT LEAD, which is the missing
#: axis rather than a finer version of one the head already has.
LSTD_FEATURE_CHANNELS = N_MEASURED

#: Head input width when conditioning is on. Re-exported so callers size the head
#: from one place instead of adding the two constants themselves.
COND_FEATURE_CHANNELS = BASE_FEATURE_CHANNELS + COND_DIM


def head_in_channels(
    cond: bool = False,
    noise: bool = False,
    logs: bool = False,
    coords: bool = False,
    grad: bool = False,
    pool: bool = False,
    cond_width: int = COND_DIM,
    fluc: bool = False,
    lstd: int = 0,
) -> int:
    """Input width of :class:`BoundHead` for a feature configuration.

    One definition, because the training loop, the inference factory and the offline
    scale-head fitter all have to agree on it, and PROTOCOL 5 records two independent
    definitions of a head's input width disagreeing and silently building the wrong shape.

    ``cond_width`` exists because that is exactly what happened: ``bound_features`` appends the
    conditioning vector at **whatever width it is handed** while this function added the constant
    ``COND_DIM``, so ``--cond-mode operating_point_tau`` -- which appends ``tau`` as a fourth
    component (``features.cond_dim``) -- built the head one channel too narrow and no interval
    head could be fitted on a tau-conditioned backbone at all. Pass ``features.cond_dim(mode)``
    when the mode is known; the default reproduces every existing caller.
    """
    n = BASE_FEATURE_CHANNELS
    if logs:
        n += LOG_FEATURE_CHANNELS
    if coords:
        n += COORD_FEATURE_CHANNELS
    if noise:
        n += NOISE_FEATURE_CHANNELS
    if grad:
        n += GRAD_FEATURE_CHANNELS
    if pool:
        n += POOL_FEATURE_CHANNELS
    if fluc:
        n += FLUC_FEATURE_CHANNELS
    if lstd:
        n += LSTD_FEATURE_CHANNELS
    if cond:
        n += cond_width
    return n


def log_grad_energy(f: torch.Tensor) -> torch.Tensor:
    """``log |grad f|^2`` over the two spatial axes of a ``(..., H, W, C)`` tensor, summed over C.

    Forward differences, edge-replicated by padding the last row/column with a zero difference,
    so the result has the input's shape without a `zeros_like` per axis. No ``sqrt``: the head
    reads this through a log, and ``log|g|^2 = 2 log|g|`` is a linear rescale a conv absorbs in
    its first weight. One definition, used by the training loop, the offline head fitter and the
    inference factory alike -- the same reason ``head_in_channels`` exists.
    """
    dh = f[..., 1:, :, :] - f[..., :-1, :, :]
    dw = f[..., 1:, :] - f[..., :-1, :]
    e = dh.pow(2).sum(dim=-1, keepdim=True)
    pad_h = torch.nn.functional.pad(e, (0, 0, 0, 0, 0, 1))
    e = dw.pow(2).sum(dim=-1, keepdim=True)
    pad_w = torch.nn.functional.pad(e, (0, 0, 0, 1))
    return torch.log((pad_h + pad_w).clamp_min(LOG_EPS))


def pooled_context(dep: torch.Tensor, obs_sd: torch.Tensor) -> torch.Tensor:
    """Four log-scale window reductions of the error's own scale features, as ``(B,T,H,W,4)``.

    All four are things a 5x5x5 conv cannot see and a mean can: how large this *window's*
    departure from persistence is overall, how it varies along the horizon, how it varies over
    the field, and how energetic the observed window was. Reductions keep their broadcast axis
    so each channel varies along exactly one axis of the volume.
    """
    d = dep.abs().mean(dim=-1, keepdim=True)  # (B,T,H,W,1), channel-averaged
    g = d.mean(dim=(1, 2, 3), keepdim=True).expand_as(d)  # one number per window
    lead = d.mean(dim=(2, 3), keepdim=True).expand_as(d)  # a profile along the horizon
    field = d.mean(dim=1, keepdim=True).expand_as(d)  # a map over the field
    energy = obs_sd.abs().mean(dim=-1, keepdim=True).mean(dim=(1, 2, 3), keepdim=True).expand_as(d)
    return torch.log(torch.cat([g, lead, field, energy], dim=-1).clamp_min(LOG_EPS))


def local_temporal_std(f: torch.Tensor, window: int) -> torch.Tensor:
    """Temporal standard deviation of ``f`` in a centred window, as ``(B,T,H,W,C)``.

    ``f`` is ``(B,T,H,W,C)``. Edges are handled by replicating the first and last frames, so
    every lead gets a full window and the result has no special cases at the ends -- the
    alternative, a shrinking window, makes the ends noisier exactly where the horizon matters
    most.

    Computed with average pooling over the time axis rather than by unfolding or by cumulative
    sums. Unfolding would materialise ``window`` copies of a ``(B,20,32,64,2)`` tensor inside the
    timed region; ``cumsum`` has no deterministic CUDA kernel and the scale-head fit runs under
    ``torch.use_deterministic_algorithms(True)``, so it raises rather than silently varying.
    ``avg_pool3d`` is deterministic and gives the windowed mean of the field and of its square,
    which is all a variance needs.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    # Asymmetric padding, because a symmetric ``window // 2`` on both sides only returns T
    # frames for an ODD window: an even one returns T + 1 and every downstream concat fails on
    # the lead axis (measured 2026-09-20 with ``bound_lstd=8``). Splitting the pad as
    # ``(w-1)//2`` before and ``w//2`` after gives exactly T frames for either parity, and is
    # identical to the old behaviour whenever ``window`` is odd.
    pad_l, pad_r = (window - 1) // 2, window // 2
    head = f[:, :1].expand(-1, pad_l, -1, -1, -1)
    tail = f[:, -1:].expand(-1, pad_r, -1, -1, -1)
    fp = torch.cat([head, f, tail], dim=1).permute(0, 4, 1, 2, 3)  # (B,C,T+2p,H,W)
    m1 = F.avg_pool3d(fp, (window, 1, 1), stride=1)
    m2 = F.avg_pool3d(fp * fp, (window, 1, 1), stride=1)
    var = (m2 - m1 * m1).clamp_min(0.0)
    return var.sqrt().permute(0, 2, 3, 4, 1)


def piv_noise_features(x: torch.Tensor) -> torch.Tensor:
    """Two estimates of the local PIV measurement-noise level, as ``(B, 1, H, W)`` each.

    Why this is a different signal from everything else the head sees. The *target* is a
    PIV measurement, so part of ``|pred - target|`` is measurement noise that no dynamics
    model can ever predict away -- an irreducible floor on the error, and therefore on the
    interval width that is worth paying for. The head's existing ten features are all
    about the flow (prediction, departure from persistence, temporal spreads, lead time,
    validity, operating point); none of them says how *noisy this particular window's
    measurement* is.

    Both estimators separate noise from flow by scale, in the two axes where the
    separation is cleanest:

    * **Temporal.** The resolved flow evolves smoothly between frames while the noise is
      independent, so the second difference ``x[t+1] - 2 x[t] + x[t-1]`` annihilates any
      locally-linear signal and leaves ``6 sigma^2`` of noise variance. Dividing by
      ``sqrt(6)`` recovers ``sigma``.
    * **Spatial.** True structures are smooth over the interrogation window while noise is
      pixel-uncorrelated, so the residual against a 3x3 box mean isolates it.

    Returned per (H, W) from the input window alone, hence available at inference.
    """
    u = x[..., :N_MEASURED]  # (B, T, H, W, C)
    d2 = u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]
    temporal = d2.std(dim=1).mean(dim=-1) / math.sqrt(6.0)  # (B, H, W)

    # 3x3 box mean over space, replicate-padded so edges are not special-cased.
    f = u.mean(dim=1).permute(0, 3, 1, 2)  # (B, C, H, W)
    smooth = F.avg_pool2d(F.pad(f, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    spatial = (f - smooth).abs().mean(dim=1)  # (B, H, W)
    return torch.stack([temporal, spatial], dim=1)  # (B, 2, H, W)


def bound_features(
    pred: torch.Tensor,
    x: torch.Tensor,
    valid: torch.Tensor,
    cond: torch.Tensor | None = None,
    noise: bool = False,
    logs: bool = False,
    coords: bool = False,
    grad: bool = False,
    pool: bool = False,
    fluc: bool = False,
    lstd: int = 0,
) -> torch.Tensor:
    """Features for the interval head, as ``(B, Cf, T, H, W)``.

    Everything here is available at inference: the backbone's own prediction, the
    input window, the validity mask, and the lead-time index.

    ``cond`` matters more than it looks. Under freestream normalisation ``x`` has
    already been divided by its own freestream speed, so **nothing else in this
    feature set carries the physical scale** -- while the SPS objective *is*
    physical: ``SIGMA_GLOBAL`` is a fixed constant in m/s, and the head's
    non-dimensional width gets multiplied back by ``u_fs`` before the penalty
    applies. Blind to Reynolds, the head can only learn a single compromise width
    across a ladder spanning 7.5x, and measurably does. On ``val_time``, binned by
    the freestream speed of each window:

    ===============  ===========  ==========  =====
    u_fs bin         width/sigma  coverage    sps
    ===============  ===========  ==========  =====
    0.073 - 0.130          0.255       0.930  54.48
    0.130 - 0.194          0.389       0.933  47.50
    0.194 - 0.294          0.558       0.926  40.04
    0.294 - 0.385          0.725       0.896  32.45
    ===============  ===========  ==========  =====

    Coverage is flat while SPS falls 22 points, purely because ``width/sigma``
    triples. The best width multiplier per bin is 0.85 / 0.70 / 0.50 / 0.50 --
    monotonically tighter as Reynolds rises, which is a function of Re that the head
    needs Re in order to express.

    ``cond[0]`` is normalised log-Reynolds, monotone in the freestream speed, so
    passing ``cond`` supplies exactly the missing scale. Passing ``None`` reproduces
    the original 10-channel behaviour, which the checkpoints trained before this
    change still require.
    """
    b, t = pred.shape[0], pred.shape[1]
    anchor = x[:, -1:]

    departure = pred[..., :N_MEASURED] - anchor[..., :N_MEASURED]
    pred_sd = pred[..., :N_MEASURED].std(dim=1, keepdim=True).expand(-1, t, -1, -1, -1)
    obs_sd = x[..., :N_MEASURED].std(dim=1, keepdim=True).expand(-1, t, -1, -1, -1)
    feats = [
        pred[..., :N_MEASURED],  # what we predicted
        departure,  # how far from persistence
        pred_sd,
        obs_sd,
    ]
    stacked = torch.cat(feats, dim=-1)  # (B,T,H,W,8)

    # Lead time, normalised. Predictability falls across the horizon, so the head
    # needs to know how far ahead each frame is.
    lead = torch.linspace(0.0, 1.0, t, device=pred.device, dtype=pred.dtype)
    lead = lead.view(1, t, 1, 1, 1).expand(b, t, pred.shape[2], pred.shape[3], 1)
    valid_c = valid.view(b, 1, valid.shape[-2], valid.shape[-1], 1).expand(b, t, -1, -1, 1)
    stacked = torch.cat([stacked, lead, valid_c], dim=-1)  # (B,T,H,W,10)

    if logs:
        # log-magnitudes of the three scale features; see LOG_FEATURE_CHANNELS.
        lg = [torch.log(v.abs().clamp_min(LOG_EPS)) for v in (departure, pred_sd, obs_sd)]
        stacked = torch.cat([stacked, *lg], dim=-1)

    if coords:
        h, w = pred.shape[2], pred.shape[3]
        rows = torch.linspace(-1.0, 1.0, h, device=pred.device, dtype=pred.dtype)
        cols = torch.linspace(-1.0, 1.0, w, device=pred.device, dtype=pred.dtype)
        rr = rows.view(1, 1, h, 1, 1).expand(b, t, h, w, 1)
        cc = cols.view(1, 1, 1, w, 1).expand(b, t, h, w, 1)
        stacked = torch.cat([stacked, rr, cc], dim=-1)

    if grad:
        # Where the field is steep, not merely where it is large. `pred`'s gradient varies
        # over the horizon; the observed last frame's is a static difficulty map.
        gp = log_grad_energy(pred[..., :N_MEASURED])
        go = log_grad_energy(x[:, -1:, ..., :N_MEASURED]).expand(b, t, -1, -1, -1)
        stacked = torch.cat([stacked, gp, go], dim=-1)

    if pool:
        stacked = torch.cat([stacked, pooled_context(departure, obs_sd)], dim=-1)

    if fluc:
        # Signed departure from the prediction's own temporal mean. `pred_sd` above is the
        # magnitude of exactly this quantity pooled over the horizon; this is the per-lead
        # signed version, which the head has never had.
        pm = pred[..., :N_MEASURED]
        stacked = torch.cat([stacked, pm - pm.mean(dim=1, keepdim=True)], dim=-1)

    if lstd:
        # Per-lead temporal spread. See LSTD_FEATURE_CHANNELS for why a WINDOWED spread is a
        # different signal from `pred_sd` rather than a refinement of it.
        stacked = torch.cat([stacked, local_temporal_std(pred[..., :N_MEASURED], lstd)], dim=-1)

    if noise:
        n = piv_noise_features(x).permute(0, 2, 3, 1)  # (B, H, W, 2)
        stacked = torch.cat([stacked, n.unsqueeze(1).expand(b, t, -1, -1, -1)], dim=-1)

    if cond is not None:
        # Broadcast the per-sample conditioning vector over time and space.
        c = cond.to(pred.dtype).view(b, 1, 1, 1, -1).expand(b, t, pred.shape[2], pred.shape[3], -1)
        stacked = torch.cat([stacked, c], dim=-1)
    return stacked.permute(0, 4, 1, 2, 3).contiguous()


class BoundHead(nn.Module):
    """Small 3D conv net mapping features to per-element interval offsets.

    Deliberately cheap. ``time_score`` is a budget we spend rather than a prize we
    win -- a trivial model already scores ~99 on that axis -- so the head is sized
    at roughly a sixth of FNO's forward cost (~1 GFLOP at width 16) instead of
    whatever would maximise SPS in isolation.
    """

    def __init__(
        self,
        in_channels: int = 10,
        width: int = 16,
        depth: int = 3,
        mode: str = "offsets",
        film: int = 0,
        cond_dim: int = COND_DIM,
        kt: int = 3,
    ) -> None:
        """``film`` routes the operating point through FiLM instead of, or as well as, planes.

        ``kt`` is the convolutions' kernel size along the LEAD axis: 3 (the shipped head, every
        conv mixes neighbouring leads) or 1 (each lead's plane is convolved on its own, with the
        lead index still an input plane). Added 2026-09-26 for the cheaper-head campaign: the head
        is 25% of the call's kernel time and its FLOPs scale with ``kt``, while the width it
        predicts varies slowly along the lead axis. Parameter names do not change with ``kt``.

        Asked for by Bruno on 2026-09-16. Today the conditioning vector is broadcast into
        ``cond_dim`` CONSTANT planes and concatenated with the other features, so at the first
        convolution it can only add a term -- three of the head's channels carry numbers with no
        spatial content, multiplied across every pixel of every frame. FiLM instead predicts a
        per-channel scale and shift from the vector and applies them to the hidden activations.

        ``0`` off (planes only, the shipped behaviour and the only mode whose parameter names
        match an existing checkpoint); ``1`` the cond planes are dropped from the convolution's
        input and the vector reaches the head only through FiLM; ``2`` both.

        Initialised to IDENTITY, not to zero: the modulation predicts a deviation ``(1 + scale)``
        from an unconditioned block. The backbone uses the same convention for the reason
        FINDINGS 9 records -- an adaLN-Zero variant that gated its branch shut converged fast and
        was then overtaken by its own unconditioned control. With identity init an on/off
        comparison is not an initialisation lottery.
        """
        super().__init__()
        if mode not in ("offsets", "scale", "logerr", "logerr_tau", "quantile"):
            raise ValueError(f"mode must be offsets|scale|logerr|logerr_tau|quantile, got {mode!r}")
        self.mode = mode
        if film not in (0, 1, 2):
            raise ValueError(f"film must be 0|1|2, got {film!r}")
        self.film = film
        if kt not in (1, 3):
            raise ValueError(f"kt must be 1|3, got {kt!r}")
        self.kt = kt
        ks, pd = (kt, 3, 3), (kt // 2, 1, 1)
        #: Width of the conditioning vector at the END of the feature tensor. `bound_features`
        #: appends it last, and it is constant over (T, H, W), so the head can recover it by
        #: slicing rather than by changing every caller's signature.
        self.cond_dim = cond_dim if film else 0
        layers: list[nn.Module] = []
        # `film=1` means the vector arrives only through the modulation, so its planes are not
        # also fed to the convolution.
        c_in = in_channels - (cond_dim if film == 1 else 0)
        for _ in range(depth - 1):
            layers += [nn.Conv3d(c_in, width, kernel_size=ks, padding=pd), nn.GELU()]
            c_in = width
        # ``offsets``: two offsets (below, above) per measured channel -- the head chooses
        # the width itself. ``scale``: one positive error *scale* per channel, from which
        # the width is derived by the closed-form rule, so the head estimates a quantity
        # it can actually be right about and the width choice carries no fitted constant.
        # ``logerr``: the same one scale per channel, but returned *as* the half-width at
        # unit multiplier, leaving the sizing to a fitted :class:`~.width_rule.WidthRule`.
        # That split exists because the two halves of the problem have different best
        # solutions: the scale is a regression, well served by least squares on
        # ``log|err|``, while the multiplier is a one-dimensional maximisation of the
        # scorer's own credit and needs no model at all.
        # ``logerr_tau``: the same scale, plus a second plane per channel holding the
        # predicted SPREAD of ``log|err|`` about it. Trained by the Gaussian likelihood of
        # ``log|err| ~ Normal(log s, tau**2)``, of which the ``logerr`` head is the
        # ``tau = const`` case -- so the two are nested and the pair is comparable. The
        # spread is a *shape* signal for the width rule to bin on, never a width itself:
        # nothing parametric reaches the interval, for the reason FINDINGS 14 gives.
        # ``quantile``: two planes per channel again, but the first is a SIGNED position of the
        # lower end relative to the prediction and the second the log of the band's width, so
        # ``lower = pred + d`` and ``upper = lower + w``. The band cannot invert (``w > 0``) and,
        # unlike ``offsets``, it need not contain the prediction -- which the scorer permits and
        # which is the "the point model is biased here" diagnostic of the quantile design
        # (Bruno, 2026-09-16). Returned as ``(below, above) = (-d, d + w)`` so every consumer
        # that reads ``pred - below`` / ``pred + above`` is unchanged.
        n_out = (2 if mode in ("offsets", "logerr_tau", "quantile") else 1) * N_MEASURED
        self.out = nn.Conv3d(c_in, n_out, kernel_size=ks, padding=pd)
        self.body = nn.Sequential(*layers)
        if film:
            # One projection per hidden convolution, zero-initialised so the head starts as its
            # own unconditioned twin. `self.body` keeps its parameter names either way, so a
            # `film=0` checkpoint is unaffected by this branch existing.
            self.film_proj = nn.ModuleList(
                [nn.Linear(cond_dim, 2 * width) for _ in range(max(depth - 1, 0))]
            )
            for proj in self.film_proj:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

        # Start at a sensible width instead of an arbitrary one: zero the final
        # weights and put the initial scale entirely in the bias.
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, float(torch.log(torch.tensor(INIT_HALF_WIDTH))))
        if mode == "logerr_tau":
            # Per-group biases, so the spread plane starts at the measured 0.32 rather than at
            # the half-width. One `constant_` for both would start tau 6x too small, and a head
            # that starts with a near-zero spread gets a gradient of ~1/tau**2 on its first step.
            with torch.no_grad():
                self.out.bias[N_MEASURED:].fill_(float(torch.log(torch.tensor(INIT_TAU))))
        if mode == "quantile":
            # Start as the symmetric band of half-width INIT_HALF_WIDTH: lower end at -0.05,
            # width 0.10. The position plane is linear, so its bias is the value itself.
            with torch.no_grad():
                self.out.bias[:N_MEASURED].fill_(-INIT_HALF_WIDTH)
                self.out.bias[N_MEASURED:].fill_(
                    float(torch.log(torch.tensor(2.0 * INIT_HALF_WIDTH)))
                )

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``offsets`` mode: ``(below, above)``. ``scale`` mode: ``(scale, scale)``.

        The scale is returned twice so callers keep one return shape; the symmetric width
        is derived from it by :meth:`BoundedModel.forward`, which is where ``u_fs`` (and
        hence the dimensionless ``rho``) is available.
        """
        if self.film:
            n = feats.shape[1]
            # Constant over (T, H, W) by construction, so one element per channel recovers it.
            cond = feats[:, n - self.cond_dim :, 0, 0, 0]
            h = feats[:, : n - self.cond_dim] if self.film == 1 else feats
            i = 0
            for layer in self.body:
                h = layer(h)
                if isinstance(layer, nn.Conv3d):
                    scale, shift = self.film_proj[i](cond).chunk(2, dim=-1)
                    h = h * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
                    i += 1
            h = self.out(h)
        else:
            h = self.out(self.body(feats))
        h = h.permute(0, 2, 3, 4, 1)
        if self.mode == "quantile":
            d = h[..., :N_MEASURED].clamp(-QUANTILE_SHIFT_MAX, QUANTILE_SHIFT_MAX)
            w = h[..., N_MEASURED:].clamp(LOG_WIDTH_MIN, LOG_WIDTH_MAX + math.log(2.0)).exp()
            return -d, d + w
        if self.mode == "logerr_tau":
            # Clamped apart, because the two planes are different quantities in different units:
            # a half-width in non-dimensional velocity, and a spread in nats of log-error.
            scale = h[..., :N_MEASURED].clamp(LOG_WIDTH_MIN, LOG_WIDTH_MAX).exp()
            tau = h[..., N_MEASURED:].clamp(LOG_TAU_MIN, LOG_TAU_MAX).exp()
            return scale, tau
        h = h.clamp(LOG_WIDTH_MIN, LOG_WIDTH_MAX).exp()
        if self.mode in ("scale", "logerr"):
            return h, h
        return h[..., :N_MEASURED], h[..., N_MEASURED:]


class BoundedModel(nn.Module):
    """A backbone plus an interval head, returning ``prediction``/``lower``/``upper``.

    ``freeze_backbone`` is the cheap path: it banks the SPS gain on top of an
    already-trained model without risking its other four subscores, and it lets the
    head be evaluated as an isolated change. Training both together may do slightly
    better but confounds the two effects.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: BoundHead | None = None,
        freeze_backbone: bool = True,
        use_cond: bool = False,
        noise: bool = False,
        logs: bool = False,
        coords: bool = False,
        grad: bool = False,
        pool: bool = False,
        fluc: bool = False,
        lstd: int = 0,
        state_frames: int = 0,
    ) -> None:
        super().__init__()
        # ``stream_state`` pseudo-frames (TrainConfig.stream_state) are prepended to the input
        # for the BACKBONE only; the head's observation features (obs_sd, the PIV-noise second
        # differences, the validity mask) are statistics of the measured window and are fitted
        # offline against the eval sets' plain 20-frame inputs, so the head must read the same
        # 20 frames at serving. 0 -- every existing checkpoint -- is byte-identical behaviour.
        self.state_frames = int(state_frames)
        self.backbone = backbone
        self.noise = noise
        self.logs = logs
        self.coords = coords
        self.grad = grad
        self.pool = pool
        self.fluc = fluc
        self.lstd = lstd
        self.head = head if head is not None else BoundHead()
        if getattr(self.head, "mode", "offsets") == "logerr_tau":
            # The second plane is a spread in log units, not an upper offset, and `forward`
            # below would size an interval with it. The tau head is an OFFLINE instrument
            # (`train.scale_fit`, `scripts/sps/sps_shape.py`); nothing serves it directly.
            raise ValueError(
                "a logerr_tau head cannot be served through BoundedModel: its second output "
                "is the spread of log|err|, not an interval offset"
            )
        self.freeze_backbone = freeze_backbone
        # Off by default so checkpoints trained before the conditioning channels were
        # added still build with a 10-channel head and load cleanly.
        self.use_cond = use_cond
        #: Interval-centre shrink ``c`` (FINDINGS 196i): the interval is built about
        #: ``m + c * (pred - m)`` -- ``m`` the window's time mean of the measured channels --
        #: while ``prediction`` stays the backbone's own field. A serving option set after
        #: construction (``make_predict(centre_shrink=...)``), never trained; 1.0 leaves every
        #: code path exactly as before. See :meth:`shrunk_centre`.
        self.centre_shrink: float = 1.0
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):  # noqa: D102
        super().train(mode)
        if self.freeze_backbone:
            # Keep the frozen backbone in eval mode so its BatchNorm running stats
            # do not drift while only the head is being trained.
            self.backbone.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        u_fs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """``u_fs`` is required only in ``scale`` mode.

        The closed-form width depends on ``rho = s * u_fs / sigma`` -- a *dimensionless*
        ratio -- while the model works non-dimensionally and ``SIGMA_GLOBAL`` is physical.
        Omitting the conversion mis-scales rho by up to 17x across the Reynolds ladder,
        the same trap ``sps_loss`` and ``interval.optimal_half_width`` both document. So
        it is an error rather than a silent default.
        """
        if self.freeze_backbone:
            with torch.no_grad():
                pred, _, _ = split_output(self.backbone(x, cond))
            pred = pred.detach()
        else:
            pred, _, _ = split_output(self.backbone(x, cond))

        if self.state_frames:
            # The backbone above read the full input, state pseudo-frames included; every
            # head feature below reads the measured window only (see __init__).
            x = x[:, self.state_frames :]
        # The field the interval is built about. ``centre is pred`` unless a centre shrink is
        # set, in which case the head reads the shrunk field (it was fitted on that field's
        # errors, `scripts/sps/shrink_cache.py`) and the offsets are laid about it.
        centre = shrunk_centre(pred, self.centre_shrink)
        valid = (x[..., :N_MEASURED].abs().sum(dim=(1, -1)) > 0).to(pred.dtype)
        below, above = self.head(
            bound_features(
                centre,
                x,
                valid,
                cond=cond if self.use_cond else None,
                noise=self.noise,
                logs=self.logs,
                coords=self.coords,
                grad=self.grad,
                pool=self.pool,
                fluc=self.fluc,
                lstd=self.lstd,
            )
        )

        if getattr(self.head, "mode", "offsets") == "scale":
            if u_fs is None:
                raise ValueError(
                    "scale mode needs u_fs: the closed-form width depends on the "
                    "dimensionless rho = s*u_fs/sigma, and sigma is physical."
                )
            scale = below  # the head returns the same tensor twice in scale mode
            k = u_fs.reshape(-1, *([1] * (scale.ndim - 1))).clamp_min(1e-8)
            rho = (scale * k) / SIGMA_GLOBAL
            half = torch_optimal_half_width_factor(rho) * scale
            below = above = half

        # The pressure channel is never scored (the scorer slices to the measured
        # channels), but the bounds are still set equal to the prediction there so
        # lower <= prediction <= upper holds over the whole array. Leaving them at
        # zero would break that invariant and make ordering_penalty fire on an
        # unscored channel, masking a genuine violation on u or v.
        offset_below = torch.zeros_like(pred)
        offset_above = torch.zeros_like(pred)
        offset_below[..., :N_MEASURED] = below
        offset_above[..., :N_MEASURED] = above
        out = {
            "prediction": pred,
            "lower": centre - offset_below,
            "upper": centre + offset_above,
        }
        if centre is not pred:
            # The serving path sizes the interval (width rule, guard, floor) about this
            # field, not about ``prediction``. Absent at shrink 1.0 so the return value is
            # unchanged for every existing consumer.
            out["centre"] = centre
        return out


def shrunk_centre(pred: torch.Tensor, c: float) -> torch.Tensor:
    """``m + c * (pred - m)`` on the measured channels, ``m`` the time mean over the window.

    The interval centre of FINDINGS 196i. Returns ``pred`` itself (not a copy) at ``c == 1``,
    which is what lets every caller test ``centre is pred``. The spelling here -- a mean over
    dim 1 of the ``[..., :N_MEASURED]`` view, then ``m + c * (p - m)`` written into a clone --
    is the one ``fast_serve.FastT1`` reproduces op for op; change one and the other.
    """
    if float(c) == 1.0:
        return pred
    p = pred[..., :N_MEASURED]
    m = p.mean(dim=1, keepdim=True)
    centre = pred.clone()
    centre[..., :N_MEASURED] = m + float(c) * (p - m)
    return centre


def scale_bounds(out: dict[str, torch.Tensor], factor: float) -> dict[str, torch.Tensor]:
    """Widen or narrow the predicted intervals about the prediction.

    Used by the post-hoc calibration scan: the head learns the *shape* of the error
    field, and a single multiplier then places the overall width at the SPS optimum
    measured against the official scorer. Separating shape from scale means the
    calibration cannot be fooled by a head that is well-ranked but mis-sized.
    """
    pred = out["prediction"]
    return {
        "prediction": pred,
        "lower": pred - (pred - out["lower"]) * factor,
        "upper": pred + (out["upper"] - pred) * factor,
    }


__all__ = [
    "BoundHead",
    "BoundedModel",
    "bound_features",
    "head_in_channels",
    "scale_bounds",
    "shrunk_centre",
]
