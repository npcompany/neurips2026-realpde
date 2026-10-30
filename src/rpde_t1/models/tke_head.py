"""A second-moment head: predict the *variance* map the backbone cannot.

An L2-trained surrogate converges to the conditional mean. That makes it unbiased in
the first moment and badly biased in the second: ours retains only 0.36-0.44 of the
target's fluctuation energy. ``tke_score`` grades exactly that second moment, and it is
the weakest of the five axes with an oracle ceiling of 98.7 against a current 71.1.

The obvious fix is refuted. A *uniform* fluctuation gain of 1.4 moves tke 71.1 -> 59.5
and 2.0 moves it to 40.5, because the predicted variance map is **mis-localised**
(spatial correlation 0.758 against the target map) rather than merely under-amplitude.
Amplifying a mis-localised map amplifies its error.

So the head predicts a **per-pixel log-ratio** correction instead:

    r(h, w)   = log( T_hat / ke(pred) )        <- what this module learns
    gain      = exp(r / 2)
    q         = mean_t(pred) + (pred - mean_t(pred)) * gain

Three properties fall out of writing it this way, and each is load-bearing:

* **Scale-free.** A ratio cancels the freestream normalisation, so the head never has to
  represent the 7.5x Reynolds ladder in its own units -- the failure mode that cost the
  interval head ~22 SPS between its lowest and highest freestream quartile.
* **It strictly generalises the geometric blend.** That closed-form estimator is
  ``r = w * log(ke_input / ke_pred)`` for a constant ``w``, which is one particular
  linear function of one of this head's input channels. So the head cannot do worse than
  the blend except by overfitting, and the blend is the honest baseline to beat.
* **``mvpe`` is invariant by construction**, since the gain is constant in time and
  ``mvpe`` time-averages before comparing.

Training is cheap for a reason worth stating: the objective depends on the backbone only
through ``ke(pred)``, a ``(H, W)`` map per window. So the backbone runs **once** over the
training set, the maps are cached, and the head then trains on 32x64 images. Minutes, not
the GPU-hour a backbone-in-the-loop version would need.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .fluctuations import DEFAULT_MAX_GAIN, apply_fluctuation_gain

#: Channels :func:`tke_features` produces, in order. Anything sizing a layer from this
#: must read the length rather than hardcode it.
FEATURE_NAMES = (
    "log_ke_input_over_pred",
    "log_ke_pred",
    "valid",
    "airfoil_sdf",
    "log_re",
    "aoa",
    "tau",
    # Per-component planes (2026-09-17). The first seven are the shipped head's whole input,
    # so the four below are APPENDED: every existing head slices ``[:want]`` and never sees
    # them, and the shipped single-gain head keeps its exact 7-channel architecture through
    # :data:`N_BASE_FEATURES`. They are zero unless per-component energy maps are supplied.
    "log_ke_input_over_pred_u",
    "log_ke_input_over_pred_v",
    "log_ke_pred_u",
    "log_ke_pred_v",
)

#: Input width of the shipped single-gain head: the planes before the per-component ones.
N_BASE_FEATURES = 7


def _log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def tke_features(
    ke_pred: np.ndarray,
    ke_input: np.ndarray,
    valid: np.ndarray,
    sdf: np.ndarray,
    cond: np.ndarray,
    drop: tuple[str, ...] = (),
    ke_pred_c: np.ndarray | None = None,
    ke_input_c: np.ndarray | None = None,
) -> np.ndarray:
    """``(N, C, H, W)`` float32 features for the head.

    ``ke_pred`` / ``ke_input`` are ``(N, H, W)`` energy maps, ``valid`` is ``(N, H, W)``
    or ``(H, W)``, ``sdf`` is ``(N, H, W)`` or ``(H, W)``, and ``cond`` is ``(N, >=2)``
    holding normalised log-Re and AoA. ``ke_pred_c`` / ``ke_input_c`` are the
    ``(N, H, W, 2)`` per-component maps (:func:`predictor.kinetic_energy_map_components`);
    when given they fill the four trailing per-component planes, otherwise those are zero.

    ``log_ke_pred`` is centred per sample rather than passed raw: its absolute level is
    the freestream scale, which the ratio target does not depend on, so leaving it in
    would hand the head a nuisance variable that varies 7.5x across the ladder.

    ``drop`` **zeroes** the named channels rather than removing them, so the head keeps a
    fixed input width and an ablated checkpoint still loads into the standard
    architecture. Ablating ``log_re``/``aoa`` is the one that matters: those planes are
    constant per sample, so the head can fit an operating-point-dependent correction that
    has nothing to extrapolate on at an unseen Reynolds number.
    """
    unknown = set(drop) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature(s) {sorted(unknown)}; have {FEATURE_NAMES}")
    n = ke_pred.shape[0]
    hw = ke_pred.shape[1:]
    ratio = _log(ke_input) - _log(ke_pred)
    level = _log(ke_pred)
    level = level - level.mean(axis=(1, 2), keepdims=True)

    def _b(a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a, (n, *hw)).astype(np.float32)

    planes = [
        ratio.astype(np.float32),
        level.astype(np.float32),
        _b(valid),
        _b(sdf),
        _b(np.asarray(cond)[:, 0].reshape(n, 1, 1)),
        _b(np.asarray(cond)[:, 1].reshape(n, 1, 1)),
        # ROADMAP N40: the head needs a *horizon* feature, because the spread of
        # ke(target)/ke(input) grows strongly with Re while its mean is flat, so one
        # learned correction cannot suit both ends of the ladder. tau -- the window's own
        # fluctuation decorrelation time -- is that horizon measured rather than proxied,
        # and unlike log_re/aoa it is not constant per cell, so an unseen operating point
        # still gets a value drawn from the range the head was fitted on.
        #
        # Zeros when the conditioning vector does not carry it. That keeps the input width
        # fixed, exactly as ``drop`` does, so a head fitted with tau and one fitted without
        # share an architecture and the ablation is a zeroed channel rather than a
        # different model.
        _b(
            np.asarray(cond)[:, 3].reshape(n, 1, 1)
            if np.asarray(cond).shape[1] > 3
            else np.zeros((n, 1, 1), dtype=np.float32)
        ),
    ]
    if ke_pred_c is not None and ke_input_c is not None:
        # Same two constructions as the pooled planes, per component: the log energy ratio,
        # and the log level centred per sample so the freestream scale drops out.
        ratio_c = _log(ke_input_c) - _log(ke_pred_c)
        level_c = _log(ke_pred_c)
        level_c = level_c - level_c.mean(axis=(1, 2), keepdims=True)
        planes += [
            ratio_c[..., 0].astype(np.float32),
            ratio_c[..., 1].astype(np.float32),
            level_c[..., 0].astype(np.float32),
            level_c[..., 1].astype(np.float32),
        ]
    else:
        planes += [np.zeros((n, *hw), dtype=np.float32)] * 4
    out = np.stack(planes, axis=1).astype(np.float32)
    for name in drop:
        out[:, FEATURE_NAMES.index(name)] = 0.0
    return out


class TkeHead(nn.Module):
    """Small 2D CNN mapping :func:`tke_features` to a per-pixel log-ratio.

    Deliberately tiny. The information it has to add over the geometric blend is a
    spatial correction, not a new representation of the flow, and the interval head --
    the one component here that has actually banked points -- did its job with 13k
    parameters. ``time_score`` is a budget, so a large head would spend more than the
    axis is worth.

    The last layer is zero-initialised, so an untrained head is the **identity** (gain
    exactly 1) rather than noise. Training then starts from the backbone's own map and
    can only move away from it deliberately.
    """

    def __init__(
        self,
        in_channels: int = N_BASE_FEATURES,
        width: int = 32,
        depth: int = 3,
        out_channels: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(depth):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, out_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def per_component(self) -> bool:
        """Two output log-ratios, one per velocity component, instead of one shared."""
        return int(self.out.out_channels) > 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(N, C, H, W)`` features -> ``(N, H, W)`` log-ratio, or ``(N, H, W, 2)`` for a
        per-component head (component last, matching the field layout the gain is applied to)."""
        y = self.out(self.body(features))
        return y.squeeze(1) if y.shape[1] == 1 else y.permute(0, 2, 3, 1)


def apply_gain(prediction: np.ndarray, log_ratio: np.ndarray, max_gain: float = DEFAULT_MAX_GAIN):
    """Rescale fluctuations by ``exp(log_ratio / 2)``, preserving the time mean.

    The head predicts a per-pixel log-ratio of energies; the gain on the *field* is the
    square root of that, hence the half. Applying it is
    :func:`~rpde_t1.models.fluctuations.apply_fluctuation_gain`, shared with the
    closed-form blend so the two cannot drift apart in how they apply their answer --
    only in how they compute it.
    """
    gain = np.exp(0.5 * np.asarray(log_ratio, dtype=np.float32))
    return apply_fluctuation_gain(prediction, gain, max_gain=max_gain)


#: Default shrinkage of the learned head, and the blend exponent it is shrunk toward.
#: **Both were chosen on the selection split, not on a reported one.** Measured on
#: `val_extrap` when these defaults were set: blend alone +2.75 tke, head alone +1.78,
#: this mixture +4.06 -- the head is strong in distribution and fragile at the Reynolds
#: ladder ends, the blend is weak and robust, and the failure modes are close to
#: independent.
#:
#: **These defaults are NOT what ships, and the reasoning above is superseded for the
#: current lineage (FINDINGS 80d).** On it the closed-form blend is negative at every
#: exponent (`val_extrap` -1.44 at 0.3, -2.98 at 0.5, -5.52 at 0.75), so the shipped
#: artifacts pass `blend 0` with `shrink 0.7` -- which makes
#: :func:`inference_log_ratio` reduce to ``r = 0.7 * r_head``, i.e. shrinkage toward the
#: **identity** and not toward the blend. Its selection table, on the corner, is
#: shrink 1.0 +0.067 final / 0.7 +0.109 / 0.6 +0.104, and the whole of the 0.7-over-1.0
#: gain is rel_l2 (-0.170 against -0.328) rather than tke. So the surviving purpose of
#: the shrink is to cap the rel_l2 cost of a head fitted on `tke` alone; read it as a
#: post-hoc stand-in for an objective that prices rel_l2, not as a mixture with the
#: blend. Callers that want the shipped behaviour must pass both values explicitly.
DEFAULT_SHRINK = 0.75
DEFAULT_BLEND = 0.5


@torch.no_grad()
def inference_log_ratio(
    prediction: np.ndarray,
    input_window: np.ndarray,
    cond: np.ndarray,
    head: TkeHead | None = None,
    blend: float = DEFAULT_BLEND,
    shrink: float = DEFAULT_SHRINK,
    device: str | torch.device = "cpu",
    uses_tau: bool = False,
) -> np.ndarray:
    """The log-ratio to rescale by, at inference, from what a submission actually has.

    Every input is available on a scored call: the prediction, the 20 observed frames, and
    the operating point estimated from them (``predictor.infer_conditioning``). No labels,
    no metadata.

    With ``head=None`` this is exactly the closed-form blend, so the two paths differ only
    in whether a head is present -- there is no separate blend-only code to drift.
    """
    # ``airfoil_sdf_fast``, not ``airfoil_sdf``: the exact version costs 52 ms per call
    # and is called per sample, which measured as -7.7 time_score -- far more than this
    # head is worth. See geometry.sdf_table.
    from ..geometry import airfoil_sdf_fast  # noqa: PLC0415  (numpy-only, ships in bundle)
    from .predictor import kinetic_energy_map_components  # noqa: PLC0415

    ke_pred_c = kinetic_energy_map_components(prediction)
    ke_input_c = kinetic_energy_map_components(input_window)
    ke_pred, ke_input = ke_pred_c.sum(-1), ke_input_c.sum(-1)
    ratio = _log(ke_input) - _log(ke_pred)
    if head is None:
        return (blend * ratio).astype(np.float32)

    n, h, w = ke_pred.shape
    # The mask is read off the input window, the same way training reads it: a pixel with
    # no measurement is identically zero in u and v across all 20 frames.
    #
    # The two channels are reduced SEPARATELY rather than as ``abs(x[..., :2]).sum((1, -1))``.
    # That slice is strided over a 3-channel array and measured **0.954 ms/window** against
    # 0.162 for this form -- 5.9x, and 28% of this function's whole cost (FINDINGS 63m). The
    # predicate is unchanged, not merely close: both sides sum non-negative terms, and a sum
    # of non-negatives is zero exactly when every term is. This is the same rewrite
    # ``features.estimate_batch`` records at 11x, on the same array.
    #
    # Do NOT reduce to a single frame instead. The comment above licenses it and it is wrong:
    # measured, a one-frame mask differs from this one on **6,634 of 430,080** pixels.
    valid = (
        (np.abs(input_window[..., 0]).sum(axis=1) + np.abs(input_window[..., 1]).sum(axis=1)) != 0
    ).astype(np.float32)
    sdf = np.stack([airfoil_sdf_fast(float(cond[i, 1]) * 20.0, h, w) for i in range(n)])

    # The head's conditioning is built here rather than taken from the backbone's, because
    # the two need not be the same width: tau is worth nothing to the backbone (measured)
    # and is the horizon feature the head wants, so the deployable combination is a
    # 3-vector backbone and a 4-vector head. Recomputing it from the window the head was
    # already handed keeps that from being a configuration to get wrong -- a head fitted
    # with tau and served the backbone's narrower vector would silently read zeros.
    cond = np.asarray(cond, dtype=np.float32)
    want = int(head.body[0].weight.shape[1])
    # ``uses_tau`` rather than a guess from ``want``: see :func:`head_uses_tau`. Serving this
    # plane to a head fitted on zeros shifts the log ratio by up to 0.31, and computing it costs
    # 0.302 ms/window that a zeroed channel does not need.
    if uses_tau and want > FEATURE_NAMES.index("tau") and cond.shape[1] <= 3:
        from ..features import TAU_MAX_FRAMES, decorrelation_time_batch  # noqa: PLC0415

        tau = np.minimum(decorrelation_time_batch(input_window) / TAU_MAX_FRAMES, 1.0)
        cond = np.concatenate([cond, tau.astype(np.float32).reshape(-1, 1)], axis=1)

    # Sliced to what this head was built for. The feature list only ever grows at the end,
    # so the first ``want`` planes of today's features are exactly the ones an older head
    # was fitted on -- which is what lets a head predating a channel keep working.
    percomp = want > N_BASE_FEATURES
    feats = tke_features(
        ke_pred,
        ke_input,
        valid,
        sdf,
        cond,
        ke_pred_c=ke_pred_c if percomp else None,
        ke_input_c=ke_input_c if percomp else None,
    )[:, :want]
    r_head = head.to(device).eval()(torch.from_numpy(feats).to(device)).cpu().numpy()
    if r_head.ndim == 4:  # per-component head: the pooled blend term broadcasts over components
        ratio = ratio[..., None]
    return (shrink * r_head + (1.0 - shrink) * blend * ratio).astype(np.float32)


def tke_static_planes(cond, h: int, w: int, device, uses_tau: bool = False, input_window=None):
    """The head's per-trajectory CONSTANT feature planes: sdf, and the conditioning columns.

    Four of the head's seven planes depend only on ``cond`` -- the airfoil sdf (a table lookup
    keyed on angle of attack) and the three broadcast conditioning columns. On Track 2 a
    trajectory is one operating point and ``FrozenPort`` freezes its estimate after three
    windows, so these are constant for the whole stream and can be built once.

    That matters for more than speed. Built inside a ``torch.compile(mode="reduce-overhead")``
    region they would be materialised during CUDA-graph capture and then FROZEN at their
    warm-up values for every later replay -- silently wrong predictions rather than an error.
    Hoisting them here is what lets the head run inside the compiled Track-2 pipeline at all.

    Returns ``(n, 4, h, w)`` in the plane order ``inference_log_ratio_torch`` stacks them.
    """
    import torch  # noqa: PLC0415

    from ..geometry import airfoil_sdf_fast  # noqa: PLC0415

    cond = np.asarray(cond, dtype=np.float32)
    if uses_tau and cond.shape[1] <= 3 and input_window is not None:
        from ..features import TAU_MAX_FRAMES, decorrelation_time_batch  # noqa: PLC0415

        tau = np.minimum(decorrelation_time_batch(input_window) / TAU_MAX_FRAMES, 1.0)
        cond = np.concatenate([cond, tau.astype(np.float32).reshape(-1, 1)], axis=1)
    n = cond.shape[0]
    sdf = np.stack([airfoil_sdf_fast(float(cond[i, 1]) * 20.0, h, w) for i in range(n)])
    out = np.zeros((n, 4, h, w), dtype=np.float32)
    out[:, 0] = sdf.astype(np.float32)
    for k, j in enumerate((0, 1, 3)):
        if cond.shape[1] > j:
            out[:, k + 1] = cond[:, j].reshape(n, 1, 1)
    return torch.from_numpy(out).to(device)


def inference_log_ratio_torch(
    prediction,
    input_window,
    cond,
    head=None,
    blend: float = None,  # type: ignore[assignment]
    shrink: float = None,  # type: ignore[assignment]
    uses_tau: bool = False,
    input_device: tuple | None = None,
    static_planes=None,
):
    """Torch twin of :func:`inference_log_ratio`, for a device-resident prediction.

    **Why a twin and not a rewrite.** 63m decomposes the deployed second-moment path and finds
    the conv forward is 0.044 ms/window -- 1% of it -- while the two ``kinetic_energy_map``
    calls and the gain application are 2.4 ms of host numpy on a 103 MB array. Both are pure
    elementwise reductions, and the chunk loop already holds the prediction on the device, so
    they belong there. What must NOT move is the *definition*: PROTOCOL 5's opening trap is two
    definitions of the same thing disagreeing, found five times in this tree. So the numpy
    function above stays the definition, this mirrors it operation for operation, and
    ``tests/test_tke_head.py`` asserts they agree on real eval windows.

    ``cond`` stays numpy: the SDF lookup and ``tau`` are per-window host quantities measured
    at 0.008 and 0.302 ms/window respectively, and the last is zero once ``tau`` is dropped --
    which 63f says to do, having measured it worth +0.05 ``tke``.

    ``input_device`` is ``(window, u_fs)`` with the window already resident and **divided by
    the freestream speed**, which is what the chunk loop hands the backbone. Passing it skips
    the two costs that dominate what is left: the host mask, 0.163 ms/window, and a second
    host->device copy of the same 103 MB array, 0.111 -- together 67% of this function
    (measured on the shipped `w8/d2` head at N=147). Both quantities are recoverable exactly
    rather than approximately: the mask is a test against zero, which a positive scale cannot
    change, and the energy map is quadratic, so multiplying by ``u_fs^2`` restores physical
    units. ``input_window`` remains the definition and is used whenever this is ``None``.
    """
    import torch  # noqa: PLC0415

    from .flow import kinetic_energy_map_components as ke_torch_c  # noqa: PLC0415

    blend = DEFAULT_BLEND if blend is None else blend
    shrink = DEFAULT_SHRINK if shrink is None else shrink
    dev = prediction.device

    def _tlog(t):
        return torch.log(torch.clamp(t, min=1e-12))

    ke_pred_c = ke_torch_c(prediction)
    if input_device is not None:
        xd, u_fs = input_device
        ke_input_c = ke_torch_c(xd) * (u_fs.reshape(-1, 1, 1, 1) ** 2)
    else:
        xd = None
        ke_input_c = ke_torch_c(torch.from_numpy(np.ascontiguousarray(input_window)).to(dev))
    ke_pred, ke_input = ke_pred_c.sum(-1), ke_input_c.sum(-1)
    ratio = _tlog(ke_input) - _tlog(ke_pred)
    if head is None:
        return (blend * ratio).float()

    n, h, w = ke_pred.shape
    # Same predicate and the same two-channel reduction as the numpy path, for the reason
    # recorded there: the strided three-channel slice is 5.9x slower and a one-frame mask is
    # unsound.
    if xd is not None:
        valid = ((xd[..., 0].abs().sum(dim=1) + xd[..., 1].abs().sum(dim=1)) != 0).to(torch.float32)
    else:
        valid = torch.from_numpy(
            (
                (
                    np.abs(input_window[..., 0]).sum(axis=1)
                    + np.abs(input_window[..., 1]).sum(axis=1)
                )
                != 0
            ).astype(np.float32)
        ).to(dev)
    want = int(head.body[0].weight.shape[1])
    level = _tlog(ke_pred)
    level = level - level.mean(dim=(1, 2), keepdim=True)
    dynamic = torch.stack([ratio.float(), level.float(), valid], dim=1)

    if static_planes is None:
        # The definition. Everything here is host work keyed on ``cond`` alone, which is why
        # ``tke_static_planes`` exists to hoist it out of a compiled region -- see its docstring.
        static_planes = tke_static_planes(
            cond, h, w, dev, uses_tau=uses_tau, input_window=input_window
        )
    planes = [dynamic, static_planes.to(dynamic.dtype)]
    if want > N_BASE_FEATURES:
        # The per-component planes, in FEATURE_NAMES order, mirroring ``tke_features``.
        ratio_c = _tlog(ke_input_c) - _tlog(ke_pred_c)
        level_c = _tlog(ke_pred_c)
        level_c = level_c - level_c.mean(dim=(1, 2), keepdim=True)
        planes.append(
            torch.stack(
                [ratio_c[..., 0], ratio_c[..., 1], level_c[..., 0], level_c[..., 1]], dim=1
            ).to(dynamic.dtype)
        )
    feats = torch.cat(planes, dim=1)[:, :want]
    with torch.no_grad():
        r_head = head(feats)
    if r_head.dim() == 4:
        ratio = ratio[..., None]
    return (shrink * r_head + (1.0 - shrink) * blend * ratio).float()


def training_log_ratio(pred, x, cond, head, valid=None):
    """The head's log-ratio on a TRAINING batch, differentiable through head AND prediction.

    The third twin, for the end-to-end fine-tune (`TrainConfig.tke_head_e2e`): the numpy
    function is the definition, ``inference_log_ratio_torch`` mirrors it for serving under
    ``no_grad``, and this one keeps the graph so the shipped objective evaluated on the
    head-corrected output reaches both the head's weights and the backbone's. Feature planes
    are the same seven in the same order (``tau`` zeroed — a head trained here must be saved
    with a non-tau ``cond_mode`` so serving zeroes it too); no shrink and no blend, because a
    head trained against the full objective prices its own ``rel_l2`` and the shrink exists
    only to cap a head fitted on ``tke`` alone (see ``DEFAULT_SHRINK``'s note).

    Every feature is scale-free (a ratio, a centred level, masks, conditioning), so training
    on normaliser units and serving physical units is the same function — the property the
    module's docstring records as load-bearing.

    ``x`` is the model input in the same units as ``pred``; ``cond`` is the batch's
    conditioning (estimated when ``--cond-source estimate``, which matches what serving
    feeds the head). ``valid`` is accepted for symmetry but the mask is recomputed from
    ``x`` exactly as serving recomputes it, so train and serve cannot drift.
    """
    import torch  # noqa: PLC0415

    from .flow import kinetic_energy_map as ke_torch  # noqa: PLC0415

    def _tlog(t):
        return torch.log(torch.clamp(t, min=1e-12))

    ke_pred = ke_torch(pred)
    ke_input = ke_torch(x)
    ratio = _tlog(ke_input) - _tlog(ke_pred)
    level = _tlog(ke_pred)
    level = level - level.mean(dim=(1, 2), keepdim=True)
    mask = ((x[..., 0].abs().sum(dim=1) + x[..., 1].abs().sum(dim=1)) != 0).to(torch.float32)
    n, h, w = ke_pred.shape
    static = tke_static_planes(
        np.asarray(cond.detach().cpu().numpy(), dtype=np.float32), h, w, pred.device
    )
    want = int(head.body[0].weight.shape[1])
    feats = torch.cat(
        [torch.stack([ratio.float(), level.float(), mask], dim=1), static.to(ratio.dtype)],
        dim=1,
    )[:, :want]
    return head(feats)


def apply_gain_torch(prediction, log_ratio, max_gain: float = DEFAULT_MAX_GAIN):
    """Torch twin of :func:`apply_gain`; the gain on the field is the square root, hence /2."""
    import torch  # noqa: PLC0415

    from .fluctuations import apply_fluctuation_gain_torch  # noqa: PLC0415

    return apply_fluctuation_gain_torch(prediction, torch.exp(0.5 * log_ratio), max_gain=max_gain)


def head_uses_tau(path) -> bool:
    """Was the head at ``path`` FITTED with a live ``tau`` plane?

    Cannot be inferred from the weights, which is the whole reason this exists.
    :func:`tke_features` always emits every plane and zeroes ``tau`` when the conditioning
    vector does not carry it, so a head fitted with tau and one fitted with the plane zeroed
    have the same input width. ``inference_log_ratio`` used to guess from that width and so
    served live tau to heads trained on zeros -- a train/serve skew measured at up to 0.31 in
    the log ratio (FINDINGS 63m).

    Absent key -> ``False``, i.e. keep the plane zeroed. That is the safe default because it
    reproduces the analysis path every existing head was fitted and *selected* on.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return str(payload.get("cond_mode", "")) == "operating_point_tau"


def load_head(path, device: str | torch.device = "cpu", arm: str = "full") -> TkeHead:
    """Rebuild a head from a checkpoint written by ``train tke-head``.

    Accepts both the single-head and multi-arm layouts, because the ablation sweep writes
    the second and a bundle should not care which produced its weights.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["heads"][arm] if "heads" in payload else payload["head"]
    # Read from the weights, not from ``len(FEATURE_NAMES)``: the feature list grows, and a
    # head fitted before it did must still rebuild at the width it was trained with rather
    # than fail to load the day a channel is added.
    head = TkeHead(
        in_channels=int(state["body.0.weight"].shape[1]),
        width=payload.get("width", 32),
        depth=payload.get("depth", 3),
        out_channels=int(state["out.weight"].shape[0]),
    )
    head.load_state_dict(state)
    return head.to(device).eval()


def tke_rel_l2(
    log_ratio: torch.Tensor,
    ke_pred: torch.Tensor,
    ke_target: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """The scorer's own tke error, differentiated with respect to the head.

    After rescaling, the prediction's energy map *is* ``exp(log_ratio) * ke_pred`` -- the
    gain is applied to fluctuations and energy is their square -- so the scored quantity
    can be written in closed form without touching the backbone. That identity is what
    makes this trainable on cached maps, and it is checked in
    ``tests/test_tke_head.py::test_rescaling_realises_the_predicted_energy_map``.

    For a per-component head ``log_ratio`` and ``ke_pred`` are ``(N, H, W, 2)`` and the
    scored map is the sum over components of each one's rescaled energy; ``ke_target`` is
    always the pooled ``(N, H, W)`` map, because that is all the scorer sees.
    """
    est = torch.exp(log_ratio) * ke_pred
    if est.dim() == 4:
        est = est.sum(-1)
    a = (est * valid).reshape(est.shape[0], -1)
    b = (ke_target * valid).reshape(ke_target.shape[0], -1)
    return (
        torch.linalg.vector_norm(a - b, dim=1) / torch.linalg.vector_norm(b, dim=1).clamp_min(eps)
    ).mean()


def blend_log_ratio(features: np.ndarray, blend: float) -> np.ndarray:
    """The closed-form geometric blend, expressed in the head's own output space.

    Lets the trained head and the zero-training baseline be scored through one code
    path, which is the only way the comparison between them means anything.
    """
    return blend * features[:, FEATURE_NAMES.index("log_ke_input_over_pred")]
