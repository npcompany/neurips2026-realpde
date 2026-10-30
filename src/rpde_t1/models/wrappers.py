"""Adapters that give every model one interface, and one inference path.

Two problems this solves.

**Interface.** The vendored baselines take a bare ``(B, T, H, W, C)`` tensor and
know nothing about operating-point conditioning; our own models want ``(x, cond)``
and may emit interval bounds. Everything here presents the same call signature,
so the training loop, the validation harness and the submission wrapper never
branch on model type.

**Residual prediction.** Measured on ``val_extrap``: the CNO baseline is ~3x *worse*
than persistence at lead 1 (rel-L2 0.115 vs 0.043) -- it cannot reproduce the frame
it was just handed. Predicting a delta from the last observed frame makes short
leads nearly free and lets capacity go to the late leads, where predictability
actually falls (from ~86% of the variance at lead 1 to ~33% at lead 20).
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from .. import T_OUT, C


class ConditionedModel(Protocol):
    """What the training loop requires of a model."""

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]: ...


class BaselineAdapter(nn.Module):
    """Wrap a vendored baseline (CNO / FNO / Transolver) as ``(x, cond)``.

    With ``cond_planes=0`` -- the default, and what every baseline arm before 2026-08-22
    ran -- ``cond`` is accepted and ignored, because these architectures have no slot for
    it and ``vendor/`` is never edited. That reproduces the published baselines exactly,
    which is the point of retraining them ourselves.

    **It is also a handicap the rest of the pipeline does not share**, and one worth
    ~+0.12 by FINDINGS 70l: the recipe conditions on (log-Re, AoA, is_sim) and these arms
    could not, so a baseline *loss* was never clean evidence. ``cond_planes`` closes it
    without touching ``vendor/``: the conditioning vector is broadcast to constant input
    planes and concatenated onto the channel axis, so the baseline's own first layer
    learns to read it. Same device as ``geom_channels`` (a field the model convolves with
    rather than a per-channel affine) and as ``stformer``'s ``cond_content``, and the
    width is accounted for in exactly the same place --
    ``factory.input_shape_with_geometry`` -- so the first layer is sized for it at
    construction and a checkpoint still rebuilds from its own config.

    Constant planes rather than FiLM because FiLM needs a hook inside the trunk and this
    needs nothing inside the trunk at all, which is what keeps ``vendor/`` untouched.
    """

    def __init__(self, model: nn.Module, name: str = "baseline", cond_planes: int = 0) -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.cond_planes = cond_planes

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if not self.cond_planes:
            del cond
            return self.model(x)
        if cond is None:
            # The submission path may omit it; zeros are what an unconditioned sample
            # looks like, and they are also what the width was sized for.
            cond = x.new_zeros(x.shape[0], self.cond_planes)
        if cond.shape[-1] != self.cond_planes:
            raise ValueError(
                f"{self.name} was built for {self.cond_planes} conditioning planes but got a "
                f"vector of width {cond.shape[-1]}; cond_mode and the checkpoint disagree"
            )
        planes = cond[:, None, None, None, :].expand(-1, *x.shape[1:-1], -1)
        return self.model(torch.cat([x, planes.to(x.dtype)], dim=-1))


def _anchor(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The last observed frame, on the physical channels only.

    A backbone that emits more than the ``C`` physical channels (the variance-map channel of
    ``models/kmap.py``) gets a zero anchor on the extra ones: the map is a ratio, not a field,
    and the input's fourth channel -- if any -- is a geometry plane, not something to add.
    """
    n = min(out.shape[-1], C)
    a = x[:, -1:, ..., :n]
    if out.shape[-1] > n:
        a = torch.nn.functional.pad(a, (0, out.shape[-1] - n))
    return a


class ResidualWrapper(nn.Module):
    """Predict a delta from the last observed frame instead of absolute fields.

    ``out = x[:, -1:] + backbone(x, cond)``, broadcast over the output horizon.

    The last input frame is a strong prior at short lead times and a weak one at
    long lead times, which is exactly the structure of this task. It also means an
    untrained model starts at persistence rather than at zero -- a far better
    initialisation given persistence scores 81.2 on ``val_extrap``.

    Bounds, when the backbone emits them, are treated as offsets around the
    prediction rather than absolute levels, so the anchor cancels.
    """

    def __init__(self, backbone: nn.Module, enabled: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.enabled = enabled

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        out = self.backbone(x, cond) if _takes_cond(self.backbone) else self.backbone(x)
        if not self.enabled:
            return out
        if isinstance(out, dict):
            pred = out["prediction"] + _anchor(x, out["prediction"])
            result = {"prediction": pred}
            if "lower" in out:
                result["lower"] = pred - out["lower"].abs()
                result["upper"] = pred + out["upper"].abs()
            return result
        return out + _anchor(x, out)


class StateNormWrapper(nn.Module):
    """Standardise the window by the trajectory's own causal statistics, and restore after.

    The ``stream_state`` backbone feeds the running mean/std maps to the stem as two extra
    frames and lets the network work out what to do with them. This does the strong version
    instead: it USES them as the normaliser.

    .. code-block::

        m, s  = the state frames (running mean / temporal std over revealed frames)
        x'    = (x - m) / clamp(s, floor)          # what the backbone sees, 20 frames
        y     = backbone(x') * clamp(s, floor) + m # restored before anything downstream

    The point is stationarity. Freestream division fixes the global amplitude but leaves every
    per-pixel difference between operating points -- where the wake sits, how strong the shear
    layer is -- for the network to absorb. Dividing by the trajectory's own per-pixel statistics
    removes that, so the backbone only ever models fluctuation DYNAMICS and sees roughly the
    same distribution at Re 3750 and Re 26700. It also subsumes the mean-flow correction by
    construction: the first moment is restored exactly, not estimated.

    Three consequences that make this cheap rather than invasive:

    * the backbone is a plain 20-frame model again -- no widened stem, so the stem cost the
      state used to carry goes away;
    * the restore happens INSIDE this wrapper, so the interval head, the second-moment head,
      the width rule and the whole serving path see the usual space and need no change;
    * the dataset needs no change either: it already supplies the state frames.

    ``floor`` guards the division where the PIV mask leaves exact zeros (9-12% of ``u`` pixels)
    and where ``s`` is genuinely tiny in the dead region, which would otherwise amplify noise
    without bound. It is in freestream-normalised units, matching the input.

    ``mode`` selects how much of the affine is used. ``"meanstd"`` is the full standardisation
    described above. ``"mean"`` subtracts and restores the running MEAN only, leaving the scale
    at 1 -- which is the variant the measurement asks for: the full version lost 0.63-0.66 `tke`
    (the scale ties the predicted fluctuation amplitude to the trajectory's HISTORICAL variance,
    while `tke` is scored against THIS window's realised variance) against only 0.05-0.11
    `rel_l2`, so the damage is entirely in the scale half. Mean-only keeps the part that removes
    the per-trajectory first moment -- the mean-flow correction, done exactly and at training
    time instead of estimated at serving -- and drops the part that hurts.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_state: int = 2,
        floor: float = 0.05,
        mode: str = "meanstd",
    ) -> None:
        super().__init__()
        if mode not in ("meanstd", "mean"):
            raise ValueError(f"mode must be 'meanstd' or 'mean', got {mode!r}")
        self.backbone = backbone
        self.n_state = int(n_state)
        self.floor = float(floor)
        self.mode = mode

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean, scale)`` from the state frames, broadcastable over the lead axis."""
        c = x.shape[-1]
        m = x[:, 0:1, ..., :2]
        s = (
            x[:, 1:2, ..., :2].clamp_min(self.floor)
            if self.mode == "meanstd"
            else torch.ones_like(x[:, 1:2, ..., :2])
        )
        if c > 2:  # leave the unscored pressure slot alone: shift 0, scale 1
            pad_m = torch.zeros_like(x[:, 0:1, ..., 2:])
            pad_s = torch.ones_like(x[:, 0:1, ..., 2:])
            m = torch.cat((m, pad_m), dim=-1)
            s = torch.cat((s, pad_s), dim=-1)
        return m, s

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        m, s = self._stats(x)
        xs = (x[:, self.n_state :] - m) / s
        out = self.backbone(xs, cond) if _takes_cond(self.backbone) else self.backbone(xs)
        if isinstance(out, dict):
            pred = out["prediction"] * s + m
            result = {"prediction": pred}
            if "lower" in out:
                # Bounds are offsets around the prediction (the ResidualWrapper convention),
                # so they carry the scale but not the shift.
                result["lower"] = out["lower"] * s
                result["upper"] = out["upper"] * s
            return result
        return out * s + m


class StateAnchorWrapper(nn.Module):
    """Blend the residual anchor per lead between the last frame and the trajectory mean.

    The residual anchor is the last observed frame -- a strong prior at lead 1 and a weak one at
    lead 20, because by then the flow has decorrelated (the integral timescale is 1.1-9 frames,
    FINDINGS 3). At long leads the trajectory's MEAN is the better reference, and the
    ``stream_state`` frames supply it causally. This makes the anchor

    .. code-block::

        anchor(lead) = last_frame + w(lead) * (state_mean - last_frame)

    with ``w`` a learned per-lead vector **initialised at 0**, so the model starts exactly at
    today's behaviour and any departure has to be earned by the gradient. ``w = 1`` would anchor
    fully on the trajectory mean; nothing constrains it to [0, 1].

    That initialisation is the whole design. Two attempts to use the state as an IMPOSED
    transform have now failed -- full standardisation cost 0.11-0.12 priced, its scale half
    costing 0.63 `tke` -- while using it as an INPUT the network can weigh gained +0.195 on the
    board. So this is written as something learned and reversible rather than fixed: if the
    trajectory mean is not a better anchor, ``a`` stays at 1 and the arm costs nothing but the
    parameters.

    Sits INSIDE the state-frame split: it reads the state from the full input, and hands the
    backbone whatever its inner wrapper expects.
    """

    def __init__(self, backbone: nn.Module, n_state: int = 2, t_out: int = T_OUT) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_state = int(n_state)
        #: Per-lead weight on the move FROM the last frame TOWARDS the trajectory mean.
        #: Zero-initialised, so step 0 is EXACTLY today's model -- the same discipline as every
        #: other conditioning pathway here and as the state's own zero-init stem columns. A
        #: sigmoid parameterisation was tried first and rejected: sigmoid(4) = 0.982 leaves an
        #: 0.018 shift at init, so "starts at today's behaviour" would have been almost-true
        #: rather than true, and an arm that begins by perturbing the model cannot attribute
        #: what it then learns.
        self.weight = nn.Parameter(torch.zeros(t_out))

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        window = x[:, self.n_state :]
        out = self.backbone(x, cond) if _takes_cond(self.backbone) else self.backbone(x)
        pred = out["prediction"] if isinstance(out, dict) else out
        n = min(pred.shape[-1], C)
        last = window[:, -1:, ..., :n]
        mean = x[:, 0:1, ..., :n]
        w = self.weight.view(1, -1, 1, 1, 1)
        shift = w * (mean - last)                  # exactly 0 at init
        if pred.shape[-1] > n:
            shift = torch.nn.functional.pad(shift, (0, pred.shape[-1] - n))
        if isinstance(out, dict):
            p = pred + shift
            result = {"prediction": p}
            if "lower" in out:
                result["lower"] = out["lower"] + shift
                result["upper"] = out["upper"] + shift
            return result
        return pred + shift


class StateRefineWrapper(nn.Module):
    """Also hand the state to the FULL-RESOLUTION refinement, not only to the stem.

    The state currently enters once, at the stem, and has to survive the patch bottleneck: the
    trunk tokenises 2x2 patches, so a per-pixel mean-flow map is compressed before anything
    downstream sees it. The refinement runs at full resolution and is where fine structure is
    actually produced (FINDINGS 117's skip exists for exactly this reason), so that is the
    second place the state could be used at its native resolution.

    Implemented as an input-side wrapper rather than a change to STFormer: the state frames are
    appended to the window as EXTRA CHANNELS on every frame, which the stem's first convolution
    already knows how to consume (the channel axis is what ``widen_grouped_stem`` widens), so no
    architecture surgery and no second code path through the trunk.

    Honest prior, written before running it: this is a ROUTING change, and routing changes have
    now failed twice in this campaign -- the learned per-lead anchor (idea 3) recovered the
    physics exactly (the weight grew with lead, corr +0.91) and still scored +0.001, because the
    backbone could already reach the same function through its own residual. Expect a null; the
    reason to run it is that it is the last of the four and costs one fine-tune.
    """

    def __init__(self, backbone: nn.Module, n_state: int = 2) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_state = int(n_state)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        state = x[:, : self.n_state, ..., :2]                     # (B, n_state, H, W, 2)
        window = x[:, self.n_state :]                             # (B, T, H, W, C)
        planes = state.permute(0, 2, 3, 1, 4).reshape(
            state.shape[0], 1, *state.shape[2:4], self.n_state * 2
        ).expand(-1, window.shape[1], -1, -1, -1)
        xs = torch.cat((window, planes), dim=-1)
        return self.backbone(xs, cond) if _takes_cond(self.backbone) else self.backbone(xs)


class DampedResidualWrapper(nn.Module):
    """Predict a delta from a *damped-persistence* reference instead of the last frame.

    ``ref_t = w_t * x_last + (1 - w_t) * mean_conf(x)`` with ``w_t = exp(-t / tau)`` for lead
    ``t = 1..T`` and the window mean weighted by a confidence mask (a pixel counts in a frame
    when ``u`` or ``v`` is non-zero there; PIV holes and dropout are exact zeros, so a pixel
    that is never valid references zero). This is the reference field ``tubelet_former.py``
    carries inside itself, lifted to the pipeline's wrapper so any backbone can be trained
    against it with ``--residual --residual-mode damped`` and nothing else changed. FINDINGS
    114 measured the reference alone at +0.4 to +1.4 ceiling over persistence.

    Same contract as :class:`ResidualWrapper`: ``backbone`` attribute (so state-dict keys are
    identical), ``out = ref + backbone(x, cond)``, bounds as offsets around the prediction.
    No parameters and no buffers, so a checkpoint carries nothing extra.
    """

    def __init__(self, backbone: nn.Module, tau: float = 3.0) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.backbone = backbone
        self.tau = float(tau)

    def reference(self, x: torch.Tensor, c_out: int, t_out: int) -> torch.Tensor:
        """``(B, T_out, H, W, c_out)`` reference, in ``x``'s dtype."""
        xin = x[..., :c_out].float()
        conf = ((x[..., 0] != 0) | (x[..., 1] != 0)).to(xin.dtype)[..., None]  # (B,T,H,W,1)
        mass = conf.sum(dim=1, keepdim=True)
        mean = (xin * conf).sum(dim=1, keepdim=True) / mass.clamp_min(1e-6)
        last = xin[:, -1:]
        lead = torch.arange(1, t_out + 1, device=x.device, dtype=xin.dtype)
        w = torch.exp(-lead / self.tau).view(1, t_out, 1, 1, 1)
        return (w * last + (1.0 - w) * mean).to(x.dtype)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        out = self.backbone(x, cond) if _takes_cond(self.backbone) else self.backbone(x)
        pred_raw = out["prediction"] if isinstance(out, dict) else out
        ref = self.reference(x, pred_raw.shape[-1], pred_raw.shape[1])
        if isinstance(out, dict):
            pred = pred_raw + ref
            result = {"prediction": pred}
            if "lower" in out:
                result["lower"] = pred - out["lower"].abs()
                result["upper"] = pred + out["upper"].abs()
            return result
        return out + ref



#: Planes :class:`NbrPriorWrapper` appends: neighbour-prior mean u, mean v, std u, std v, and the
#: mean distance (in grid steps) of the neighbours it blended.
NBR_PRIOR_PLANES = 5
#: ``nbr_prior_mode`` -> number of planes appended.
NBR_PRIOR_MODES = {"all": 5, "std": 3}
#: The operating-point grid's steps, the unit of the neighbour metric: Reynolds rungs are 1275
#: apart (linear, 3750 ... 27975), angles 5 degrees.
NBR_RE_STEP, NBR_AOA_STEP = 1275.0, 5.0


class NbrPriorWrapper(nn.Module):
    """Hand the backbone the statistics of the NEAREST OTHER operating points it was trained on.

    FINDINGS 224. A 20-frame window says little about where a cell's fluctuation energy sits over
    a whole trajectory (the model uses 5-10 frames, FINDINGS 157), yet the per-pixel temporal std
    map of a cell correlates 0.985 with the map one Reynolds rung away (0.83 one angle step away).
    So the training cells' climatology is a strong prior for an operating point the model has
    never seen -- which is what the private test is. This wrapper looks it up from the (estimated)
    operating point in ``cond``: the ``k`` nearest table cells in grid steps, EXCLUDING any cell
    within half a step (the cell itself, when it is in the table), blended by inverse squared
    distance, and appended as constant planes on every frame. Excluding the cell itself at
    training AND at inference is what makes a training window look like a held-out one: the
    prior always comes from somewhere else.

    The table (``nbr_table``: per-cell time-mean and std of u, v, freestream-normalised, zero in
    the mask; ``nbr_coords``: (Re, AoA)) is a buffer, so it travels in the checkpoint and the
    served model needs no data. Input-side, inside the residual anchor, like
    :class:`StateRefineWrapper`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_cells: int,
        h: int = 32,
        w: int = 64,
        k: int = 4,
        mode: str = "all",
    ) -> None:
        super().__init__()
        if mode not in NBR_PRIOR_MODES:
            raise ValueError(f"nbr_prior_mode must be one of {sorted(NBR_PRIOR_MODES)}")
        self.backbone = backbone
        self.k = int(k)
        #: ``all``: mean u, mean v, std u, std v, distance; ``std``: std u, std v, distance.
        self.mode = mode
        self.register_buffer("nbr_table", torch.zeros(n_cells, h, w, 4))
        self.register_buffer("nbr_coords", torch.zeros(n_cells, 2))
        #: Training only: the batch's LABEL (Re, AoA), set by the loop before each forward, so the
        #: cell itself is excluded exactly. Without it (evaluation, serving) the rule is by
        #: distance to the estimate: no table cell within half a grid step is ever used.
        self._label: torch.Tensor | None = None

    def set_label(self, re: torch.Tensor | None, aoa: torch.Tensor | None) -> None:
        self._label = None if re is None else torch.stack((re.float(), aoa.float()), dim=1)

    @torch.no_grad()
    def set_table(self, coords: torch.Tensor, table: torch.Tensor) -> None:
        self.nbr_coords.copy_(coords)
        self.nbr_table.copy_(table)

    def planes(self, cond: torch.Tensor) -> torch.Tensor:
        """``(B, H, W, NBR_PRIOR_PLANES)`` from ``cond``'s normalised (log Re, AoA)."""
        from ..features import AOA_MAX, RE_MAX, RE_MIN  # noqa: PLC0415

        lr0, lr1 = float(torch.log(torch.tensor(RE_MIN))), float(torch.log(torch.tensor(RE_MAX)))
        c = cond[:, :2].float()
        re = torch.exp(c[:, 0] * (lr1 - lr0) + lr0)
        aoa = c[:, 1] * AOA_MAX
        dr = (re[:, None] - self.nbr_coords[None, :, 0]) / NBR_RE_STEP
        da = (aoa[:, None] - self.nbr_coords[None, :, 1]) / NBR_AOA_STEP
        d = torch.sqrt(dr * dr + da * da)                                 # (B, N)
        if self.training and self._label is not None:
            lab = self._label.to(d.device)
            dl = torch.sqrt(
                ((lab[:, None, 0] - self.nbr_coords[None, :, 0]) / NBR_RE_STEP) ** 2
                + ((lab[:, None, 1] - self.nbr_coords[None, :, 1]) / NBR_AOA_STEP) ** 2
            )
            own = dl < 0.5
        else:
            own = d < 0.5
        d = torch.where(own, torch.full_like(d, float("inf")), d)         # never the cell itself
        k = min(self.k, d.shape[1] - 1)
        dk, ik = torch.topk(d, k, dim=1, largest=False)                   # (B, k)
        wts = 1.0 / dk.clamp_min(1e-3) ** 2
        wts = wts / wts.sum(dim=1, keepdim=True)
        prior = (self.nbr_table[ik] * wts[:, :, None, None, None]).sum(dim=1)  # (B, H, W, 4)
        dist = (dk * wts).sum(dim=1)[:, None, None, None].expand(*prior.shape[:3], 1)
        if self.mode == "std":
            prior = prior[..., 2:4]
        return torch.cat((prior, dist), dim=-1)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if cond is None:
            raise ValueError("NbrPriorWrapper needs the operating point in cond")
        p = self.planes(cond).to(x.dtype)[:, None].expand(-1, x.shape[1], -1, -1, -1)
        xs = torch.cat((x, p), dim=-1)
        return self.backbone(xs, cond) if _takes_cond(self.backbone) else self.backbone(xs)

def _takes_cond(module: nn.Module) -> bool:
    import inspect

    try:
        return "cond" in inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False


def call_model(
    model: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor | None = None,
    u_fs: torch.Tensor | None = None,
):
    """Call a model, passing ``u_fs`` only if its forward accepts it.

    Only the closed-form (``scale``) interval head needs it, and threading it through
    every backbone's signature to serve one head would be the wrong trade. Introspection
    keeps the training loop, ``quick_eval`` and the submission path on one call site.
    """
    if u_fs is not None and _takes(model, "u_fs"):
        return model(x, cond, u_fs)
    return model(x, cond)


def _takes(module: nn.Module, name: str) -> bool:
    import inspect

    # ``torch.compile`` returns an OptimizedModule whose ``forward`` is a generic
    # ``(*args, **kwargs)`` wrapper, so introspecting it would silently answer False and
    # drop ``u_fs`` -- which the closed-form interval head needs. Unwrap to the original.
    target = getattr(module, "_orig_mod", module)
    try:
        return name in inspect.signature(target.forward).parameters
    except (TypeError, ValueError):
        return False


def split_output(
    out: torch.Tensor | dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Normalise a model's return value into ``(prediction, lower, upper)``."""
    if isinstance(out, dict):
        return out["prediction"], out.get("lower"), out.get("upper")
    return out, None, None
