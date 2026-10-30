"""Auxiliary decoders on a shared trunk, and the wrapper that attaches them.

The premise of the study these serve: the shipped sim pretrain optimises the task alone
(``w_field=1``, every other weight zero), and the question is whether *additional*
supervision on the same trunk produces a better 20->20 forecaster. So every head here is a
training-time device with two properties that make the comparison mean something.

**They read the features the task head reads.** Each head is attached to the output of the
U-Net's ``final_conv[0]`` -- the ``dim``-channel, full-resolution, all-frames
representation whose only other consumer is the 1x1 convolution that emits the prediction.
An auxiliary loss can therefore only help by changing *that* representation, which is the
mechanism being tested. A head bolted onto the bottleneck, or given its own decoder stack,
would instead be testing whether extra parameters help.

**They cost nothing at inference.** :meth:`AuxModel.forward` skips them entirely outside
training mode, so a scored forward pass runs the baseline architecture and nothing else;
and :func:`strip_aux` returns a state dict the un-wrapped backbone loads, which is what
makes "zero inference cost" a checkable claim rather than an assertion.

The heads, and what each is for:

``fine``
    The full ``64x128`` field the recording grid carries, at the ``20`` predicted frames.
    Deliberately ill-posed -- the fine field is not determined by an aliased ``32x64``
    input, so MSE yields the conditional mean. The failure mode is therefore degeneracy,
    not divergence: if that conditional mean is smooth the head collapses to bilinear
    upsampling and teaches the trunk nothing. :func:`upsample_baseline` exists to test for
    exactly that, and the result decides whether this head is worth anything.

``subgrid``
    Per-coarse-cell variance of the native field inside each ``2x2`` block. Better posed
    than ``fine`` on the same information -- a conditional *variance* is a function of the
    coarse field -- an eighth of the output width, and pointed at the second moment, which
    is the axis a conditional-mean predictor is known to under-serve.

``recon``
    The clean input window, where it was hidden by the mask. Masked spatio-temporal
    modelling, in the sense that the trunk must carry enough of the flow to inpaint its own
    input while still forecasting.

``paux``
    Pressure over the predicted frames, from a p-free input. Well-posed physics rather than
    hallucination (``lap p = -rho div(u . grad u)``), and it needs a global receptive field,
    which the bottleneck attention is there to supply.

``opreg``
    The operating point ``(log-Re, AoA)``, mean-pooled over the trunk's token axis rather
    than read per pixel -- unlike the other three heads this is not a spatial pretext task,
    it is a per-sample regression. Sim and real share this target but not label noise or
    realisation detail, so a shared head that predicts it well from *both* domains is
    supervising the embedding to keep exactly the content the joint stage's hypothesis is
    about: whether the trunk represents a window by its cell identity, in a form the two
    domains can both be read through. Needs a token-sequence trunk (``(B, N, C)``, e.g.
    STFormer's final ``norm``) rather than the dense ``(B, C, T, H, W)`` site the other
    three heads read, so :func:`find_trunk` resolves its width differently and
    :class:`AuxModel` skips the dense-field permute for it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .. import N_MEASURED, SUB_S
from .transformer import unpatchify

#: Where each backbone exposes the representation the output projection consumes. Adding a
#: backbone to this study means adding one entry; the alternative -- guessing a module by
#: position -- silently attaches the heads to the wrong tensor and still trains.
#:
#: ``STFormer``'s site is its final ``norm`` -- the post-LayerNorm token sequence that
#: both the coarse head and the refinement path read (``stformer.py``'s own ``forward``),
#: matching the site ``loop.FeatureTap``/``align_weight`` already tap for the same
#: architecture (FINDINGS 67e-bis).
TRUNK_SITES: dict[str, str] = {"UNet3d": "final_conv.0", "STFormer": "norm"}

#: Head names, in a fixed order so a config dict and a state dict agree.
AUX_HEADS = ("fine", "subgrid", "recon", "paux", "opreg")


def aux_weights(config: dict[str, Any]) -> dict[str, float]:
    """Which heads ``config`` asks for, and at what loss weight.

    ``recon`` is gated on the *mask* rather than on its own weight: reconstructing an
    unmasked input is the identity map on a channel the trunk was handed, which trains a
    head to do nothing and dilutes the task loss doing it.
    """
    w = {
        "fine": float(config.get("w_fine", 0.0)),
        "subgrid": float(config.get("w_subgrid", 0.0)),
        "recon": max(
            float(config.get("w_recon", 0.0)),
            float(config.get("w_recon_tke", 0.0)),
            float(config.get("w_recon_acf", 0.0)),
        )
        if config.get("aux_mask", "")
        else 0.0,
        "paux": float(config.get("w_paux", 0.0)),
        "opreg": float(config.get("w_opreg", 0.0)),
    }
    return {k: v for k, v in w.items() if v > 0}


def _resolve(module: nn.Module, path: str) -> nn.Module:
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def find_trunk(model: nn.Module) -> tuple[nn.Module, int]:
    """The module whose output the auxiliary heads read, and its channel count.

    Searches ``model`` for any backbone named in :data:`TRUNK_SITES`, so it works whether
    the backbone is bare or already inside a residual wrapper.

    Width comes from the resolved module's own shape rather than a table entry, so it can
    never drift from the checkpoint that built it. A ``LayerNorm`` site (STFormer's token
    sequence, ``(B, N, C)``) reads its ``normalised_shape``; anything else is assumed
    convolutional and reads the last ``Conv3d``'s output width (UNet3d's ``(B, C, T, H, W)``
    site), the same two tensor layouts :func:`domain_moment_loss` already has to
    distinguish for the same reason (FINDINGS 67e-bis).
    """
    for module in model.modules():
        site = TRUNK_SITES.get(type(module).__name__)
        if site is None:
            continue
        trunk = _resolve(module, site)
        if isinstance(trunk, nn.LayerNorm):
            width = trunk.normalized_shape[0]
        else:
            width = next(
                m.out_channels for m in reversed(list(trunk.modules())) if isinstance(m, nn.Conv3d)
            )
        return trunk, width
    raise ValueError(
        f"No backbone in {type(model).__name__} has a registered trunk site. "
        f"Known: {sorted(TRUNK_SITES)}."
    )


class _Head(nn.Module):
    """One 3x3 spatial mixing layer, then a pointwise projection.

    Two layers, not one: a pure 1x1 head can only re-weight channels the trunk already
    computed at that cell, so it would be too weak to represent a sub-grid quantity and the
    experiment would measure head capacity instead of the auxiliary signal. Two is also the
    ceiling -- the point is to make the *trunk* carry the information, and a deep head lets
    it not.
    """

    def __init__(self, width: int, out_channels: int) -> None:
        super().__init__()
        out = nn.Conv3d(width, out_channels, 1)
        # Zero-init the projection, so the head starts by predicting nothing and each
        # auxiliary term starts at its own "no information" value -- mean(target^2) for the
        # MSE terms, exactly 1.0 for the relative one. Measured without it, the ``subgrid``
        # term opened at a relative L2 of **31.6** against a field term of 0.72: a randomly
        # initialised head emitting O(1) values against a ~1e-5 target dominates the first
        # hundred steps and shoves the trunk somewhere the task loss then has to undo. The
        # same discipline as a zero-init adaLN gate, and for the same reason.
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.net = nn.Sequential(
            nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FineHead(nn.Module):
    """Decode ``SUB_S x`` the trunk's spatial resolution, by pixel shuffle.

    Sub-pixel convolution rather than a transposed convolution: at this resolution the
    checkerboard artefacts of a stride-2 transpose land squarely in the high-wavenumber band
    the head exists to supervise, which would make the target it teaches an artefact of the
    layer that teaches it.
    """

    def __init__(self, width: int, out_channels: int = N_MEASURED, factor: int = SUB_S) -> None:
        super().__init__()
        self.factor = factor
        self.out_channels = out_channels
        self.head = _Head(width, out_channels * factor * factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = x.shape
        y = self.head(x).view(b, self.out_channels, self.factor, self.factor, t, h, w)
        # (B, C, r1, r2, T, H, W) -> (B, C, T, H, r1, W, r2) -> (B, C, T, H*r1, W*r2)
        y = y.permute(0, 1, 4, 5, 2, 6, 3).reshape(
            b, self.out_channels, t, h * self.factor, w * self.factor
        )
        return y


class OpHead(nn.Module):
    """Regresses ``(log-Re, AoA)`` from a mean-pooled token-sequence trunk activation.

    Two linear layers around one nonlinearity -- enough to be more than a per-channel
    readout without giving the *head* the capacity to memorise what the *trunk* should be
    carrying, the same balance :class:`_Head` strikes for the dense pretext tasks. The final
    layer is zero-initialised for the same reason as every other head here: at step 0 the
    term is a constant (``mean(target^2)``, since the prediction is exactly zero) rather
    than a random guess that could dominate early training.

    Reads a ``(B, N, C)`` token sequence and pools over ``N`` -- the shape a token-sequence
    trunk (STFormer's ``norm``) emits, not the ``(B, C, T, H, W)`` dense field the other
    heads here read. Rejects anything else outright: silently pooling the wrong axis of a
    dense field would average over *space*, not tokens, and would still return a number.
    """

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"OpHead reads a (B, tokens, C) token sequence, got shape {tuple(x.shape)}"
            )
        return self.net(x.mean(dim=1))


class TokenReconHead(nn.Module):
    """Decode STFormer's post-norm tokens into the clean observed velocity window."""

    channels_last = True

    def __init__(
        self,
        width: int,
        patch: tuple[int, int, int],
        shape: tuple[int, int, int],
        out_channels: int = N_MEASURED,
    ) -> None:
        super().__init__()
        self.patch, self.shape, self.out_channels = patch, shape, out_channels
        volume = patch[0] * patch[1] * patch[2]
        self.proj = nn.Linear(width, volume * out_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TokenReconHead needs (B,tokens,C), got {tuple(x.shape)}")
        return unpatchify(self.proj(x), self.patch, self.shape, self.out_channels)


class AuxModel(nn.Module):
    """A backbone plus auxiliary heads, transparent to everything downstream.

    ``forward`` returns the backbone's own output unchanged in eval mode, and in training
    mode a dict carrying ``prediction`` (plus any bound keys the backbone emitted) and
    ``aux``, a name->tensor dict the loss consumes. All aux tensors are channels-last
    ``(B, T, H, W, C)``, matching every other tensor that crosses this boundary.
    """

    def __init__(self, backbone: nn.Module, config: dict[str, Any]) -> None:
        super().__init__()
        self.backbone = backbone
        weights = aux_weights(config)
        trunk, width = find_trunk(backbone)
        self.trunk_width = width

        heads: dict[str, nn.Module] = {}
        if "fine" in weights:
            heads["fine"] = FineHead(width)
        if "subgrid" in weights:
            heads["subgrid"] = _Head(width, N_MEASURED)
        if "recon" in weights:
            stformer = next((m for m in backbone.modules() if type(m).__name__ == "STFormer"), None)
            if isinstance(trunk, nn.LayerNorm):
                if stformer is None:
                    raise ValueError("token reconstruction needs the STFormer owning the trunk")
                t, h, w, _ = stformer.in_shape
                heads["recon"] = TokenReconHead(width, stformer.patch, (t, h, w))
            else:
                heads["recon"] = _Head(width, N_MEASURED)
        if "paux" in weights:
            heads["paux"] = _Head(width, 1)
        if "opreg" in weights:
            heads["opreg"] = OpHead(width)
        self.aux = nn.ModuleDict(heads)

        self._feat: torch.Tensor | None = None
        trunk.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        # Only while the heads will actually be read: holding a reference to a
        # full-resolution activation through every eval batch is a needless 20 MB.
        self._feat = output if (self.training and len(self.aux)) else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        out = self.backbone(x, cond)
        if not self.training or not len(self.aux):
            return out
        feat = self._feat
        if feat is None:
            raise RuntimeError("trunk hook did not fire; the backbone was not the one wrapped")
        self._feat = None
        # Every dense head returns (B, C, T, H, W); opreg returns (B, 2) -- a per-sample
        # regression has no spatial axes left to permute into channels-last.
        aux = {
            name: (
                y
                if y.dim() == 2 or getattr(self.aux[name], "channels_last", False)
                else y.permute(0, 2, 3, 4, 1)
            )
            for name, y in ((n, h(feat)) for n, h in self.aux.items())
        }
        result = {"aux": aux}
        result.update(out if isinstance(out, dict) else {"prediction": out})
        return result


def strip_aux(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """An :class:`AuxModel` state dict, reduced to the plain backbone's.

    Drops ``aux.*`` and unwraps the ``backbone.`` prefix, so the checkpoint loads into the
    architecture a submission builds. This is the concrete form of the claim that the
    auxiliary objectives are free at inference.
    """
    return {k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")}


def upsample_baseline(coarse: torch.Tensor, factor: int = SUB_S) -> torch.Tensor:
    """Bilinear upsample of a ``(B, T, H, W, C)`` prediction, as the ``fine`` head's control.

    The degeneracy test for ``FineHead``: if its error is no better than this, the head has
    learned interpolation and the auxiliary loss is a no-op dressed as supervision. Reported
    rather than assumed -- it is the documented failure mode of the idea.
    """
    b, t, h, w, c = coarse.shape
    flat = coarse.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    up = nn.functional.interpolate(flat, scale_factor=factor, mode="bilinear", align_corners=False)
    return up.reshape(b, t, c, h * factor, w * factor).permute(0, 1, 3, 4, 2)
