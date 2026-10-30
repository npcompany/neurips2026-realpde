"""Alternative uncertainty estimators, for comparison against the trained bound head.

Why compare at all. The head reaches Spearman 0.67 against the true per-element
``|error|``, and that ceiling — not its calibration — is what caps SPS at ~43 against
an oracle's 57.8. Post-hoc recalibration recovers +0.7 of that 14.7, because a
monotone transform cannot change a ranking. So the question is whether a *different
source* of uncertainty ranks the errors better, not whether the mapping can be tuned.

Three estimators, all producing per-element intervals around the **same deterministic
prediction**, so the other four subscores are untouched and only SPS moves:

* :func:`laplace_variance` — last-layer Laplace. Post-hoc, no retraining: fit a
  Gaussian posterior over the final linear layer's weights from the training Fisher,
  then propagate it to an output variance. Captures **epistemic** uncertainty only.
* :func:`mc_dropout_variance` — MC dropout over the final hidden features. Cheap
  because only the last layer is resampled, so the expensive trunk runs once.
* the trained ``BoundHead`` (see :mod:`rpde_t1.models.bounds`) — a learned variance
  head trained directly on the SPS functional.

An expectation worth stating up front, so the result is interpretable either way:
this task's uncertainty is dominated by **aleatoric** error — turbulence is genuinely
unpredictable at lead 20, where only ~33% of the variance is explainable. Laplace and
MC dropout both estimate *epistemic* uncertainty, which for a converged network on
in-distribution data is the small part. Both therefore need an added aleatoric term to
be competitive at all, and the interesting question is whether their *shape* across
space and lead time ranks errors better than the head's.

**Caveat on MC dropout.** Sampling dropout at test time on a network trained without
it changes the mean prediction and is not the method as published. The FNO baseline
has no dropout modules, so the honest comparison needs a dropout-trained backbone;
:func:`mc_dropout_variance` on a standard checkpoint is a lower bound on what the
method can do, and is labelled as such wherever it is reported.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn


def _feature_hook(layer: nn.Module, store: dict) -> torch.utils.hooks.RemovableHandle:
    """Capture a layer's input, which is the feature vector the last layer consumes."""

    def hook(_module, inputs, _output):
        store["phi"] = inputs[0].detach()

    return layer.register_forward_hook(hook)


def _is_readout(module: nn.Module) -> bool:
    """A Linear, or a 1x1 convolution -- which is a Linear over channels."""
    if isinstance(module, nn.Linear):
        return True
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return all(k == 1 for k in module.kernel_size)
    return False


def readout_in_features(layer: nn.Module) -> int:
    """Input width of a readout, whichever type it is."""
    return layer.in_features if isinstance(layer, nn.Linear) else layer.in_channels


def flatten_readout_features(layer: nn.Module, raw: torch.Tensor) -> torch.Tensor:
    """``(N, F)`` features from whatever layout the readout consumes.

    A ``Linear`` takes ``(..., F)``; a convolution takes ``(B, F, *spatial)``. Reshaping
    a conv's input as if the features were trailing would mix channels with space and
    silently produce a covariance over the wrong axis.
    """
    f = readout_in_features(layer)
    if isinstance(layer, nn.Linear):
        return raw.reshape(-1, f)
    return raw.movedim(1, -1).reshape(-1, f)


def last_layer(model: nn.Module, example: tuple | None = None) -> nn.Module:
    """The network's **readout** -- the last Linear or 1x1 conv actually executed.

    Resolved by execution order when ``example`` is given, because module *registration*
    order is not forward order. That distinction is not academic: on the 3D U-Net the
    registration-order rule selects ``mid_block2.mlp.1``, a FiLM conditioning projection
    in the middle of the trunk, and a Laplace posterior fitted there would be reported as
    predictive uncertainty while describing something else entirely. It happens to be
    right for the FNO baseline, where ``fc2`` is last in both orders, which is why it went
    unnoticed.

    Without ``example`` it falls back to registration order, preserving the original
    behaviour for callers that cannot supply an input.
    """
    candidates = [m for m in model.modules() if _is_readout(m)]
    if not candidates:
        raise ValueError("No Linear or 1x1-conv readout found; last-layer methods need one.")
    if example is None:
        return candidates[-1]

    order: list[nn.Module] = []
    handles = [m.register_forward_hook(lambda mod, *_: order.append(mod)) for m in candidates]
    try:
        with torch.no_grad():
            model(*example)
    finally:
        for h in handles:
            h.remove()
    if not order:
        raise ValueError("No readout executed during the probe forward pass.")
    return order[-1]


@torch.no_grad()
def fit_laplace(
    model: nn.Module,
    batches: Iterator[dict[str, torch.Tensor]],
    device: str | torch.device = "cuda",
    prior_precision: float = 1.0,
    noise_var: float = 1.0,
    max_elements: int = 2_000_000,
    probe: tuple | None = None,
) -> torch.Tensor:
    """Posterior covariance of the last layer's weights, as ``(F, F)``.

    A GGN/Fisher Laplace approximation under a Gaussian likelihood: the posterior
    precision is ``sum_n phi_n phi_n^T / noise_var + prior_precision * I``. Only the
    feature second moment is needed, so this is one pass over training data and a
    single ``F x F`` inverse -- ``F`` is 128 for the FNO baseline.

    Shared across output channels, which assumes the same feature covariance drives
    each. That is exact for an isotropic likelihood and adequate here, since ``u`` and
    ``v`` are scored with equal weight.
    """
    model = model.to(device).eval()
    layer = last_layer(model, example=probe) if probe is not None else last_layer(model)
    store: dict[str, torch.Tensor] = {}
    handle = _feature_hook(layer, store)

    n_features = readout_in_features(layer)
    moment = torch.zeros(n_features, n_features, dtype=torch.float64, device=device)
    seen = 0
    try:
        for batch in batches:
            model(batch["input"].to(device), batch["cond"].to(device))
            phi = flatten_readout_features(layer, store["phi"]).double()
            moment += phi.T @ phi
            seen += phi.shape[0]
            if seen >= max_elements:
                break
    finally:
        handle.remove()

    precision = moment / max(noise_var, 1e-12)
    precision += prior_precision * torch.eye(n_features, dtype=torch.float64, device=device)
    return torch.linalg.inv(precision).float()


@torch.no_grad()
def laplace_variance(
    model: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor,
    covariance: torch.Tensor,
    chunk: int = 1_000_000,
) -> torch.Tensor:
    """Epistemic variance per element, ``phi^T Sigma phi``, shaped like the output.

    Chunked over flattened elements: a single batch carries ~5M feature vectors of
    length 128, and forming them all at once would need tens of GB.
    """
    layer = last_layer(model, example=(x, cond))
    store: dict[str, torch.Tensor] = {}
    handle = _feature_hook(layer, store)
    try:
        out = model(x, cond)
        out = out["prediction"] if isinstance(out, dict) else out
    finally:
        handle.remove()

    phi = flatten_readout_features(layer, store["phi"])
    var = torch.empty(phi.shape[0], device=phi.device, dtype=phi.dtype)
    cov = covariance.to(phi.device, phi.dtype)
    for i in range(0, phi.shape[0], chunk):
        block = phi[i : i + chunk]
        var[i : i + chunk] = ((block @ cov) * block).sum(dim=-1)
    shape = (
        store["phi"].shape[:-1]
        if isinstance(layer, nn.Linear)
        else (store["phi"].shape[0], *store["phi"].shape[2:])
    )
    return var.reshape(shape)


@torch.no_grad()
def mc_dropout_variance(
    model: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor,
    n_samples: int = 16,
    p: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mean, variance)`` over dropout samples applied to the final features.

    The trunk runs **once**; dropout is resampled on the captured features and pushed
    through the last layer only. That keeps the cost at ``n_samples`` matmuls rather
    than ``n_samples`` full forward passes -- which matters because ``time_score``
    would otherwise collapse: FNO is 4.6 ms/sample on the evaluation GPU, so 16 true
    passes would take it from 90.6 to roughly 55.

    The price is that only last-layer weight uncertainty is explored, so this is a
    lower bound on full MC dropout rather than a substitute for it.
    """
    layer = last_layer(model)
    store: dict[str, torch.Tensor] = {}
    handle = _feature_hook(layer, store)
    try:
        model(x, cond)
    finally:
        handle.remove()

    phi = store["phi"]
    total = torch.zeros(*phi.shape[:-1], layer.out_features, device=phi.device, dtype=phi.dtype)
    total_sq = torch.zeros_like(total)
    for _ in range(n_samples):
        mask = (torch.rand(phi.shape, device=phi.device, generator=generator) > p).to(phi.dtype) / (
            1.0 - p
        )
        sample = layer(phi * mask)
        total += sample
        total_sq += sample * sample
    mean = total / n_samples
    var = (total_sq / n_samples - mean * mean).clamp_min(0.0)
    return mean, var


def bounds_from_variance(
    prediction: torch.Tensor,
    variance: torch.Tensor,
    k: float = 2.0,
    aleatoric: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric interval ``prediction +/- k * sqrt(variance + aleatoric)``.

    ``aleatoric`` is a fitted homoscedastic floor. Without it a purely epistemic
    estimate is far too narrow: the network is confident, and most of the error here
    is irreducible turbulence rather than parameter uncertainty. ``k`` and
    ``aleatoric`` are the two knobs to calibrate against SPS on a validation split --
    the comparison against the trained head is only fair if all three estimators get
    their scale tuned the same way.
    """
    scale = k * torch.sqrt(variance + aleatoric)
    return prediction - scale, prediction + scale


# --------------------------------------------------------------------------- #
# Dropout injection, for a *valid* MC-dropout comparison
# --------------------------------------------------------------------------- #
#
# MC dropout is only a posterior approximation if dropout was present during
# training. Sampling it at test time on a network trained without it measures the
# sensitivity of a layer to random masking -- dominated by feature and weight norms --
# not predictive uncertainty. Our first attempt did exactly that and scored Spearman
# 0.396 against the true error, which is what such a proxy deserves.
#
# The vendored FNO has no Dropout modules and vendor/ is never edited, so dropout is
# injected with forward hooks instead. Hooks hold no parameters, so ``state_dict`` is
# unchanged: a dropout-trained checkpoint still loads into the plain architecture, and
# can still be initialised from a checkpoint trained without dropout.


#: Per-architecture dropout sites. Matched on any component of the dotted module path, since the
#: backbone is normally nested under a wrapper.
#:
#: The FNO entry is the original: the pointwise convolutions inside each Fourier block and the
#: first head layer. The spectral weights are deliberately excluded -- dropping frequencies removes
#: whole modes rather than units, a much harsher prior.
#:
#: ``mlp`` covers the transformer families (STFormer, STFormer3D, the plain transformer, ModPCT),
#: whose blocks are ``blocks.<i>.{norm1,attn,norm2,mlp,modulation}``. Only the **MLP** is a site,
#: and that choice is the same one the FNO entry makes: dropping attention outputs removes whole
#: token interactions, and dropping the ``modulation`` linear would randomly disable the
#: conditioning path that FINDINGS 23z-ter measured as this architecture's one conditioning win.
DROPOUT_TARGETS: dict[str, tuple[str, ...]] = {
    "fno": ("convs", "fc1"),
    "transformer": ("mlp",),
    # ``unet2d.Branch``: after the first conv of each residual branch, before the FiLM'd norm.
    "unet2d": ("conv1",),
}


def dropout_targets(model: nn.Module) -> tuple[str, ...]:
    """Which module names carry dropout for this model, chosen by what it actually contains.

    Dispatching on structure rather than on a config string keeps this working for any wrapper
    (residual, bounds, aux) and for architectures added later, and it fails loudly rather than
    silently installing zero hooks -- which is what ``install_dropout`` raising exists to prevent.
    """
    names = {part for n, _ in model.named_modules() for part in n.split(".")}
    if "blocks" in names and "mlp" in names:
        return DROPOUT_TARGETS["transformer"]
    if "branch" in names and "conv1" in names:
        return DROPOUT_TARGETS["unet2d"]
    return DROPOUT_TARGETS["fno"]


#: Residual *branches* a stochastic-depth hook may drop, best-known-first.
#:
#: These are whole sublayers, not units: ``ConditionedBlock.forward`` is
#: ``x = x + (1+g1) * attn(...)`` then ``x + (1+g2) * mlp(...)``, so zeroing one branch's output
#: for a sample turns that block into the identity for it -- which is what stochastic depth *is*.
#: ``attn_s``/``attn_t`` are ``FactorizedBlock``'s two attention branches (STFormer3D).
#:
#: ``modulation`` is deliberately absent, for the same reason ``DROPOUT_TARGETS`` excludes it:
#: dropping it would randomly disable the conditioning path FINDINGS 23z-ter measured as this
#: architecture's one conditioning win, which is an ablation rather than a regulariser.
#: Element dropout zeroes activations; channel dropout (``Dropout2d``) zeroes whole feature maps
#: of a ``(B, C, H, W)`` output, the conv-net form -- neighbouring pixels of one map are so
#: correlated that dropping single activations regularises little. Falls back to element
#: dropout on outputs that are not 4-D.
DROPOUT_KINDS: tuple[str, ...] = ("element", "channel")


def install_dropout(
    model: nn.Module,
    p: float = 0.1,
    targets: tuple[str, ...] | None = None,
    kind: str = "element",
) -> list[torch.utils.hooks.RemovableHandle]:
    """Inject dropout after the named submodules, returning the hook handles.

    ``targets`` defaults to :func:`dropout_targets`, which picks the sites the model actually has.
    Before that existed this function hardcoded FNO's names and **raised on every transformer**,
    so ``TrainConfig.dropout`` was unusable on the entire STFormer lineage -- a flag that had been
    declared, documented and never exercised on the architecture now under evaluation.

    Dropout follows ``model.training``, so it is active while fitting and off at eval.
    Use :func:`mc_sampling` to force it on for test-time sampling.
    """
    if kind not in DROPOUT_KINDS:
        raise ValueError(f"dropout kind must be one of {DROPOUT_KINDS}, got {kind!r}")
    if targets is None:
        targets = dropout_targets(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    model._mc_dropout_active = False  # type: ignore[attr-defined]

    def hook(_module, _inputs, output):
        active = model.training or getattr(model, "_mc_dropout_active", False)
        if kind == "channel" and output.dim() == 4:
            return torch.nn.functional.dropout2d(output, p=p, training=active)
        return torch.nn.functional.dropout(output, p=p, training=active)

    for name, child in model.named_modules():
        # Match on any component of the dotted path, not just the first or last: the
        # backbone is usually nested under a wrapper, so a real target looks like
        # ``backbone.model.convs.0`` rather than ``convs.0``.
        if any(part in targets for part in name.split(".")) and isinstance(
            child, (nn.Conv3d, nn.Conv2d, nn.Conv1d, nn.Linear)
        ):
            handles.append(child.register_forward_hook(hook))
    if not handles:
        raise ValueError(f"No submodule matched {targets}; nothing to drop out.")
    return handles


#: Residual-branch names that stochastic depth drops. A ``ConditionedBlock`` is
#: ``x = x + (1+g1)*attn(...)`` then ``x + (1+g2)*mlp(...)``, so dropping the *output of these
#: two submodules* is exactly drop-path on that block -- the ``(1+g)`` modulation then scales a
#: zeroed branch to zero, and the skip connection carries the sample through untouched.
#: ``branch`` is ``unet2d.ResBlock``'s residual branch (``x + branch(x, cond)``), the CNN
#: analogue of the two transformer branches above.
DROP_PATH_TARGETS: tuple[str, ...] = ("attn", "mlp", "attn_s", "attn_t", "branch")


def install_drop_path(
    model: nn.Module,
    p: float = 0.1,
    targets: tuple[str, ...] = DROP_PATH_TARGETS,
    keep_holder: list | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Inject stochastic depth on the residual branches, returning the hook handles.

    Drop-path differs from :func:`install_dropout` in *granularity*, and that is the whole
    point of having both. Dropout zeroes individual activations inside the MLP, so every block
    still contributes to every sample; drop-path zeroes a block's entire branch for a whole
    sample, so the network trains as an ensemble of depths. On a residual stack the second is
    the stronger regulariser per unit of ``p``, which is why the conventional dose is similar
    (0.1) despite the much larger thing being removed.

    Per **sample**, not per element: the mask is broadcast over every axis but the batch, which
    is what makes this depth-ensembling rather than a coarse dropout. Survivors are scaled by
    ``1/(1-p)`` so the expected branch output is unchanged and inference needs no correction.

    Hooks carry no parameters and follow ``model.training``, so a drop-path-trained checkpoint
    loads into the plain architecture and runs as the identity at eval -- the same property that
    lets ``TrainConfig.dropout`` ship without touching the inference bundle.

    One granularity caveat, recorded rather than hidden: the mask is taken over dim 0 of the
    branch output. For ``ConditionedBlock`` that dim is the batch, so the drop is exactly
    per-sample. ``FactorizedBlock`` calls its spatial branch on a tensor whose dim 0 is
    ``batch * groups``, so there it is per (sample, group) -- a finer-grained variant of the same
    idea. Only the former is what the 2D arms use.

    **Per-cell (per-sample) dose.** ``keep_holder`` is an optional one-element list carrying a
    per-sample keep vector ``(B,)`` in ``(0, 1]`` (``keep = 1 - p_i``). When present, non-empty,
    and shaped to the batch dim of the branch output, each sample is dropped at its own rate --
    the mechanism behind an operating-point-dosed drop-path (more regularisation where the model
    over-fits, less at the hard corner where it needs depth). ``p`` above is then the *maximum*
    dose, used only for the ``[0, 1)`` guard and as the scalar fallback. The caller updates
    ``keep_holder[0]`` each step; a stale or mis-shaped vector (e.g. an auxiliary forward at a
    different batch size, or before the first update) falls back to the scalar ``p`` draw, so the
    hook is always well-defined.
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"drop_path p must be in [0, 1), got {p}")
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(_module, _inputs, output):
        if not model.training or p <= 0.0:
            return output
        if keep_holder is not None and keep_holder and keep_holder[0] is not None:
            kv = keep_holder[0]
            if kv.shape[0] == output.shape[0]:
                keep = kv.to(dtype=output.dtype, device=output.device).view(
                    (-1,) + (1,) * (output.dim() - 1)
                )
                mask = torch.bernoulli(keep.expand(output.shape[0], *([1] * (output.dim() - 1))))
                return output * mask / keep
        keep = 1.0 - p
        shape = (output.shape[0],) + (1,) * (output.dim() - 1)
        mask = torch.empty(shape, dtype=output.dtype, device=output.device).bernoulli_(keep)
        return output * mask / keep

    for name, child in model.named_modules():
        # The branch module itself, matched on the LAST path component -- not the Linears inside
        # it. `install_dropout` matches any component and picks Linear/Conv leaves, which is the
        # right rule for element dropout and the wrong one here: hooking `mlp.fc1` would drop a
        # sample's hidden activations, not its branch.
        if name.rsplit(".", 1)[-1] in targets and not isinstance(child, (nn.Linear, nn.Conv2d)):
            handles.append(child.register_forward_hook(hook))
    if not handles:
        raise ValueError(f"No residual branch matched {targets}; nothing to drop.")
    return handles


class mc_sampling:  # noqa: N801  (context manager, lowercase by convention here)
    """Force injected dropout on inside a ``no_grad`` eval block."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def __enter__(self) -> nn.Module:
        self.model._mc_dropout_active = True  # type: ignore[attr-defined]
        return self.model

    def __exit__(self, *exc: object) -> None:
        self.model._mc_dropout_active = False  # type: ignore[attr-defined]
