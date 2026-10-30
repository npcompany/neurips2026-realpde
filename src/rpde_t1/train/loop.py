"""The training loop, plus model construction and in-loop validation.

Two kinds of evaluation exist here and they serve different purposes:

* :func:`quick_eval` -- cheap, in-loop, non-dimensional. Used only for model
  selection and progress logging.
* the official scorer (see ``rpde_t1.train.evaluate``) -- the number we report.

Keeping them separate matters: the in-loop figure is convenient but is computed in
normalised space over a window subset, so it is *not* comparable to a leaderboard
score. Anything quoted as a score goes through the vendored scorer.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .. import N_MEASURED, SUB_S, T_IN, T_OUT, C, n_mask_planes
from ..data.dataset import REAL_RE_HI, REAL_RE_LO, make_dataset
from ..data.masking import sample_input_mask
from ..data.splits import SIM_ALL_BUT_REAL_VAL, SIM_HIFI, SIM_ONLY_ROWS, is_sim_split
from ..data.survey import base_kind, phase_kinds
from ..features import (
    COND_DIM,
    RE_MAX,
    RE_MIN,
    SIM_DOMAINS,
    cond_dim,
    n_flow_channels,
    parse_flow_channels,
)
from ..geometry import n_geometry_channels, parse_geom_channels
from ..models.aux import aux_weights, strip_aux, upsample_baseline
from ..models.factory import (
    BASELINES,
    OURS,
    STATE_WRAPPER_FLAGS,
    backbone_state_frames,
    build_backbone,
    stream_state_frames,
    wrap_model,
)
from ..models.kmap import widen_output_channels
from ..models.wrappers import call_model, split_output
from ..paths import REPO_ROOT, VENDOR_KIT
from .config import TrainConfig
from .head_credit import frozen_head, head_credit_loss, re_notch, re_taper
from .losses import (
    amplitude_free_field_loss,
    ceiling_loss,
    domain_correlation_loss,
    domain_moment_loss,
    energy_spectrum_loss,
    enstrophy_loss,
    error_homogeneity_loss,
    field_loss,
    frame_normalized_field_loss,
    helmholtz_field_loss,
    interval_calibration_loss,
    interval_credit_loss,
    kmap_e2e_log_ratio,
    kmap_loss,
    lead_weighted_field_loss,
    mean_field_loss,
    mean_fluct_field_loss,
    mvpe_loss,
    opreg_loss,
    ordering_penalty,
    pvar_loss,
    relative_field_loss,
    relative_l2,
    rstress_shape_loss,
    spectral_loss,
    sps_loss,
    subgrid_variance,
    tke_amplitude_loss,
    tke_huber_loss,
    tke_loss,
    tke_shape_loss,
    tke_temporal_profile_loss,
    unpredictability_gate,
    vorticity_loss,
    weighted_mse,
)
from .seeds import seed_everything, seed_worker
from .wandb_log import WandbRun, score_row, train_row, val_row

#: Re-exported from ``models.factory``, which owns model construction. They are not
#: redeclared here: two lists meant two registries to keep in step, and callers import
#: ``BASELINES`` from whichever module is already in scope.
__all__ = [
    "BASELINES",
    "OURS",
    "build_model",
    "geom_names",
    "train",
    "widen_cond_proj",
    "widen_fc0",
    "widen_init_conv",
]

#: Models that silently break under fp16 autocast, with the reason. Both were found
#: the hard way and cost a training run each, so they are recorded rather than
#: worked around.
#:
#: The Transolver failure is the nastier of the two: the loss goes NaN but
#: ``GradScaler`` then skips *every* optimiser step, so training "completes",
#: weights stay finite (they are still the initialisation), and the validation
#: numbers look merely poor instead of broken. Only the identical metrics across two
#: separate stages gave it away.
FP16_UNSAFE = {
    "fno": "cuFFT needs power-of-two signal sizes; FNO pads to (26, 38, 70)",
    "transolver": "Physics-Attention reduces over 40,960 tokens and overflows fp16",
}


#: log-Re anchors for the per-cell drop-path dose, computed once.
_LR_MIN, _LR_MAX = math.log(RE_MIN), math.log(RE_MAX)
_LR_LO, _LR_HI = math.log(REAL_RE_LO), math.log(REAL_RE_HI)


def drop_path_keep(cond0: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Per-sample keep probability ``1 - p_i`` for an operating-point-dosed drop-path.

    ``cond0`` is the conditioning vector's first component -- normalised log-Re in ``[0, 1]``
    (``features.conditioning_features``: ``(log re - log RE_MIN)/(log RE_MAX - log RE_MIN)``), so
    it is the same quantity for a labelled or an estimated operating point. The dose ``p_i`` runs
    linearly in log Re from ``lo`` at Re 3750 to ``hi`` at Re 26700, clamped outside -- the tensor
    form of ``dataset.re_dose`` on the real ladder. Returns ``1 - p_i`` (what the drop-path hook
    multiplies and divides by), so a simulated window at cond0 outside the real range clamps to an
    end dose rather than extrapolating one.
    """
    log_re = cond0.to(torch.float32) * (_LR_MAX - _LR_MIN) + _LR_MIN
    t = ((log_re - _LR_LO) / (_LR_HI - _LR_LO)).clamp(0.0, 1.0)
    p = lo + (hi - lo) * t
    return 1.0 - p


def geom_names(cfg: TrainConfig) -> tuple[str, ...]:
    """Geometry channel groups this config asks for, parsed once."""
    return parse_geom_channels(getattr(cfg, "geom_channels", ""))


def flow_names(cfg: TrainConfig) -> tuple[str, ...]:
    """Derived flow-plane groups this config asks for (``features.FLOW_CHANNELS``)."""
    return parse_flow_channels(getattr(cfg, "flow_channels", ""))


def dt_strides(cfg: TrainConfig) -> tuple[int, ...]:
    """All-to-all temporal strides this config asks for (``TrainConfig.dt_aug``)."""
    spec = getattr(cfg, "dt_aug", "") or ""
    return tuple(int(s) for s in spec.replace(" ", "").split(",") if s)


def dt_ratio_range(cfg: TrainConfig) -> tuple[float, ...]:
    """``(lo, hi)`` of the simulated time-step augmentation (``dt_ratio_aug``), or ``()``."""
    spec = getattr(cfg, "dt_ratio_aug", "") or ""
    vals = tuple(float(s) for s in spec.replace(" ", "").split(",") if s)
    if vals and len(vals) != 2:
        raise ValueError(f"dt_ratio_aug must be 'lo,hi', got {spec!r}")
    return vals


def widen_fc0(state: dict, n_extra: int) -> dict:
    """Insert ``n_extra`` zero input columns into FNO's ``fc0``, in the right place.

    FNO's only layer that sees the input channel count is ``fc0 = Linear(C_in + 3, width)``,
    where the ``+3`` is FNO's own coordinate grid **concatenated after** the data channels
    (``x = cat((x, grid), -1)`` in ``FNO3d.forward``). So new data channels belong at index
    ``C_in``, *not* appended at the end -- appending would silently permute the pretrained
    weights onto the wrong inputs, which fails no test and merely trains worse.

    With the columns zeroed the widened model is **bit-identical** to the original on the
    original channels regardless of what the new ones contain (verified:
    ``max|f_3(x) - f_10([x, junk])| = 0``), so a three-channel pretrain transfers exactly
    and the geometry starts as a no-op the fine-tune can grow into.
    """
    w = state.get("model.fc0.weight")
    if w is None or n_extra <= 0:
        return state
    c_in = w.shape[1] - 3  # the trailing three are the coordinate grid
    pad = w.new_zeros(w.shape[0], n_extra)
    state = dict(state)
    state["model.fc0.weight"] = torch.cat([w[:, :c_in], pad, w[:, c_in:]], dim=1)
    return state


#: Cumulative unfreeze stages (`FT.2`), shallowest first. Each stage is a set of substrings
#: matched against parameter names; a stage trains its own patterns *plus every shallower
#: stage's*, so the ladder is nested by construction and `all` is ordinary full fine-tuning.
#:
#: The premise is that 22.2M parameters against 69 real operating points is the wrong ratio,
#: and FINDINGS 28g is why it is worth testing now: a full-parameter fine-tune erases a
#: 6.47-point pretrain difference within 100 steps, so "how much of the trunk may move" is
#: the only lever measured to change that rate rather than fight it.
#:
#: Norm affines join at the first rung deliberately -- they are the cheapest per-parameter
#: way to re-fit a distribution shift, and leaving them frozen makes stage 1 a conditioning
#: probe rather than an adapter.
#:
#: **`norms` and `cond` are separate rungs because on this architecture they differ by 20x.**
#: NEW_ITEMS `CN.2` calls adaLN "the `FT.2` adapter" and `FT.2` describes rung 1 as "a few
#: 100k params", but STFormer's per-block modulation is `Linear(dim, 6*dim)` -- 0.89M a block,
#: 7.1M over 8 blocks, a third of the model. So the description and the mechanism name
#: different things here, and both get a rung: `norms` is the few-100k adapter the item
#: intends, `cond` is the adaLN pathway the mechanism intends.
UNFREEZE_LADDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("norms", ("norm", "cond_proj", "cond_const", "cond_pos", "film_", "attn_temp")),
    ("cond", ("modulation",)),
    ("io", ("stem", "embed", "pos", "head", "to_map", "refine")),
)


def apply_freeze_patterns(model: nn.Module, spec: str) -> tuple[int, int]:
    """Freeze every parameter whose name contains one of the comma-separated substrings.

    Returns ``(frozen, total)`` parameter counts; refuses a spec that matches nothing, since a
    freeze that silently froze nothing would be a control wearing an arm's name.
    """
    patterns = tuple(t for t in spec.split(",") if t)
    frozen = total = 0
    for name, prm in model.named_parameters():
        total += prm.numel()
        if any(pat in name for pat in patterns):
            prm.requires_grad_(False)
            frozen += prm.numel()
    if frozen == 0:
        raise ValueError(f"freeze_patterns={spec!r} matched no parameter of this model")
    return frozen, total


def llrd_groups(model: nn.Module, decay: float, backbone_scale: float, head_mult: float) -> list:
    """AdamW param groups for layer-wise lr decay, one per depth of the bare backbone.

    The backbone must expose ``lr_depth(name) -> int`` and ``n_depths``; the wrapper prefixes
    (``backbone.``, ``model.``) are stripped before the lookup. Depth ``d`` of ``n`` gets
    ``backbone_scale * decay ** (n - 1 - d)``; the output ``head`` keeps ``head_mult``.
    """
    bare = model
    while not hasattr(bare, "lr_depth") and hasattr(bare, "backbone"):
        bare = bare.backbone
    while not hasattr(bare, "lr_depth") and hasattr(bare, "model"):
        bare = bare.model
    if not hasattr(bare, "lr_depth"):
        raise ValueError(f"llrd needs a backbone with lr_depth(); {type(bare).__name__} has none")
    n = bare.n_depths
    by_depth: dict[int, list] = {}
    head = []
    for name, prm in bare.named_parameters():
        if not prm.requires_grad:
            continue
        if name.startswith("head"):
            head.append(prm)
        else:
            by_depth.setdefault(bare.lr_depth(name), []).append(prm)
    groups = [
        {"params": ps, "lr_scale": backbone_scale * decay ** (n - 1 - d)}
        for d, ps in sorted(by_depth.items())
    ]
    groups.append({"params": head, "lr_scale": head_mult})
    print(
        f"llrd={decay}: {len(groups) - 1} depth groups, stem at "
        f"{backbone_scale * decay ** (n - 1):.3g}x, head at {head_mult:g}x"
    )
    return groups


def unfreeze_patterns(spec: str) -> tuple[str, ...] | None:
    """Name substrings a stage spec allows to train, or ``None`` for "everything".

    ``spec`` is ``norms`` | ``cond`` | ``io`` | ``blocks:N`` | ``all`` (or empty).
    ``blocks:N`` adds the **last** ``N`` transformer blocks to everything ``io`` allows,
    because a fine-tune adapts the readout end of a network rather than its feature extractor.
    """
    spec = spec.strip()
    if not spec or spec == "all":
        return None
    stage, _, arg = spec.partition(":")
    names = [name for name, _ in UNFREEZE_LADDER]
    if stage not in (*names, "blocks"):
        raise ValueError(f"unfreeze must be one of {[*names, 'blocks:N', 'all']}, got {spec!r}")
    if arg and stage != "blocks":
        # Silently ignoring it would accept ``cond:2`` and train something other than what
        # was asked for, which is the failure this whole function is defensive about.
        raise ValueError(f"only blocks takes an argument; got {spec!r}")
    depth = names.index(stage) + 1 if stage in names else len(names)
    allowed: list[str] = []
    for _, patterns in UNFREEZE_LADDER[:depth]:
        allowed.extend(patterns)
    if stage == "blocks":
        n = int(arg) if arg else 1
        if n < 1:
            raise ValueError(f"blocks:N needs N >= 1, got {spec!r}")
        allowed.append(f"__last{n}blocks__")
    return tuple(allowed)


def apply_unfreeze(model: nn.Module, spec: str) -> tuple[int, int]:
    """Freeze every parameter a stage spec does not name. Returns ``(trainable, total)``.

    Raises if the spec matches nothing. That is the point of returning counts at all: this
    is exactly the class of change PROTOCOL 5 records silently no-op'ing three times, and a
    ladder that freezes everything trains a constant while still writing a plausible score.
    """
    allowed = unfreeze_patterns(spec)
    if allowed is None:
        return sum(p.numel() for p in model.parameters()), sum(
            p.numel() for p in model.parameters()
        )

    # ``blocks:N`` is resolved here rather than in the pattern list because which indices are
    # "last" is a property of the model, not of the spec.
    block_pats: list[str] = []
    marker = next((a for a in allowed if a.startswith("__last")), None)
    if marker is not None:
        n = int(marker[len("__last") : -len("blocks__")])
        idx = sorted(
            {
                int(part)
                for name, _ in model.named_parameters()
                if ".blocks." in f".{name}"
                for part in [f".{name}".split(".blocks.")[1].split(".")[0]]
                if part.isdigit()
            }
        )
        if not idx:
            raise ValueError("unfreeze blocks:N asked for transformer blocks; this model has none")
        block_pats = [f"blocks.{i}." for i in idx[-n:]]

    patterns = [a for a in allowed if not a.startswith("__last")] + block_pats
    total = trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        p.requires_grad_(any(pat in name for pat in patterns))
        trainable += p.numel() if p.requires_grad else 0
    if trainable == 0:
        raise ValueError(
            f"unfreeze={spec!r} matched no parameter of this model. Patterns were {patterns}; "
            f"first few parameter names are {[n for n, _ in model.named_parameters()][:6]}."
        )
    return trainable, total


def check_init_coverage(
    model: nn.Module,
    missing: list[str],
    unexpected: list[str],
    source: str,
    min_matched_frac: float = 0.5,
) -> None:
    """Refuse an ``--init-from`` load that matched too little to be a partial load.

    The load is deliberately non-strict, and three legitimate cases depend on that: a
    bound head is freshly initialised by design, a JEPA pretrain carries only an
    encoder, and a geometry-widened FNO gains input columns. Every one of those leaves a
    *small* remainder.

    What ``strict=False`` does **not** hide is a shape mismatch: a key present in both
    with different dimensions still raises from torch, so a mistyped width was never the
    silent case (pinned by ``test_a_width_mismatch_already_raises_without_the_guard``).
    The silent case is a *structural* difference -- a different depth, a different module
    set, the wrong ``--model`` -- where every key that exists matches and the rest are
    merely absent. That trains a mostly randomly-initialised model without complaint and
    scores plausibly, which is FINDINGS 12's wiring incident with a different cause: four
    of five pieces of plumbing present, the config file agreeing with the intent, and 3.3
    GPU-h spent before two identical rows of a score table gave it away.

    Half is far above any legitimate partial load and far below a mismatch.
    """
    n_total = len(model.state_dict())
    matched = n_total - len(missing)
    print(
        f"initialised from {source} "
        f"(matched={matched}/{n_total} missing={len(missing)} unexpected={len(unexpected)})"
    )
    if matched < min_matched_frac * n_total:
        raise RuntimeError(
            f"--init-from matched only {matched} of {n_total} tensors in {source}. That is "
            f"an architecture mismatch, not a partial load: check --model and --arch "
            f"against that run's config.json. First missing: {list(missing)[:5]}"
        )


def widen_init_conv(state: dict, want_in: int, key: str = "init_conv.weight") -> dict:
    """Pad UNet3d's stem to ``want_in`` input channels with **zeros**.

    The U-Net counterpart of :func:`widen_fc0`, and needed for the same reason: a 3-channel sim
    pretrain has to initialise a fine-tune whose input carries extra planes (the masking
    visibility channel, or geometry). ``load_state_dict`` raises on a size mismatch even with
    ``strict=False``, so without this the transfer is not merely lossy -- it is impossible.

    Appended rather than inserted, because ``WindowDataset`` builds the input as the three data
    channels followed by the visibility plane and then geometry, with nothing after. (FNO needs
    insertion at ``c_in`` instead, since it concatenates its own coordinate grid last -- which is
    the trap ``widen_fc0`` documents.)

    Zeroed, so the widened model is bit-identical to the original on the original channels
    whatever the new ones contain: the pretrain transfers exactly and the new plane starts as a
    no-op the fine-tune can grow into. A checkpoint already wide enough is returned untouched.
    """
    w = state.get(key)
    if w is None or w.shape[1] >= want_in:
        return state
    pad = w.new_zeros(w.shape[0], want_in - w.shape[1], *w.shape[2:])
    state = dict(state)
    state[key] = torch.cat([w, pad], dim=1)
    return state


#: Stem-parameter suffixes ``widen_grouped_stem`` widens: STFormer's ``Conv2d`` over folded
#: time slots, STFormer3D's factorised ``Conv3d``, and the simulated copy of either that
#: ``domain_interface=True`` adds. Every one that is present is widened; a run without the
#: split simply has fewer.
GROUPED_STEM_KEYS = (
    "stem.0.weight",
    "stem_sim.0.weight",
    "stem.0.0.weight",
    "stem_sim.0.0.weight",
)


def widen_grouped_stem(state: dict, c_old: int, want_in: int) -> dict:
    """Widen an STFormer stem, whose input columns are ``patch_t`` interleaved copies of ``C``.

    The third member of the ``widen_*`` family and the one with the trap. ``STFormer._to_groups``
    folds the window as ``(B, T, H, W, C) -> (B*gt, patch_t*C, H, W)`` **channel-fastest within a
    time slot**, so the stem's columns run ``[t0 u, t0 v, t0 p, t1 u, ...]``. Appending
    ``n_extra`` columns at the end would put every new plane after the last frame's pressure and
    shift nothing else -- which loads without error and silently feeds the pretrained weights a
    permuted input, exactly the failure :func:`widen_fc0` exists to prevent on FNO.

    So the zeros are **inserted after each block of ``c_old``**, once per time slot.
    Nothing here is read from the config. ``want_in`` comes from the *freshly built model's own*
    stem, ``patch_t`` is derived as ``have / c_old`` and the new channel count as
    ``want_in / patch_t``, so the layout is reconstructed from the two shapes that have to agree
    rather than from a default that would have to be kept in sync with the architecture forever
    (``evaluate.POST_HOC_DEFAULTS`` is the standing cost of getting that wrong). The two
    divisions are the check: a stem whose width is not a multiple of the old channel count also
    carries ``cond_content`` planes, where the placement is no longer inferable, and that raises
    instead of guessing. STFormer3D's factorised ``Conv3d`` stem takes ``C`` columns directly, so
    it derives ``patch_t = 1`` and this reduces to a plain append.

    Zeroed, so the widened model is bit-identical to the pretrain on the original channels
    whatever the new planes contain: the input arm's step 0 *is* the control's step 0, and the
    A/B measures what the model learns from the plane rather than an initialisation difference.
    A checkpoint already wide enough is returned untouched.
    """
    state = dict(state)
    for key in [k for k in state if any(k.endswith(s) for s in GROUPED_STEM_KEYS)]:
        w = state[key]
        have = w.shape[1]
        if have >= want_in:
            continue
        if have % c_old or want_in % (have // c_old):
            raise ValueError(
                f"{key}: cannot place new columns in a stem of {have} -> {want_in} input "
                f"channels at c_old={c_old}. A stem carrying cond_content planes is not "
                f"inferable this way; re-pretrain rather than widening."
            )
        patch_t = have // c_old
        n_extra = want_in // patch_t - c_old
        pad = w.new_zeros(w.shape[0], n_extra, *w.shape[2:])
        state[key] = torch.cat(
            [torch.cat([w[:, i * c_old : (i + 1) * c_old], pad], dim=1) for i in range(patch_t)],
            dim=1,
        )
    return state


def widen_stem_history(state: dict, t_old: int, t_new: int) -> dict:
    """Warm-start a longer input context from a shorter one, **bit-exactly**.

    STFormer folds its ``t_in`` frames into the stem's input columns time-slot-major and
    channel-fastest (``[t0 u, t0 v, t0 p, t1 u, ...]``, see :func:`widen_grouped_stem`), so a
    20-frame stem has ``20 * C`` columns and a 40-frame one ``40 * C``. The pretrained columns
    are placed on the slots of the **most recent** ``t_old`` frames and the older frames' slots
    are zero, so at step 0 the widened model computes exactly what the source computes on the
    last ``t_old`` frames whatever the older frames hold: the fine-tune delta is the value of
    the history and nothing else. Nothing but the stem's first convolution changes -- tokens,
    blocks, head and refinement are shared by construction (the token grid is the output's).

    Raises on anything it cannot place: a stem whose width is not ``t_old * C`` for an integer
    ``C``, or a ``temporal_anchor`` weight (``(t_out, t_in - 1)``), which has no zero-preserving
    placement and would need re-pretraining.
    """
    if t_new < t_old:
        raise ValueError(f"cannot shrink the input context from {t_old} to {t_new} frames")
    if t_new == t_old:
        return state
    state = dict(state)
    if any(k.endswith("anchor_weights") for k in state):
        raise ValueError("temporal_anchor weights cannot be widened to a longer context")
    for key in [k for k in state if any(k.endswith(s) for s in GROUPED_STEM_KEYS)]:
        w = state[key]
        have = w.shape[1]
        if have % t_old:
            raise ValueError(
                f"{key}: {have} stem columns is not a multiple of t_in={t_old}; a stem carrying "
                f"cond_content planes cannot be re-timed this way"
            )
        c = have // t_old
        pad = w.new_zeros(w.shape[0], (t_new - t_old) * c, *w.shape[2:])
        state[key] = torch.cat([pad, w], dim=1)
    return state


def widen_cond_proj(state: dict, want_in: int) -> dict:
    """Pad every conditioning projection to ``want_in`` input features with **zeros**.

    The third member of the family above, for the *conditioning* vector rather than the
    input field: ``cond_mode="operating_point_tau"`` appends the decorrelation time, so the
    first ``Linear`` of each conditioning MLP grows from 3 columns to 4 and a pretrain
    trained without it cannot be loaded at all.

    Zeroing the new column is what makes the resulting A/B one variable. The widened model
    is bit-identical to the pretrain *whatever tau is*, so the fine-tune starts from exactly
    the arm it is being compared against and has to earn any use it makes of the new
    feature -- rather than starting from a different random perturbation of it.

    Keyed by shape rather than by name because the four backbones call this layer four
    different things (``cond_proj``, ``scalar_embed``, and both under a ``backbone.``
    prefix once wrapped); what identifies it is that it is a 2-D weight whose input width
    is the conditioning width. A checkpoint already wide enough is returned untouched.
    """
    out = dict(state)
    for key, w in state.items():
        if not key.endswith(".weight") or w.ndim != 2 or w.shape[1] >= want_in:
            continue
        if w.shape[1] != COND_DIM:
            continue
        out[key] = torch.cat([w, w.new_zeros(w.shape[0], want_in - w.shape[1])], dim=1)
    return out


def nbr_prior_cells(cfg: TrainConfig) -> list[tuple[int, int]]:
    """The real cells ``cfg.train_split`` trains on: ``nbr_prior``'s table (FINDINGS 224). Never a
    held-out cell, so a validation operating point can only ever see its neighbours."""
    from ..data import dataset as _ds  # noqa: PLC0415 - avoids a cycle at import time

    if cfg.train_kind != "real":
        raise ValueError("nbr_prior is defined for the real fine-tune (train_kind real)")
    split = cfg.train_split
    every = [(_ds.parse_params(p).re, _ds.parse_params(p).aoa) for p in _ds.all_cells()]
    if split == "all":
        return every
    if split.startswith("all_but_"):
        held = set(_ds.split_cells(split[len("all_but_") :]))
        return [c for c in every if c not in held]
    return list(_ds.split_cells(split))


def fill_nbr_table(model: nn.Module, cells: list[tuple[int, int]]) -> None:
    """Write the cells' mean/std climatology into every ``NbrPriorWrapper`` in ``model``."""
    from ..data.climatology import build_climatology, build_climatology_std  # noqa: PLC0415
    from ..models.wrappers import NbrPriorWrapper  # noqa: PLC0415

    mean, std = build_climatology("real"), build_climatology_std("real")
    table = torch.from_numpy(np.stack([np.concatenate([mean[c], std[c]], -1) for c in cells]))
    coords = torch.tensor([[float(r), float(a)] for r, a in cells])
    found = 0
    for mod in model.modules():
        if isinstance(mod, NbrPriorWrapper):
            mod.set_table(coords.to(mod.nbr_coords.device), table.to(mod.nbr_table.device))
            found += 1
    if not found:
        raise ValueError("nbr_prior is set but the model has no NbrPriorWrapper")


def build_model(cfg: TrainConfig, in_shape=None, out_shape=(20, 32, 64, 3)) -> nn.Module:
    """Construct a model for training, from ``models.factory``.

    The architecture itself is built by the same code an inference bundle runs, so a
    checkpoint written here can always be rebuilt there. Only the two things that make
    no sense off a training config live in this function: disabling fp16 for the
    architectures that break under it, and injecting dropout hooks into the backbone
    before it is wrapped.
    """
    if in_shape is None:
        # ``stream_state`` pseudo-frames widen the stem exactly as a longer raw context does;
        # ``models.factory.stream_state_frames`` is the one home for the count.
        # ``backbone_state_frames`` is the one place that knows which state wrappers consume
        # the pseudo-frames before the stem (``state_norm`` uses them as the normaliser,
        # ``state_planes`` re-emits them as channels) and which leave them as frames.
        in_shape = (cfg.t_in + backbone_state_frames(asdict(cfg)), 32, 64, 3)
    if getattr(cfg, "kmap_channel", False) and out_shape[-1] == C:
        # The variance-map channel (models/kmap.py): frames 0..2 plus one map channel.
        out_shape = (*out_shape[:-1], C + 1)
    if getattr(cfg, "moment_joint", False) and out_shape[-1] == C:
        if getattr(cfg, "moment_of", ""):
            raise ValueError("moment_joint and moment_of are exclusive")
        out_shape = (*out_shape[:-1], C + N_MEASURED)  # frames + log sigma^2 (u, v)
    if getattr(cfg, "moment_of", ""):
        # A moment network emits log sigma^2 for u and v; it is not a field, so no residual.
        out_shape = (*out_shape[:-1], N_MEASURED)
        if cfg.residual:
            print("note: moment_of set -- residual wrapper forced off (the output is log sigma^2)")
            cfg.residual = False
    config = asdict(cfg)
    model = build_backbone(config, kit_dir=VENDOR_KIT, in_shape=in_shape, out_shape=out_shape)

    if cfg.model in FP16_UNSAFE and cfg.amp:
        print(f"note: fp16 autocast disabled for {cfg.model} ({FP16_UNSAFE[cfg.model]})")
        cfg.amp = False

    if cfg.dropout > 0:
        # On the bare backbone, before wrapping: the hooks are attached per module, and
        # the interval head is a separate estimator that must not be sampled with it.
        from ..models.uq import install_dropout  # noqa: PLC0415

        sites = tuple(t for t in cfg.dropout_sites.split(",") if t) or None
        n = len(install_dropout(model, p=cfg.dropout, targets=sites, kind=cfg.dropout_kind))
        print(
            f"note: {cfg.dropout_kind} dropout p={cfg.dropout} injected at {n} sites, "
            f"targets={sites or 'default'}"
        )

    dp_holder: list | None = None
    if cfg.drop_path > 0:
        # Same placement argument as dropout above, and independent of it: the two hook
        # different modules (branch output vs the Linears inside it) and compose.
        from ..models.uq import install_drop_path  # noqa: PLC0415

        # A one-element holder carries the per-sample keep vector for an operating-point-dosed
        # drop-path (cfg.drop_path_hi); None keeps the flat scalar dose. The training loop fills
        # holder[0] each step from the batch's Reynolds (see `drop_path_keep`).
        dp_holder = [None] if cfg.drop_path_hi is not None else None
        n = len(install_drop_path(model, p=cfg.drop_path, keep_holder=dp_holder))
        dose = f" -> {cfg.drop_path_hi} per-cell (re_dose)" if cfg.drop_path_hi is not None else ""
        print(f"note: drop_path p={cfg.drop_path}{dose} injected at {n} residual branches")

    wrapped = wrap_model(model, config)
    if dp_holder is not None:
        # Reachable from the loop, which holds the wrapped model; the hook closes over the same
        # list object, so wrapping does not hide it.
        wrapped._dp_keep_holder = dp_holder  # type: ignore[attr-defined]
    return wrapped


#: Simulated splits a joint stage may draw its auxiliary half from. Restricted to this list
#: rather than any split name: `sim_val_*` are the sets the sim benchmark scores on, and
#: training on one of them silently is the failure PROTOCOL 5 opens with.
#: `sim_all_but_real_val` is the grid minus real's validation Reynolds rows (splits.py);
#: `sim_hifi` is the grid minus the low-incidence, high-Reynolds corner where simulation is
#: measurably least faithful (FINDINGS 175), which is a fidelity cut rather than a coverage one.
SIM_SPLIT_CHOICES = ("sim_train", "sim_all", SIM_ALL_BUT_REAL_VAL, SIM_ONLY_ROWS, SIM_HIFI)


def joint_sim_real(real_ds, cfg: TrainConfig) -> tuple[ConcatDataset, WeightedRandomSampler]:
    """Mix a simulated minority into every real fine-tuning batch (``joint_sim_frac``).

    A weighted sampler over ``ConcatDataset`` rather than two loaders read in step: the
    mixture is then exact in expectation at *any* batch size, and a step is one optimiser
    step regardless of the fraction, so a joint arm and a real-only arm at the same
    ``steps`` have seen the same number of gradients.

    The simulated half is built the way the **pretrain's** sim data was built, read off the
    checkpoint being fine-tuned rather than respecified here. An anchor to the pretraining
    distribution is only an anchor if it is that same distribution: a fluctuation-rescaled
    pretrain anchored to unrescaled sim would pull the weights toward data they were never
    fitted on, which is a distribution shift wearing a regulariser's clothes.

    Both parts of that shape can be overridden when the question is instead "which simulated
    data is the best auxiliary source": ``joint_sim_kind`` for the grid and
    ``joint_rescale_fluct`` for the fluctuation amplitude. Neither has a *value* that means
    "inherit" -- ``""`` and ``None`` do -- so an override is always deliberate.
    """
    shape: dict[str, object] = {"zero_p": cfg.zero_p, "rescale_fluct": cfg.rescale_fluct}
    sim_kind = "sim"
    # Same rule as the two booleans, and for the same reason: pressure supervision is part of
    # what the simulated half *is*, so a p-supervised pretrain's anchor keeps it. Naming
    # ``--p-target`` on the joint stage overrides, which is how "pretrain only" and
    # "pretrain + joint" are separable arms.
    p_target = cfg.p_target
    if cfg.init_from:
        pre = torch.load(cfg.init_from, map_location="cpu").get("config", {})
        shape = {k: bool(pre.get(k, v)) for k, v in shape.items()}
        p_target = p_target or str(pre.get("p_target", ""))
        if pre.get("train_kind") in SIM_DOMAINS:
            sim_kind = pre["train_kind"]
    shape["p_target"] = p_target
    # The escape hatch the inheritance above never had. Applied *after* it, so ``None`` keeps the
    # anchor semantics and an explicit value wins -- which is the only way `--rescale-fluct` can
    # reach the one stage that has simulated data to rescale (FINDINGS 52j).
    if cfg.joint_rescale_fluct is not None:
        shape["rescale_fluct"] = bool(cfg.joint_rescale_fluct)
    if cfg.joint_sim_kind:
        # Explicit choice of auxiliary source, overriding what the pretrain used.
        if cfg.joint_sim_kind not in SIM_DOMAINS:
            raise ValueError(f"joint_sim_kind must be one of {sorted(SIM_DOMAINS)}")
        sim_kind = cfg.joint_sim_kind
    # Defaults to ``sim_train`` and not ``sim_all``: the sim grid covers the Reynolds rows
    # real holds out, and feeding them here shows the model those operating points in the
    # simulated domain, which makes val_interp and val_extrap in-sample as *conditioning*
    # even though no real target is in the batch. ``joint_sim_split`` opens that up for the
    # question "is the whole simulated grid the better anchor", and its docstring records
    # what the two OOD splits then mean.
    # Validated rather than trusted: a typo in this string would otherwise fall through to the
    # `sim_train` branch and train a different mixture than the config records.
    if cfg.joint_sim_split not in SIM_SPLIT_CHOICES:
        raise ValueError(f"joint_sim_split must be one of {sorted(SIM_SPLIT_CHOICES)}")
    sim_ds = make_dataset(
        cfg.joint_sim_split,
        kind=sim_kind,
        stride=cfg.stride,
        t_in=cfg.t_in,
        stream_state=cfg.stream_state,
        normalizer=cfg.normalizer,
        # The sim-only augmentations belong HERE, not on the real dataset above, which
        # gates them off for being real. Without this forwarding, `--noise-scale 0.1` on a
        # joint fine-tune reaches nothing at all while config.json records it -- FINDINGS
        # section 12's failure exactly. This is the one place a joint run sees sim data, so
        # it is the one place the sim2real measurement bridge can be applied.
        noise_scale=cfg.noise_scale,
        mask_prob=cfg.mask_prob,
        geom_channels=geom_names(cfg),
        flow_channels=flow_names(cfg),
        cond_mode=cfg.cond_mode,
        cond_source=cfg.cond_source,
        # The time-step augmentation reaches the joint stage's simulated half here; the real
        # half takes ``dt_aug`` (whole steps) through the real dataset above.
        dt_ratio_aug=dt_ratio_range(cfg),
        dt_ratio_p1=cfg.dt_ratio_p1,
        seed=cfg.seed,
        **shape,  # type: ignore[arg-type]
    )
    f = cfg.joint_sim_frac
    print(
        f"joint fine-tune: {len(real_ds)} real + {len(sim_ds)} "
        f"{cfg.joint_sim_split}/{sim_kind} windows, "
        f"sim fraction {f:.2f}"
        + (f" annealing to {cfg.joint_sim_frac_end:.2f}" if cfg.joint_sim_frac_end >= 0 else "")
        + f", sim built with {shape}"
    )
    sampler = WeightedRandomSampler(
        sim_fraction_weights(len(real_ds), len(sim_ds), f),
        num_samples=len(real_ds),
        replacement=True,
        # Seeded for the reason ``re_tilted_sampler`` states: unseeded, the realised mixture
        # is drawn from whatever the global CPU stream happens to hold at each epoch, so the
        # sim/real composition of an arm was not a function of its seed alone.
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    return ConcatDataset([real_ds, sim_ds]), sampler


def sim_fraction_weights(n_real: int, n_sim: int, f: float) -> list[float]:
    """Per-window sampling weights giving a simulated share of exactly ``f`` in expectation."""
    return [(1.0 - f) / n_real] * n_real + [f / n_sim] * n_sim


def retune_sim_fraction(loader: DataLoader, cfg: TrainConfig, progress: float) -> float | None:
    """Move the joint sampler's simulated share along the anneal. Returns it, or None.

    Called where the training loop rebuilds the loader's iterator, because that is the only
    point a ``WeightedRandomSampler`` re-reads its weights -- it draws a whole epoch of indices
    in one ``multinomial``. Doing it per step would look like a finer schedule in the config and
    be the same staircase in the data, which is the class of silent disagreement PROTOCOL 5
    keeps finding.
    """
    if cfg.joint_sim_frac_end < 0 or cfg.joint_sim_frac <= 0:
        return None
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, WeightedRandomSampler):
        return None
    start, end = cfg.joint_sim_frac, cfg.joint_sim_frac_end
    f = start + (end - start) * min(1.0, max(0.0, progress))
    sizes = getattr(loader.dataset, "cumulative_sizes", None)
    if not sizes or len(sizes) != 2:
        raise ValueError(
            "joint_sim_frac_end needs the two-part real+sim ConcatDataset that "
            "joint_sim_real builds; got a dataset with cumulative_sizes="
            f"{sizes!r}. Refusing to train on a mixture that is not the one configured."
        )
    n_real, n_sim = sizes[0], sizes[1] - sizes[0]
    sampler.weights = torch.as_tensor(sim_fraction_weights(n_real, n_sim, f), dtype=torch.double)
    return f


def re_tilted_sampler(dataset, tilt: float, seed: int = 0) -> WeightedRandomSampler:
    """Draw windows with probability proportional to ``(Re / Re_max) ** tilt``.

    The continuous form of "train a specialist on the fast cells". A hard split answers the
    same question but costs the rest of the ladder and needs a router at inference, on the
    axis FINDINGS 5f measures as the *smooth* one -- adjacent-Re climatology maps correlate
    0.935 against 0.735 for adjacent AoA. The tilt keeps every window reachable and makes
    "how specialised" a dial rather than a fork, so one sweep interpolates between the
    generalist and the specialist instead of comparing two incomparable training sets.

    What it is for: the per-window study found the model's deficit concentrated at high Re
    and *not* explained by missing input information (conditioning on the decorrelation
    time changed nothing). Two hypotheses survive -- the flow's phase is genuinely
    unrecoverable at 20 frames there, or capacity is being spent elsewhere on the ladder.
    Only the second one moves under reweighting, so this separates them.

    Sampled **with replacement** and for exactly ``len(dataset)`` draws, so a step is the
    same number of gradients at every tilt and the arms differ in composition alone. The
    generator is seeded from ``cfg.seed`` for the same reason the rest of the loop is:
    an A/B whose data order also changed is two variables.

    ``tilt=0`` is not merely equivalent to the unweighted loader -- it is bypassed by the
    caller, so the control arm keeps plain shuffling and stays bit-comparable to every run
    that predates this function.
    """
    re = _dataset_window_re(dataset)
    weights = (re / re.max()) ** float(tilt)
    weights = weights / weights.sum()
    effective = float(1.0 / np.square(weights).sum())
    print(
        f"re_tilt={tilt:g}: {len(re)} windows, effective sample size {effective:.0f} "
        f"({100 * effective / len(re):.1f}%), weight ratio {weights.max() / weights.min():.3g}"
    )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(re),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _dataset_window_re(dataset) -> np.ndarray:
    """Per-window Reynolds number, seeing through the phase-augmentation concatenation."""
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([_dataset_window_re(d) for d in dataset.datasets])
    return dataset.window_re()


def make_loaders(cfg: TrainConfig) -> tuple[DataLoader, dict[str, DataLoader]]:
    if cfg.zero_p and cfg.p_target:
        # Contradictory, and it fails in the direction that looks like a control. ``p_target``
        # blanks the input plane itself and then *restores* a gauged target, deliberately
        # overriding ``zero_p`` -- so passing both does not turn pressure off, it leaves it on
        # while the command line reads as though it were off. Measured cost of not checking: a
        # `pftc` control arm launched with both flags is a bit-identical duplicate of the `pft`
        # arm it is supposed to be the control for, and nothing downstream disagrees.
        raise ValueError(
            "zero_p and p_target contradict: p_target already blanks the input plane and "
            "supervises a gauged target, so --zero-p cannot switch it off. Pass exactly one "
            "(--zero-p for a pressure-free arm, --p-target gauge for a supervised one)."
        )
    if cfg.mask_prob > 0 and cfg.zero_p:
        # ``mask_prob`` is the organizers' pressure dropout: with that probability it zeroes the
        # simulated ``p`` plane so the model sees the real data's missing-pressure case. Under
        # ``zero_p`` that plane is *already* zero in both input and target, so the branch sets zero
        # to zero and the flag is a strict no-op -- and in a config with no other stochastic
        # augmentation it does not even perturb the random stream in a way anything reads.
        #
        # Measured consequence, and it is why this raises rather than warns:
        # ``scripts/train/chain_real.sh`` hardcodes ``--zero-p`` on the sim pretrain, and
        # ``scripts/train/pretrain_levers.sh`` layers ``--mask-prob 0.1`` on top of it, so ROADMAP
        # N25's
        # refutation of this flag was measured on an arm that saw **bit-identical data to its
        # control** (FINDINGS 52o).
        raise ValueError(
            "mask_prob is the pressure dropout and zero_p already zeroes that plane, so the two "
            "together are a no-op that reads as a configured arm. Drop --zero-p to make the "
            "dropout meaningful, or drop --mask-prob."
        )
    train_ds = make_dataset(
        cfg.train_split,
        kind=cfg.train_kind,
        stride=cfg.stride,
        t_in=cfg.t_in,
        **({"t_out": 2 * T_OUT} if cfg.rollout_w > 0 else {}),
        normalizer=cfg.normalizer,
        mask_prob=cfg.mask_prob,
        mask_swap_prob=cfg.mask_swap_prob,
        noise_scale=cfg.noise_scale,
        real_noise_scale=cfg.real_noise_scale,
        fluct_aug=cfg.fluct_aug,
        aoa_jitter_deg=cfg.aoa_jitter_deg,
        fluct_aug_hi=cfg.fluct_aug_hi,
        tgt_amp_aug=cfg.tgt_amp_aug,
        tgt_amp_aug_lo=cfg.tgt_amp_aug_lo,
        tgt_fluct_aug=cfg.tgt_fluct_aug,
        tgt_fluct_aug_hi=cfg.tgt_fluct_aug_hi,
        mean_aug=cfg.mean_aug,
        zero_p=cfg.zero_p,
        rescale_fluct=cfg.rescale_fluct,
        geom_channels=geom_names(cfg),
        flow_channels=flow_names(cfg),
        cond_mode=cfg.cond_mode,
        cond_source=cfg.cond_source,
        piv_noise=cfg.piv_noise,
        piv_noise_target=cfg.piv_noise_target,
        aux_mask_mode=cfg.aux_mask,
        aux_mask_ratio=cfg.aux_mask_ratio,
        aux_mask_plane=cfg.aux_mask_plane,
        aux_p=cfg.w_paux > 0,
        p_target=cfg.p_target,
        # Both resolution heads are supervised from the native cache, so either one asks
        # for it. Requested here rather than per-head so the dataset reads the array once.
        fine=cfg.w_fine > 0 or cfg.w_subgrid > 0,
        dt_aug=dt_strides(cfg),
        dt_ratio_aug=dt_ratio_range(cfg),
        dt_ratio_p1=cfg.dt_ratio_p1,
        history_dropout=cfg.history_dropout,
        recent_dropout=cfg.recent_dropout,
        recent_dropout_max=cfg.recent_dropout_max,
        stream_state=cfg.stream_state,
        state_dropout=cfg.state_dropout,
        state_dropout_max=cfg.state_dropout_max,
        seed=cfg.seed,
    )
    if cfg.phase_aug:
        # The scored phase plus the three unscored ones, concatenated: four views of the
        # same trajectories, so a window is 4x more likely to be novel at a given step.
        # Only the train loader -- the validation loaders below stay on phase (0, 0),
        # because that is the phase the competition scores.
        # EVERY shaping flag the scored phase gets, the other three must get too. Omitting
        # them is invisible on real -- rescale_fluct is sim-only and zero_p is a no-op where
        # there is no pressure -- but on a *simulated* pretrain it silently trains 3/4 of the
        # windows on unrescaled, pressure-supervised data, which is a different dataset
        # rather than a different phase of the same one.
        extra = [
            make_dataset(
                cfg.train_split,
                kind=kind,
                stride=cfg.stride,
                t_in=cfg.t_in,
                normalizer=cfg.normalizer,
                mask_prob=cfg.mask_prob,
                mask_swap_prob=cfg.mask_swap_prob,
                noise_scale=cfg.noise_scale,
                real_noise_scale=cfg.real_noise_scale,
                fluct_aug=cfg.fluct_aug,
                fluct_aug_hi=cfg.fluct_aug_hi,
                tgt_amp_aug=cfg.tgt_amp_aug,
                tgt_amp_aug_lo=cfg.tgt_amp_aug_lo,
                tgt_fluct_aug=cfg.tgt_fluct_aug,
                tgt_fluct_aug_hi=cfg.tgt_fluct_aug_hi,
                mean_aug=cfg.mean_aug,
                zero_p=cfg.zero_p,
                rescale_fluct=cfg.rescale_fluct,
                geom_channels=geom_names(cfg),
                flow_channels=flow_names(cfg),
                cond_mode=cfg.cond_mode,
                cond_source=cfg.cond_source,
                # The auxiliary-objective flags obey the same rule as the block comment
                # above, and for a sharper reason than the sim/real ones: an unmasked phase
                # has no visibility plane, so its input is 3 channels against the scored
                # phase's 4 and ``ConcatDataset`` collation fails outright -- while ``fine``
                # and ``aux_p`` would simply be absent from 3/4 of the batches and take the
                # loss with them. Neither degrades quietly; both break, which is the better
                # failure but still a break.
                piv_noise=cfg.piv_noise,
                piv_noise_target=cfg.piv_noise_target,
                aux_mask_mode=cfg.aux_mask,
                aux_mask_ratio=cfg.aux_mask_ratio,
                aux_mask_plane=cfg.aux_mask_plane,
                aux_p=cfg.w_paux > 0,
                p_target=cfg.p_target,
                fine=cfg.w_fine > 0 or cfg.w_subgrid > 0,
                dt_aug=dt_strides(cfg),
                dt_ratio_aug=dt_ratio_range(cfg),
                dt_ratio_p1=cfg.dt_ratio_p1,
                history_dropout=cfg.history_dropout,
                recent_dropout=cfg.recent_dropout,
                recent_dropout_max=cfg.recent_dropout_max,
                stream_state=cfg.stream_state,
                state_dropout=cfg.state_dropout,
                state_dropout_max=cfg.state_dropout_max,
                seed=cfg.seed,
            )
            for kind in phase_kinds(base_kind(cfg.train_kind))
        ]
        print(
            f"phase augmentation: {len(train_ds)} windows -> {len(train_ds) + sum(map(len, extra))}"
        )
        train_ds = ConcatDataset([train_ds, *extra])

    sampler = None
    if cfg.synth_path:
        raise ValueError("synth_path (data/synthetic.py) is not included in this package")
    if cfg.joint_sim_frac > 0:
        if sampler is not None:
            raise ValueError(
                "joint_sim_frac and synth_path both reweight the batch; running them together "
                "would make the realised mixture the product of two samplers rather than "
                "either configured fraction. Pick one."
            )
        train_ds, sampler = joint_sim_real(train_ds, cfg)
    elif cfg.re_tilt != 0.0 and sampler is None:
        sampler = re_tilted_sampler(train_ds, cfg.re_tilt, cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        # Explicit, for the reason ``re_tilted_sampler`` gives: an A/B whose data order
        # also changed is two variables. Left to itself ``RandomSampler`` draws its seed
        # from the *global* CPU stream at every ``__iter__``, so the epoch order is a
        # function of how many CPU draws happened earlier -- and model construction is one
        # of them. Two arms differing only in architecture therefore saw different data
        # orders, and the difference read as part of the effect.
        generator=torch.Generator().manual_seed(cfg.seed),
        # See ``seeds.seed_worker``: the augmentation generator lives on the dataset and is
        # forked, so without this every worker draws the identical stream.
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
    )
    # Validation uses a coarse stride: these loaders exist for model selection, and
    # 20 frames between window starts already gives non-overlapping targets.
    # The split name decides the grid, not ``cfg.train_kind``: a sim-trained model is
    # validated on sim, and a sim-pretrained-then-real-fine-tuned one on real, and both
    # are expressed by naming the splits.
    # A sim-trained run validates on the same simulated grid it trains on, so that a
    # ``sim_resampled`` pretrain is not selected against windows on the other clock.
    #
    # ``rescale_fluct`` rides along onto the simulated validation splits for the same
    # reason: it is a *distribution* change, not an augmentation, so a run that trains on
    # rescaled sim and validates on raw sim is measuring a train/val mismatch on top of
    # whatever it meant to measure. That mismatch is worth ~0.36 of the 0.665 tke error
    # this baseline reported. Real splits never take it -- there is nothing to rescale.
    sim_kind = cfg.train_kind if cfg.train_kind in SIM_DOMAINS else "sim"
    val_loaders = {
        split: DataLoader(
            make_dataset(
                split,
                kind=sim_kind if is_sim_split(split) else "real",
                stride=40,
                t_in=cfg.t_in,
                stream_state=cfg.stream_state,
                normalizer=cfg.normalizer,
                zero_p=cfg.zero_p and is_sim_split(split),
                rescale_fluct=cfg.rescale_fluct and is_sim_split(split),
                # For the same reason as ``rescale_fluct`` above, and it is the same defect
                # §22a found: a run that trains on gauge-removed, ``u_fs**2``-scaled pressure
                # and validates on the raw absolute channel is measuring a train/val
                # mismatch on top of whatever it meant to measure. The validation ``loss``
                # column carries the pressure term, so this reaches a selection criterion.
                p_target=cfg.p_target if is_sim_split(split) else "",
                geom_channels=geom_names(cfg),
                flow_channels=flow_names(cfg),
                cond_mode=cfg.cond_mode,
                cond_source=cfg.cond_source,
                # A ``dt_ratio_aug`` run validates its simulated splits on the REAL clock
                # (ratio pinned at 1): that is the units cache's clock and the one the
                # augmentation is centred on, so the diagnostic reads the same task.
                dt_ratio_aug=(1.0, 1.0) if is_sim_split(split) and dt_ratio_range(cfg) else (),
                # The visibility plane, all ones. A masked-modelling run must still be
                # *validated* on the intact input -- masking is the training device, not
                # the task -- and the plane has to be there for the shapes to agree.
                aux_mask_mode=cfg.aux_mask,
                aux_mask_ratio=0.0,
                aux_mask_plane=cfg.aux_mask_plane,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=min(2, cfg.num_workers),
        )
        for split in cfg.val_splits
    }
    return train_loader, val_loaders


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to a tenth of the peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def auxiliary_losses(
    aux: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: TrainConfig
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Loss contributions of the auxiliary heads, and their diagnostics.

    Each term is weighted so that ``--w-* 1.0`` puts it on the same footing as the field
    term: each is normalised by the count of the elements it owns (``opreg``'s two numbers
    per sample; the dense heads' pixels). That is the only way ``--w-fine 1.0`` and
    ``--w-recon 1.0`` mean comparable things, which the study needs since it compares them.

    ``fine_vs_upsample`` is logged, not optimised. It is the ratio of the fine head's error
    to that of bilinearly upsampling the model's own coarse prediction, and it is the
    documented failure mode of the idea: a ratio at or above 1 means the head learned
    interpolation and the trunk was taught nothing, so the arm should be read as a null
    regardless of how its task metric moved.
    """
    total: torch.Tensor | None = None
    parts: dict[str, float] = {}
    # (H, W) -> (1, 1, H, W, 1): PIV-masked cells are identically zero in every auxiliary
    # target too, so including them would pay the head for predicting a constant.
    valid = batch["valid"][:, None, :, :, None]

    def add(name: str, weight: float, value: torch.Tensor) -> None:
        nonlocal total
        total = value * weight if total is None else total + value * weight
        parts[name] = float(value)

    if "recon" in aux:
        # Hidden cells only. A visible cell is copied through the input, so scoring it
        # would reward the identity map and dilute the term that is doing the work.
        hidden = batch["aux_mask"][..., None]
        add(
            "recon",
            cfg.w_recon,
            weighted_mse(aux["recon"], batch["recon_target"][..., :N_MEASURED], hidden * valid),
        )
        if cfg.w_recon_tke > 0 or cfg.w_recon_acf > 0:
            pred = aux["recon"].float()
            target = batch["recon_target"][..., :N_MEASURED].float()
            hidden = batch["aux_mask"].bool()
            hidden_t = hidden[..., None]
            count = hidden_t.sum(dim=1).clamp_min(1)
            target_mean = (target * hidden_t).sum(dim=1, keepdim=True) / count[:, None]
        if cfg.w_recon_tke > 0:
            moment_mask = torch.ones_like(hidden_t) if cfg.recon_tke_scope == "full" else hidden_t
            moment_count = moment_mask.sum(dim=1).clamp_min(1)
            moment_mean = (target * moment_mask).sum(dim=1, keepdim=True) / moment_count[:, None]
            target_energy = (
                ((target - moment_mean).square() * moment_mask).sum(dim=1) / moment_count
            ).sum(-1)
            # Predict RMS fluctuation amplitude DIRECTLY from one decoder channel. Computing
            # variance of a zero-initialised decoded sequence has zero derivative at zero and
            # leaves the objective pinned at relative error 1. The direct amplitude target is
            # equivalent second-moment information but has a live gradient at initialisation.
            pred_moment = pred[..., 0].mean(dim=1)
            target_moment = (
                target_energy
                if cfg.recon_tke_target == "energy"
                else target_energy.clamp_min(1e-12).sqrt()
            )
            # Tube masking hides a spatial location for the full window. ``any`` also gives a
            # well-defined fallback for other mask geometries without counting visible pixels.
            energy_valid = hidden.any(dim=1) & (batch["valid"] > 0.5)
            if cfg.recon_tke_spatial == "global":
                spatial_count = energy_valid.sum(dim=(1, 2)).clamp_min(1)
                pred_moment = (pred_moment * energy_valid).sum(dim=(1, 2)) / spatial_count
                target_moment = (target_moment * energy_valid).sum(dim=(1, 2)) / spatial_count
                energy_valid = torch.ones_like(pred_moment, dtype=torch.bool)
            add(
                "recon_tke",
                cfg.w_recon_tke,
                relative_l2(
                    pred_moment * energy_valid,
                    target_moment * energy_valid,
                ).mean(),
            )
        if cfg.w_recon_acf > 0:
            pair = hidden[:, 1:] & hidden[:, :-1]
            pair_t = pair[..., None]
            pair_count = pair_t.sum(dim=1).clamp_min(1)
            centered = target - target_mean
            lag_cov = ((centered[:, 1:] * centered[:, :-1] * pair_t).sum(dim=1) / pair_count).sum(
                -1
            )
            lag_energy = (
                ((centered[:, :-1].square() * pair_t).sum(dim=1) / pair_count)
                .sum(-1)
                .clamp_min(1e-8)
            )
            target_acf = lag_cov / lag_energy
            pred_acf = pred[..., 1].mean(dim=1)
            pair_valid = pair.any(dim=1) & (batch["valid"] > 0.5)
            add(
                "recon_acf",
                cfg.w_recon_acf,
                relative_l2(pred_acf * pair_valid, target_acf * pair_valid).mean(),
            )

    if "fine" in aux:
        fine_target = batch["target_fine"][..., :N_MEASURED]
        fine_valid = valid.repeat_interleave(SUB_S, dim=2).repeat_interleave(SUB_S, dim=3)
        add("fine", cfg.w_fine, weighted_mse(aux["fine"], fine_target, fine_valid))

    if "subgrid" in aux:
        # Relative, not MSE -- and this one is not a style choice. Sub-grid variance is
        # ~1e-5 where the field is O(1), so a plain MSE lands at ~1e-9 against the field
        # term's ~1e-3: measured at initialisation the term was already 30x too small, and
        # it shrinks from there, so the arm would have been a no-op wearing a weight. A
        # per-sample relative L2 is O(1), is scale-free across a Reynolds ladder over which
        # sub-grid energy spans orders of magnitude, and is the exact form ``tke_score``
        # takes -- which is the metric this head exists to serve.
        target = subgrid_variance(batch["target_fine"][..., :N_MEASURED], factor=SUB_S)
        add(
            "subgrid",
            cfg.w_subgrid,
            relative_l2(aux["subgrid"] * valid, target * valid).mean(),
        )

    if "paux" in aux:
        # Per-sample gate: the one simulated cell whose pressure is on a defective absolute
        # gauge contributes nothing rather than 99.9% of the loss mass (splits.py).
        gate = batch["has_p_aux"].reshape(-1, 1, 1, 1, 1) * valid
        add("paux", cfg.w_paux, weighted_mse(aux["paux"], batch["target_p"], gate))

    if "opreg" in aux:
        add("opreg", cfg.w_opreg, opreg_loss(aux["opreg"], batch["cond"]))

    return total, parts


def fine_degeneracy(
    aux_fine: torch.Tensor, pred: torch.Tensor, batch: dict[str, torch.Tensor]
) -> float:
    """Fine-head error divided by the error of bilinearly upsampling ``pred``.

    Below 1 the head resolves sub-grid content the coarse prediction does not carry; at or
    above 1 it has learned interpolation. Computed under ``no_grad`` on the batch already in
    flight, so it costs one upsample.
    """
    with torch.no_grad():
        fine_target = batch["target_fine"][..., :N_MEASURED]
        fine_valid = batch["valid"][:, None, :, :, None]
        fine_valid = fine_valid.repeat_interleave(SUB_S, dim=2).repeat_interleave(SUB_S, dim=3)
        head = weighted_mse(aux_fine, fine_target, fine_valid)
        base = weighted_mse(
            upsample_baseline(pred[..., :N_MEASURED].float()), fine_target, fine_valid
        )
        return float(head / base.clamp_min(1e-12))


def distill_target(
    teacher: torch.Tensor, truth: torch.Tensor, alpha: float, part: str = ""
) -> torch.Tensor:
    """The field target a distilled step trains on: ``alpha*teacher + (1-alpha)*truth``.

    ``part`` optionally Reynolds-splits the teacher first, over the time axis of an
    ``(B, T, H, W, C)`` window: ``"mean"`` keeps the teacher's temporal mean and takes the
    fluctuation from ``truth``, ``"fluct"`` does the reverse, ``""`` uses the whole field.
    Both splits recompose to a full window, so nothing downstream changes shape.

    Only the **measured** channels are distilled. The teacher is fine-tuned on real data, where
    pressure is unsupervised, so its ``p`` channel carries no information -- overwriting the
    simulated half's ground-truth ``p`` with it would replace real physics supervision with
    noise. On a real-only batch ``has_p`` is all zero and ``temporal_tke`` reads channels 0-1,
    so the ``p`` column of the target is never read and this is bit-identical to overwriting the
    whole tensor; it only bites once a distilled stage draws a simulated half (FINDINGS 64h).

    A pure function of its arguments -- no RNG, no module state -- which is what keeps a
    distilled run inside ``seeds.seed_everything``'s bit-for-bit contract.
    """
    if part:
        t_mean, u_mean = teacher.mean(1, keepdim=True), truth.mean(1, keepdim=True)
        teacher = t_mean + (truth - u_mean) if part == "mean" else u_mean + (teacher - t_mean)
    out = truth.clone()
    out[..., :N_MEASURED] = (
        alpha * teacher[..., :N_MEASURED] + (1.0 - alpha) * truth[..., :N_MEASURED]
    )
    return out


def _validate_mean_fluct_config(cfg: TrainConfig) -> None:
    if not np.isfinite(cfg.mean_fluct_lambda) or cfg.mean_fluct_lambda < 0:
        raise ValueError("mean_fluct_lambda must be finite and nonnegative")
    if cfg.mean_fluct_robust and cfg.mean_fluct_lambda <= 0:
        raise ValueError(
            "mean_fluct_robust only modifies mean_fluct_lambda > 0; it would be ignored"
        )
    if cfg.mean_fluct_lambda > 0:
        if cfg.amplitude_free or cfg.lead_decay > 0 or cfg.helm_lambda != 1.0:
            raise ValueError("mean_fluct_lambda cannot combine with another field-loss replacement")
        if cfg.w_field <= 0 or (cfg.bounds and cfg.freeze_backbone):
            raise ValueError("mean_fluct_lambda requires an active field loss")


#: The frozen mean-optimal model a moment network is trained against (``TrainConfig.moment_of``),
#: set once by :func:`train` and read by :func:`moment_loss`; module-level because the loss is
#: called from the step and from ``quick_eval`` with the same ``cfg`` and neither carries modules.
_MOMENT_TARGET: dict[str, nn.Module | None] = {"model": None}

#: The frozen absorb head (``TrainConfig.absorb_head``), loaded once per path and device.


def shuffle_cond(cond: torch.Tensor | None, p: float) -> torch.Tensor | None:
    """``TrainConfig.cond_drop``: with probability ``p`` per sample, another sample's
    operating-point vector (a random permutation of the batch). ``p <= 0`` returns ``cond``."""
    if cond is None or p <= 0:
        return cond
    perm = torch.randperm(cond.shape[0], device=cond.device)
    keep = (torch.rand(cond.shape[0], device=cond.device) >= p)[:, None].to(cond.dtype)
    return keep * cond + (1.0 - keep) * cond[perm]


def moment_loss(out, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    """Gaussian negative log-likelihood of the frozen model's residuals under the predicted
    ``log sigma^2``: ``0.5 * (s + r^2 * exp(-s))`` on u and v over the PIV-valid cells. Minimised
    by ``sigma^2 = E[r^2 | input]``, the posterior variance (FINDINGS 153n)."""
    mu_model = _MOMENT_TARGET["model"]
    if mu_model is None:
        raise RuntimeError("moment_of is set but no frozen mean model was loaded")
    s, _, _ = split_output(out)
    with torch.no_grad():
        mu, _, _ = split_output(
            call_model(mu_model, batch["input"], batch["cond"], batch.get("u_fs"))
        )
    r2 = (batch["target"][..., :N_MEASURED].float() - mu[..., :N_MEASURED].float()).square()
    s = s[..., :N_MEASURED].float()
    m = batch["valid"][:, None, :, :, None].float()
    n = (m.sum() * s.shape[1] * N_MEASURED).clamp_min(1.0)  # valid cells x leads x channels
    nll = 0.5 * (s + r2 * torch.exp(-s))
    total = (nll * m).sum() / n
    parts = {
        "total": float(total),
        "nll": float(total),
        "log_var": float((s * m).sum() / n),
        "resid2": float((r2 * m).sum() / n),
    }
    return total, parts


class _StreamCorr:
    """Training-time image of the shipped Track-2 mean-flow correction (``stream_corr_lam``).

    Serving applies ``pred + lam(n) * mean over revealed windows of mean_t(target - pred)``,
    frozen after 16 windows (rpde_t2.ttt.TrajectoryCalibratedPort, bias_mode='meanflow'). Here
    the same per-trajectory first-moment residual is estimated as an EMA over training steps of
    the live model's own detached residuals, and added (stop-gradient) to the prediction before
    the loss — so the backbone is graded on the composite system the stream actually serves and
    stops re-learning the moment the correction owns. The EMA starts at zero, which mirrors the
    stream's warm-up shrinkage; ``lam`` is the frozen-regime effective strength
    (0.75 * 16/18 = 0.667). Approximations, recorded: the EMA tracks the live model while the
    stream corrects the EMA-of-weights model, and training-step time replaces stream position.
    State is not checkpointed — it re-warms in ~1/(1-decay) steps after a resume.
    """

    def __init__(self, lam: float, decay: float):
        self.lam, self.decay = float(lam), float(decay)
        self.ema: torch.Tensor | None = None  # (n_traj, H, W, C), grows lazily

    def _ensure(self, n: int, like: torch.Tensor) -> None:
        if self.ema is None:
            self.ema = like.new_zeros((n, *like.shape[1:]))
        elif self.ema.shape[0] < n:
            self.ema = torch.cat(
                [self.ema, like.new_zeros((n - self.ema.shape[0], *like.shape[1:]))]
            )

    def bias(self, traj: torch.Tensor) -> torch.Tensor | None:
        """``(B, H, W, C)`` stop-gradient bias for the batch; None before any update."""
        if self.ema is None:
            return None
        idx = traj.clamp(max=self.ema.shape[0] - 1)
        return self.lam * self.ema[idx]

    @torch.no_grad()
    def update(self, traj: torch.Tensor, out: Any, batch: dict) -> None:
        pred = split_output(out)[0].detach()
        target = batch["target"]
        r = (target - pred[..., : target.shape[-1]]).float().mean(dim=1)  # lead-averaged
        self._ensure(int(traj.max()) + 1, r)
        uniq, inv = torch.unique(traj, return_inverse=True)
        sums = r.new_zeros((len(uniq), *r.shape[1:]))
        sums.index_add_(0, inv, r)
        cnt = torch.bincount(inv, minlength=len(uniq)).to(r.dtype).view(-1, 1, 1, 1)
        self.ema[uniq] = self.decay * self.ema[uniq] + (1.0 - self.decay) * (sums / cnt)


def rollout_select(
    batch: dict[str, torch.Tensor], aoa_min: float, re_min: float = 0.0
) -> tuple[dict[str, torch.Tensor], float]:
    """``rollout_aoa_min`` / ``rollout_re_min`` (FINDINGS 224h): the rollout batch restricted to
    the windows whose label angle is at least ``aoa_min`` and label Reynolds number at least
    ``re_min``, and their share of the batch. Every per-window tensor is indexed; anything else
    passes through. Both at 0 return the batch untouched with share 1.0, so the ungated term is
    exactly the one 224f/h measured."""
    if aoa_min <= 0 and re_min <= 0:
        return batch, 1.0
    keep = batch["aoa"] >= aoa_min
    if re_min > 0:
        keep = keep & (batch["re"] >= re_min)
    n = keep.shape[0]
    idx = keep.nonzero().squeeze(1)
    k = int(idx.numel())
    if k == n:
        return batch, 1.0
    sub = {
        key: v[idx] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n else v
        for key, v in batch.items()
    }
    return sub, k / n


def compute_losses(
    out,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
    progress: float = 1.0,
    training: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    _validate_mean_fluct_config(cfg)
    if getattr(cfg, "moment_of", ""):
        return moment_loss(out, batch)
    pred, lower, upper = split_output(out)
    target, valid = batch["target"], batch["valid"]
    # A map channel (models/kmap.py) rides along as channel 3; every field term below reads the
    # frames only, and the channel is graded by its own term.
    pred_all = pred
    if pred.shape[-1] > target.shape[-1]:
        pred = pred[..., : target.shape[-1]]
    if getattr(cfg, "moment_joint", False):
        logvar = pred_all[..., target.shape[-1] : target.shape[-1] + N_MEASURED]
    # Correction-in-the-loop (``stream_corr_lam``): the fit loop injects the per-trajectory
    # mean-flow bias and every term below grades the corrected composite. Constant over the
    # lead axis, so the tke-family terms — variance about the temporal mean — are untouched
    # by construction; only the first-moment terms feel it. Stop-gradient by construction
    # (_StreamCorr builds it from detached residuals under no_grad).
    stream_bias = batch.get("stream_bias")
    if stream_bias is not None:
        pred = pred + stream_bias.unsqueeze(1).to(pred.dtype)
    # Under distillation ``target`` is the teacher's field; ``distill_tke_truth`` routes the
    # tke-family terms at the preserved ground truth instead (falls back to ``target`` when there
    # is no teacher, so the flag is a no-op outside distillation).
    tke_target = batch["truth"] if (cfg.distill_tke_truth and "truth" in batch) else target

    parts: dict[str, float] = {}
    total = pred.new_zeros(())

    # The prediction the dense tke terms grade: bare, or through a frozen head (196k). With
    # ``tke_loop_head_field`` the field term grades it too; the band credit keeps the bare
    # prediction, because the serving interval is centred on it.
    tke_pred = pred
    if cfg.tke_loop_head_field and not cfg.tke_loop_head:
        raise ValueError("tke_loop_head_field needs --tke-loop-head (a frozen head)")
    if cfg.tke_loop_head:
        raise ValueError("tke_loop_head (train/absorb.py) is not included in this package")
    fpred = tke_pred if cfg.tke_loop_head_field else pred

    if not (cfg.bounds and cfg.freeze_backbone):
        # With a frozen backbone the prediction cannot change, so a field term would
        # contribute only a constant and a misleading loss curve.
        # Two flags here replace `field_loss` outright rather than adding to it, so a run
        # setting both would silently get whichever the branch order favours. Refuse instead:
        # PROTOCOL 5's trap list is mostly flags that were accepted and then not honoured.
        if cfg.helm_lambda != 1.0 and cfg.lead_decay > 0:
            raise ValueError(
                "helm_lambda and lead_decay both replace the field term; set at most one"
            )
        if cfg.field_norm != "none" and (
            cfg.helm_lambda != 1.0
            or cfg.lead_decay > 0
            or cfg.amplitude_free
            or cfg.mean_fluct_lambda > 0
        ):
            raise ValueError(
                "field_norm replaces the field term and is exclusive with helm_lambda, "
                "lead_decay, amplitude_free and mean_fluct_lambda; set at most one"
            )
        if cfg.mean_fluct_lambda > 0:
            fl = mean_fluct_field_loss(
                fpred,
                target,
                fluct_weight=cfg.mean_fluct_lambda,
                has_p=batch.get("has_p"),
                huber_delta=cfg.huber_delta,
                huber_mix=cfg.huber_mix,
                p_valid=valid[:, None, :, :, None] if cfg.p_target else None,
                w_p=cfg.w_p,
                robust_fluct=cfg.mean_fluct_robust,
            )
        elif cfg.helm_lambda != 1.0:
            # A reweighting of the same MSE in an orthogonal basis, so it REPLACES the field
            # term rather than adding to it -- lambda = 1 reproduces it exactly (Parseval).
            # It covers the measured channels only, so refuse a batch carrying pressure
            # rather than silently dropping that supervision.
            if batch.get("has_p") is not None and float(batch["has_p"].sum()) > 0:
                raise ValueError(
                    "helm_lambda != 1 covers the measured channels only; it cannot be used on "
                    "a batch with pressure supervision (sim or joint). Use a real-only arm."
                )
            fl = helmholtz_field_loss(fpred, target, lam=cfg.helm_lambda)
        elif cfg.amplitude_free:
            # Sim-stage only, and `train` refuses it on real data -- the whole point is that real
            # is where the amplitude comes from, so applying it there would delete the signal
            # rather than protect it.
            fl = amplitude_free_field_loss(
                fpred,
                target,
                has_p=batch.get("has_p"),
                huber_delta=cfg.huber_delta,
                huber_mix=cfg.huber_mix,
            )
        elif cfg.field_norm == "frame":
            # Bruno's per-frame normalised, masked field term (2026-09-17): same huber
            # elementwise error, each frame divided by its own detached RMS, valid cells only.
            fl = frame_normalized_field_loss(
                fpred,
                target,
                valid,
                has_p=batch.get("has_p"),
                huber_delta=cfg.huber_delta,
                huber_mix=cfg.huber_mix,
                p_valid=valid[:, None, :, :, None] if cfg.p_target else None,
                w_p=cfg.w_p,
            )
        elif cfg.lead_decay > 0:
            # Tilted accuracy pressure (§42). Pressure is *redistributed*, not added: the weights
            # have mean 1, so this is not a dose change wearing a different name. Deliberately
            # exclusive with the huber options rather than composed with them: `lead_decay` was
            # measured as a null (§42 follow-up) while `huber_delta` is §28's largest single
            # lever, so a silent composition would put a dead knob in front of a live one.
            fl = lead_weighted_field_loss(fpred, target, decay=cfg.lead_decay)
        else:
            fl = field_loss(
                fpred,
                target,
                has_p=batch.get("has_p"),
                huber_delta=cfg.huber_delta,
                huber_mix=cfg.huber_mix,
                p_valid=valid[:, None, :, :, None] if cfg.p_target else None,
                w_p=cfg.w_p,
            )
        total = total + cfg.w_field * fl
        parts["field"] = float(fl)
        if cfg.p_target and batch.get("has_p") is not None:
            # The pressure contribution, by difference. Logged because §22's discipline is
            # that a term is *verified active at convergence* rather than assumed: a p term
            # that has fallen to a hundredth of the field loss is a null wearing a weight,
            # and this is the only place that is visible. A second elementwise pass over a
            # (B, T, H, W, 1) tensor, so it costs nothing next to the backward.
            parts["field_p"] = float(fl) - float(
                mean_fluct_field_loss(
                    fpred,
                    target,
                    fluct_weight=cfg.mean_fluct_lambda,
                    huber_delta=cfg.huber_delta,
                    huber_mix=cfg.huber_mix,
                    robust_fluct=cfg.mean_fluct_robust,
                )
                if cfg.mean_fluct_lambda > 0
                else field_loss(fpred, target, huber_delta=cfg.huber_delta, huber_mix=cfg.huber_mix)
            )

    if getattr(cfg, "moment_joint", False):
        # The variance head's own term: Gaussian NLL of the frames' residual, the residual
        # detached so the frames keep their field / tke objectives untouched.
        r2 = (target[..., :N_MEASURED].float() - pred[..., :N_MEASURED].detach().float()).square()
        sv = logvar.float()
        m = valid[:, None, :, :, None].float()
        n = (m.sum() * sv.shape[1] * N_MEASURED).clamp_min(1.0)
        nll = (0.5 * (sv + r2 * torch.exp(-sv)) * m).sum() / n
        total = total + cfg.w_moment * nll
        parts["nll"] = float(nll)
        parts["log_var"] = float((sv * m).sum() / n)
    if cfg.tke_gate not in ("none", "huber", "both", "pred"):
        raise ValueError(
            f"tke_gate must be 'none', 'huber', 'both' or 'pred', got {cfg.tke_gate!r}"
        )
    tke_gate = None
    if cfg.tke_gate != "none" and (cfg.w_tke_huber > 0 or cfg.w_tke_shape > 0):
        tke_gate = unpredictability_gate(
            pred, tke_target, valid=valid, invert=cfg.tke_gate == "pred"
        )
        # Verified active at convergence, 22's discipline: a gate that has collapsed to
        # uniform is a null wearing a flag, and this is the only place that is visible.
        parts["tke_gate_sd"] = float(tke_gate.std())

    if cfg.w_tke > 0:
        tl = tke_loss(
            tke_pred,
            tke_target,
            valid=valid,
            lead_dropout=cfg.tke_lead_dropout if training else 0.0,
        )
        total = total + cfg.w_tke * tl
        parts["tke"] = float(tl)

    if cfg.w_tke_shape > 0:
        sh = tke_shape_loss(
            tke_pred,
            tke_target,
            valid=valid,
            gate=tke_gate if cfg.tke_gate == "both" else None,
            lead_dropout=cfg.tke_lead_dropout if training else 0.0,
        )
        total = total + cfg.w_tke_shape * sh
        parts["tke_shape"] = float(sh)

    if cfg.w_tke_amp > 0:
        amp = tke_amplitude_loss(
            pred,
            tke_target,
            valid=valid,
            delta=cfg.tke_amp_delta,
            lead_dropout=cfg.tke_lead_dropout if training else 0.0,
        )
        total = total + cfg.w_tke_amp * amp
        parts["tke_amp"] = float(amp)

    if cfg.w_errhomog > 0 and not (cfg.bounds and cfg.freeze_backbone):
        # Shape the backbone's error to be locally homogeneous (predictable from its neighbourhood),
        # which is the SPS interval-efficiency bound (FINDINGS 207). Inert with a frozen backbone
        # (the error field cannot change), same guard as the field term.
        eh = error_homogeneity_loss(pred, target, valid=valid, k=cfg.errhomog_k)
        total = total + cfg.w_errhomog * eh
        parts["errhomog"] = float(eh)

    if cfg.w_tke_tprofile > 0:
        tpf = tke_temporal_profile_loss(pred, tke_target, valid=valid)
        total = total + cfg.w_tke_tprofile * tpf
        parts["tke_tprofile"] = float(tpf)

    if cfg.w_pvar > 0:
        if not cfg.p_target:
            raise ValueError(
                "w_pvar grades the temporal variance of the pressure channel, which is the "
                "drift of an arbitrary datum unless p_target='gauge' removes the gauge. "
                "Set --p-target gauge or drop --w-pvar."
            )
        pv = pvar_loss(pred, target, has_p=batch.get("has_p"), valid=valid)
        total = total + cfg.w_pvar * pv
        parts["pvar"] = float(pv)

    if cfg.w_tke_huber > 0:
        hb = tke_huber_loss(
            tke_pred,
            tke_target,
            valid=valid,
            delta=cfg.tke_huber_delta,
            normalize=cfg.tke_huber_normalize,
            gate=tke_gate,
            lead_dropout=cfg.tke_lead_dropout if training else 0.0,
        )
        total = total + cfg.w_tke_huber * hb
        parts["tke_huber"] = float(hb)

    if cfg.w_kmap > 0:
        if pred_all.shape[-1] <= C:
            raise ValueError("w_kmap > 0 needs --kmap-channel: the output has no map channel")
        kl = kmap_loss(pred_all, batch["input"], tke_target, valid=valid, base=cfg.kmap_base)
        total = total + cfg.w_kmap * kl
        parts["kmap"] = float(kl)

    if cfg.w_credit > 0:
        cr = interval_credit_loss(pred, target, batch["u_fs"], k=cfg.credit_k)
        total = total + cfg.w_credit * cr
        parts["credit"] = float(cr)

    if cfg.w_headcredit > 0:
        if not cfg.headcredit_arm:
            raise ValueError("w_headcredit > 0 needs --headcredit-arm (a fitted interval head)")
        fh = frozen_head(cfg.headcredit_arm, pred.device, batch["cond"].shape[-1])
        half = fh.half_width(pred.detach(), batch["input"], valid, batch["cond"], batch["u_fs"])
        taper = None
        if cfg.headcredit_lo != 1.0:
            if "re" not in batch:
                raise ValueError("headcredit_lo needs the dataset's per-window 're' label")
            taper = re_taper(batch["re"].to(pred.device), cfg.headcredit_lo)
        if cfg.headcredit_notch_lo != 1.0:
            if "re" not in batch:
                raise ValueError("headcredit_notch_lo needs the dataset's per-window 're' label")
            notch = re_notch(
                batch["re"].to(pred.device),
                cfg.headcredit_notch_lo,
                cfg.headcredit_notch_re,
                cfg.headcredit_notch_width,
            )
            taper = notch if taper is None else taper * notch
        hc = head_credit_loss(
            pred, target, half, batch["u_fs"], tau=cfg.headcredit_tau, weight=taper
        )
        total = total + cfg.w_headcredit * hc
        parts["headcredit"] = float(hc)
        if training and cfg.headcredit_track_lr > 0:
            # After the width is taken, so this step's credit is graded by the head as it was.
            parts["headcredit_track"] = fh.track_step(
                pred, target, batch["input"], valid, batch["cond"], cfg.headcredit_track_lr
            )

    if cfg.w_rstress > 0:
        rs = rstress_shape_loss(pred, target, valid=valid)
        total = total + cfg.w_rstress * rs
        parts["rstress"] = float(rs)

    if cfg.w_spectral > 0:
        sl = spectral_loss(pred, target)
        total = total + cfg.w_spectral * sl
        parts["spectral"] = float(sl)

    if cfg.w_espec > 0:
        es = energy_spectrum_loss(pred, target)
        total = total + cfg.w_espec * es
        parts["espec"] = float(es)

    if cfg.w_relfield > 0:
        rf = relative_field_loss(pred, target)
        total = total + cfg.w_relfield * rf
        parts["relfield"] = float(rf)

    if cfg.w_meanfield > 0:
        mf = mean_field_loss(pred, target)
        total = total + cfg.w_meanfield * mf
        parts["meanfield"] = float(mf)

    if cfg.w_absorb > 0:
        raise ValueError("w_absorb (train/absorb.py) is not included in this package")

    if cfg.w_ceiling > 0:
        cl = ceiling_loss(pred, target, valid=valid, huber_delta=cfg.ceiling_huber)
        total = total + cfg.w_ceiling * cl
        parts["ceiling"] = float(cl)

    if cfg.w_vorticity > 0:
        vl = vorticity_loss(pred, target, valid=valid)
        total = total + cfg.w_vorticity * vl
        parts["vorticity"] = float(vl)

    if cfg.w_enstrophy > 0:
        el = enstrophy_loss(pred, target, valid=valid)
        total = total + cfg.w_enstrophy * el
        parts["enstrophy"] = float(el)

    if cfg.w_mvpe > 0:
        ml = mvpe_loss(pred, target)
        total = total + cfg.w_mvpe * ml
        parts["mvpe"] = float(ml)

    if cfg.bounds and lower is not None and upper is not None:
        if cfg.w_calib > 0 and progress < cfg.calib_warmup_frac:
            # Decayed over the warm-up window so the SPS term takes over smoothly.
            ramp = 1.0 - progress / max(1e-9, cfg.calib_warmup_frac)
            cl = interval_calibration_loss(lower, upper, pred, target)
            total = total + cfg.w_calib * ramp * cl
            parts["calib"] = float(cl)
        if cfg.w_sps > 0:
            bl = sps_loss(lower, upper, target, batch["u_fs"])
            total = total + cfg.w_sps * bl
            parts["sps"] = float(bl)
        op = ordering_penalty(lower, upper)
        total = total + 10.0 * op
        parts["order"] = float(op)

    aux = out.get("aux") if isinstance(out, dict) else None
    if aux:
        aux_total, aux_parts = auxiliary_losses(aux, batch, cfg)
        if aux_total is not None:
            total = total + aux_total
        parts.update(aux_parts)
        if "fine" in aux:
            parts["fine/vs_upsample"] = fine_degeneracy(aux["fine"], pred, batch)

    parts["total"] = float(total)
    return total, parts


#: Modules whose output is the encoder's deepest representation, best first. Matched by
#: name suffix so a wrapped model (residual, bounds) taps the same place a bare one does.
#:
#: ``norm`` is last and is the transformer entry: ``STFormer.forward`` ends its block stack
#: with ``h = self.norm(h)`` and feeds *both* the coarse head and the refinement path from
#: it, so it is the one shared deepest representation. Without it ``align_weight`` raised on
#: the entire STFormer lineage -- the same "declared, documented, never exercised on the
#: architecture under evaluation" gap ``uq.dropout_targets`` exists to close. It stays last
#: because it is the least specific name here: a U-Net that also has a ``norm`` must still
#: tap its bottleneck, and the earlier entries win.
BOTTLENECK_MODULES = ("mid_block2", "mid_block1", "middle", "norm")


class FeatureTap:
    """Capture one module's output on every forward pass.

    A forward hook rather than a changed ``forward`` signature: the alignment term needs an
    interior activation, and threading one through every model in the factory would put a
    training-only concern into the architecture that inference also builds.
    """

    def __init__(self, model: nn.Module, names: tuple[str, ...] = BOTTLENECK_MODULES) -> None:
        modules = dict(model.named_modules())
        for name in names:
            hits = [(k, m) for k, m in modules.items() if k.split(".")[-1] == name]
            if hits:
                self.name, target = hits[-1]
                break
        else:
            raise ValueError(
                f"no module named any of {names} in {type(model).__name__}; the alignment "
                f"term has nothing to tap"
            )
        # Where the channels are, read off the module rather than assumed. A LayerNorm
        # normalises over the trailing dims, so its output is ``(B, tokens, C)``; a
        # convolutional block's is ``(B, C, ...)``. ``domain_moment_loss`` is per-channel and
        # on the wrong axis it still returns a number, so this has to come from the model.
        self.channel_dim = -1 if isinstance(target, nn.LayerNorm) else 1
        self.value: torch.Tensor | None = None
        target.register_forward_hook(self._store)

    def _store(self, _module, _inputs, output) -> None:
        # A block may return a tuple; the activation is the first element by convention.
        self.value = output[0] if isinstance(output, tuple) else output


class Ema:
    """Exponential moving average of a model's state, kept in fp32.

    Tracks the whole ``state_dict`` rather than only the parameters, so the averaged
    checkpoint is a complete model: a normalisation buffer copied from the final iterate
    while its scale parameter is an average of 4000 of them would be a mismatched pair.
    Integer entries are copied rather than averaged, which is what a step counter or a
    boolean mask needs.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def state_dict(self, like: nn.Module) -> dict[str, torch.Tensor]:
        """The average, cast back to each entry's original dtype."""
        ref = like.state_dict()
        return {k: v.to(ref[k].dtype) for k, v in self.shadow.items()}


def l2sp_penalty(model: nn.Module, ref: dict[str, torch.Tensor]) -> torch.Tensor:
    """``sum ||w - w_init||^2`` over the trainable weights present in ``ref``.

    Only parameters, not buffers: a running statistic is not a fitted direction, and
    penalising its drift would fight the data rather than the forgetting.
    """
    total = None
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in ref:
            continue
        term = (p - ref[name]).pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("l2sp > 0 but no trainable parameter matched the reference weights")
    return total


@torch.no_grad()
def quick_eval(
    model: nn.Module,
    loader: DataLoader,
    device,
    max_windows: int = 512,
    cfg: TrainConfig | None = None,
    progress: float = 1.0,
) -> dict:
    """In-loop diagnostics in non-dimensional space. NOT a leaderboard score.

    When ``cfg`` is passed the *training* loss is also evaluated on this split, with the
    identical :func:`compute_losses` the loop optimises, so ``loss`` here and ``train/loss``
    on the dashboard are the same quantity and their gap is a genuine over-fitting signal.
    The auxiliary pretext terms never enter it -- their heads are skipped in eval mode -- so
    it is the field/physics objective, which is the part that has a validation counterpart.
    ``cfg`` is optional so a standalone re-scorer can get the non-dimensional metrics
    without a config in hand.

    **Bounded by windows, not by batches.** An earlier version stopped after a fixed 16
    *batches*, so the evaluated window count was ``16 * batch_size`` and two runs at
    different batch sizes silently measured different subsets. Measured cost of that:
    a U-Net fine-tune read ``val_extrap`` rel-L2 **0.0840** over the first 128 of 147
    windows and **0.0946** over all 147 -- a 12.6% error, on the split that decides
    things, in the direction that flatters the model. And because the validation loaders
    are unshuffled and the trajectories sort by (AoA, Re), the excluded tail is a
    specific corner of the operating-point grid rather than a random sample, which is
    exactly the composition bias ``PROTOCOL.md`` opens with.

    The default covers every standing validation set whole (the largest, ``val_time``,
    holds 337 windows at stride 40), so in practice this is a full pass; the cap remains
    only so a future, larger set cannot silently make in-loop evaluation expensive.

    Averaging is per **window**, not per batch: a mean of batch means would re-weight a
    short trailing batch as heavily as a full one.
    """
    model.eval()
    rel_sum = tke_sum = loss_sum = 0.0
    seen = 0
    for batch in loader:
        take = min(int(batch["input"].shape[0]), max_windows - seen)
        if take <= 0:
            break
        batch = {k: v[:take].to(device, non_blocking=True) for k, v in batch.items()}
        out = call_model(model, batch["input"], batch["cond"], batch.get("u_fs"))
        pred, _, _ = split_output(out)
        rel_sum += relative_l2(pred[..., :2], batch["target"][..., :2]).sum().item()
        # tke_loss already reduces over the batch, so weight it back up by the count.
        tke_sum += tke_loss(pred, batch["target"], valid=batch["valid"]).item() * take
        if cfg is not None:
            # Same reduction as rel/tke: compute_losses returns a batch mean, so scale by
            # the window count and divide by the total below -- a per-window average, not
            # a mean of batch means that would over-weight a short trailing batch.
            loss_sum += compute_losses(out, batch, cfg, progress, training=False)[1]["total"] * take
        seen += take
        if seen >= max_windows:
            break
    model.train()
    n = max(1, seen)
    # ``n_windows`` is recorded so a number in history.json can be checked for coverage
    # rather than assumed complete.
    result = {"rel_l2": rel_sum / n, "tke": tke_sum / n, "n_windows": seen}
    if cfg is not None:
        result["loss"] = loss_sum / n
    return result


def _fmt_eta(seconds: float) -> str:
    """``h:mm:ss`` for a remaining-time estimate, so a long run's log is scannable."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def official_scores(model: nn.Module, cfg: TrainConfig, device) -> dict[str, dict] | None:
    """Leaderboard subscores on ``cfg.val_splits`` for the live model, or ``None`` on failure.

    Never fatal, for the reason W&B logging is not (see ``wandb_log``): a six-hour run must
    not die because an eval set is unbuilt or the scorer subprocess hiccupped. The model is
    switched to eval for the scoring pass and restored to train regardless of outcome.
    """
    from .evaluate import score_model  # noqa: PLC0415 -- breaks a loop<->evaluate import cycle

    try:
        model.eval()
        return score_model(model, cfg, splits=cfg.val_splits, device=device)
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must never end a training run
        print(f"note: in-loop official scoring failed ({type(exc).__name__}: {exc}); disabling it")
        return None
    finally:
        model.train()


def train(cfg: TrainConfig) -> Path:
    """Run a training job. Returns the output directory."""
    # First statement in the function: ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS
    # handle is created, so this has to precede the first matmul rather than merely the
    # first parameter draw.
    seed_everything(cfg.seed, cfg.deterministic)
    _validate_mean_fluct_config(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Checked before anything is built or written: silently dropping the term would leave
    # config.json claiming a regularisation the run never applied, which is FINDINGS 12's
    # failure mode -- an artefact that agrees with the intent while the data does not.
    if cfg.l2sp > 0 and not cfg.init_from:
        raise ValueError("l2sp > 0 needs init_from: there is no reference point without it")
    if cfg.amplitude_free and cfg.train_kind not in SIM_DOMAINS:
        raise ValueError(
            "amplitude_free grades the fluctuation SHAPE and discards its amplitude, which is "
            f"correct for sim and destroys the signal on real (train_kind={cfg.train_kind!r}). "
            "Use it on the sim pretrain; stage iii is where the amplitude is learned."
        )
    if (cfg.align_weight > 0 or cfg.coral_weight > 0) and cfg.joint_sim_frac <= 0:
        raise ValueError(
            "domain alignment needs joint_sim_frac > 0: a single-domain batch has no "
            "second distribution to align to, and the term would silently be zero"
        )

    out_dir = Path(cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = str(out_dir)

    # Built *before* the config is written, because build_model corrects fields it
    # cannot honour -- notably it forces ``amp=False`` for the FP16_UNSAFE models. Saving
    # first left config.json claiming fp16 for every FNO and Transolver run while the
    # checkpoint's embedded config recorded the truth: two sources of record disagreeing
    # about what was executed.
    if cfg.nbr_prior:
        # Before build_model: the wrapper's table is sized by the cell list (FINDINGS 224).
        cfg.nbr_cells = [list(c) for c in nbr_prior_cells(cfg)]
    model = build_model(cfg).to(device)
    nbr_mods: list = []
    if cfg.nbr_prior:
        from ..models.wrappers import NbrPriorWrapper  # noqa: PLC0415

        fill_nbr_table(model, [tuple(c) for c in cfg.nbr_cells])
        nbr_mods = [m_ for m_ in model.modules() if isinstance(m_, NbrPriorWrapper)]
        print(f"nbr_prior: table of {len(cfg.nbr_cells)} {cfg.train_split} cells")
    if cfg.moment_of:
        from .evaluate import load_checkpoint  # noqa: PLC0415 -- evaluate imports this module

        mu_model, mu_cfg = load_checkpoint(cfg.moment_of, device=device)
        mu_model.eval()
        for prm in mu_model.parameters():
            prm.requires_grad_(False)
        _MOMENT_TARGET["model"] = mu_model
        print(f"moment_of: frozen mean model {cfg.moment_of} ({mu_cfg.model}, {mu_cfg.arch})")
    cfg.save()
    n_params = sum(p.numel() for p in model.parameters())

    # Started here rather than at the top of the function so the config W&B records is
    # the corrected one -- the same reason cfg.save() sits below build_model.
    run = WandbRun.start(cfg)

    if cfg.init_from:
        state = torch.load(cfg.init_from, map_location="cpu")
        if "context" in state and "model" not in state:
            raise ValueError("JEPA encoder checkpoints (train/jepa_fit.py): loading them is not included in this package")
        sd = state["model"]
    if cfg.init_from and state is not None:
        n_geom = n_geometry_channels(geom_names(cfg))
        n_flow = n_flow_channels(flow_names(cfg))
        pre_cfg = state.get("config", {})
        # The BACKBONE's frame count, not the dataset's: under ``state_norm`` the state frames
        # are consumed by the wrapper as the normaliser and never reach the stem, so the stem
        # must NOT be widened for them (that is what backbone_state_frames encodes).
        t_pre = int(pre_cfg.get("t_in", T_IN)) + backbone_state_frames(pre_cfg)
        t_new = cfg.t_in + backbone_state_frames(asdict(cfg))
        if t_new != t_pre:
            # A 20-frame checkpoint initialising a 40-frame run — or a plain checkpoint
            # initialising a ``stream_state`` one: the pretrained stem columns land on the most
            # recent frames' slots, zeros on the older ones (which is where the state
            # pseudo-frames go, so at step 0 the widened model IS the source model). Done first
            # so the channel-widening steps below see a stem whose layout is already
            # ``t_in * C``.
            if cfg.model not in ("stformer",):
                raise ValueError(f"t_in={t_new} warm start from t_in={t_pre} is STFormer-only")
            sd = widen_stem_history(sd, t_pre, t_new)
        if n_geom and cfg.model == "fno":
            # A 3-channel pretrain being widened for geometry channels.
            sd = widen_fc0(sd, n_geom)
        # ``state_planes`` re-emits the state as extra CHANNELS on every frame, which is the
        # same stem-widening situation as a new flow or visibility plane: zeros in the new
        # columns, so the widened model is bit-identical to the source on the original ones.
        n_state_planes = 2 * stream_state_frames(cfg.stream_state) if cfg.state_planes else 0
        if (
            n_flow
            or n_mask_planes(cfg.aux_mask, cfg.aux_mask_plane)
            or n_state_planes
            or (cfg.nbr_prior and not pre_cfg.get("nbr_prior", False))
        ) and cfg.model in (
            "stformer",
            "stformer3d",
        ):
            # New flow planes and the masked-modelling visibility plane both widen the grouped
            # stem. ``c_old`` describes the SOURCE checkpoint, before either new plane family;
            # including the requested mask plane here mis-infers patch_t (60/4 instead of 60/3)
            # and produces a tensor that still cannot load.
            want = next(
                w.shape[1]
                for k, w in model.state_dict().items()
                if any(k.endswith(s) for s in GROUPED_STEM_KEYS)
            )
            sd = widen_grouped_stem(sd, C + n_geom, want)
        if cond_dim(cfg.cond_mode) > COND_DIM:
            # A pretrain conditioned on (log-Re, AoA, is_sim) initialising a run that also
            # reads the window's decorrelation time.
            sd = widen_cond_proj(sd, cond_dim(cfg.cond_mode))
        if cfg.model == "unet":
            # Same situation on the U-Net stem, and it also covers the masking visibility plane:
            # a task-only pretrain initialising a masked-modelling fine-tune is 3 channels
            # feeding a 4-channel stem, which raises rather than degrading.
            want = 3 + n_geom + n_mask_planes(cfg.aux_mask, cfg.aux_mask_plane)
            sd = widen_init_conv(sd, want)
        if any(k.startswith("aux.") for k in sd) and not aux_weights(asdict(cfg)):
            # An auxiliary-objective checkpoint being used to *initialise* a run that has no
            # auxiliary heads -- the second stage of a two-stage recipe, where the pretext
            # objective is dropped and only the task remains. ``AuxModel.aux`` is a sibling
            # of ``AuxModel.backbone``, not nested under it, so its keys are a bare ``aux.``
            # branch; the *backbone* keys are one ``backbone.`` prefix deeper than whatever
            # ``model.backbone`` (bare or residual-wrapped) itself expects. A plain load
            # would report every backbone parameter missing and, under strict=False, silently
            # train from scratch: the exact silent-failure class this project has already
            # paid 3.3 GPU-h for. Strip the wrapper, and let the count printed below prove
            # the transfer happened.
            sd = strip_aux(sd)
            print(f"stripped auxiliary heads from {cfg.init_from}: {len(sd)} tensors kept")
        if aux_weights(asdict(cfg)) and not any(k.startswith("aux.") for k in sd):
            # The other direction: a plain checkpoint (no auxiliary heads -- e.g. the shared
            # sim pretrain) initialising a run that *adds* one, such as a joint stage that
            # turns on ``w_opreg``. ``AuxModel.backbone`` is named ``backbone`` regardless of
            # what it wraps (bare or residual), so the source state dict -- saved one wrapping
            # level shallower -- is one ``backbone.`` prefix short, the mirror image of the
            # strip above.
            sd = {f"backbone.{k}": v for k, v in sd.items()}
            print(f"prefixed {cfg.init_from} for the auxiliary-head wrapper: {len(sd)} tensors")
        # Each input-side state wrapper adds one nesting level OUTSIDE the residual anchor, so
        # a checkpoint saved without it is exactly that many ``backbone.`` prefixes short --
        # the same situation as the auxiliary-head and bound-head branches above, and the same
        # silent failure if unhandled (every tensor missing, strict=False, a fine-tune that
        # quietly trains from scratch). Counted rather than hard-coded so a third wrapper does
        # not repeat the bug a third time.
        pre_state = state.get("config", {})
        n_wrap = sum(
            bool(getattr(cfg, flag)) and not bool(pre_state.get(flag, False))
            for flag in STATE_WRAPPER_FLAGS
        )
        for _ in range(n_wrap):
            sd = {f"backbone.{k}": v for k, v in sd.items()}
        if n_wrap:
            print(f"prefixed {cfg.init_from} for {n_wrap} state wrapper(s): {len(sd)} tensors")
        if cfg.bounds and not any(k.startswith("backbone.") for k in sd):
            # A plain checkpoint being wrapped for bound training: re-key it under
            # the BoundedModel's backbone prefix so the head stays freshly initialised.
            sd = {f"backbone.{k}": v for k, v in sd.items()}
        if (
            cfg.moment_of
            and sd
            and all(k.startswith("backbone.") for k in sd)
            and not any(k.startswith("backbone.") for k in model.state_dict())
        ):
            # A moment network warm-started from a residual-wrapped mean model (FINDINGS 153n):
            # drop the wrapper prefix and leave the head fresh -- its 40 log-variance channels
            # do not match the mean model's 60 field channels, and a shape mismatch would raise
            # even under strict=False.
            msd = model.state_dict()
            sd = {k[len("backbone.") :]: v for k, v in sd.items()}
            sd = {k: v for k, v in sd.items() if k not in msd or msd[k].shape == v.shape}
            print(
                f"moment warm start: wrapper prefix dropped, {len(sd)} tensors from {cfg.init_from}"
            )
        if cfg.kmap_channel:
            # A 3-channel checkpoint initialising a run with the variance-map channel: the two
            # output layers are widened with zeros in the new slots (models/kmap.py). A source
            # that already has the channel is left alone.
            sd = widen_output_channels(sd, model.state_dict())
        if cfg.nbr_prior:
            # The neighbour table is this run's (its cells), never the source checkpoint's.
            for k, v in model.state_dict().items():
                if k.endswith(("nbr_table", "nbr_coords")):
                    sd[k] = v
        missing, unexpected = model.load_state_dict(sd, strict=False)
        check_init_coverage(model, missing, unexpected, cfg.init_from)

    # A split-interface model (`STFormer(domain_interface=True)`) initialised from an ordinary
    # checkpoint gets its real branch from the file and its simulated branch from scratch,
    # which would make the split an initialisation lottery rather than a measurement. Copying
    # real -> sim makes the two branches identical at step 0, so the split model *is* its
    # shared-interface twin until training separates them.
    if cfg.init_from and getattr(model, "domain_interface", False):
        print(f"synced {model.sync_domain_interface()} tensors onto the simulated interface")

    # After the load, because freezing a parameter that was about to be overwritten by the
    # pretrained value would freeze it at its *initialisation* instead (`FT.2`).
    if cfg.freeze_patterns:
        n_frozen, n_all = apply_freeze_patterns(model, cfg.freeze_patterns)
        print(
            f"freeze_patterns={cfg.freeze_patterns}: {n_frozen / 1e6:.3f}M of "
            f"{n_all / 1e6:.2f}M frozen"
        )
    if cfg.unfreeze:
        if not cfg.init_from:
            raise ValueError(
                "unfreeze needs init_from: freezing part of a randomly-initialised model "
                "trains an adapter on top of noise, which is not what the ladder measures"
            )
        n_train, n_total = apply_unfreeze(model, cfg.unfreeze)
        print(
            f"unfreeze={cfg.unfreeze}: training {n_train / 1e6:.3f}M of "
            f"{n_total / 1e6:.2f}M parameters ({100 * n_train / n_total:.1f}%)"
        )

    # Captured after the load and before the first step, so it is the pretrained point
    # itself. Detached clones on the training device: the penalty is evaluated every step.
    ref_state: dict[str, torch.Tensor] | None = None
    if cfg.l2sp > 0:
        ref_state = {k: v.detach().clone() for k, v in model.named_parameters()}

    ema = Ema(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    tap = FeatureTap(model) if cfg.align_weight > 0 or cfg.coral_weight > 0 else None

    # End-to-end second-moment head (config.tke_head_e2e): a zero-init w8/d2 TkeHead whose
    # mean-preserving gain is applied to the frames INSIDE the step, so the whole objective
    # below grades the corrected output and gradients reach both modules. w8/d2 is the shipped
    # head configuration held fixed (AGENTS: ablate weights, not configuration).
    if cfg.kmap_e2e and cfg.tke_head_e2e:
        raise ValueError(
            "kmap_e2e and tke_head_e2e are two trained gain sources in one step; "
            "a run with both would attribute nothing to either -- pick one"
        )
    if cfg.kmap_e2e and not cfg.kmap_channel:
        raise ValueError("kmap_e2e spends the map channel; it needs --kmap-channel")

    e2e_head = None
    e2e_head_ema = None
    if cfg.tke_head_e2e:
        if cfg.bounds:
            raise ValueError(
                "tke_head_e2e corrects the frames the interval widths would calibrate against; "
                "train the bounds stage on the corrected checkpoint instead"
            )
        from ..models.tke_head import TkeHead, training_log_ratio  # noqa: PLC0415

        e2e_head = TkeHead(width=8, depth=2).to(device)
        e2e_head_ema = Ema(e2e_head, cfg.ema_decay) if cfg.ema_decay > 0 else None
        print("note: end-to-end second-moment head (w8/d2, zero-init = identity) in the objective")

    # `w_jepa`: a JEPA co-objective. The live model sees the MASKED input while an EMA copy sees
    # the clean one, and the term is the MSE between their deepest shared representations. Unlike
    # `align_weight` (which matches per-channel moments ACROSS DOMAINS in one batch) this matches a
    # corrupted view to a whole view of the SAME sample, so it is a prediction task rather than a
    # distribution constraint.
    #
    # A separate module rather than `Ema`'s state-dict shadow, because the target has to be
    # *forwarded* and a shadow cannot be. Its parameters never receive gradients; it is advanced by
    # EMA of the live weights, which is what keeps the objective from collapsing to a constant.
    jepa_tap = jepa_target = jepa_target_tap = None
    if cfg.w_jepa > 0:
        import copy as _copy  # noqa: PLC0415

        jepa_rng = np.random.default_rng(cfg.seed + 9173)
        jepa_tap = FeatureTap(model)
        jepa_target = _copy.deepcopy(model).to(device)
        for prm in jepa_target.parameters():
            prm.requires_grad_(False)
        jepa_target.eval()
        jepa_target_tap = FeatureTap(jepa_target)
        # The EFFECTIVE mask, not the config field: the term falls back to `tube` at ratio 0.5
        # when `aux_mask` is unset, and a log that printed the empty field would misreport what
        # the run actually did -- the class of disagreement PROTOCOL 5 opens with.
        print(
            f"jepa co-objective: w={cfg.w_jepa} ema={cfg.jepa_ema} "
            f"mask={cfg.aux_mask or 'tube'}@{cfg.aux_mask_ratio or 0.5} "
            f"tapping {jepa_tap.name!r} (in-loop masking, no visibility plane)"
        )
    if tap is not None:
        print(
            f"latent alignment: tapping {tap.name!r} (channel_dim={tap.channel_dim}), "
            f"moment_weight={cfg.align_weight} correlation_weight={cfg.coral_weight}"
        )

    train_loader, val_loaders = make_loaders(cfg)
    if cfg.recur_k > 0:
        raise ValueError("recurrent chains (train/recurrent.py) is not included in this package")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if e2e_head is not None:
        n_trainable += sum(p.numel() for p in e2e_head.parameters())

    # A jointly-trained bound head needs a different rate from the backbone it sits on:
    # the head is 13k randomly-initialised parameters that want the ~3e-3 a frozen run
    # uses, while the backbone is already fine-tuned and is wrecked at that rate. One
    # shared rate is either too high for one or too low for the other, which would make
    # a frozen-vs-joint comparison say nothing about freezing.
    # ``lr_scale`` rides along on each group because the schedule below rewrites every
    # group's ``lr`` each step; without it the two rates would collapse into one.
    head_params = [p for p in getattr(model, "head", nn.Module()).parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in trainable if id(p) not in head_ids]
    groups = [
        {"params": backbone_params, "lr_scale": cfg.backbone_lr_scale},
        {"params": head_params, "lr_scale": cfg.head_lr_mult},
    ]
    if cfg.llrd != 1.0:
        groups = llrd_groups(model, cfg.llrd, cfg.backbone_lr_scale, cfg.head_lr_mult)
    if cfg.trunk_lr_scale != 1.0:
        # The trunk is the complement of the `io` rung, read from `UNFREEZE_LADDER` rather than
        # respelled here: a second copy of that pattern list would drift from the ladder's, and
        # then `trunk_lr_scale=0` and `unfreeze=io` would silently name different parameters
        # while both claiming to be "freeze the trunk".
        iface = unfreeze_patterns("io") or ()
        iface_ids = {
            id(p)
            for name, p in model.named_parameters()
            if p.requires_grad and any(pat in name for pat in iface)
        }
        trunk = [p for p in backbone_params if id(p) not in iface_ids]
        rest = [p for p in backbone_params if id(p) in iface_ids]
        if not trunk:
            raise ValueError("trunk_lr_scale set but no parameter is outside the io interface")
        groups = [
            {"params": trunk, "lr_scale": cfg.backbone_lr_scale * cfg.trunk_lr_scale},
            {"params": rest, "lr_scale": cfg.backbone_lr_scale},
            {"params": head_params, "lr_scale": cfg.head_lr_mult},
        ]
        n_trunk = sum(p.numel() for p in trunk)
        print(
            f"trunk_lr_scale={cfg.trunk_lr_scale}: {n_trunk / 1e6:.3f}M of "
            f"{n_trainable / 1e6:.2f}M trainable parameters at "
            f"{cfg.backbone_lr_scale * cfg.trunk_lr_scale:g}x the base rate"
        )
    groups = [g for g in groups if g["params"]]
    if e2e_head is not None:
        # Its own group AFTER the backbone groups are built from `trainable` (extending that
        # list first would put these params in two groups). lr_scale 1.0 because the schedule
        # below rewrites every group's lr from that key; the freeze/lr-scale knobs are backbone
        # anatomy this 3k-parameter identity-init head has no part in. Extended into `trainable`
        # here so grad clipping and the norm log see the whole optimised set.
        groups.append({"params": list(e2e_head.parameters()), "lr_scale": 1.0})
        trainable = trainable + list(e2e_head.parameters())
    if cfg.weight_decay > 0:
        # Decoupled decay on the matrices only: LayerNorm gains, biases and every other 1-D
        # tensor are exempt, the usual AdamW convention. Split here rather than in the group
        # builders above so ``lr_scale`` and the freeze ladder are untouched; every recorded
        # run has weight_decay 0 and takes the other branch unchanged.
        split = []
        for g in groups:
            decay = [p for p in g["params"] if p.ndim > 1]
            no_decay = [p for p in g["params"] if p.ndim <= 1]
            if decay:
                split.append({**g, "params": decay, "weight_decay": cfg.weight_decay})
            if no_decay:
                split.append({**g, "params": no_decay, "weight_decay": 0.0})
        groups = split
        n_dec = sum(p.numel() for g in groups if g["weight_decay"] > 0 for p in g["params"])
        print(f"note: weight_decay={cfg.weight_decay} on {n_dec / 1e6:.3f}M params (ndim>1 only)")
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.train_compile:
        raise ValueError("train_compile (the recurrent chain step) is not included in this package")
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device == "cuda")

    print(
        f"model={cfg.model} residual={cfg.residual} bounds={cfg.bounds} "
        f"params={n_params / 1e6:.2f}M trainable={n_trainable / 1e6:.3f}M device={device}"
    )
    eff = cfg.batch_size * cfg.grad_accum
    print(
        f"train windows={len(train_loader.dataset)} steps={cfg.steps} "
        f"bs={cfg.batch_size}x{cfg.grad_accum}=eff {eff}"
    )
    run.summary(
        {
            "model/params": n_params,
            "model/trainable": n_trainable,
            "model/amp": cfg.amp,
            "data/train_windows": len(train_loader.dataset),
            "data/effective_batch": eff,
        }
    )

    # Optimiser steps per pass over the data, so the log line can report progress in epochs
    # as well as steps -- with stride-1 windows overlapping 20-fold, "epoch" is a loose unit,
    # but it is the one the reader thinks in when choosing eval_every.
    steps_per_epoch = max(1, len(train_loader) // max(1, cfg.grad_accum))

    history: list[dict] = []
    scores_history: list[dict] = []
    best = math.inf
    # Official scoring is a count of evenly-spaced points, converted to a step interval
    # here; e.g. 20 points over 5000 steps scores every 250. ``score_ok`` latches off after
    # the first failure (an unbuilt eval set, say) so a broken scoring setup costs one note,
    # not one per point.
    score_interval = max(1, cfg.steps // cfg.score_points) if cfg.score_points > 0 else 0
    score_ok = cfg.score_points > 0
    step = 0
    micro = 0
    opt.zero_grad(set_to_none=True)
    teachers: list[nn.Module] = []
    if cfg.distill_head and not cfg.distill_from:
        raise ValueError("distill_head corrects the teachers' prediction; it needs distill_from")
    if cfg.distill_from:
        from ..models.factory import load_for_inference  # noqa: PLC0415

        for c in (s.strip() for s in cfg.distill_from.split(",")):
            if not c:
                continue
            # ``kit_dir`` is what lets a VENDORED baseline be a teacher: `cno`/`fno`/
            # `transolver` are rebuilt by ``load_baseline``, which is only importable once the
            # kit is on ``sys.path``. Without it a mixed-architecture teacher list raises
            # ModuleNotFoundError at load time -- so the ensemble-distillation feature could
            # never actually reach the three released baselines it was written to consume.
            tm, _ = load_for_inference(c, kit_dir=VENDOR_KIT, device=device)
            tm.eval()
            for p in tm.parameters():
                p.requires_grad_(False)
            teachers.append(tm)
        print(f"distillation: {len(teachers)} teachers, alpha={cfg.distill_alpha}")

    if cfg.state_lr_mult != 1.0:
        # Scale the gradient of ONLY the stem columns that read the state pseudo-frames. They
        # are zero-initialised (widen_stem_history pads at the FRONT, so they are the first
        # n_frames * C columns) while every other weight arrives pretrained, which makes them
        # the undertrained part of the model at any fine-tune length. A hook rather than a
        # parameter group because they are columns of one weight tensor, not a separate module.
        if not cfg.stream_state:
            raise ValueError("state_lr_mult needs --stream-state; there are no state columns")
        n_cols = stream_state_frames(cfg.stream_state) * C
        hooked = []
        for name, param in model.named_parameters():
            if any(name.endswith(k) for k in GROUPED_STEM_KEYS):
                mult = float(cfg.state_lr_mult)
                param.register_hook(
                    lambda g, n=n_cols, m=mult: torch.cat((g[:, :n] * m, g[:, n:]), dim=1)
                )
                hooked.append(name)
        if not hooked:
            raise ValueError("state_lr_mult found no grouped stem weight to hook")
        print(f"state-lr: x{cfg.state_lr_mult} on {n_cols} state columns of {hooked}")

    stream_corr = None
    if cfg.stream_corr_lam > 0:
        # A real-only stage: sim windows and phase copies carry trajectory ids from other
        # datasets (colliding index spaces), and distillation rewrites batch["target"] before
        # the residual would read it.
        if cfg.joint_sim_frac > 0 or cfg.phase_aug or teachers:
            raise ValueError(
                "stream_corr_lam needs a plain real-only stage: no joint_sim_frac, no "
                "phase_aug, no distillation (trajectory ids and targets would not be the "
                "stream's own)"
            )
        stream_corr = _StreamCorr(cfg.stream_corr_lam, cfg.stream_corr_ema)
        print(f"stream-corr: lam={cfg.stream_corr_lam} ema={cfg.stream_corr_ema}")

    t0 = time.perf_counter()
    model.train()
    # Per-cell drop-path keep vector holder (None unless cfg.drop_path_hi is set); the drop-path
    # hook closes over this same list, so the loop fills it each step (see `drop_path_keep`).
    dp_keep_holder = getattr(model, "_dp_keep_holder", None)

    while step < cfg.steps:
        # Before the iterator is built, which is the only moment the weighted sampler re-reads
        # its weights. Printed rather than assumed: the whole point of the arm is the schedule,
        # so the schedule has to be visible in the log next to the step it took effect at.
        annealed = retune_sim_fraction(train_loader, cfg, step / max(1, cfg.steps))
        if annealed is not None:
            print(f"  sim fraction -> {annealed:.3f} at step {step}")
        for batch in train_loader:
            if step >= cfg.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg) * g.get("lr_scale", 1.0)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if cfg.rollout_w > 0:
                # ``rollout_w`` (FINDINGS 224d): the loader hands 2 x T_OUT target frames; the
                # first T_OUT are the ordinary target, the rest are what the model must predict
                # from its OWN forecast fed back as the next input window.
                batch["target2"] = batch["target"][:, T_OUT:]
                batch["target"] = batch["target"][:, :T_OUT]

            # Operating-point-dosed drop-path: fill the hook's per-sample keep vector from this
            # batch's Reynolds. Set here (once per step, on the batch the forward will read) so the
            # main forward and any auxiliary forward on the same batch share the draw's rate; a
            # forward on a differently-sized tensor falls back to the flat dose inside the hook.
            if dp_keep_holder is not None:
                dp_keep_holder[0] = drop_path_keep(
                    batch["cond"][:, 0], cfg.drop_path, cfg.drop_path_hi
                )

            if teachers:
                # Replace the field target with the teachers' mean prediction (blended with
                # the truth by ``distill_alpha``). Computed in fp32 outside autocast so the
                # target is stable; the student still trains under autocast below.
                with torch.no_grad():
                    tp = torch.stack(
                        [
                            split_output(
                                call_model(tm, batch["input"], batch["cond"], batch.get("u_fs"))
                            )[0]
                            for tm in teachers
                        ]
                    ).mean(0)
                    if cfg.distill_head:
                        raise ValueError("distill_head (train/absorb.py) is not included in this package")
                # Keep the ground truth before overwriting, so tke terms can anchor to reality
                # rather than to the teacher's (variance-deficient) prediction -- see
                # ``distill_tke_truth`` and ``compute_losses``.
                batch["truth"] = batch["target"]
                batch["target"] = distill_target(
                    tp, batch["target"], cfg.distill_alpha, cfg.distill_part
                )

            if stream_corr is not None:
                sb = stream_corr.bias(batch["traj"])
                if sb is not None:
                    batch["stream_bias"] = sb

            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                model_input = batch["input"]
                if cfg.recon_clean_field:
                    if not cfg.aux_mask or (
                        cfg.w_recon <= 0 and cfg.w_recon_tke <= 0 and cfg.w_recon_acf <= 0
                    ):
                        raise ValueError(
                            "recon_clean_field needs aux_mask and a reconstruction objective"
                        )
                    if geom_names(cfg) or flow_names(cfg):
                        raise ValueError(
                            "recon_clean_field is not yet defined with geometry/flow planes"
                        )
                    clean_input = model_input.clone()
                    clean_input[..., :C] = batch["recon_target"]
                    if cfg.aux_mask_plane:
                        clean_input[..., C] = 1.0
                    out = call_model(model, clean_input, batch["cond"], batch.get("u_fs"))
                    loss, parts = compute_losses(out, batch, cfg, progress=step / max(1, cfg.steps))
                    # `compute_losses` saw the clean-view auxiliary activation. Remove that
                    # identity reconstruction and replace it with the masked-view loss.
                    loss = loss - cfg.w_recon * parts.get("recon", 0.0)
                    loss = loss - cfg.w_recon_tke * parts.get("recon_tke", 0.0)
                    loss = loss - cfg.w_recon_acf * parts.get("recon_acf", 0.0)
                    if step / max(1, cfg.steps) < cfg.recon_stop_frac:
                        masked_out = call_model(
                            model, model_input, batch["cond"], batch.get("u_fs")
                        )
                        masked_aux, masked_parts = auxiliary_losses(masked_out["aux"], batch, cfg)
                        assert masked_aux is not None
                        loss = loss + masked_aux
                        parts.update(masked_parts)
                    else:
                        parts["recon_stopped"] = 1.0
                    parts["total"] = float(loss)
                elif cfg.recur_k > 0:
                    raise ValueError("recurrent chains (train/recurrent.py) is not included in this package")
                else:
                    if nbr_mods:
                        for m_ in nbr_mods:
                            m_.set_label(batch["re"], batch["aoa"])
                    cond_in = shuffle_cond(batch["cond"], cfg.cond_drop)
                    out = call_model(model, model_input, cond_in, batch.get("u_fs"))
                    out = call_model(model, model_input, batch["cond"], batch.get("u_fs"))
                    r_e2e = None
                    if e2e_head is not None:
                        # batch["input"], not model_input: the head must read the window
                        # serving will hand it, not a training-time augmentation of it.
                        r_e2e = training_log_ratio(out, batch["input"], batch["cond"], e2e_head)
                    elif cfg.kmap_e2e:
                        r_e2e = kmap_e2e_log_ratio(out, batch["input"], base=cfg.kmap_base)
                    if r_e2e is not None:
                        from ..models.fluctuations import (  # noqa: PLC0415
                            apply_fluctuation_gain_torch as _gain,
                        )

                        out = _gain(out, torch.exp(0.5 * r_e2e))
                    loss, parts = compute_losses(out, batch, cfg, progress=step / max(1, cfg.steps))
                    if cfg.rdrop_alpha > 0:
                        # R-drop: a second forward with independent drop-path masks; average
                        # the two task losses and penalise their u,v disagreement.
                        out2 = call_model(model, model_input, batch["cond"], batch.get("u_fs"))
                        loss2, _rp = compute_losses(
                            out2, batch, cfg, progress=step / max(1, cfg.steps)
                        )
                        _p1 = split_output(out)[0][..., :N_MEASURED].float()
                        _p2 = split_output(out2)[0][..., :N_MEASURED].float()
                        _cons = torch.nn.functional.mse_loss(_p1, _p2)
                        loss = 0.5 * (loss + loss2) + cfg.rdrop_alpha * _cons
                        parts["rdrop_cons"] = float(_cons)
                        parts["total"] = float(loss)
                    if cfg.rollout_w > 0:
                        # Feed the forecast back (the full 3-channel frames, in the same
                        # freestream units) and score the next window against the real frames.
                        # The gradient runs through both calls, so the first forecast is pushed
                        # to be a window the model can continue from -- a damped, energy-poor
                        # forecast continues worse, which is the pressure on the energy deficit.
                        pred1 = out["prediction"] if isinstance(out, dict) else out
                        # As a measured window looks: zero in the PIV mask, no pressure (real
                        # windows carry none).
                        vm = batch["valid"][:, None, :, :, None].to(pred1.dtype)
                        uv = pred1[..., :N_MEASURED] * vm
                        x2 = torch.cat((uv, torch.zeros_like(pred1[..., N_MEASURED:])), dim=-1)
                        b2 = dict(batch, input=x2, target=batch["target2"])
                        b2, share = rollout_select(b2, cfg.rollout_aoa_min, cfg.rollout_re_min)
                        if share > 0:
                            out2 = call_model(model, b2["input"], b2["cond"], b2.get("u_fs"))
                            cfg2 = (
                                dc_replace(cfg, w_field=0.0, w_headcredit=0.0)
                                if cfg.rollout_mode == "tke"
                                else cfg
                            )
                            loss2, parts2 = compute_losses(
                                out2, b2, cfg2, progress=step / max(1, cfg.steps)
                            )
                            w2 = cfg.rollout_w if share == 1.0 else cfg.rollout_w * share
                            loss = loss + w2 * loss2
                            parts["rollout"] = float(loss2)
                            parts["rollout_field"] = parts2.get("field", float("nan"))
                        parts["rollout_share"] = share
                        parts["total"] = float(loss)
                    if r_e2e is not None:
                        # 22's discipline: verified active at convergence. |r| stuck at ~0
                        # means the gain never left the identity and the arm is the control.
                        parts["e2e_r_abs"] = float(r_e2e.detach().abs().mean())
                    if cfg.log_domain_split and "cond" in batch and batch["cond"].shape[1] > 2:
                        # Detached diagnostic: per-domain train rel_l2 (real vs sim), split on the
                        # is_sim flag cond[:, 2]. Never touches loss/grad/RNG. Uses the same
                        # per-sample metric as quick_eval (relative_l2 on the two scored channels),
                        # so rel_real is comparable to the val_* rel_l2 columns.
                        with torch.no_grad():
                            _p = split_output(out)[0].detach().float()
                            _rl = relative_l2(_p[..., :2], batch["target"][..., :2].float())
                            _is_sim = batch["cond"][:, 2] > 0.5
                            if (~_is_sim).any():
                                parts["rel_real"] = float(_rl[~_is_sim].mean())
                            if _is_sim.any():
                                parts["rel_sim"] = float(_rl[_is_sim].mean())

            if stream_corr is not None:
                # AFTER the loss (the applied bias is the pre-update estimate, causal in
                # training time), from the live model's detached prediction.
                stream_corr.update(batch["traj"], out, batch)

            if tap is not None:
                # ``cond[..., 2]`` is the is_sim flag the dataset already carries, so the
                # split is the loader's own notion of domain rather than a second one.
                assert tap.value is not None, "the alignment hook did not fire"
                if cfg.align_weight > 0:
                    al = domain_moment_loss(tap.value, batch["cond"][:, 2], tap.channel_dim)
                    loss = loss + cfg.align_weight * al
                    parts["align"] = float(al)
                if cfg.coral_weight > 0:
                    coral = domain_correlation_loss(tap.value, batch["cond"][:, 2], tap.channel_dim)
                    loss = loss + cfg.coral_weight * coral
                    parts["coral"] = float(coral)
                parts["total"] = float(loss)

            if jepa_target is not None:
                # MASKED view through the live model, CLEAN view through the EMA target.
                #
                # The mask is applied HERE rather than by the dataset's `aux_mask`, and that is a
                # deliberate narrowing. `aux_mask` appends a visibility plane, which changes the
                # input width -- so a task-only pretrain no longer loads, and the stem-widening
                # path that would fix it computes its `c_old` including the very plane being added
                # and refuses. Masking in the loop keeps the input width identical to every other
                # arm, so `--init-from` works and this is a one-variable change.
                #
                # The cost, recorded rather than hidden: with no visibility plane the model cannot
                # tell a masked cell from a PIV measurement hole, which is also zero. Both are
                # "no data here", so the ambiguity is mild -- but it means the model is not told
                # WHERE to predict, only asked to encode the whole field consistently.
                b, t, h, w, _ = batch["input"].shape
                # One mask per SAMPLE, not one per batch: a shared mask would let the model key on
                # a fixed hidden pattern within a step. `True` is hidden, hence the `~`.
                hid = np.stack(
                    [
                        sample_input_mask(
                            jepa_rng, cfg.aux_mask or "tube", cfg.aux_mask_ratio or 0.5, t, h, w
                        )
                        for _ in range(b)
                    ]
                )
                keep = torch.from_numpy(~hid).to(batch["input"].device)
                masked = batch["input"] * keep[..., None].to(batch["input"].dtype)
                with torch.no_grad():
                    call_model(jepa_target, batch["input"], batch["cond"], batch.get("u_fs"))
                    clean_repr = jepa_target_tap.value
                with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                    call_model(model, masked, batch["cond"], batch.get("u_fs"))
                assert jepa_tap.value is not None, "the jepa hook did not fire on the live model"
                assert clean_repr is not None, "the jepa hook did not fire on the target"
                jl = torch.nn.functional.mse_loss(
                    jepa_tap.value.float(), clean_repr.float().detach()
                )
                loss = loss + cfg.w_jepa * jl
                parts["jepa"] = float(jl)
                parts["total"] = float(loss)

            if ref_state is not None:
                # Outside the autocast block: a sum of 6M squared differences in fp16
                # overflows long before the differences themselves are large.
                pen = l2sp_penalty(model, ref_state)
                loss = loss + cfg.l2sp * pen
                parts["l2sp"] = float(pen)
                parts["total"] = float(loss)

            # Scale down so accumulated gradients average rather than sum, keeping
            # the effective learning rate independent of grad_accum.
            scaler.scale(loss / cfg.grad_accum).backward()
            micro += 1
            if micro < cfg.grad_accum:
                continue

            micro = 0
            if cfg.sam_rho > 0:
                # Sharpness-aware minimisation (``TrainConfig.sam_rho``): ascend to
                # w + rho * g / ||g|| along this batch's gradient, take the gradient THERE on the
                # same batch, come back, and let the optimiser step with that gradient. The
                # gradients are still loss-scaled, so the norm is unscaled by hand; GradScaler's
                # own unscale_ may run only once per step and belongs to the clip below.
                if (
                    cfg.grad_accum != 1
                    or cfg.recon_clean_field
                    or cfg.recur_k > 0
                    or tap is not None
                    or jepa_target is not None
                    or ref_state is not None
                    or stream_corr is not None
                    or e2e_head is not None
                    or cfg.kmap_e2e
                    or cfg.headcredit_track_lr > 0
                ):
                    raise ValueError("sam_rho supports the plain training step only")
                inv = 1.0 / float(scaler.get_scale()) if scaler.is_enabled() else 1.0
                with torch.no_grad():
                    gnorm = float(
                        torch.norm(
                            torch.stack(
                                [p_.grad.detach().float().norm() for p_ in trainable
                                 if p_.grad is not None]
                            )
                        )
                    ) * inv
                if math.isfinite(gnorm) and gnorm > 0:
                    k = cfg.sam_rho * inv / (gnorm + 1e-12)
                    eps: list[torch.Tensor | None] = []
                    with torch.no_grad():
                        for p_ in trainable:
                            if p_.grad is None:
                                eps.append(None)
                                continue
                            e = (p_.grad.detach().float() * k).to(p_.dtype)
                            p_.add_(e)
                            eps.append(e)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                        out_sam = call_model(model, model_input, batch["cond"], batch.get("u_fs"))
                        loss_sam, _ = compute_losses(
                            out_sam, batch, cfg, progress=step / max(1, cfg.steps)
                        )
                    scaler.scale(loss_sam).backward()
                    with torch.no_grad():
                        for p_, e in zip(trainable, eps, strict=True):
                            if e is not None:
                                p_.sub_(e)
                    parts["sam_loss"] = float(loss_sam)
                    parts["sam_gnorm"] = gnorm
                # A non-finite norm is an overflowed fp16 step: GradScaler skips it as usual.
            # ``clip_grad_norm_`` returns the pre-clip total norm, which is the quantity
            # worth watching -- so it is captured, not discarded. When clipping is off it is
            # still computed on log steps only (max_norm=inf clips nothing), so the signal is
            # there without paying for it every step.
            grad_norm: float | None = None
            log_now = step % cfg.log_every == 0
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip))
            elif log_now:
                scaler.unscale_(opt)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, math.inf))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                # After the optimiser step and once per *optimiser* step, not per
                # micro-batch: an update per micro-batch would make the effective decay
                # depend on grad_accum, so two runs differing only in accumulation would
                # average over different horizons.
                ema.update(model)
            if e2e_head_ema is not None:
                # The served pair is (ema backbone, ema head): averaging one side of a
                # co-adapted pair would serve a head tuned to weights it never sees.
                e2e_head_ema.update(e2e_head)

            if jepa_target is not None:
                # Same placement argument as `ema` above: once per OPTIMISER step, so the decay
                # horizon does not silently depend on grad_accum. Done in-place under no_grad so
                # the target never enters the autograd graph.
                with torch.no_grad():
                    d = cfg.jepa_ema
                    for tp, lp in zip(
                        jepa_target.state_dict().values(),
                        model.state_dict().values(),
                        strict=True,
                    ):
                        if tp.dtype.is_floating_point:
                            tp.mul_(d).add_(lp.detach(), alpha=1.0 - d)
                        else:
                            tp.copy_(lp)

            if log_now:
                rate = (step + 1) / (time.perf_counter() - t0)
                epoch = step / steps_per_epoch
                eta = _fmt_eta((cfg.steps - step) / rate) if rate > 0 else "?"
                lr_now = lr_at(step, cfg)
                print(
                    f"  step {step:6d}/{cfg.steps}  ep {epoch:5.1f}  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in parts.items())
                    + f"  lr={lr_now:.2e}"
                    + (f"  |g|={grad_norm:.2f}" if grad_norm is not None else "")
                    + f"  {rate:.1f} it/s  eta {eta}"
                )
                run.log(train_row(parts, lr_now, rate, grad_norm, epoch), step=step)

            if step > 0 and step % cfg.eval_every == 0:
                progress = step / max(1, cfg.steps)
                metrics = {
                    s: quick_eval(model, dl, device, cfg=cfg, progress=progress)
                    for s, dl in val_loaders.items()
                }
                record = {"step": step, "train": parts, "val": metrics}
                print(
                    f"  eval  step {step}: "
                    + "  ".join(
                        f"{s}[loss={m['loss']:.4f} rel={m['rel_l2']:.4f} tke={m['tke']:.4f}]"
                        for s, m in metrics.items()
                    )
                )
                history.append(record)
                selector = metrics.get("val_interp", next(iter(metrics.values())))
                score = selector["rel_l2"] + selector["tke"]
                row = val_row(metrics)
                row["select_score"] = score
                run.log(row, step=step)
                if score < best:
                    best = score
                    _save(out_dir / "best.pt", model, cfg, step, record)
                    run.summary({"best/step": step, "best/selector": best})

            # Official scoring rides its own schedule, not eval_every's, so ``score_points``
            # lands exactly N points over the run regardless of how often validation runs.
            # Kept out of history.json (whose records are the quick_eval diagnostic) and in
            # its own scores.json, so a leaderboard number never sits in a field readers take
            # for the non-dimensional one.
            if score_ok and step > 0 and step % score_interval == 0:
                scores = official_scores(model, cfg, device)
                if scores is None:
                    score_ok = False  # unbuilt eval set or a broken scorer: stop retrying
                else:
                    scores_history.append({"step": step, "scores": scores})
                    run.log(score_row(scores), step=step)
                    print(
                        f"  score step {step}: "
                        + "  ".join(
                            f"{s}[rel={v.get('rel_l2_score', 0):.1f} "
                            f"tke={v.get('tke_score', 0):.1f} "
                            f"mvpe={v.get('mvpe_score', 0):.1f}]"
                            for s, v in scores.items()
                        )
                    )

            if cfg.snapshot_every > 0 and step > 0 and step % cfg.snapshot_every == 0:
                # Ingredients for a trajectory soup (``train soup``). Kept as separate
                # files rather than accumulated in place because which *stretch* of the
                # trajectory to average is the question being asked -- and because
                # ``best.pt`` is one checkpoint under a selection rule, not a series.
                _save(out_dir / f"snap_{step:06d}.pt", model, cfg, step, {"step": step})
            step += 1

    metrics = {s: quick_eval(model, dl, device, cfg=cfg) for s, dl in val_loaders.items()}
    history.append({"step": step, "val": metrics, "final": True})
    _save(out_dir / "last.pt", model, cfg, step, history[-1])

    # The last iterate is the checkpoint the protocol scores, so it is always scored when
    # scoring is on -- whether or not the final step fell on a ``score_points`` boundary --
    # and *before* the EMA weights below overwrite the model in place.
    if score_ok:
        final_scores = official_scores(model, cfg, device)
        if final_scores is not None:
            scores_history.append({"step": step, "scores": final_scores, "final": True})
            run.log(score_row(final_scores), step=step)
            run.summary({f"final_{k}": v for k, v in score_row(final_scores).items()})
            print(
                "  final score: "
                + "  ".join(
                    f"{s}[rel={v.get('rel_l2_score', 0):.1f} "
                    f"tke={v.get('tke_score', 0):.1f} "
                    f"mvpe={v.get('mvpe_score', 0):.1f}]"
                    for s, v in final_scores.items()
                )
            )

    if ema is not None:
        # Written as a second complete checkpoint rather than replacing last.pt, so the
        # averaged and final iterates are both scoreable from one run and the comparison
        # is paired: same batches, same seed, same everything but the averaging.
        ema_state = ema.state_dict(model)
        torch.save(
            {"model": ema_state, "config": cfg.__dict__, "step": step, "record": history[-1]},
            out_dir / "ema.pt",
        )
        model.load_state_dict(ema_state)
        ema_metrics = {s: quick_eval(model, dl, device, cfg=cfg) for s, dl in val_loaders.items()}
        history.append({"step": step, "val": ema_metrics, "final": True, "ema": True})
        print(
            f"  ema (decay {cfg.ema_decay}): "
            + "  ".join(
                f"{s}[rel={m['rel_l2']:.4f} tke={m['tke']:.4f}]" for s, m in ema_metrics.items()
            )
        )
        run.summary({f"ema/{k}": v for k, v in val_row(ema_metrics).items()})
    if cfg.w_headcredit > 0 and cfg.headcredit_track_lr > 0:
        # The tracked interval head as it ended, for the record; the shipped head is refitted.
        from .head_credit import _CACHE

        for fh in _CACHE.values():
            torch.save(
                {
                    "head": {k: v.cpu() for k, v in fh.head.state_dict().items()},
                    "config": fh.cfg.__dict__,
                    "track_steps": fh.track_steps,
                },
                out_dir / "headcredit_tracked.pt",
            )
    if e2e_head is not None:
        # `train tke-head`'s payload layout, so load_head/head_uses_tau/`score --tke-head`
        # and the packaging path serve this head unchanged. "head" is the EMA average (the
        # served pair is ema.pt + this); the final iterate rides along for last.pt reads.
        # cond_mode never names tau: training_log_ratio zeroes that plane, and a payload
        # claiming tau would make serving feed the head a channel it was trained without.
        torch.save(
            {
                "head": {
                    k: v.cpu()
                    for k, v in (
                        e2e_head_ema.state_dict(e2e_head)
                        if e2e_head_ema is not None
                        else e2e_head.state_dict()
                    ).items()
                },
                "head_raw": {k: v.cpu() for k, v in e2e_head.state_dict().items()},
                "width": 8,
                "depth": 2,
                "cond_mode": "operating_point",
                "e2e": True,
            },
            out_dir / "tke_head_e2e.pt",
        )
        print(f"  e2e second-moment head -> {out_dir / 'tke_head_e2e.pt'}")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if scores_history:
        (out_dir / "scores.json").write_text(json.dumps(scores_history, indent=2), encoding="utf-8")
    minutes = (time.perf_counter() - t0) / 60
    print(f"done in {minutes:.1f} min -> {out_dir}")
    print(
        "final: "
        + "  ".join(f"{s}[rel={m['rel_l2']:.4f} tke={m['tke']:.4f}]" for s, m in metrics.items())
    )
    # The final row is what ``last.pt`` was written from, and last.pt -- not best.pt -- is
    # the checkpoint the protocol scores, so it is summarised as well as logged.
    final = val_row(metrics)
    run.log(final, step=step)
    run.summary({"perf/minutes": minutes, **{f"final/{k}": v for k, v in final.items()}})
    run.finish()
    return out_dir


def _save(path: Path, model: nn.Module, cfg: TrainConfig, step: int, record: dict) -> None:
    torch.save(
        {"model": model.state_dict(), "config": cfg.__dict__, "step": step, "record": record}, path
    )
