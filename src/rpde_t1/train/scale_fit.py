"""Fit an error-scale head on cached frozen-backbone outputs, then size it with the width rule.

Two decisions define this module, and both came out of measurement rather than taste.

**The objective is a regression, not the SPS relaxation.** ``losses.sps_loss`` replaces the
scorer's hard inside/outside indicator with sigmoids at a *fixed absolute* temperature
``tau=0.02``. Widths here span three decades, so that temperature is enormous relative to a
narrow interval and negligible relative to a wide one, and the relaxation consequently
over-rewards widening. That is the mechanism behind FINDINGS 10's fitted 0.60 multiplier --
which this project reproduced on U-Net, FNO and CNO alike, and which is *also* optimal on
in-sample windows, so it was never the held-out-error story it was recorded as. Predicting
``log|err|`` by least squares has no temperature, no vanishing gradient where the interval
sits far from the target, and a dense target at every element.

**Sizing is separated from ranking.** The head outputs a scale; a
:class:`~rpde_t1.models.width_rule.WidthRule` fitted against the scorer's own credit turns
that scale into an interval. The two halves have different best solutions -- one is a
conditional-mean regression, the other a one-dimensional maximisation with no model in it --
and solving them together is what the single fitted constant was papering over.

The backbone is frozen, so its outputs are cached once per split (``scripts/sps/sps_cache.py``)
and every arm here trains on tensors rather than on forward passes: ~2 minutes instead of
~12, which is what makes an honest fit/select/report protocol affordable per arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .. import N_MEASURED, SIGMA_GLOBAL
from ..features import COND_DIM
from ..models.bounds import BoundHead, bound_features, head_in_channels
from ..models.width_rule import WidthRule, fit_prescaled
from ..score.sps_lab import Cached, sps_breakdown
from .seeds import seed_everything


@dataclass
class ScaleFitConfig:
    """Everything an arm varies. Defaults are the configuration selected in FINDINGS 14."""

    steps: int = 1500
    batch_size: int = 4
    lr: float = 3e-3
    weight_decay: float = 0.0
    width: int = 32
    depth: int = 3
    cond: bool = True
    noise: bool = False
    logs: bool = True
    coords: bool = False
    #: Spatial-derivative channels; see ``bounds.GRAD_FEATURE_CHANNELS``.
    grad: bool = False
    #: Signed departure of the prediction from its own temporal mean; see
    #: ``bounds.FLUC_FEATURE_CHANNELS``. Added 2026-09-16 (Bruno).
    fluc: bool = False
    #: Window, in frames, for a PER-LEAD temporal spread of the prediction; 0 disables it. See
    #: ``bounds.LSTD_FEATURE_CHANNELS``: the head's existing two spread features are pooled over
    #: the whole horizon and broadcast to every lead, while the error grows 3.5x across it.
    lstd: int = 0
    #: Route the operating point through FiLM: 0 planes only (shipped), 1 FiLM instead of planes,
    #: 2 both. See :class:`~rpde_t1.models.bounds.BoundHead`. Added 2026-09-16 (Bruno).
    film: int = 0
    #: Kernel size of the head's convolutions along the lead axis, 3 or 1; see
    #: :class:`~rpde_t1.models.bounds.BoundHead`. Added 2026-09-26 (cheaper-head campaign).
    kt: int = 3
    #: Pooled window context; see ``bounds.POOL_FEATURE_CHANNELS``.
    pool: bool = False
    #: ``l2`` is least squares on ``log|err|`` (lognormal ML); ``l1`` is its median
    #: counterpart, which is the more robust target given a kurtosis of ~25. ``nll`` is the
    #: two-parameter version: the head also predicts the SPREAD of ``log|err|`` and is fitted
    #: by the Gaussian likelihood of ``log|err| ~ Normal(log s, tau**2)``, which contains
    #: ``l2`` as the constant-``tau`` case. The spread is not a width -- it is a candidate
    #: second binning axis for the width rule (``scripts/sps/sps_shape.py``).
    #: ``pinball`` is the quantile design (Bruno, 2026-09-16): a ``quantile`` head whose two
    #: ends are fitted as the ``tau_lo`` and ``tau_hi`` conditional quantiles of the signed
    #: error by the pinball loss, so skew and tail are learned per element instead of being
    #: pooled by the width rule. No log, no size target.
    loss: str = "l2"
    #: Quantile levels of the two ends under ``loss="pinball"``; unused otherwise.
    tau_lo: float = 0.05
    tau_hi: float = 0.95
    seed: int = 0
    #: Bins for the width rule fitted on top. ``1`` reduces to a per-channel constant,
    #: which measured as good as any finer table on held-out data.
    n_rho: int = 1
    n_lead: int = 1
    asymmetric: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)


class ScaleDataset:
    """Cached windows as torch tensors, kept in fp16 on the host and cast per batch."""

    def __init__(
        self, caches: list[Cached], device: str, teacher: list[np.ndarray] | None = None
    ) -> None:
        self.device = device
        self.pred = torch.from_numpy(np.concatenate([c.pred for c in caches])).half()
        self.x = torch.from_numpy(np.concatenate([c.inputs for c in caches])).half()
        self.target = torch.from_numpy(np.concatenate([c.target for c in caches])).half()
        self.cond = torch.from_numpy(np.concatenate([c.cond for c in caches])).float()
        self.u_fs = torch.from_numpy(np.concatenate([c.u_fs for c in caches])).float()
        # A teacher head's physical error scale on the same windows (`predict_scale`'s output),
        # the regression target of a distilled fit; None for the ordinary fit on `log|err|`.
        self.tscale = None
        if teacher is not None:
            self.tscale = torch.from_numpy(np.concatenate(teacher)).half()
            if self.tscale.shape[:-1] != self.pred.shape[:-1]:
                raise ValueError(
                    f"teacher scales {tuple(self.tscale.shape)} do not match the cached "
                    f"predictions {tuple(self.pred.shape)}"
                )

    def __len__(self) -> int:
        return self.pred.shape[0]

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        d = self.device
        out = {
            "pred": self.pred[idx].to(d).float(),
            "x": self.x[idx].to(d).float(),
            "target": self.target[idx].to(d).float(),
            "cond": self.cond[idx].to(d),
            "u_fs": self.u_fs[idx].to(d),
        }
        if self.tscale is not None:
            out["tscale"] = self.tscale[idx].to(d).float()
        return out


def _nondim(b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cached arrays are physical; the head works non-dimensionally, as at training time."""
    s = b["u_fs"].reshape(-1, 1, 1, 1, 1).clamp_min(1e-8)
    return b["pred"] / s, b["x"] / s, b["target"] / s


def head_mode(loss: str) -> str:
    """The head architecture an objective needs. One definition, shared with the merge."""
    return {"nll": "logerr_tau", "pinball": "quantile"}.get(loss, "logerr")


def pinball(u: torch.Tensor, tau: float) -> torch.Tensor:
    """``rho_tau(u) = tau*u`` for ``u >= 0`` and ``(tau-1)*u`` for ``u < 0``, elementwise.

    Its expected value over ``u = y - q`` is minimised at the ``tau``-quantile of ``y``: the
    derivative is ``-tau P(y > q) + (1 - tau) P(y < q)``, zero when ``P(y < q) = tau``.
    """
    return torch.maximum(tau * u, (tau - 1.0) * u)


def build_head(cfg: ScaleFitConfig, cond_width: int = COND_DIM) -> BoundHead:
    """``cond_width`` comes from the cache's own conditioning vectors rather than a constant:
    ``operating_point_tau`` makes them 4-wide and a hard-coded 3 builds the wrong first layer."""
    return BoundHead(
        in_channels=head_in_channels(
            cfg.cond,
            cfg.noise,
            cfg.logs,
            cfg.coords,
            cfg.grad,
            cfg.pool,
            cond_width,
            cfg.fluc,
            cfg.lstd,
        ),
        width=cfg.width,
        depth=cfg.depth,
        film=cfg.film,
        cond_dim=cond_width,
        kt=cfg.kt,
        # The objective decides the head's shape: only ``nll`` and ``pinball`` have a second
        # output to fit, and building the wider head for the other losses would leave half of
        # it untrained.
        mode=head_mode(cfg.loss),
    )


def train_scale_head(
    cfg: ScaleFitConfig,
    fit: list[Cached],
    device: str = "cuda",
    log_every: int = 250,
    log: list | None = None,
    teacher: list[np.ndarray] | None = None,
) -> BoundHead:
    """Least-squares fit of ``log(scale)`` onto ``log|err|`` over the scored elements.

    ``teacher``, one array per fitting cache in the same order (a fitted head's
    :func:`predict_scale` output, physical units), turns the fit into a DISTILLATION: the
    target becomes the teacher's own log scale instead of ``log|err|``. Same loss, same
    element set. The point is a smaller head fitted to a smooth dense target rather than to
    the raw error magnitude, whose log has a kurtosis near 25 (2026-09-26). ``pinball``
    fits have no scale target and refuse it.
    """
    seed_everything(cfg.seed)
    if teacher is not None and cfg.loss == "pinball":
        raise ValueError("a pinball fit has no scale target to distil")
    ds = ScaleDataset(fit, device, teacher=teacher)
    head = build_head(cfg, fit[0].cond.shape[-1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    g = torch.Generator().manual_seed(cfg.seed)

    head.train()
    for step in range(cfg.steps):
        idx = torch.randint(0, len(ds), (cfg.batch_size,), generator=g)
        b = ds.batch(idx)
        pred, x, target = _nondim(b)
        valid = (x[..., :N_MEASURED].abs().sum(dim=(1, -1)) > 0).to(pred.dtype)
        feats = bound_features(
            pred,
            x,
            valid,
            cond=b["cond"] if cfg.cond else None,
            noise=cfg.noise,
            logs=cfg.logs,
            coords=cfg.coords,
            grad=cfg.grad,
            pool=cfg.pool,
            fluc=cfg.fluc,
            lstd=cfg.lstd,
        )
        scale, tau = head(feats)

        err = (target[..., :N_MEASURED] - pred[..., :N_MEASURED]).detach()
        # The scorer's own element set. Fitting on the PIV hole would regress the scale
        # onto a target the metric ignores, and the hole is 5-11% of the field.
        scored = b["target"][..., :N_MEASURED] != 0.0
        if cfg.loss == "pinball":
            # A ``quantile`` head returns ``(below, above)``: the lower end is ``pred - below``
            # and the upper ``pred + above``, so the residuals of the two ends against the
            # target are ``err + below`` and ``err - above``. Signed error, no log.
            below, above = scale, tau
            per = pinball(err + below, cfg.tau_lo) + pinball(err - above, cfg.tau_hi)
            loss = (per * scored).sum() / scored.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            if log is not None and (step % log_every == 0 or step == cfg.steps - 1):
                log.append({"step": step, "loss": float(loss.detach())})
            continue
        if "tscale" in b:
            # the teacher's scale is physical (predict_scale multiplies by u_fs); back to the
            # non-dimensional units the head works in
            s_fs = b["u_fs"].reshape(-1, 1, 1, 1, 1).clamp_min(1e-8)
            y = torch.log((b["tscale"] / s_fs).clamp_min(1e-6))
        else:
            y = torch.log(err.abs().clamp_min(1e-6))
        d = torch.log(scale.clamp_min(1e-9)) - y
        if cfg.loss == "nll":
            # Gaussian negative log likelihood of `log|err|`, constant dropped. The `log(tau)`
            # term is what stops the spread running away to cover everything: widening tau
            # cheapens the first term and is paid for here.
            per = 0.5 * (d / tau.clamp_min(1e-6)).pow(2) + torch.log(tau.clamp_min(1e-6))
        else:
            per = d.pow(2) if cfg.loss == "l2" else d.abs()
        loss = (per * scored).sum() / scored.sum().clamp_min(1.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if log is not None and (step % log_every == 0 or step == cfg.steps - 1):
            log.append({"step": step, "loss": float(loss.detach())})
    return head.eval()


def finetune_credit(
    head: BoundHead,
    cfg: ScaleFitConfig,
    fit: list[Cached],
    k: float,
    steps: int = 400,
    lr: float = 3e-4,
    beta: float = 0.15,
    device: str = "cuda",
    log: list | None = None,
) -> BoundHead:
    """Fine-tune a regression-trained scale head on the scorer's own credit.

    Why this can add anything at all. The regression stage fits ``log|err|`` by least squares,
    which is the right objective for a *conditional mean* and the wrong one for SPS: the metric
    does not care equally about every element. An element whose error is far outside any
    sensible interval is unreachable and its squared log-error is wasted capacity; an element
    near the coverage boundary is where a small change in the predicted scale flips real credit.
    Least squares cannot express that; the credit can.

    Why this is not the objective that produced the 0.60 constant. ``losses.sps_loss`` softens
    the coverage indicator with a **fixed absolute** temperature (``tau=0.02`` non-dimensional).
    Widths here span three decades, so that temperature is enormous relative to a narrow
    interval and negligible relative to a wide one, and the relaxation therefore over-rewards
    widening -- which is what a fitted multiplier of 0.60 was correcting. Here the temperature
    is **relative**: ``sigmoid((w - |e|) / (beta * w))``. At ``|e| = w`` it is 0.5 whatever the
    magnitude of ``w``, and the transition occupies a fixed *fraction* of the width, so the
    relaxation is scale-invariant and carries no width-dependent bias.

    Two further differences from that failed attempt: it starts from a good solution, so the
    vanishing gradient far from the boundary never bites, and the sizing multiplier ``k`` is
    held fixed (detached, refit between rounds), so the head is asked only to improve the
    *ranking* while the sizing stays where the credit maximisation already put it.
    """
    ds = ScaleDataset(fit, device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(cfg.seed + 1)
    sigma = float(SIGMA_GLOBAL)
    head.train()
    for step in range(steps):
        idx = torch.randint(0, len(ds), (cfg.batch_size,), generator=g)
        b = ds.batch(idx)
        pred, x, target = _nondim(b)
        valid = (x[..., :N_MEASURED].abs().sum(dim=(1, -1)) > 0).to(pred.dtype)
        scale, _ = head(
            bound_features(
                pred,
                x,
                valid,
                cond=b["cond"] if cfg.cond else None,
                noise=cfg.noise,
                logs=cfg.logs,
                coords=cfg.coords,
                grad=cfg.grad,
                pool=cfg.pool,
                fluc=cfg.fluc,
                lstd=cfg.lstd,
            )
        )
        err = (target[..., :N_MEASURED] - pred[..., :N_MEASURED]).detach().abs()
        u = b["u_fs"].reshape(-1, 1, 1, 1, 1)
        w = k * scale  # non-dimensional half-width
        w_phys = (w * u).clamp_min(1e-9)  # sigma is physical
        soft_inside = torch.sigmoid((w - err) / (beta * w).clamp_min(1e-9))
        credit = soft_inside * torch.exp(-2.0 * w_phys / sigma)

        scored = b["target"][..., :N_MEASURED] != 0.0
        loss = -(credit * scored).sum() / scored.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if log is not None and (step % 100 == 0 or step == steps - 1):
            log.append({"ft_step": step, "neg_credit": float(loss.detach())})
    return head.eval()


@torch.no_grad()
def predict_scale(
    head: BoundHead, c: Cached, cfg: ScaleFitConfig, device: str = "cuda", batch: int = 16
) -> np.ndarray:
    """The head's physical error scale on a cached split, ``(N,T,H,W,N_MEASURED)``."""
    ds = ScaleDataset([c], device)
    out = np.empty(c.measured_shape, dtype=np.float32)
    for s in range(0, len(ds), batch):
        idx = torch.arange(s, min(s + batch, len(ds)))
        b = ds.batch(idx)
        pred, x, _ = _nondim(b)
        valid = (x[..., :N_MEASURED].abs().sum(dim=(1, -1)) > 0).to(pred.dtype)
        scale, _ = head(
            bound_features(
                pred,
                x,
                valid,
                cond=b["cond"] if cfg.cond else None,
                noise=cfg.noise,
                logs=cfg.logs,
                coords=cfg.coords,
                grad=cfg.grad,
                pool=cfg.pool,
                fluc=cfg.fluc,
                lstd=cfg.lstd,
            )
        )
        # Back to physical units, the units of SIGMA_GLOBAL and of the cached errors.
        out[s : s + len(idx)] = (scale * b["u_fs"].reshape(-1, 1, 1, 1, 1)).float().cpu().numpy()
    return out


@torch.no_grad()
def predict_offsets(
    head: BoundHead, c: Cached, cfg: ScaleFitConfig, device: str = "cuda", batch: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """The head's band as ``(below, above)`` offsets from the prediction, physical units.

    ``lower = pred - below`` and ``upper = pred + above``, each ``(N,T,H,W,N_MEASURED)``. For a
    one-sided head both arrays are the scale of :func:`predict_scale`; for a ``quantile`` head
    ``below`` may be negative where the whole band sits above the prediction.
    """
    ds = ScaleDataset([c], device)
    lo = np.empty(c.measured_shape, dtype=np.float32)
    up = np.empty(c.measured_shape, dtype=np.float32)
    for s in range(0, len(ds), batch):
        idx = torch.arange(s, min(s + batch, len(ds)))
        b = ds.batch(idx)
        pred, x, _ = _nondim(b)
        valid = (x[..., :N_MEASURED].abs().sum(dim=(1, -1)) > 0).to(pred.dtype)
        below, above = head(
            bound_features(
                pred,
                x,
                valid,
                cond=b["cond"] if cfg.cond else None,
                noise=cfg.noise,
                logs=cfg.logs,
                coords=cfg.coords,
                grad=cfg.grad,
                pool=cfg.pool,
                fluc=cfg.fluc,
                lstd=cfg.lstd,
            )
        )
        k = b["u_fs"].reshape(-1, 1, 1, 1, 1)
        lo[s : s + len(idx)] = (below * k).float().cpu().numpy()
        up[s : s + len(idx)] = (above * k).float().cpu().numpy()
    return lo, up


@torch.no_grad()
def predict_tau(
    head: BoundHead, c: Cached, cfg: ScaleFitConfig, device: str = "cuda", batch: int = 16
) -> np.ndarray:
    """The head's predicted spread of ``log|err|`` on a cached split, ``(N,T,H,W,N_MEASURED)``.

    **Not** multiplied by ``u_fs``, unlike :func:`predict_scale`. ``tau`` is a spread in nats of
    a log-ratio, so it is already dimensionless; scaling it would make it a function of the
    freestream speed, which is exactly the extrapolation hazard ``width_rule`` refuses a
    ``u_fs`` binning axis for.
    """
    if head.mode != "logerr_tau":
        raise ValueError(f"predict_tau needs a logerr_tau head, got mode {head.mode!r}")
    ds = ScaleDataset([c], device)
    out = np.empty(c.measured_shape, dtype=np.float32)
    for s in range(0, len(ds), batch):
        idx = torch.arange(s, min(s + batch, len(ds)))
        b = ds.batch(idx)
        pred, x, _ = _nondim(b)
        valid = (x[..., :N_MEASURED].abs().sum(dim=(1, -1)) > 0).to(pred.dtype)
        _, tau = head(
            bound_features(
                pred,
                x,
                valid,
                cond=b["cond"] if cfg.cond else None,
                noise=cfg.noise,
                logs=cfg.logs,
                coords=cfg.coords,
                grad=cfg.grad,
                pool=cfg.pool,
                fluc=cfg.fluc,
                lstd=cfg.lstd,
            )
        )
        out[s : s + len(idx)] = tau.float().cpu().numpy()
    return out


def size_and_score(
    scales: dict[str, np.ndarray],
    fit: list[Cached],
    targets: dict[str, Cached],
    cfg: ScaleFitConfig,
) -> tuple[WidthRule, dict[str, dict]]:
    """Fit the width rule on the fitting splits' scales, then score every target split.

    The candidate scale is mapped onto the fitting set's median ``|err|`` before the rule is
    fitted, and the same constant is applied when scoring. Without it an arm whose output
    level happens to sit far from the error scale pins its multiplier at the edge of the
    search grid and reports the *grid bound* as its optimum -- measured on a width-64 head,
    which read 43.62 instead of a real number for exactly this reason.
    """
    err_fit = np.concatenate([c.err for c in fit])
    half_fit = np.concatenate([scales[c.split] for c in fit])
    scored_fit = np.concatenate([c.scored for c in fit])

    rule = fit_prescaled(
        err_fit,
        half_fit,
        scored_fit,
        n_rho=cfg.n_rho,
        n_lead=cfg.n_lead,
        asymmetric=cfg.asymmetric,
    )
    out = {}
    for name, c in targets.items():
        s = scales[name]
        lo, hi = c.pred.copy(), c.pred.copy()
        lo[..., :N_MEASURED] = c.pred[..., :N_MEASURED] - s
        hi[..., :N_MEASURED] = c.pred[..., :N_MEASURED] + s
        lo, hi = rule.apply(c.pred, lo, hi)
        out[name] = sps_breakdown(c.pred, lo, hi, c.target, q=c.q()).as_dict()
    return rule, out


def save_arm(
    path: Path,
    head: nn.Module,
    cfg: ScaleFitConfig,
    rule: WidthRule,
    scores: dict,
    scales: dict[str, np.ndarray] | None = None,
) -> Path:
    """Persist an arm, including its predicted scale fields if given.

    Caching the scales matters more than it looks: with them, every *width rule* variant --
    bin counts, asymmetry, lead binning -- is a pure numpy experiment on a fixed head, so the
    sizing half of the problem can be explored without a GPU and without retraining anything.
    Without them, changing a bin count means re-running a forward pass.
    """
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"head": head.state_dict(), "config": cfg.__dict__, "scores": scores},
        path / "head.pt",
    )
    if scales:
        np.savez(path / "scales.npz", **{k: v.astype(np.float32) for k, v in scales.items()})
    rule.to_json(path / "width_rule.json")
    (path / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return path


def merge_scale_head(backbone_ckpt: Path, arm_dir: Path) -> dict:
    """One checkpoint from a frozen backbone and an offline-fitted scale head.

    The head is fitted on *cached* backbone outputs, so it lives in its own file and the
    two only ever meet at inference. Everything downstream -- the packager, the official
    scorer, ``load_for_inference`` -- takes a single checkpoint, so the join has to happen
    somewhere; doing it here means the mapping from a :class:`ScaleFitConfig` to the
    ``bound_*`` config keys ``models.factory.wrap_model`` reads exists once. Two
    independent spellings of a head's shape is the failure PROTOCOL 5 records, and
    ``head_in_channels`` exists for the same reason.

    The backbone checkpoint is normally itself a ``BoundedModel`` carrying the older
    width-trained head. That head is *dropped*, not nested: the scale head replaces it.
    """
    payload = torch.load(Path(backbone_ckpt), map_location="cpu", weights_only=False)
    arm = torch.load(Path(arm_dir) / "head.pt", map_location="cpu", weights_only=False)
    cfg = arm["config"]

    # Decide the nesting from the CONFIG, not from the key names. ``wrap_model`` applies the
    # residual anchor first and the bound head last, and both wrappers name their child
    # ``backbone`` -- so ``any(k.startswith("backbone."))`` cannot tell "already has a bound head"
    # from "has a residual anchor and no bound head". A ``--residual`` checkpoint trained without
    # ``--bounds`` looks bound-wrapped, gets no prefix added, and then fails to load into the
    # ``BoundedModel(ResidualWrapper(backbone))`` that `build_for_inference` builds: expected
    # ``backbone.backbone.pos``, got ``backbone.pos``. That blocked every residual artefact from
    # being packaged at all, and it surfaced only in `sps_assemble.py`'s rebuild check.
    #
    # The config answers it exactly: the merged checkpoint always gains a bound head, so exactly
    # one ``backbone.`` prefix is needed iff the source did not already carry one.
    #
    # ``head.`` is still only stripped when a bound head is really present: in a bare backbone
    # ``head`` may be one of the architecture's own submodules -- STFormer has a ``head`` output
    # projection -- and stripping it there deletes weights the model needs.
    if payload.get("config", {}).get("bounds", False):
        state = {k: v for k, v in payload["model"].items() if not k.startswith("head.")}
    else:
        state = {f"backbone.{k}": v for k, v in payload["model"].items()}
    state.update({f"head.{k}": v for k, v in arm["head"].items()})

    config = dict(payload.get("config", {}))
    config.update(
        bounds=True,
        freeze_backbone=True,
        bound_mode=head_mode(cfg.get("loss", "l2")),
        bound_cond=cfg["cond"],
        bound_noise=cfg["noise"],
        bound_logs=cfg["logs"],
        bound_coords=cfg["coords"],
        # ``.get`` and not ``[...]``: every other key here predates the arm dirs on disk, but
        # ``grad`` does not, so a head fitted before it existed must still merge.
        bound_grad=cfg.get("grad", False),
        bound_pool=cfg.get("pool", False),
        # Same `.get` reasoning as `grad`: these three postdate the arm dirs on disk, and a head
        # fitted before they existed must still merge. Carrying them is not optional -- a bundle
        # that omits `bound_fluc`/`bound_lstd` rebuilds the head at the WRONG INPUT WIDTH, and one
        # that omits `bound_film` rebuilds it without its modulation, which for `film=2` has the
        # right width and would load with only missing keys. PROTOCOL 5 records what silently
        # building the wrong head shape costs.
        bound_fluc=cfg.get("fluc", False),
        bound_lstd=cfg.get("lstd", 0),
        bound_film=cfg.get("film", 0),
        bound_width=cfg["width"],
        bound_depth=cfg["depth"],
        # `.get`, as for `grad`: arm dirs written before the lead-axis kernel size existed.
        bound_kt=cfg.get("kt", 3),
        # Provenance goes in ``extra`` because ``TrainConfig(**raw)`` takes the config dict
        # as keyword arguments: a top-level key it does not declare is a TypeError, and the
        # checkpoint then loads through ``models.factory`` (which takes a plain dict) but
        # not through ``train.evaluate.load_checkpoint``. Half-loadable is worse than either.
        extra={**config.get("extra", {}), "scale_head": dict(cfg)},
    )
    return {"model": state, "config": config, "scores": arm.get("scores", {})}


__all__ = [
    "ScaleDataset",
    "ScaleFitConfig",
    "build_head",
    "head_in_channels",
    "head_mode",
    "merge_scale_head",
    "pinball",
    "predict_offsets",
    "predict_scale",
    "predict_tau",
    "save_arm",
    "size_and_score",
    "train_scale_head",
]
