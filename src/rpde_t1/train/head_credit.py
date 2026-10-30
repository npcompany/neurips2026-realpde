"""Band credit through a FROZEN interval head, as a training signal for the backbone (FINDINGS 214).

FINDINGS 213 found that the shipped interval head already knows each pixel's local error level as
well as the true neighbouring errors do, and that what costs band credit is the spread of the
error inside that level. Only the prediction can change that spread. ``losses.interval_credit_loss``
(``--w-credit``) rewards the credit an *oracle* band sized at the realised error would earn, which
favours many near-exact pixels and a few hopeless ones -- a WIDER spread for a known level, the
opposite of what a real head can use. This term instead grades the prediction against the band a
real head would actually draw:

    h        = half-width from the frozen head + its width rule, physical units, DETACHED
    inside   = sigmoid((h - |e|) / (tau * h))            a smooth "truth is in the band"
    credit   = inside * exp(-2 h / sigma)                 the scorer's per-element credit
    loss     = - mean over scored elements of credit

The width is detached on purpose. The interval head is refitted after every backbone change
(``g_price.sh``), so a backbone that learned to steer a frozen head's width would be gaming a head
that is thrown away. With the width fixed the only gradient is on the centre, and it is largest
for errors just outside the band: the backbone is asked to bring those inside, i.e. to make its
error size consistent with the level the head predicts.

``headcredit_track_lr`` > 0 (FINDINGS 223) lets the head keep learning while the backbone trains,
instead of staying at the control's fit: after each step's width is taken, the head takes one
optimiser step of its OWN objective (the offline ``train_scale_head`` regression of ``log scale``
onto ``log|err|``, loss from the arm's config) on the current backbone's detached errors. The
width seen by the band credit therefore tracks the backbone it grades, rather than the backbone
the head was fitted on. The backbone still gets no gradient through the width, and the width rule
stays the arm's. The tracked head is a training device only; the shipped head is refitted.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .. import N_MEASURED, SIGMA_GLOBAL
from ..models.bounds import bound_features
from ..models.width_rule import WidthRule
from .scale_fit import ScaleFitConfig, build_head

_CACHE: dict[tuple[str, str, int], FrozenHead] = {}


class FrozenHead:
    """A fitted interval head (``sps_apply.py --arm-dir`` layout) evaluated on training tensors."""

    def __init__(self, arm_dir: str | Path, device: torch.device, cond_width: int):
        arm = Path(arm_dir)
        payload = torch.load(arm / "head.pt", map_location="cpu", weights_only=False)
        raw = dict(payload["config"])
        known = set(ScaleFitConfig.__dataclass_fields__)
        self.cfg = ScaleFitConfig(**{k: v for k, v in raw.items() if k in known})
        self.head = build_head(self.cfg, cond_width=cond_width)
        self.head.load_state_dict(payload["head"])
        self.head.to(device).eval().requires_grad_(False)
        rule = WidthRule.from_json(arm / "width_rule.json")
        self.rule = rule if rule.is_uniform else rule.to_uniform()
        self.opt: torch.optim.Optimizer | None = None
        self.track_steps = 0

    def _features(self, pred, x, valid, cond) -> torch.Tensor:
        c = self.cfg
        return bound_features(
            pred.float(),
            x.float(),
            valid.float(),
            cond=cond.float() if c.cond else None,
            noise=c.noise,
            logs=c.logs,
            coords=c.coords,
            grad=c.grad,
            pool=c.pool,
            fluc=c.fluc,
            lstd=c.lstd,
        )

    def track_step(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x: torch.Tensor,
        valid: torch.Tensor,
        cond: torch.Tensor,
        lr: float,
    ) -> float:
        """One step of the head's own fitting objective on these (detached) errors; its loss.

        Same objective as ``scale_fit.train_scale_head`` for the ``l1``/``l2`` losses the
        shipped heads use: ``log scale`` regressed onto ``log|err|`` over the scored elements,
        non-dimensional units, gradient clipped at 1, constant ``lr`` (AdamW, the arm's weight
        decay)."""
        if self.cfg.loss not in ("l1", "l2"):
            raise ValueError(f"headcredit tracking supports l1/l2 heads, not {self.cfg.loss!r}")
        if self.opt is None:
            self.head.requires_grad_(True)
            self.opt = torch.optim.AdamW(
                self.head.parameters(), lr=lr, weight_decay=self.cfg.weight_decay
            )
        pred, target = pred.detach().float(), target.detach().float()
        with torch.enable_grad(), torch.autocast(pred.device.type, enabled=False):
            self.head.train()
            scale, _ = self.head(self._features(pred, x.detach(), valid.detach(), cond.detach()))
            err = (target[..., :N_MEASURED] - pred[..., :N_MEASURED]).abs()
            d = torch.log(scale.clamp_min(1e-9)) - torch.log(err.clamp_min(1e-6))
            per = d.pow(2) if self.cfg.loss == "l2" else d.abs()
            scored = (target[..., :N_MEASURED] != 0.0).to(per.dtype)
            loss = (per * scored).sum() / scored.sum().clamp_min(1.0)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            self.opt.step()
            self.head.eval()
        self.track_steps += 1
        return float(loss.detach())

    @torch.no_grad()
    def half_width(
        self,
        pred: torch.Tensor,
        x: torch.Tensor,
        valid: torch.Tensor,
        cond: torch.Tensor,
        u_fs: torch.Tensor,
    ) -> torch.Tensor:
        """Physical half-width ``(B,T,H,W,N_MEASURED)`` for a non-dimensional ``pred``/``x``.

        Same spelling as ``scale_fit.predict_scale`` followed by ``WidthRule.apply_torch``, the
        path the packaged bundle serves; ``tests/test_head_credit.py`` pins the two together.
        """
        scale, _ = self.head(self._features(pred, x, valid, cond))
        s = u_fs.float().reshape(-1, 1, 1, 1, 1)
        p = pred.float() * s
        scale = scale * s
        lo, hi = p.clone(), p.clone()
        lo[..., :N_MEASURED] = p[..., :N_MEASURED] - scale
        hi[..., :N_MEASURED] = p[..., :N_MEASURED] + scale
        lo, hi = self.rule.apply_torch(p, lo, hi)
        return 0.5 * (hi - lo)[..., :N_MEASURED]


def frozen_head(arm_dir: str, device: torch.device, cond_width: int) -> FrozenHead:
    """One loaded head per (arm dir, device, conditioning width), reused across steps."""
    key = (str(arm_dir), str(device), int(cond_width))
    if key not in _CACHE:
        _CACHE[key] = FrozenHead(arm_dir, device, cond_width)
    return _CACHE[key]


def re_taper(re: torch.Tensor, lo: float, hi: float = 1.0) -> torch.Tensor:
    """``data.dataset.re_dose`` on a batch: linear in log Re, ``lo`` at 3750 to ``hi`` at 26700."""
    from ..data.dataset import REAL_RE_HI, REAL_RE_LO

    t = (torch.log(re) - math.log(REAL_RE_LO)) / (math.log(REAL_RE_HI) - math.log(REAL_RE_LO))
    return lo + (hi - lo) * t.clamp(0.0, 1.0)


def re_notch(
    re: torch.Tensor, lo: float, centre: float = 16500.0, width: float = 0.5
) -> torch.Tensor:
    """A dip in log Re: ``lo`` at ``centre``, rising linearly in ``|log(Re / centre)|`` to 1 at
    ``width`` (natural-log units) and beyond. ``lo = 1`` is flat."""
    d = (torch.log(re) - math.log(centre)).abs() / width
    return lo + (1.0 - lo) * d.clamp(0.0, 1.0)


def head_credit_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    half: torch.Tensor,
    u_fs: torch.Tensor,
    tau: float = 0.25,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative smoothed band credit at a fixed (detached) physical half-width ``half``.

    ``weight`` is an optional per-sample multiplier ``(B,)`` (the Reynolds taper); the mean is
    still over every scored element, so a weight below one weakens the term rather than
    re-normalising it away."""
    s = u_fs.reshape(-1, *([1] * (pred.ndim - 1))).clamp_min(1e-8)
    err = (pred[..., :N_MEASURED] - target[..., :N_MEASURED]).abs() * s
    h = half.detach().clamp_min(1e-8)
    inside = torch.sigmoid((h - err) / (tau * h))
    credit = inside * torch.exp(-2.0 * h / SIGMA_GLOBAL)
    scored = (target[..., :N_MEASURED] != 0.0).to(credit.dtype)
    num = credit * scored
    if weight is not None:
        num = num * weight.reshape(-1, *([1] * (num.ndim - 1))).to(num.dtype)
    return -num.sum() / scored.sum().clamp_min(1.0)
