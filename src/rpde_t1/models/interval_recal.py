"""A LEARNED recurrent interval calibrator for the Track-2 stream (the learned twin of 192's).

``models.interval_autocal`` picks width multipliers by the SPS credit the revealed windows would
have earned, over a 29-point grid per pixel -- worth ~+0.055 sps on ``hard5`` and parked for its
~1.4 ms/step credit tables. This learns the same map from a handful of per-pixel statistics of the
revealed error: per (pixel, channel), a GRU cell shared across pixels folds, window by window,

* per lead bucket (4 x 5 leads): mean log ``|e| / h``, the coverage ``|e| <= g h`` at each
  ``cov_grid`` multiple ``g`` of the shipped half-width ``h``, and (``h_feature``) mean
  ``log(h / sigma)`` -- the credit's width penalty makes the best multiplier depend on it;
* optionally (``context``) the same features 3x3-pooled and pooled over the whole field;

and its state emits a log multiplier per lead bucket, ``tanh``-bounded to [1/2, 2], applied to
the shipped half-width around the shipped centre. Window k is served from windows < k.

Two front ends share :class:`RecurrentCalibrator`: ``forward`` runs whole cached trajectories at
once (``scripts/t2/t2_recal_lab.py`` trains it that way), and :class:`IntervalRecal` is the
batch-1 serving step, with every piece of state preallocated so ``step`` has no host branch and
captures as one CUDA graph -- the same contract as ``IntervalAutocal``, and the same place in
``make_gpu_predict``. ``tests/test_interval_recal.py`` pins that the two agree.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .. import H, W

N_BUCKETS = 4
T_OUT = 20
LOG2 = math.log(2.0)


def lead_bucket(device=None) -> torch.Tensor:
    return torch.arange(T_OUT, device=device) * N_BUCKETS // T_OUT


class RecurrentCalibrator(nn.Module):
    """Per-(pixel, channel) recurrent map: revealed-error statistics -> log width multipliers."""

    def __init__(
        self,
        hidden: int = 16,
        c: int = 2,
        arch: str = "gru",
        context: bool = True,
        cov_grid: tuple[float, ...] = (1.0,),
        sigma: float | None = None,
        observe_every: int = 1,
        signed: bool = False,
        shift_head: bool = False,
    ) -> None:
        super().__init__()
        if arch not in ("gru", "mean"):
            raise ValueError(f"arch must be 'gru' or 'mean', got {arch!r}")
        # ``signed``: also feed the per-bucket mean SIGNED error over the half-width (the centre
        # shift's signal, 192's beta shift); ``shift_head``: a second output per bucket, a shift
        # of BOTH bounds in units of the half-width (tanh-bounded to +-1/2), zero-initialised.
        self.signed, self.shift_head = bool(signed), bool(shift_head)
        # Fold in only every ``observe_every``-th revealed window (window k folded when
        # ``k % observe_every == 0``): the error's size persists within a trajectory (192b), so
        # skipping observations halves the serving cost for a small share of the gain.
        self.observe_every = int(observe_every)
        self.cov_grid = tuple(float(g) for g in cov_grid)
        # ``sigma`` is required: the width feature ``log(h / sigma)`` is part of every served
        # calibrator, and the feature layout below is spelled without any test on it -- the
        # clean-bundle converter folded ``self.sigma is not None`` to False on the 2026-09-25
        # night packaging (state_dict 87 -> 75 inputs), so nothing here branches on the value.
        if sigma is None:
            raise ValueError("RecurrentCalibrator needs sigma (the scorer's SIGMA_GLOBAL)")
        self.sigma = float(sigma)
        n_parts = 2 + len(self.cov_grid) + (1 if signed else 0)  # lr, covs, lh[, signed]
        self.n_feat = n_parts * N_BUCKETS
        n_in = 3 * self.n_feat + c + 1  # own, 3x3-pooled, field-pooled; channel one-hot; scored
        # ``mean`` is the no-recurrence ablation: a running mean of the same inputs (plus the
        # evidence weight n/(n+5)) through one tanh layer -- a learned closed-form statistic.
        self.cell = nn.GRUCell(n_in, hidden) if arch == "gru" else nn.Linear(n_in + 1, hidden)
        self.out = nn.Linear(hidden, N_BUCKETS)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # step 0: the shipped width everywhere
        if self.shift_head:
            self.out_shift = nn.Linear(hidden, N_BUCKETS)
            nn.init.zeros_(self.out_shift.weight)
            nn.init.zeros_(self.out_shift.bias)
        self.hidden, self.c, self.arch, self.context = hidden, c, arch, context
        self.register_buffer("bucket", lead_bucket(), persistent=False)
        # Device-resident constants: a host->device copy inside the step would break the capture.
        self.register_buffer("grid", torch.tensor(self.cov_grid), persistent=False)

    def config(self) -> dict:
        return {
            "hidden": self.hidden,
            "c": self.c,
            "arch": self.arch,
            "context": self.context,
            "cov_grid": list(self.cov_grid),
            "sigma": self.sigma,
            "observe_every": self.observe_every,
            "signed": self.signed,
            "shift_head": self.shift_head,
        }

    # -- one revealed window ----------------------------------------------------------------
    def features(self, ae, h, scored, e=None) -> torch.Tensor:
        """``(B, 20, H, W, C)`` revealed window -> ``(B, H, W, C, n_feat + 1)``.

        Written as a few large batched ops rather than one small op per statistic: at batch 1
        the serving step is kernel-count-bound, not arithmetic-bound. ``e`` (signed error) is
        read only by a ``signed`` calibrator.
        """
        lr = torch.log((ae + 1e-3 * h + 1e-8) / (h + 1e-8)).clamp(-6.0, 3.0)
        g = self.grid.to(ae.dtype).view(-1, *([1] * ae.dim()))
        parts = [lr[None], (ae[None] <= g * h[None]).to(ae.dtype)]
        parts.append(torch.log(h / self.sigma + 1e-8).clamp(-12.0, 3.0)[None])
        if self.signed:
            if e is None:
                raise ValueError("a signed calibrator needs the signed error")
            parts.append((e / (h + 1e-8)).clamp(-3.0, 3.0)[None])
        v = torch.cat(parts, dim=0) * scored[None]  # (P, B, 20, H, W, C)
        # Buckets are 4 contiguous runs of 5 leads, so a bucket sum is a reshape and a reduction
        # -- no gather, no one-hot matmul, no data-dependent shape (the step captures as a graph).
        # Columns are bucket-major -- for each bucket j: lr, each cov, [lh] -- the order the
        # lab's first version appended them in.
        p_, b_, t_, hh, ww, cc = v.shape
        per = T_OUT // N_BUCKETS
        n = scored.view(b_, N_BUCKETS, per, hh, ww, cc).sum(2).clamp_min(1.0)  # (B, 4, H, W, C)
        f = v.view(p_, b_, N_BUCKETS, per, hh, ww, cc).sum(3) / n[None]  # (P, B, 4, H, W, C)
        f = f.permute(1, 3, 4, 5, 2, 0).reshape(b_, hh, ww, cc, -1)
        frac = scored.mean(1)[..., None]
        return torch.cat([f, frac], dim=-1)

    def inputs(self, f: torch.Tensor) -> torch.Tensor:
        b, hh, ww, c, nf = f.shape
        own, frac = f[..., :-1], f[..., -1:]
        onehot = torch.eye(c, device=f.device).view(1, 1, 1, c, c).expand(b, hh, ww, c, c)
        if not self.context:  # the ablation: each pixel reads only its own history
            z = torch.zeros_like(own)
            return torch.cat([own, z, z, onehot, frac], dim=-1)
        x = own.permute(0, 3, 4, 1, 2).reshape(b, c * (nf - 1), hh, ww)
        pooled = nn.functional.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        pooled = pooled.reshape(b, c, nf - 1, hh, ww).permute(0, 3, 4, 1, 2)
        wsum = frac.sum((1, 2), keepdim=True).clamp_min(1e-6)
        field = (own * frac).sum((1, 2), keepdim=True) / wsum
        field = field.expand_as(own)
        return torch.cat([own, pooled, field, onehot, frac], dim=-1)

    def advance(self, state, xsum, x, r: torch.Tensor):
        """One recurrent step. ``r`` is the number of windows folded in AFTER this one (>= 1)."""
        if self.arch == "gru":
            return self.cell(x, state), xsum
        xsum = xsum + x
        ev = (r / (r + 5.0)).expand(x.shape[0], 1)
        return torch.tanh(self.cell(torch.cat([xsum / r, ev], dim=-1))), xsum

    def log_mult(self, state: torch.Tensor, lead_shape) -> torch.Tensor:
        """``(N, hidden)`` state -> ``(N, 20)`` log multipliers (bucket -> lead)."""
        return (torch.tanh(self.out(state)) * LOG2)[:, self.bucket]

    # -- whole cached trajectories (the lab) -------------------------------------------------
    def forward(self, ae, h, scored, e=None):
        """``(n_traj, n_win, 20, H, W, C)`` -> causal multipliers of the same shape (and, for a
        ``shift_head`` calibrator, the centre shift in units of ``h`` as a second tensor)."""
        n, nw, t, hh, ww, c = ae.shape
        state = ae.new_zeros(n * hh * ww * c, self.hidden)
        xsum = ae.new_zeros(n * hh * ww * c, 3 * self.n_feat + c + 1)
        mults, shifts = [], []
        folded = 0
        for k in range(nw):
            logm = self.log_mult(state, None).view(n, hh, ww, c, T_OUT)
            mults.append(torch.exp(logm.permute(0, 4, 1, 2, 3)))
            if self.shift_head:
                sh = (0.5 * torch.tanh(self.out_shift(state)))[:, self.bucket]
                shifts.append(sh.view(n, hh, ww, c, T_OUT).permute(0, 4, 1, 2, 3))
            if k < nw - 1 and k % self.observe_every == 0:
                folded += 1
                ek = None if e is None else e[:, k]
                x = self.inputs(self.features(ae[:, k], h[:, k], scored[:, k], ek))
                x = x.reshape(-1, x.shape[-1])
                state, xsum = self.advance(state, xsum, x, ae.new_tensor(float(folded)))
        if self.shift_head:
            return torch.stack(mults, dim=1), torch.stack(shifts, dim=1)
        return torch.stack(mults, dim=1)


def save_calibrator(model: RecurrentCalibrator, path) -> None:
    torch.save({"config": model.config(), "state_dict": model.state_dict()}, path)


def load_calibrator(path, device="cpu") -> RecurrentCalibrator:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(blob["config"])
    cfg.setdefault("observe_every", 1)
    cfg.setdefault("signed", False)
    cfg.setdefault("shift_head", False)
    model = RecurrentCalibrator(**{**cfg, "cov_grid": tuple(cfg["cov_grid"])})
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


class IntervalRecalReference:
    """Batch-1 serving of a :class:`RecurrentCalibrator`, in the evaluator's target space.

    The straightforward spelling (any arch, any context), kept as the reference the fast
    :class:`IntervalRecal` is tested against and as its fallback.

    Same wire conventions as ``IntervalAutocal``: ``out_scale``/``out_shift`` map physical to
    target space, ``in_mean``/``in_std`` recover the revealed PHYSICAL target from the next input
    window, only the measured channels ``[:c]`` are re-sized, pressure bounds pass through.
    """

    def __init__(
        self,
        model: RecurrentCalibrator,
        device,
        out_scale: torch.Tensor,
        out_shift: torch.Tensor,
        in_mean: torch.Tensor,
        in_std: torch.Tensor,
        c: int = 2,
        h: int = H,
        w: int = W,
    ) -> None:
        self.dev = torch.device(device)
        self.model = model.to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.c = int(c)
        f32 = dict(device=self.dev, dtype=torch.float32)
        self.a = out_scale.reshape(-1)[: self.c].to(**f32)
        self.b = out_shift.reshape(-1)[: self.c].to(**f32)
        self.mi = in_mean.reshape(-1)[: self.c].to(**f32)
        self.si = in_std.reshape(-1)[: self.c].to(**f32)
        n = h * w * self.c
        self.hw = (h, w)
        self._state = torch.zeros((n, model.hidden), **f32)
        self._xsum = torch.zeros((n, 3 * model.n_feat + self.c + 1), **f32)
        self._c_prev = torch.zeros((T_OUT, h, w, self.c), **f32)  # physical centre
        self._h_prev = torch.zeros((T_OUT, h, w, self.c), **f32)  # physical base half-width
        self._has_prev = torch.zeros((), **f32)  # 1 once a window has been served
        self._r = torch.zeros((), **f32)  # revealed windows folded in

    def reset(self) -> None:
        for t in (self._state, self._xsum, self._c_prev, self._h_prev, self._has_prev, self._r):
            t.zero_()

    @torch.no_grad()
    def observe(self, x_eval: torch.Tensor) -> None:
        """Fold in the previous window's revealed target (= this input window, stride 20)."""
        t = x_eval[0, ..., : self.c].float() * self.si + self.mi
        scored = ((t.abs() > 1e-6) * self._has_prev).float()
        ae = (t - self._c_prev).abs()
        x = self.model.inputs(self.model.features(ae[None], self._h_prev[None], scored[None]))
        x = x.reshape(-1, x.shape[-1])
        r = self._r + self._has_prev
        new_state, new_xsum = self.model.advance(self._state, self._xsum, x, r.clamp_min(1.0))
        keep = self._has_prev  # before the first served window nothing was revealed
        self._state.copy_(keep * new_state + (1 - keep) * self._state)
        self._xsum.copy_(keep * new_xsum + (1 - keep) * self._xsum)
        self._r.copy_(r)

    @torch.no_grad()
    def apply(self, lo: torch.Tensor, hi: torch.Tensor):
        """Re-size target-space bounds ``(1, 20, H, W, C)``; remembers the base for ``observe``."""
        c = self.c
        lo_m, hi_m = lo[..., :c].float(), hi[..., :c].float()
        cen, half = 0.5 * (lo_m + hi_m), 0.5 * (hi_m - lo_m)
        self._c_prev.copy_(((cen - self.b) / self.a)[0])
        self._h_prev.copy_((half / self.a)[0])
        self._has_prev.fill_(1.0)
        hh, ww = self.hw
        logm = self.model.log_mult(self._state, None).view(hh, ww, c, T_OUT)
        mult = torch.exp(logm.permute(3, 0, 1, 2))[None]  # (1, 20, H, W, c)
        new_half = half * mult
        lo2 = torch.cat([(cen - new_half).to(lo.dtype), lo[..., c:]], dim=-1)
        hi2 = torch.cat([(cen + new_half).to(hi.dtype), hi[..., c:]], dim=-1)
        return lo2, hi2

    def step(self, x_eval: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
        """``observe`` then ``apply``: one serving step, graph-capturable (no host sync)."""
        self.observe(x_eval)
        return self.apply(lo, hi)

    # The predictor's serving hook switches graphs by a host count of calls since reset: the
    # first call of a trajectory has nothing revealed, so it only applies (identical here, where
    # observe is a no-op before the first served window); there is no frozen form.
    step_frozen = step

    def step_first(self, x_eval: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
        return self.apply(lo, hi)


def _gru_from_gates(gi: torch.Tensor, gh: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """``nn.GRUCell``'s update from precomputed gates (biases included); one kernel on CUDA."""
    if gi.is_cuda:
        return torch.ops.aten._thnn_fused_gru_cell(gi, gh, h)[0]
    ir, iz, i_n = gi.chunk(3, dim=1)
    hr, hz, h_n = gh.chunk(3, dim=1)
    r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
    return (1 - z) * torch.tanh(i_n + r * h_n) + z * h


class IntervalRecal:
    """The serving step in about 45 kernels instead of ~100, for the shipped variant.

    At batch 1 the step is kernel-count-bound (FINDINGS 208: 168 kernels, 270 us of real work on
    the T4), so every statistic is written in place into preallocated buffers, the constant
    parts of the GRU input (the channel one-hot, and the zeroed context slots of a no-context
    calibrator) are folded into a per-row bias, the GRU runs as two ``addmm`` and one fused cell
    kernel, and the first call of a trajectory (nothing revealed) is a separate ``step_first``
    chosen by the caller's host count -- so no step carries a "has anything been revealed"
    blend. The error ratio is formed once, ``q = |e| / (h + 1e-8)``, and every feature is read
    off it; that differs from the reference's spelling only below 1e-6 (pinned by
    ``tests/test_interval_recal.py``).

    Only the GRU / no-context calibrator takes this path; anything else is served by
    :class:`IntervalRecalReference` (see :func:`make_interval_recal`).
    """

    def __init__(
        self,
        model: RecurrentCalibrator,
        device,
        out_scale: torch.Tensor,
        out_shift: torch.Tensor,
        in_mean: torch.Tensor,
        in_std: torch.Tensor,
        c: int = 2,
        h: int = H,
        w: int = W,
        shift_beta: float = 0.0,
        shift_n0: float = 5.0,
    ) -> None:
        if model.arch != "gru" or model.context:
            raise ValueError("the fast step serves the GRU / no-context calibrator only")
        if shift_beta and type(self) is IntervalRecal:
            raise ValueError("the closed-form centre shift is served by the fused merged step only")
        # FINDINGS 192's closed-form centre shift, on top of the learned multipliers (2026-09-25
        # night lab: +0.005..+0.013 sps at beta 0.25, n0 5): both bounds move by
        # ``beta * S / (r + n0)`` where ``S`` sums the revealed windows' per-pixel mean signed
        # error (leads pooled, against the shipped centre) and ``r`` counts them.
        self.shift_beta, self.shift_n0 = float(shift_beta), float(shift_n0)
        self.dev = torch.device(device)
        model = model.to(self.dev).eval()
        self.model, self.c, self.hw = model, int(c), (h, w)
        f32 = dict(device=self.dev, dtype=torch.float32)
        a = out_scale.reshape(-1)[:c].to(**f32)
        b = out_shift.reshape(-1)[:c].to(**f32)
        self.a, self.b = a, b
        self.mi = in_mean.reshape(-1)[:c].to(**f32)
        self.si = in_std.reshape(-1)[:c].to(**f32)
        n_rows, hid = h * w * c, model.hidden
        g = torch.tensor(model.cov_grid, **f32)
        self.n_cov = len(model.cov_grid)
        self.has_lh = True  # the width feature is always present (sigma is required)
        # Planes of ``_v``: lr, covs, lh[, signed error / h] -- the lab's feature order.
        self.signed, self.shift_head = bool(model.signed), bool(model.shift_head)
        self.i_lh = 1 + self.n_cov
        self.n_parts = 2 + self.n_cov + int(self.signed)
        self.inv_sigma = 1.0 / model.sigma
        self.grid = g.view(-1, 1, 1, 1, 1)
        # The GRU input is [own (4 * n_parts, bucket-major), context (zeros), onehot (c), frac].
        wih = model.cell.weight_ih.detach().to(**f32)  # (3H, n_in)
        nf = 4 * self.n_parts
        n_in = wih.shape[1]
        own_cols = list(range(nf))
        frac_col = n_in - 1
        onehot_cols = list(range(n_in - 1 - c, n_in - 1))
        self.w_small_t = torch.cat([wih[:, own_cols], wih[:, [frac_col]]], dim=1).t().contiguous()
        chan = torch.arange(n_rows, device=self.dev) % c  # rows are (h, w, c), c fastest
        bias = model.cell.bias_ih.detach().to(**f32)[None] + wih[:, onehot_cols].t()[chan]
        self.bias_rows = bias.contiguous()  # (rows, 3H)
        self.w_hh_t = model.cell.weight_hh.detach().to(**f32).t().contiguous()
        self.b_hh = model.cell.bias_hh.detach().to(**f32)
        # The width logits and, for a ``shift_head`` calibrator, the shift logits, as ONE GEMM:
        # columns [:4] are the log-multiplier logits, [4:] the shift logits.
        w_out = model.out.weight.detach().to(**f32)
        b_out = model.out.bias.detach().to(**f32)
        if self.shift_head:
            w_out = torch.cat([w_out, model.out_shift.weight.detach().to(**f32)], dim=0)
            b_out = torch.cat([b_out, model.out_shift.bias.detach().to(**f32)], dim=0)
        self.w_out_t = w_out.t().contiguous()  # (H, 4 or 8)
        self.b_out = b_out
        # State and scratch, all preallocated.
        self._state = torch.zeros((n_rows, hid), **f32)
        self._c_prev = torch.zeros((T_OUT, h, w, c), **f32)  # physical centre
        self._h_eps = torch.ones((T_OUT, h, w, c), **f32)  # physical half-width + 1e-8
        self._v = torch.zeros((self.n_parts, T_OUT, h, w, c), **f32)  # lr, covs, lh planes
        self._x = torch.zeros((n_rows, nf + 1), **f32)  # GRU input: own + frac
        self._cen = torch.zeros((1, T_OUT, h, w, c), **f32)
        self._half = torch.zeros((1, T_OUT, h, w, c), **f32)
        self._one = torch.ones((), **f32)
        # 1 once ``apply_`` has served a window: the MERGED step (``observe_gated``) reads it so
        # a trajectory's first call, which has nothing revealed, leaves the state untouched
        # without a host branch -- one graph serves every step.
        self._hp = torch.zeros((), **f32)
        self._S = torch.zeros((h, w, c), **f32)  # running signed-error sum (physical)
        self._r = torch.zeros((), **f32)  # revealed windows folded into ``_S``

    def reset(self) -> None:
        self._state.zero_()
        self._hp.zero_()
        self._S.zero_()
        self._r.zero_()

    def state_buffers(self) -> list[torch.Tensor]:
        """Every buffer a step mutates: a CUDA-graph capture snapshots and restores them."""
        return [self._state, self._c_prev, self._h_eps, self._v, self._hp, self._S, self._r]

    @torch.no_grad()
    def observe(self, x_eval: torch.Tensor, gated: bool = False) -> None:
        """Fold in the previous window's revealed target (never on a trajectory's first call).

        ``gated``: blend the new state in by ``_hp`` (0 before the first served window), so the
        call is harmless on a trajectory's first step -- the merged single-graph contract.
        """
        c, (hh, ww) = self.c, self.hw
        t = torch.addcmul(self.mi, x_eval[0, ..., :c].float(), self.si)  # physical target
        sc = t.abs().gt_(1e-6).float()
        d = t.sub_(self._c_prev)
        v = self._v
        if self.signed:
            torch.div(d, self._h_eps, out=v[-1]).clamp_(-3.0, 3.0)
        q = d.abs_().div_(self._h_eps)
        torch.add(q, 1e-3, out=v[0])
        v[0].log_().clamp_(-6.0, 3.0)
        cov = v[1 : 1 + self.n_cov]
        torch.sub(self.grid, q, out=cov)
        cov.heaviside_(self._one)  # 1{q <= g}
        v.mul_(sc)  # the lh plane was written by the previous ``apply``
        per = T_OUT // N_BUCKETS
        n = sc.view(N_BUCKETS, per, hh, ww, c).sum(1).clamp_min_(1.0)  # (4, H, W, c)
        f = v.view(self.n_parts, N_BUCKETS, per, hh, ww, c).sum(2).div_(n)  # (P, 4, H, W, c)
        nf = 4 * self.n_parts
        self._x[:, :nf].view(hh, ww, c, N_BUCKETS, self.n_parts).copy_(f.permute(2, 3, 4, 1, 0))
        self._x[:, nf].view(hh, ww, c).copy_(sc.mean(0))
        gi = torch.addmm(self.bias_rows, self._x, self.w_small_t)
        gh = torch.addmm(self.b_hh, self._state, self.w_hh_t)
        self._fold_state(_gru_from_gates(gi, gh, self._state), gated)

    def _fold_state(self, new_state: torch.Tensor, gated: bool) -> None:
        if gated:
            self._state.copy_(torch.lerp(self._state, new_state, self._hp))
        else:
            self._state.copy_(new_state)

    def observe_gated(self, x_eval: torch.Tensor) -> None:
        self.observe(x_eval, gated=True)

    @torch.no_grad()
    def apply_(self, lo: torch.Tensor, hi: torch.Tensor) -> None:
        """``apply`` written back INTO ``lo``/``hi`` (the pipeline's own output buffers), so the
        merged graph's outputs stay the buffers the predictor copies to the host in one go."""
        lo2, hi2 = self.apply(lo, hi)
        c = self.c
        lo[..., :c].copy_(lo2[..., :c])
        hi[..., :c].copy_(hi2[..., :c])
        self._hp.fill_(1.0)

    def step_merged(self, x_eval: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> None:
        """The one-graph step: gated observe, then apply in place. Same result as
        ``step_first`` on a trajectory's first call and ``step`` after it."""
        self.observe_gated(x_eval)
        self.apply_(lo, hi)

    @torch.no_grad()
    def apply(self, lo: torch.Tensor, hi: torch.Tensor):
        c, (hh, ww) = self.c, self.hw
        # One cast, then mixed-dtype ops that promote to fp32 (an fp16 + fp16 add would round in
        # fp16 before landing in the fp32 buffer).
        lo_f, hi_c = lo[..., :c].float(), hi[..., :c]
        cen = torch.add(lo_f, hi_c, out=self._cen).mul_(0.5)
        half = torch.sub(hi_c, lo_f, out=self._half).mul_(0.5)
        torch.sub(cen[0], self.b, out=self._c_prev).div_(self.a)
        torch.div(half[0], self.a, out=self._h_eps)
        lh = self._v[self.i_lh]
        torch.mul(self._h_eps, self.inv_sigma, out=lh)
        lh.add_(1e-8).log_().clamp_(-12.0, 3.0)
        self._h_eps.add_(1e-8)
        zz = torch.addmm(self.b_out, self._state, self.w_out_t).tanh_()
        m = zz[:, :N_BUCKETS].mul(LOG2).exp_()
        m = m.view(hh, ww, c, N_BUCKETS).permute(3, 0, 1, 2)[:, None]  # (4, 1, H, W, c)
        half_b = half.view(N_BUCKETS, T_OUT // N_BUCKETS, hh, ww, c)
        new_half = (half_b * m).view_as(half)
        if self.shift_head:
            sh = zz[:, N_BUCKETS:].mul(0.5).view(hh, ww, c, N_BUCKETS).permute(3, 0, 1, 2)[:, None]
            cen = cen + (half_b * sh).view_as(half)
        lo2 = torch.cat([(cen - new_half).to(lo.dtype), lo[..., c:]], dim=-1)
        hi2 = torch.cat([(cen + new_half).to(hi.dtype), hi[..., c:]], dim=-1)
        return lo2, hi2

    def step(self, x_eval: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
        self.observe(x_eval)
        return self.apply(lo, hi)

    step_frozen = step

    def step_first(self, x_eval: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
        return self.apply(lo, hi)


_OBSERVE_SRC = """
template <typename T>
void recal_observe(T x, T mi, T si, T cp, T he, T lh, {outs}) {{
    T t = x * si + mi;
    T s = fabs(t) > T(1e-6) ? T(1) : T(0);
    T q = fabs(t - cp) / he;
    T l = log(q + T(1e-3));
    l = l < T(-6) ? T(-6) : (l > T(3) ? T(3) : l);
    o0 = l * s;
{covs}
    o{lh_idx} = lh * s;
    o{s_idx} = s;
}}
"""

# The signed masked error for the centre shift, as its own one-output kernel: the jiterator
# allows at most 8 outputs per kernel and the observe kernel already has 8.
_SIGNED_SRC = """
template <typename T>
T recal_signed(T x, T mi, T si, T cp) {
    T t = x * si + mi;
    return fabs(t) > T(1e-6) ? (t - cp) : T(0);
}
"""

# ONE kernel for every observe plane (2026-09-26, the launch-count cut): the plane index is a
# broadcast input, so the output is the stacked ``(P+1, 20, H, W, c)`` planes -- lr, covs, lh,
# [signed], mask -- with no ``torch.stack`` after it. The bucket sums then come out of one
# batched matmul against a (4, 20) indicator, the mask plane giving the counts.
_PLANES_SRC = """
template <typename T>
T recal_planes(T x, T mi, T si, T cp, T he, T lh, T idx) {{
    T t = x * si + mi;
    T s = fabs(t) > T(1e-6) ? T(1) : T(0);
    T d = t - cp;
    T q = fabs(d) / he;
    int i = (int)(idx + T(0.5));
    if (i == 0) {{
        T l = log(q + T(1e-3));
        l = l < T(-6) ? T(-6) : (l > T(3) ? T(3) : l);
        return l * s;
    }}
{covs}
    if (i == {lh_idx}) return lh * s;
{signed}
    return s;
}}
"""

# The GRU cell from ONE gate GEMM over [features | state]: columns [r | z | i_n | h_n], and the
# has-previous gate folded in (``h + hp * (h_new - h)``) -- cell and lerp as one kernel.
_CELL_SRC = """
template <typename T>
T recal_cell(T gr, T gz, T gn, T hn, T h, T hp) {
    T r = T(1) / (T(1) + exp(-gr));
    T z = T(1) / (T(1) + exp(-gz));
    T n = tanh(gn + r * hn);
    T hnew = (T(1) - z) * n + z * h;
    return h + hp * (hnew - h);
}
"""

# The signed error over the half-width, clamped and masked: the ``signed`` feature plane.
_SIGNED_FEAT_SRC = """
template <typename T>
T recal_signed_feat(T x, T mi, T si, T cp, T he) {
    T t = x * si + mi;
    T s = fabs(t) > T(1e-6) ? T(1) : T(0);
    T v = (t - cp) / he;
    v = v < T(-3) ? T(-3) : (v > T(3) ? T(3) : v);
    return v * s;
}
"""

# The apply with the LEARNED shift head: ``z2`` the shift logits, a shift of both bounds by
# ``tanh(z2) / 2`` half-widths (the lab's ``shift_head`` spelling).
_APPLY_SHIFTHEAD_SRC = """
template <typename T>
void recal_apply_shifthead(T lo, T hi, T z, T z2, T a, T b,
                           T& lo2, T& hi2, T& cp, T& he, T& lh) {{
    T cen = (lo + hi) * T(0.5);
    T half = (hi - lo) * T(0.5);
    T m = exp(tanh(z) * T({log2}));
    T sh = tanh(z2) * T(0.5) * half;
    lo2 = cen + sh - half * m;
    hi2 = cen + sh + half * m;
    cp = (cen - b) / a;
    T h = half / a;
    T v = log(h * T({inv_sigma}) + T(1e-8));
    lh = v < T(-12) ? T(-12) : (v > T(3) ? T(3) : v);
    he = h + T(1e-8);
}}
"""

_APPLY_SRC = """
template <typename T>
void recal_apply(T lo, T hi, T z, T a, T b, T& lo2, T& hi2, T& cp, T& he, T& lh) {{
    T cen = (lo + hi) * T(0.5);
    T half = (hi - lo) * T(0.5);
    T m = exp(tanh(z) * T({log2}));
    lo2 = cen - half * m;
    hi2 = cen + half * m;
    cp = (cen - b) / a;
    T h = half / a;
    T v = log(h * T({inv_sigma}) + T(1e-8));
    lh = v < T(-12) ? T(-12) : (v > T(3) ? T(3) : v);
    he = h + T(1e-8);
}}
"""

# The same apply with the closed-form centre shift: ``S`` the running signed-error sum
# (physical, per pixel and channel, broadcast over the leads), ``r`` its window count.
_APPLY_SHIFT_SRC = """
template <typename T>
void recal_apply_shift(T lo, T hi, T z, T a, T b, T S, T r,
                       T& lo2, T& hi2, T& cp, T& he, T& lh) {{
    T cen = (lo + hi) * T(0.5);
    T half = (hi - lo) * T(0.5);
    T m = exp(tanh(z) * T({log2}));
    T sh = T({beta}) * S / (r + T({n0})) * a;
    lo2 = cen + sh - half * m;
    hi2 = cen + sh + half * m;
    cp = (cen - b) / a;
    T h = half / a;
    T v = log(h * T({inv_sigma}) + T(1e-8));
    lh = v < T(-12) ? T(-12) : (v > T(3) ? T(3) : v);
    he = h + T(1e-8);
}}
"""


class IntervalRecalFused(IntervalRecal):
    """:class:`IntervalRecal` with its element-wise chains fused into two runtime-compiled kernels.

    The Track-2 image has no C compiler, so Triton (``torch.compile``) cannot fuse anything
    there (FINDINGS 190); PyTorch's jiterator compiles element-wise kernels through NVRTC, which
    ships with the CUDA libraries, and was probed working in the image, under CUDA-graph capture
    (``scripts/t2/t2_image_jiterator_probe.py``). The observe chain (target, mask, ratio, log,
    every coverage test, the masked width feature) becomes ONE kernel and the apply chain
    (centre, half-width, multiplier, new bounds, remembered state) ONE more; reductions and the
    GRU's matrix products stay standard kernels. The kernels compile on the first call, which
    the serving hook runs in its untimed capture warm-up. CUDA only; needs ``sigma``.
    """

    def __init__(self, model: RecurrentCalibrator, device, **wire) -> None:
        super().__init__(model, device, **wire)
        self.shifted = bool(self.shift_beta)
        if self.shifted and self.shift_head:
            raise ValueError("the closed-form shift and the learned shift head are exclusive")
        f32 = dict(device=self.dev, dtype=torch.float32)
        P, hid = self.n_parts, model.hidden
        n_rows = self._state.shape[0]
        covs = "\n".join(
            f"    if (i == {1 + i}) return (q <= T({g!r}) ? T(1) : T(0)) * s;"
            for i, g in enumerate(model.cov_grid)
        )
        signed = ""
        if self.signed:
            signed = (
                f"    if (i == {P - 1}) {{ T v = d / he; "
                "v = v < T(-3) ? T(-3) : (v > T(3) ? T(3) : v); return v * s; }"
            )
        self._k_planes = torch.cuda.jiterator._create_jit_fn(
            _PLANES_SRC.format(covs=covs, lh_idx=self.i_lh, signed=signed)
        )
        self._k_cell = torch.cuda.jiterator._create_jit_fn(_CELL_SRC)
        if self.shifted:
            self._k_signed = torch.cuda.jiterator._create_jit_fn(_SIGNED_SRC)
        # Plane index (P planes + the mask), the (4, 20) bucket indicator, and the feature/state
        # buffer in TRANSPOSED layout: rows ``(p, b)`` for the P feature planes (bucket means),
        # then 4 rows of bucket counts (the scored fraction, folded into the weights as
        # ``w_frac / 20`` each), then the ``hid`` rows of the GRU state -- so the gate GEMM is one
        # ``addmm`` over ``_feat.t()`` and the state needs no copy into an input buffer.
        self._idx = torch.arange(P + 1, **f32).view(P + 1, 1, 1, 1, 1)
        bucket = lead_bucket(self.dev)
        self._B = (bucket[None, :] == torch.arange(N_BUCKETS, device=self.dev)[:, None]).to(
            torch.float32
        )  # (4, 20)
        self.n_cols = (P + 1) * N_BUCKETS
        self._feat = torch.zeros((self.n_cols + hid, n_rows), **f32)
        self._one_scalar = torch.ones((), **f32)
        # Weights: ``w_small_t`` is (nf + 1, 3H) with rows in the lab's bucket-major order
        # (b * P + p) and the frac row last; re-order to plane-major (p * 4 + b), spread the frac
        # row over the 4 count rows at 1/20, and append the hidden weights into a single
        # ``[r | z | i_n | h_n]`` matrix of 4H columns.
        w_small = self.w_small_t  # (nf + 1, 3H)
        perm = [b * P + p for p in range(P) for b in range(N_BUCKETS)]
        w_x = torch.cat([w_small[perm], w_small[-1:].expand(N_BUCKETS, -1) / 20.0], dim=0)
        w_h = self.w_hh_t  # (H, 3H)
        w_cat = torch.zeros((self.n_cols + hid, 4 * hid), **f32)
        w_cat[: self.n_cols, : 3 * hid] = w_x
        w_cat[self.n_cols :, : 2 * hid] = w_h[:, : 2 * hid]
        w_cat[self.n_cols :, 3 * hid :] = w_h[:, 2 * hid :]
        self.w_cat = w_cat.contiguous()
        b_hh = self.b_hh
        self.bias4 = torch.cat(
            [
                self.bias_rows[:, : 2 * hid] + b_hh[: 2 * hid],
                self.bias_rows[:, 2 * hid :],
                b_hh[2 * hid :].expand(n_rows, -1),
            ],
            dim=1,
        ).contiguous()  # (rows, 4H)
        if self.shift_head:
            self._k_app_head = torch.cuda.jiterator._create_multi_output_jit_fn(
                _APPLY_SHIFTHEAD_SRC.format(log2=repr(LOG2), inv_sigma=repr(self.inv_sigma)),
                num_outputs=5,
            )
        self._k_app = torch.cuda.jiterator._create_multi_output_jit_fn(
            _APPLY_SRC.format(log2=repr(LOG2), inv_sigma=repr(self.inv_sigma)), num_outputs=5
        )
        if self.shifted:
            self._k_app_shift = torch.cuda.jiterator._create_multi_output_jit_fn(
                _APPLY_SHIFT_SRC.format(
                    log2=repr(LOG2),
                    inv_sigma=repr(self.inv_sigma),
                    beta=repr(self.shift_beta),
                    n0=repr(self.shift_n0),
                ),
                num_outputs=5,
            )

    def reset(self) -> None:
        self._feat[self.n_cols :].zero_()
        self._hp.zero_()
        self._S.zero_()
        self._r.zero_()

    def state_buffers(self) -> list[torch.Tensor]:
        return [self._feat, self._c_prev, self._h_eps, self._v, self._hp, self._S, self._r]

    @property
    def state_t(self) -> torch.Tensor:
        """The GRU state as ``(rows, H)``, a transposed view into the feature buffer."""
        return self._feat[self.n_cols :].t()

    @torch.no_grad()
    def observe(self, x_eval: torch.Tensor, gated: bool = False) -> None:
        c, (hh, ww) = self.c, self.hw
        P, hid, C = self.n_parts, self.model.hidden, self.n_cols
        xm = x_eval[0, ..., :c].float()[None]
        # 1 kernel: every masked plane, stacked -- (P + 1, 20, H, W, c), the mask last.
        planes = self._k_planes(
            xm,
            self.mi,
            self.si,
            self._c_prev[None],
            self._h_eps[None],
            self._v[self.i_lh][None],
            self._idx,
        )
        n_rows = hh * ww * c
        # 1 kernel: bucket sums of every plane, (P + 1, 4, rows); the mask's are the counts.
        sums = torch.matmul(self._B, planes.view(P + 1, T_OUT, n_rows))
        if self.shifted:
            es = self._k_signed(xm[0], self.mi, self.si, self._c_prev)
            m = es.sum(0).div_(sums[P].sum(0).view(hh, ww, c).clamp_min_(1.0))
            self._S.addcmul_(m, self._hp)
            self._r.add_(self._hp)
        n = sums[P].clamp_min(1.0)  # (4, rows)
        feat = self._feat
        torch.div(sums[:P], n[None], out=feat[: P * N_BUCKETS].view(P, N_BUCKETS, n_rows))
        feat[P * N_BUCKETS : C].copy_(sums[P])  # the raw counts; the weights carry the 1/20
        # 1 GEMM: [r | z | i_n | h_n] over [features | state].
        g = torch.addmm(self.bias4, feat.t(), self.w_cat)  # (rows, 4H)
        h = self.state_t
        hp = self._hp if gated else self._one_scalar
        h_new = self._k_cell(
            g[:, :hid], g[:, hid : 2 * hid], g[:, 2 * hid : 3 * hid], g[:, 3 * hid :], h, hp
        )
        feat[C:].copy_(h_new.t())

    @torch.no_grad()
    def apply_(self, lo: torch.Tensor, hi: torch.Tensor) -> None:
        c, (hh, ww) = self.c, self.hw
        per = T_OUT // N_BUCKETS
        zz = torch.addmm(self.b_out, self.state_t, self.w_out_t)  # (rows, 4 [+ 4])
        z = zz[:, :N_BUCKETS].view(hh, ww, c, N_BUCKETS).permute(3, 0, 1, 2)[:, None]
        lo_f = lo[0, ..., :c].float().view(N_BUCKETS, per, hh, ww, c)
        hi_f = hi[0, ..., :c].float().view(N_BUCKETS, per, hh, ww, c)
        if self.shift_head:
            z2 = zz[:, N_BUCKETS:].view(hh, ww, c, N_BUCKETS).permute(3, 0, 1, 2)[:, None]
            lo2, hi2, cp, he, lh = self._k_app_head(lo_f, hi_f, z, z2, self.a, self.b)
        elif self.shifted:
            lo2, hi2, cp, he, lh = self._k_app_shift(
                lo_f, hi_f, z, self.a, self.b, self._S[None, None], self._r.view(1, 1, 1, 1, 1)
            )
        else:
            lo2, hi2, cp, he, lh = self._k_app(lo_f, hi_f, z, self.a, self.b)
        self._c_prev.copy_(cp.view_as(self._c_prev))
        self._h_eps.copy_(he.view_as(self._h_eps))
        self._v[self.i_lh].copy_(lh.view_as(self._c_prev))
        lo[0, ..., :c].copy_(lo2.view(T_OUT, hh, ww, c))
        hi[0, ..., :c].copy_(hi2.view(T_OUT, hh, ww, c))
        self._hp.fill_(1.0)

    @torch.no_grad()
    def apply(self, lo: torch.Tensor, hi: torch.Tensor):
        c, (hh, ww) = self.c, self.hw
        per = T_OUT // N_BUCKETS
        if self.shift_head:
            raise RuntimeError("the shift-head calibrator is served by the merged step only")
        z = torch.addmm(self.b_out, self.state_t, self.w_out_t)  # (rows, 4) logits
        z = z.view(hh, ww, c, N_BUCKETS).permute(3, 0, 1, 2)[:, None]  # (4, 1, H, W, c)
        lo_f = lo[0, ..., :c].float().view(N_BUCKETS, per, hh, ww, c)
        hi_f = hi[0, ..., :c].float().view(N_BUCKETS, per, hh, ww, c)
        lo2, hi2, cp, he, lh = self._k_app(lo_f, hi_f, z, self.a, self.b)
        self._c_prev.copy_(cp.view_as(self._c_prev))
        self._h_eps.copy_(he.view_as(self._h_eps))
        self._v[self.i_lh].copy_(lh.view_as(self._c_prev))
        shape = (1, T_OUT, hh, ww, c)
        lo2 = torch.cat([lo2.view(shape).to(lo.dtype), lo[..., c:]], dim=-1)
        hi2 = torch.cat([hi2.view(shape).to(hi.dtype), hi[..., c:]], dim=-1)
        return lo2, hi2


def make_interval_recal(
    model: RecurrentCalibrator,
    device,
    fused: bool = False,
    shift_beta: float = 0.0,
    shift_n0: float = 5.0,
    **wire,
):
    """The fast step where it applies (``fused``: its jiterator form), the reference otherwise."""
    if model.arch == "gru" and not model.context:
        if fused and torch.device(device).type == "cuda":
            return IntervalRecalFused(
                model, device, shift_beta=shift_beta, shift_n0=shift_n0, **wire
            )
        if shift_beta:
            raise ValueError("the centre shift needs the fused (CUDA, sigma) step")
        return IntervalRecal(model, device, **wire)
    if shift_beta:
        raise ValueError("the centre shift needs the fast calibrator step")
    return IntervalRecalReference(model, device, **wire)
