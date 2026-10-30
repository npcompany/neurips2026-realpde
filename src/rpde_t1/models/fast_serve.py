"""Serving-only fast path for the Track-2 STFormer bundle: the same arithmetic in fewer kernels.

At batch 1 under a CUDA graph the step is bound by kernel count, not arithmetic (FINDINGS 218):
424 kernels, 355 of them under 5 us. Many of those are element-wise chains the eager code spells
as several ops -- ``x * (1.0 + scale) + shift`` is three kernels -- and many more are functions
of the conditioning vector alone, recomputed on every replay although the operating point is
frozen for a whole trajectory. This module serves the same trained model with

* every function of ``cond`` alone (the conditioning MLP, the six adaLN maps per block, the two
  FiLM projections, the interval head's FiLM, the attention temperature) computed ONCE per
  operating point, eagerly and outside the graph, into static buffers the graph reads; and
* each element-wise chain fused into one jiterator kernel (NVRTC, so it needs no C compiler and
  runs in the evaluation image, FINDINGS 208) that performs the SAME IEEE operations in the SAME
  order. Multiply and add are spelled ``__fmul_rn``/``__fadd_rn`` so the compiler cannot contract
  them into an FMA (a plain ``a * b + c`` differs from eager in 23% of elements); each
  intermediate is rounded to the storage dtype exactly where the eager op would round it.

The output is required to be BIT-IDENTICAL to the reference ``make_gpu_predict`` pipeline, and
the predictor checks that at construction on a synthetic window, falling back to the reference
if it is not. The training code is untouched: the model's own ``forward`` stays the definition.

Only the exact shipped configuration is supported (:func:`supported`); anything else serves
through the reference pipeline.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

# One kernel per chain, computing in fp32 (PyTorch's opmath for fp32 and fp16 alike) with
# rounding-exact intrinsics. ``R(v)`` rounds an intermediate to the dtype the corresponding eager
# op would have written: the identity for fp32, and round-to-nearest-even to fp16 precision for
# fp16 -- jiterator instantiates an fp16 kernel with ``T = float``, so ``T(v)`` does not round.
_JIT_SRC = {
    # x * a + b, as ``x * (1.0 + scale) + shift`` with ``a = 1.0 + scale`` precomputed.
    "mul_add": "T m = R(__fmul_rn(float(x), float(a))); return T(R(__fadd_rn(m, float(b))));",
    # x + g * h, the gated residual ``x + (1.0 + gate) * h`` with ``g = 1.0 + gate``.
    "res": "T m = R(__fmul_rn(float(g), float(h))); return T(R(__fadd_rn(float(x), m)));",
    # F.normalize(q, dim=-1) * s: ``q / max(|q|, eps) * s``.
    "norm_scale": (
        "float r = R(__fdiv_rn(float(q), fmaxf(float(n), 1e-12f)));"
        " return T(R(__fmul_rn(r, float(s))));"
    ),
    # F.normalize(k, dim=-1): ``k / max(|k|, eps)``.
    "norm": "return T(R(__fdiv_rn(float(k), fmaxf(float(n), 1e-12f))));",
}
_JIT_ARGS = {
    "mul_add": "T x, T a, T b",
    "res": "T x, T g, T h",
    "norm_scale": "T q, T n, T s",
    "norm": "T k, T n",
}

# float -> nearest-even fp16 value, as float, by the hardware conversion (PTX ``cvt.rn.f16.f32``
# then ``cvt.f32.f16``; inline asm, so it needs no header inside NVRTC). The same function as
# torch's own fp32 -> fp16 cast: checked bit for bit on 2^25 random fp32 patterns and on every
# fp16 value and its fp32 neighbours (normals, subnormals, the overflow tie at 65520, NaN).
# Four to five times fewer instructions than the previous software version (mantissa
# arithmetic with branches), which made the fused FiLM/GELU pass instruction-bound.
_ROUND_HALF = r"""
    auto R = [](float f) -> float {
        unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
        float r; asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h)); return r; };
"""
_ROUND_NONE = "    auto R = [](float f) -> float { return f; };\n"
# ``RO``: the same rounding applied to a kernel's RESULT only (``_kernel(..., round_out=True)``),
# so an fp32 kernel can hand back the values an fp16 copy of its result would hold.
_ROUND_OUT = _ROUND_HALF.replace("auto R =", "auto RO =")
_ROUND_OUT_NONE = _ROUND_NONE.replace("auto R =", "auto RO =")

_JIT: dict[tuple[str, bool], Any] = {}


def _jit(name: str, half: bool = False):
    """The fused kernel ``name``; ``half`` rounds its intermediates to fp16."""
    fn = _JIT.get((name, half))
    if fn is None:
        from torch.cuda.jiterator import _create_jit_fn  # noqa: PLC0415

        src = (
            f"template <typename T> T {name}({_JIT_ARGS[name]}) {{\n"
            + (_ROUND_HALF if half else _ROUND_NONE)
            + f"    {_JIT_SRC[name]}\n}}"
        )
        fn = _JIT[(name, half)] = _create_jit_fn(src)
    return fn


def _mul_add(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``x * a + b``, fused when the product already has the result's dtype (eager rounds the
    product to its own dtype, which the kernel then reproduces)."""
    prod = torch.promote_types(x.dtype, a.dtype)
    if prod == torch.promote_types(prod, b.dtype):
        return _jit("mul_add", prod == torch.float16)(x, a, b)
    return x * a + b


def _res(x: torch.Tensor, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """``x + g * h``, fused when the product has the stream's dtype. Under autocast ``g * h`` is
    an fp16 product added to an fp32 stream: fusable too, rounding the product to fp16."""
    prod = torch.promote_types(g.dtype, h.dtype)
    out = torch.promote_types(x.dtype, prod)
    if out == prod:
        return _jit("res", prod == torch.float16)(x, g, h)
    if out == torch.float32 and prod == torch.float16:
        return _jit_res_mixed()(x, g, h)
    return x + g * h


def _jit_res_mixed():
    """``x + g * h`` with an fp16 product added in fp32: round the product only."""
    fn = _JIT.get(("res_mixed", True))
    if fn is None:
        from torch.cuda.jiterator import _create_jit_fn  # noqa: PLC0415

        src = (
            "template <typename T> T res_mixed(T x, T g, T h) {\n"
            + _ROUND_HALF
            + "    float m = R(__fmul_rn(float(g), float(h)));\n"
            "    return T(__fadd_rn(float(x), m));\n}"
        )
        fn = _JIT[("res_mixed", True)] = _create_jit_fn(src)
    return fn


def _norm_scale(q: torch.Tensor, n: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``F.normalize(q, dim=-1) * s`` given ``n = q.norm(2, -1, keepdim=True)``."""
    div = torch.promote_types(q.dtype, n.dtype)
    if div == torch.promote_types(div, s.dtype):
        return _jit("norm_scale", div == torch.float16)(q, n, s)
    return F.normalize(q, dim=-1) * s


def _norm(k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return _jit("norm", torch.promote_types(k.dtype, n.dtype) == torch.float16)(k, n)


def supported(model: torch.nn.Module) -> bool:
    """The exact configuration this module reproduces: BoundedModel(ResidualWrapper(STFormer))
    with adaLN blocks, qk-normalised softmax attention, FiLM on the stem and refine paths, a
    coarse linear head and the stem-skip refine -- the shipped Track-2 backbone."""
    from .bounds import BoundedModel  # noqa: PLC0415
    from .stformer import STFormer  # noqa: PLC0415
    from .wrappers import ResidualWrapper  # noqa: PLC0415

    if not isinstance(model, BoundedModel) or not model.freeze_backbone:
        return False
    if getattr(model.head, "mode", "offsets") == "scale":  # the closed-form width needs u_fs
        return False
    rw = model.backbone
    if not isinstance(rw, ResidualWrapper) or not rw.enabled:
        return False
    s = rw.backbone
    if type(s) is not STFormer:
        return False
    flags_off = (
        "cond_token",
        "rel_pos",
        "temporal_gru",
        "cond_content",
        "lead_query_decoder",
        "head_fourier",
        "reynolds",
        "temporal_anchor",
        "domain_interface",
    )
    if any(getattr(s, f, False) for f in flags_off) or getattr(s, "recur", None) is not None:
        return False
    if not (s.cond_conv and s.has_stem and s.coarse_head):
        return False
    if s.refine_enabled and not s.refine_stem_skip:
        return False
    if s.modulation != "adaln" or s.grid[0] != 1:
        return False
    if s.cond_proj is None or s.c_out != 3:
        return False
    for blk in s.blocks:
        if blk.mixer != "softmax" or blk.modulation_mode != "adaln" or blk.cond_attn:
            return False
    return True


class FastSTFormer:
    """``STFormer.forward`` for the supported configuration, with ``cond`` hoisted.

    :meth:`set_cond` recomputes the conditioning-derived tensors, in place into buffers whose
    addresses never change (a CUDA graph captured over :meth:`forward` reads them), and must be
    called whenever the conditioning vector changes. It runs the model's own submodules on
    ``cond`` in exactly the expressions ``STFormer.forward`` uses, so the values are identical.
    """

    def __init__(self, stformer: torch.nn.Module, stack_cond: bool = False) -> None:
        self.s = stformer
        self.buf: dict[str, torch.Tensor] | None = None
        self.outs: dict[str, torch.Tensor] = {}
        # The per-head temperature is a function of a parameter alone: the blocks' logit
        # scales stacked into one tensor, so the clamp and the exp run once for all of them
        # (element-wise ops, so each block's slice holds the value its own two ops would give).
        with torch.no_grad():
            ls = [blk.attn.logit_scale.detach() for blk in stformer.blocks if blk.attn.qk_norm]
            self._ls_all = torch.stack(ls, 0) if ls else None
        self._ls_row: dict[int, int] = {}
        for i, blk in enumerate(stformer.blocks):
            if blk.attn.qk_norm:
                self._ls_row[i] = len(self._ls_row)
        # ``stack_cond``: the conditioning maps as two matrix products instead of two plus
        # six -- the stem and refine FiLM weights stacked along their output rows, and the
        # blocks' adaLN weights likewise (their SiLU is the same function of the same input,
        # run once). Each output row is the same dot product; with fp16 weights cuBLAS sums
        # it the same way at either width and the values are identical (FINDINGS 236), in
        # fp32 it does not (the last bit moves), so the fp32 pipeline keeps the per-module
        # products. Copies, so the model's own parameters are untouched.
        self.stack_cond = bool(stack_cond)
        s = stformer
        with torch.no_grad():
            films = [s.film_stem] + ([s.film_refine] if s.refine_enabled else [])
            self._w_film = torch.cat([f.weight.detach() for f in films], 0)
            self._b_film = torch.cat([f.bias.detach() for f in films], 0)
            self._n_film = [f.out_features for f in films]
            self._w_mod = torch.cat([blk.modulation[1].weight.detach() for blk in s.blocks], 0)
            self._b_mod = torch.cat([blk.modulation[1].bias.detach() for blk in s.blocks], 0)

    @torch.no_grad()
    def _cond_values(self, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        """The FiLM and adaLN maps of ``cond``, as views into two stacked products (see
        ``__init__``); ``1.0 +`` is applied once over each product, the same element-wise add
        as on every chunk separately."""
        s = self.s
        c = s._conditioning(cond, cond)  # (B, dim); the first argument only sizes a None cond
        vals: dict[str, torch.Tensor] = {}
        # all the blocks' temperatures in two kernels instead of two per block
        ls_all = self._ls_all.clamp(max=_LOG100).exp() if self._ls_all is not None else None
        # A single row is the one shape where cuBLAS picks a different kernel for the wider
        # product (a gemv), and its sums differ in the last bit (FINDINGS 236); from two rows
        # on the products are identical. The per-size check in the bundle would catch a
        # difference anyway, but this keeps the values identical at every size.
        if not self.stack_cond or cond.shape[0] < 2:
            scale, shift = s.film_stem(c).chunk(2, dim=-1)
            vals["stem_a"], vals["stem_b"] = (
                (1.0 + scale[:, :, None, None]),
                shift[:, :, None, None],
            )
            if s.refine_enabled:
                scale, shift = s.film_refine(c).chunk(2, dim=-1)
                vals["ref_a"] = 1.0 + scale[:, :, None, None]
                vals["ref_b"] = shift[:, :, None, None]
            for i, blk in enumerate(s.blocks):
                sh1, sc1, g1, sh2, sc2, g2 = blk.modulation(c).chunk(6, dim=-1)
                vals[f"b{i}_sh1"], vals[f"b{i}_a1"] = sh1[:, None], (1.0 + sc1[:, None])
                vals[f"b{i}_g1"] = 1.0 + g1[:, None]
                vals[f"b{i}_sh2"], vals[f"b{i}_a2"] = sh2[:, None], (1.0 + sc2[:, None])
                vals[f"b{i}_g2"] = 1.0 + g2[:, None]
                if blk.attn.qk_norm:
                    vals[f"b{i}_ls"] = ls_all[self._ls_row[i]]
            return {k: v.contiguous() for k, v in vals.items()}
        film = F.linear(c, self._w_film, self._b_film)
        film1 = 1.0 + film
        half = self._n_film[0] // 2
        vals["stem_a"] = film1[:, :half, None, None]
        vals["stem_b"] = film[:, half : 2 * half, None, None]
        if s.refine_enabled:
            base, half = self._n_film[0], self._n_film[1] // 2
            vals["ref_a"] = film1[:, base : base + half, None, None]
            vals["ref_b"] = film[:, base + half : base + 2 * half, None, None]
        mods = F.linear(F.silu(c), self._w_mod, self._b_mod)
        mods1 = 1.0 + mods
        dim = self._w_mod.shape[0] // (6 * len(s.blocks))
        for i, blk in enumerate(s.blocks):
            o = 6 * dim * i
            vals[f"b{i}_sh1"] = mods[:, None, o : o + dim]
            vals[f"b{i}_a1"] = mods1[:, None, o + dim : o + 2 * dim]
            vals[f"b{i}_g1"] = mods1[:, None, o + 2 * dim : o + 3 * dim]
            vals[f"b{i}_sh2"] = mods[:, None, o + 3 * dim : o + 4 * dim]
            vals[f"b{i}_a2"] = mods1[:, None, o + 4 * dim : o + 5 * dim]
            vals[f"b{i}_g2"] = mods1[:, None, o + 5 * dim : o + 6 * dim]
            if blk.attn.qk_norm:
                vals[f"b{i}_ls"] = ls_all[self._ls_row[i]]
        return vals

    def set_cond(self, cond: torch.Tensor) -> None:
        vals = self._cond_values(cond)
        if self.buf is None or any(
            (self.buf[k].shape, self.buf[k].dtype) != (v.shape, v.dtype) for k, v in vals.items()
        ):
            with torch.inference_mode(False):
                self.buf = {k: v.clone() for k, v in vals.items()}
        else:
            for k, v in vals.items():
                self.buf[k].copy_(v)

    def _block(self, i: int, blk: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        b = self.buf
        mul_add, res = _mul_add, _res
        xn = mul_add(blk.norm1(x), b[f"b{i}_a1"], b[f"b{i}_sh1"])
        a = blk.attn
        if torch.is_autocast_enabled() and xn.dtype == torch.float32:
            # Autocast would cast ``xn`` once for ``to_q`` and again for ``to_kv``; cast it
            # once (the same ``.to``) and hand both linears the reduced-precision copy.
            xn = xn.to(torch.get_autocast_gpu_dtype())
        q = a._heads(a.to_q(xn))
        k, v = (a._heads(t) for t in a.to_kv(xn).chunk(2, dim=-1))
        if a.qk_norm:
            q = _norm_scale(q, q.norm(2.0, -1, keepdim=True), b[f"b{i}_ls"])
            k = _norm(k, k.norm(2.0, -1, keepdim=True))
            out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        h = a.proj(out.transpose(1, 2).flatten(2))
        x = res(x, b[f"b{i}_g1"], h)
        return res(x, b[f"b{i}_g2"], blk.mlp(mul_add(blk.norm2(x), b[f"b{i}_a2"], b[f"b{i}_sh2"])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .transformer import unpatchify  # noqa: PLC0415

        s, b = self.s, self.buf
        assert b is not None, "set_cond must run before forward"
        mul_add = _mul_add
        bsz = x.shape[0]
        gt, gh, gw = s.grid
        groups = s._to_groups(x)
        feat = mul_add(s.stem(groups), b["stem_a"], b["stem_b"])
        tokens = s.embed(feat)
        tokens = tokens.reshape(bsz, gt, -1, gh, gw).permute(0, 1, 3, 4, 2)
        # ``tokens.reshape(b, n, -1) + pos``: the add reads the permuted tokens directly and
        # writes the contiguous stream, instead of a reshape copy followed by the add.
        dim = tokens.shape[-1]
        h = self._out(
            "h", (bsz, s.n_tokens, dim), torch.promote_types(tokens.dtype, s.pos.dtype), x.device
        )
        torch.add(tokens, s.pos.view(1, 1, gh, gw, dim), out=h.view(bsz, gt, gh, gw, dim))
        for i, blk in enumerate(s.blocks):
            h = self._block(i, blk, h)
        h = s.norm(h)
        t_out, h_, w_, c = s.out_shape
        pt, ph, pw = s.patch
        if not s.refine_enabled:
            return unpatchify(s.head(h), s.patch, (t_out, h_, w_), s.c_out)
        m = s.to_map(h).reshape(bsz * gt, gh, gw, ph, pw, -1)
        m = m.permute(0, 5, 1, 3, 2, 4).reshape(bsz * gt, -1, gh * ph, gw * pw)
        m = mul_add(m, b["ref_a"], b["ref_b"])
        refine_input = torch.cat([m, feat], dim=1)
        # ``unpatchify(head(h)) + _from_groups(refine(...))`` as ONE add over the two permuted
        # views, written into the contiguous (B, T, H, W, C) result: the same sums without the
        # two layout copies. Both views index (b, gt, pt, gh, ph, gw, pw, c).
        co = s.c_out
        hv = s.head(h).reshape(bsz, gt, gh, gw, pt, ph, pw, co).permute(0, 1, 4, 2, 5, 3, 6, 7)
        rv = s.refine(refine_input).reshape(bsz, gt, pt, co, gh, ph, gw, pw)
        rv = rv.permute(0, 1, 2, 4, 5, 6, 7, 3)
        out = self._out(
            "out", (bsz, t_out, h_, w_, co), torch.promote_types(hv.dtype, rv.dtype), x.device
        )
        torch.add(hv, rv, out=out.view(bsz, gt, pt, gh, ph, gw, pw, co))
        return out

    def _out(self, name: str, shape, dtype, device) -> torch.Tensor:
        """A persistent output buffer (its address must survive into the captured graph)."""
        t = self.outs.get(name)
        if t is None or tuple(t.shape) != tuple(shape) or t.dtype != dtype:
            with torch.inference_mode(False):
                t = self.outs[name] = torch.empty(shape, dtype=dtype, device=device)
        return t


_LOG100 = 4.605170185988092  # math.log(100.0), the temperature cap in transformer.Attention


class FastBounded:
    """``BoundedModel.forward`` (frozen backbone, residual wrapper) over :class:`FastSTFormer`.

    ``__call__(x, cond, u_fs)`` has ``call_model``'s signature so the predictor's pipeline can
    use it in place of the model; ``cond`` is ignored there and read from the buffers
    :meth:`set_cond` fills.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.st = FastSTFormer(model.backbone.backbone)

    def set_cond(self, cond: torch.Tensor) -> None:
        self.st.set_cond(cond)

    def __call__(self, x: torch.Tensor, cond: torch.Tensor, u_fs: torch.Tensor | None = None):
        from .bounds import bound_features  # noqa: PLC0415
        from .wrappers import _anchor  # noqa: PLC0415

        with torch.no_grad():
            out = self.st.forward(x)
            pred = out + _anchor(x, out)
        return _bounded_tail(self.model, pred, x, cond, bound_features)


def _bounded_tail(m, pred, x, cond, bound_features):
    """The rest of ``BoundedModel.forward`` after the backbone, unchanged."""
    from .. import N_MEASURED  # noqa: PLC0415
    from .bounds import shrunk_centre  # noqa: PLC0415

    pred = pred.detach()
    if m.state_frames:
        x = x[:, m.state_frames :]
    centre = shrunk_centre(pred, getattr(m, "centre_shrink", 1.0))
    valid = (x[..., :N_MEASURED].abs().sum(dim=(1, -1)) > 0).to(pred.dtype)
    below, above = m.head(
        bound_features(
            centre,
            x,
            valid,
            cond=cond if m.use_cond else None,
            noise=m.noise,
            logs=m.logs,
            coords=m.coords,
            grad=m.grad,
            pool=m.pool,
            fluc=m.fluc,
            lstd=m.lstd,
        )
    )
    offset_below = torch.zeros_like(pred)
    offset_above = torch.zeros_like(pred)
    offset_below[..., :N_MEASURED] = below
    offset_above[..., :N_MEASURED] = above
    out = {"prediction": pred, "lower": centre - offset_below, "upper": centre + offset_above}
    if centre is not pred:
        out["centre"] = centre
    return out


# --------------------------------------------------------------------------------------------
# The whole Track-2 GPU pipeline (bridge -> state frames -> model -> width rule -> second-moment
# head -> evaluator transform), for the configuration ``make_gpu_predict`` serves on the board.
# --------------------------------------------------------------------------------------------

# Multi-output kernels. Every line mirrors one eager op of ``make_gpu_predict._pipeline`` (or of
# the function it calls), named in the comment; the op order and every rounding point are the
# eager ones, so the results are bit-identical, not merely close.
_MULTI_SRC = {
    # The two stream-state pseudo-frames (_pipeline: n, mean_e, var_e, mean_p, sd_p).
    "state": (
        "T a_sum, T a_sq, T a_n, T std2, T mean2, T s, T& mean_p, T& sd_p",
        2,
        """
    float n = fmaxf(float(a_n), 1.0f);
    float me = __fdiv_rn(float(a_sum), n);
    float var = fmaxf(__fsub_rn(__fdiv_rn(float(a_sq), n), __fmul_rn(me, me)), 0.0f);
    mean_p = T(__fdiv_rn(__fadd_rn(__fmul_rn(me, float(std2)), float(mean2)), float(s)));
    sd_p = T(__fdiv_rn(__fmul_rn(sqrtf(var), float(std2)), float(s)));
""",
    ),
    # BoundedModel offsets + the freestream denorm + the width rule's bin index
    # (_pipeline: pred/lo/hi * mul, below, above, half, log_rho, i_rho, flat).
    "width_in": (
        "T pf, T pb, T w, T s, T icl, T inv_sigma, T r0, T inv_step, T nre, T kk,"
        " T& below, T& above, T& flat",
        3,
        """
    float lof = __fmul_rn(__fsub_rn(float(pb), float(w)), float(s));
    float hif = __fmul_rn(__fadd_rn(float(pb), float(w)), float(s));
    float bl = __fsub_rn(pf, lof);
    float ab = __fsub_rn(hif, pf);
    float half = __fmul_rn(0.5f, __fadd_rn(bl, ab));
    float lr = logf(__fmul_rn(fmaxf(half, 1e-30f), float(inv_sigma)));
    float v = __fadd_rn(__fmul_rn(__fsub_rn(lr, float(r0)), float(inv_step)), 1.0f);
    float i = fminf(fmaxf(truncf(v), 0.0f), float(nre));
    below = T(bl); above = T(ab);
    flat = T(__fadd_rn(__fmul_rn(i, float(kk)), float(icl)));
""",
    ),
    # The width rule's multipliers, the ordering guard and the evaluator transform
    # (_pipeline: new_lo, new_hi, minimum/maximum, * _out_scale + _out_shift).
    "width_out": (
        "T p, T below, T above, T alo, T aup, T sc, T sh, T& lo, T& hi",
        2,
        """
    float nl = __fsub_rn(float(p), __fmul_rn(float(below), float(alo)));
    float nh = __fadd_rn(float(p), __fmul_rn(float(above), float(aup)));
    lo = T(__fadd_rn(__fmul_rn(fminf(nl, nh), float(sc)), float(sh)));
    hi = T(__fadd_rn(__fmul_rn(fmaxf(nl, nh), float(sc)), float(sh)));
""",
    ),
    # The second-moment head's two dynamic planes (inference_log_ratio_torch: ke_*_c, sums,
    # ratio, level) from the per-component temporal second moments.
    "ke": (
        "T mp0, T mp1, T mx0, T mx1, T s, T& ratio, T& level",
        2,
        """
    float u2 = __fmul_rn(float(s), float(s));
    float kp = __fadd_rn(__fmul_rn(0.5f, float(mp0)), __fmul_rn(0.5f, float(mp1)));
    float ki = __fadd_rn(__fmul_rn(__fmul_rn(0.5f, float(mx0)), u2),
                         __fmul_rn(__fmul_rn(0.5f, float(mx1)), u2));
    float lp = logf(fmaxf(kp, 1e-12f));
    ratio = T(__fsub_rn(logf(fmaxf(ki, 1e-12f)), lp));
    level = T(lp);
""",
    ),
}

_SINGLE_SRC = {
    # (f - mean_t) ** 2 (flow.kinetic_energy_map_components)
    "sqdev": ("T f, T m", "float d = __fsub_rn(float(f), float(m)); return T(__fmul_rn(d, d));"),
    # log(|v|.clamp_min(LOG_EPS)) (bound_features' log planes)
    "logabs": ("T v", "return T(logf(fmaxf(fabsf(float(v)), 1e-6f)));"),
    # FiLM then GELU: gelu(h * a + b), GELU as ATen's erf form x * 0.5 * (1 + erf(x / sqrt 2))
    # (the FiLM rounds twice, as its mul and add do; the GELU once, in opmath, as ATen's does)
    "film_gelu": (
        "T h, T a, T b",
        "float x = R(__fadd_rn(R(__fmul_rn(float(h), float(a))), float(b)));"
        " return T(R(__fmul_rn(__fmul_rn(x, 0.5f),"
        " __fadd_rn(1.0f, erff(__fmul_rn(x, 0.70710678118654752440f))))));",
    ),
    # BoundHead's logerr planes: h.clamp(LOG_WIDTH_MIN, LOG_WIDTH_MAX).exp()
    "clamp_exp": (
        "T h, T lo, T hi",
        "return T(expf(fminf(fmaxf(float(h), float(lo)), float(hi))));",
    ),
    # valid mask as a float plane: (sum |x| > 0)
    "positive": ("T v", "return T(float(v) > 0.0f ? 1.0f : 0.0f);"),
    # the gain on the fluctuations and the evaluator transform (tke head -> apply_gain_torch ->
    # apply_fluctuation_gain_torch -> * _out_scale + _out_shift)
    "gain_out": (
        "T p, T m, T rh, T ratio, T sc, T sh, T shrink, T cb, T gmax",
        """
    float r = __fadd_rn(R(__fmul_rn(float(shrink), float(rh))), __fmul_rn(float(cb), float(ratio)));
    float g = fminf(fmaxf(expf(__fmul_rn(0.5f, r)), 0.0f), float(gmax));
    float v = __fadd_rn(float(m), __fmul_rn(__fsub_rn(float(p), float(m)), g));
    return T(__fadd_rn(__fmul_rn(v, float(sc)), float(sh)));
""",
    ),
}


def _kernel(name: str, half: bool = False, round_out: bool = False, **scalars: float):
    """Kernel ``name`` from the tables above; ``half`` rounds its intermediates (``R``) to fp16,
    ``round_out`` its results (``RO``, where the kernel's source applies it)."""
    key = (name, half, round_out)
    fn = _JIT.get(key)
    if fn is None:
        from torch.cuda.jiterator import (  # noqa: PLC0415
            _create_jit_fn,
            _create_multi_output_jit_fn,
        )

        rnd = (_ROUND_HALF if half else _ROUND_NONE) + (
            _ROUND_OUT if round_out else _ROUND_OUT_NONE
        )
        if name in _MULTI_SRC:
            sig, n_out, body = _MULTI_SRC[name]
            src = f"template <typename T> void {name}({sig}) {{\n{rnd}{body}\n}}"
            fn = _create_multi_output_jit_fn(src, n_out, **scalars)
        else:
            sig, body = _SINGLE_SRC[name]
            src = f"template <typename T> T {name}({sig}) {{\n{rnd}{body}\n}}"
            fn = _create_jit_fn(src, **scalars)
        _JIT[key] = fn
    return fn


def _half_weight_copy(module: torch.nn.Module) -> torch.nn.Module:
    """A deep copy with every Linear/Conv weight and bias stored in fp16."""
    import copy  # noqa: PLC0415

    out = copy.deepcopy(module)
    for m in out.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            with torch.no_grad():
                m.weight.data = m.weight.data.half()
                if m.bias is not None:
                    m.bias.data = m.bias.data.half()
    return out


def pipeline_supported(
    model, *, lead_anchor, tke_head, tke_head_uses_tau, stream_state, zero_pressure, n_ch
) -> bool:
    """:class:`FastPipeline` reproduces one configuration: the Track-2 board bundle's."""
    from .tke_head import N_BASE_FEATURES  # noqa: PLC0415

    if not supported(model) or lead_anchor is not None or not zero_pressure or n_ch != 2:
        return False
    if stream_state != "meanvar" or model.state_frames != 2:
        return False
    if float(getattr(model, "centre_shrink", 1.0)) != 1.0:
        return False  # the Track-2 pipeline centres its interval on ``pred`` (Track 1 only)
    if tke_head is None or tke_head.per_component:
        return False
    if int(tke_head.body[0].weight.shape[1]) != N_BASE_FEATURES:
        return False
    h = model.head
    if getattr(h, "film", 0) != 1 or getattr(h, "mode", "") != "logerr":
        return False
    body = list(h.body)
    if len(body) % 2 or not all(
        isinstance(c, torch.nn.Conv3d) and isinstance(g, torch.nn.GELU) and g.approximate == "none"
        for c, g in zip(body[::2], body[1::2], strict=True)
    ):
        return False
    if not (model.logs and model.coords and model.grad and model.use_cond):
        return False
    if model.noise or model.pool or model.fluc or model.lstd:
        return False
    return True


class FastPipeline:
    """``make_gpu_predict``'s ``_pipeline`` for :func:`pipeline_supported`'s configuration.

    ``__call__(x_eval, cb, sp, acc)`` returns the same three fp16 arrays, bit for bit. Every
    function of the operating point alone is refreshed by :meth:`set_cond` (the conditioning
    MLP and the FiLM/adaLN maps, the interval head's FiLM, the second-moment head's static
    planes); the model input, the head features and the outputs live in preallocated buffers
    whose constant planes are written once.
    """

    def __init__(
        self,
        model,
        *,
        std_in,
        mean_in,
        out_scale,
        out_shift,
        sigma,
        r0,
        step,
        nre,
        n_lead,
        a_lo,
        a_up,
        i_ch,
        i_lead,
        tke_head,
        blend,
        shrink,
        fs_rows,
        half_weights: bool = False,
        check_finite: bool = False,
    ) -> None:
        import numpy as np  # noqa: PLC0415

        from .bounds import LOG_WIDTH_MAX, LOG_WIDTH_MIN  # noqa: PLC0415
        from .fluctuations import DEFAULT_MAX_GAIN  # noqa: PLC0415

        if half_weights:
            # Under fp16 autocast every linear/conv casts its fp32 weight and bias to fp16 on
            # every call: 111 copy kernels per step on the board model (FINDINGS 218). A copy of
            # the model whose linear/conv parameters are ALREADY fp16 makes each of those casts
            # a no-op; the values are the same ``.half()`` rounding autocast applies. Norms and
            # the other parameters stay fp32, as autocast runs them. The reference pipeline keeps
            # the original fp32 model, which the fp16 finiteness fallback also relies on.
            model, tke_head = _half_weight_copy(model), _half_weight_copy(tke_head)
        self.model, self.st = model, FastSTFormer(model.backbone.backbone)
        self.head, self.tke = model.head, tke_head
        self.std_in, self.mean_in = std_in, mean_in
        self.std2, self.mean2 = std_in[..., :2], mean_in[..., :2]
        self.sc2, self.sh2 = out_scale[:2].contiguous(), out_shift[:2].contiguous()
        self.a_lo, self.a_up = a_lo, a_up
        self.icl = (i_ch + i_lead).float()  # (1, T, 1, 1, 2), small integers
        f32 = np.float32
        # Division by a python float is a multiply by its fp32 reciprocal in ATen's CUDA kernel.
        self.width_k = {
            "inv_sigma": float(f32(1.0) / f32(sigma)),
            "r0": float(f32(r0)),
            "inv_step": float(f32(1.0) / f32(step)),
            "nre": float(nre),
            "kk": float(2 * n_lead),
        }
        self.gain_k = {
            "shrink": float(shrink),
            "cb": float((1.0 - shrink) * blend),
            "gmax": float(DEFAULT_MAX_GAIN),
        }
        self.clamp_k = {"lo": float(LOG_WIDTH_MIN), "hi": float(LOG_WIDTH_MAX)}
        self.fs_rows = fs_rows
        self.check_finite = check_finite
        self.nf_div = 820.0  # the state count's saturation, as _pipeline spells it
        self.buf: dict[str, torch.Tensor] | None = None
        self.film: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._cond_graph: Any = None  # None: not tried; False: failed; else the CUDAGraph
        self._cond_in: tuple[torch.Tensor, torch.Tensor] | None = None

    # --- per operating point -------------------------------------------------------------
    def _alloc(self, x_eval: torch.Tensor) -> None:
        b, t, h, w, c = x_eval.shape
        dev = x_eval.device
        with torch.inference_mode(False):
            buf = {
                "xin": torch.zeros(b, t + 2, h, w, c, device=dev),
                "feat": torch.zeros(b, 20, t, h, w, device=dev),
                "tke": torch.zeros(b, 7, h, w, device=dev),
                # prediction, lower, upper as ONE buffer: the predictor copies it to the host in
                # one transfer without stacking the three first (``stacked``)
                "out3": torch.zeros(3, b, t, h, w, c, device=dev, dtype=torch.float16),
                "finite": torch.ones((), device=dev, dtype=torch.bool),
            }
        buf["pred"], buf["lo"], buf["hi"] = buf["out3"].unbind(0)
        fv = buf["feat"].permute(0, 2, 3, 4, 1)
        # The constant planes, by bound_features' own expressions.
        lead = torch.linspace(0.0, 1.0, t, device=dev, dtype=torch.float32)
        fv[..., 8] = lead.view(1, t, 1, 1)
        rows = torch.linspace(-1.0, 1.0, h, device=dev, dtype=torch.float32)
        cols = torch.linspace(-1.0, 1.0, w, device=dev, dtype=torch.float32)
        fv[..., 16] = rows.view(1, 1, h, 1)
        fv[..., 17] = cols.view(1, 1, 1, w)
        self.buf = buf

    @torch.no_grad()
    def set_cond(self, cb: torch.Tensor, sp: torch.Tensor, like: torch.Tensor) -> None:
        """Refresh every conditioning-derived buffer for a new operating point.

        The first call runs eagerly (it allocates); it then captures the same computation as
        its own small CUDA graph, so the later refreshes -- one per new operating-point
        estimate, i.e. the first few steps of every trajectory -- are one launch instead of
        some forty. Any capture failure keeps the eager refresh.
        """
        if self._cond_graph is None or self._cond_graph is False:
            self._set_cond_eager(cb, sp, like)
            if self._cond_graph is None and cb.is_cuda:
                self._capture_cond(cb, sp)
            return
        self._cond_in[0].copy_(cb)
        self._cond_in[1].copy_(sp)
        self._cond_graph.replay()

    def _capture_cond(self, cb: torch.Tensor, sp: torch.Tensor) -> None:
        try:
            with torch.inference_mode(False):
                cin = (cb.detach().clone(), sp.detach().clone())
            cur = torch.cuda.current_stream()
            side = torch.cuda.Stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._set_cond_eager(*cin, None)
            cur.wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._set_cond_eager(*cin, None)
            self._cond_graph, self._cond_in = g, cin
        except Exception:  # noqa: BLE001 - the eager refresh is always correct
            self._cond_graph = False

    @torch.no_grad()
    def _set_cond_eager(self, cb: torch.Tensor, sp: torch.Tensor, like) -> None:
        if self.buf is None:
            self._alloc(like)
        self.st.set_cond(cb)
        hd = self.head
        cond = cb.to(torch.float32)
        vals = []
        for proj in hd.film_proj:
            scale, shift = proj(cond).chunk(2, dim=-1)
            vals.append(((1.0 + scale[:, :, None, None, None]), shift[:, :, None, None, None]))
        if self.film is None or any(
            (a.dtype, a.shape) != (va.dtype, va.shape)
            for (a, _), (va, _) in zip(self.film, vals, strict=True)
        ):
            with torch.inference_mode(False):
                self.film = [(a.clone(), s.clone()) for a, s in vals]
        else:
            for (a, s), (va, vs) in zip(self.film, vals, strict=True):
                a.copy_(va)
                s.copy_(vs)
        self.buf["tke"][:, 3:].copy_(sp.to(torch.float32))

    # --- per step ------------------------------------------------------------------------
    def __call__(self, x_eval, cb, sp, acc):
        from .bounds import log_grad_energy  # noqa: PLC0415

        b = self.buf
        xin, fv = b["xin"], b["feat"].permute(0, 2, 3, 4, 1)
        # bridge -> freestream -> normalise, into the model-input buffer after the state frames
        x_phys = _mul_add(x_eval, self.std_in, self.mean_in)
        s = x_phys[:, :, : self.fs_rows, :, 0].abs().mean(dim=(1, 2, 3)).clamp_min(1e-6)
        mul = s.view(-1, 1, 1, 1, 1)
        torch.div(x_phys, mul, out=xin[:, 2:])
        # the two stream-state pseudo-frames
        a_sum, a_sq, a_n = acc[0], acc[1], acc[2]
        mean_p, sd_p = _kernel("state")(a_sum, a_sq, a_n, self.std2, self.mean2, s.view(1, 1, 1))
        xin[:, 0, ..., :2].copy_(mean_p)
        xin[:, 1, ..., :2].copy_(sd_p)
        xin[:, 0, ..., 2].copy_((a_n / self.nf_div).clamp_max(1.0))
        # the model
        with torch.no_grad():
            out = self.st.forward(xin)
            pred = out + xin[:, -1:]
        x = xin[:, 2:]
        pb2 = pred[..., :2]
        # interval-head features, written into their channel-first planes
        fv[..., 0:2].copy_(pb2)
        torch.sub(pb2, x[:, -1:, ..., :2], out=fv[..., 2:4])
        psd = pb2.std(dim=1, keepdim=True)
        osd = x[..., :2].std(dim=1, keepdim=True)
        fv[..., 4:6].copy_(psd)
        fv[..., 6:8].copy_(osd)
        vsum = x[..., :2].abs().sum(dim=(1, -1))
        valid = _kernel("positive")(vsum)
        fv[..., 9].copy_(valid[:, None])
        logabs = _kernel("logabs")
        fv[..., 10:12].copy_(logabs(fv[..., 2:4]))
        fv[..., 12:14].copy_(logabs(psd))
        fv[..., 14:16].copy_(logabs(osd))
        fv[..., 18:19].copy_(log_grad_energy(pb2))
        fv[..., 19:20].copy_(log_grad_energy(x[:, -1:, ..., :2]))
        # the interval head (FiLM on each body conv, then the logerr planes)
        hd, film = self.head, self.film
        hh = b["feat"]
        i = 0
        for layer in hd.body:
            if isinstance(layer, torch.nn.Conv3d):
                hh = layer(hh)
                a, sh = film[i]
                hh = _kernel("film_gelu", hh.dtype == torch.float16)(hh, a, sh)
                i += 1
        hh = hd.out(hh).permute(0, 2, 3, 4, 1)
        if hh.dtype == torch.float16:
            # Under autocast ``exp`` runs in fp32 on the fp16 clamp: keep the eager spelling.
            wdt = hh.clamp(self.clamp_k["lo"], self.clamp_k["hi"]).exp()
        else:
            wdt = _kernel("clamp_exp", **self.clamp_k)(hh, **self.clamp_k)
        # the freestream denorm and the width rule
        predf = pred.float() * mul
        pf2 = predf[..., :2]
        below, above, flat = _kernel("width_in", **self.width_k)(
            pf2, pb2, wdt, mul, self.icl, **self.width_k
        )
        flat = flat.long()
        lo2, hi2 = _kernel("width_out")(
            pf2, below, above, self.a_lo[flat], self.a_up[flat], self.sc2, self.sh2
        )
        b["lo"][..., :2].copy_(lo2)
        b["hi"][..., :2].copy_(hi2)
        # the second-moment head
        xd = x[..., :2]
        m2p = _kernel("sqdev")(pf2, pf2.mean(dim=-4, keepdim=True)).mean(dim=-4)
        m2x = _kernel("sqdev")(xd, xd.mean(dim=-4, keepdim=True)).mean(dim=-4)
        ratio, level = _kernel("ke")(
            m2p[..., 0], m2p[..., 1], m2x[..., 0], m2x[..., 1], s.view(-1, 1, 1)
        )
        tf = b["tke"]
        tf[:, 0].copy_(ratio)
        torch.sub(level, level.mean(dim=(1, 2), keepdim=True), out=tf[:, 1])
        tf[:, 2].copy_(valid)
        rh = self.tke(tf)
        mean_t = pf2.mean(dim=-4, keepdim=True)
        pg = _kernel("gain_out", rh.dtype == torch.float16, **self.gain_k)(
            pf2,
            mean_t,
            rh[:, None, :, :, None],
            ratio[:, None, :, :, None],
            self.sc2,
            self.sh2,
            **self.gain_k,
        )
        b["pred"][..., :2].copy_(pg)
        if self.check_finite:
            # The fp16 guard's reduction, inside the graph: the predictor reads this flag after
            # its one device-to-host copy instead of synchronising for it first.
            torch.all(torch.isfinite(b["out3"]), out=b["finite"])
        return b["pred"], b["lo"], b["hi"]

    def finite_flag(self, pred, lo, hi) -> torch.Tensor | None:
        """This step's in-graph finiteness flag, if (pred, lo, hi) are this pipeline's output."""
        if not self.check_finite or self.stacked(pred, lo, hi) is None:
            return None
        return self.buf["finite"]

    def stacked(self, pred, lo, hi) -> torch.Tensor | None:
        """``torch.stack((pred, lo, hi))`` without the copy, when they are this pipeline's own
        output buffer (the graph replays into it); ``None`` otherwise."""
        b = self.buf
        if b is None or not (pred is b["pred"] and lo is b["lo"] and hi is b["hi"]):
            return None
        return b["out3"]


# --------------------------------------------------------------------------------------------
# Track 1: the same model and heads behind ``make_predict``'s batched, eager ``chunk``.
# --------------------------------------------------------------------------------------------

_T1_SRC = {
    # The width rule's multipliers and the ordering guard, in physical units (no evaluator
    # transform on Track 1): WidthRule.apply_torch's new_lo/new_hi/minimum/maximum, then
    # make_predict's own minimum/maximum guard, which is the identity on its output.
    # (``RO`` on the results, with round_out=True: the fp16-rounded values the fp16 output
    # blocks would hold, stored in fp32)
    "width_out_plain": (
        "T p, T below, T above, T alo, T aup, T& lo, T& hi",
        2,
        """
    float nl = __fsub_rn(float(p), __fmul_rn(float(below), float(alo)));
    float nh = __fadd_rn(float(p), __fmul_rn(float(above), float(aup)));
    lo = T(RO(fminf(nl, nh)));
    hi = T(RO(fmaxf(nl, nh)));
""",
    ),
}
_T1_SINGLE = {
    # the interval head's conv bias, FiLM and exact GELU in one pass: the bias add rounds as
    # ``add_`` does, then ``film_gelu`` unchanged
    "bias_film_gelu": (
        "T h, T c, T a, T b",
        "float x = R(__fadd_rn(float(h), float(c))); "
        "x = R(__fadd_rn(R(__fmul_rn(x, float(a))), float(b))); "
        "return T(R(__fmul_rn(__fmul_rn(x, 0.5f),"
        " __fadd_rn(1.0f, erff(__fmul_rn(x, 0.70710678118654752440f))))));",
    ),
    # the output conv's bias, the log-width clamp and the exp. ``lo``/``hi`` are float32
    # tensors so that T is float32 and the exp lands in fp32, as autocast computes the
    # reference's ``exp`` (``exp`` is on autocast's fp32 list); the add and the clamp round to
    # fp16 as the reference's fp16 ``add_``/``clamp`` do.
    "bias_clamp_exp": (
        "T h, T c, T lo, T hi",
        "float x = R(__fadd_rn(float(h), float(c))); "
        "x = R(fminf(fmaxf(x, float(lo)), float(hi))); return T(expf(x));",
    ),
    # apply_gain_torch on the physical prediction, no transform after it. ``RO`` on the result
    # (round_out=True) stores the fp16-rounded value the fp16 output block would hold, in fp32.
    "gain_plain": (
        "T p, T m, T rh, T ratio, T shrink, T cb, T gmax",
        """
    float r = __fadd_rn(R(__fmul_rn(float(shrink), float(rh))), __fmul_rn(float(cb), float(ratio)));
    float g = fminf(fmaxf(expf(__fmul_rn(0.5f, r)), 0.0f), float(gmax));
    return T(RO(__fadd_rn(float(m), __fmul_rn(__fsub_rn(float(p), float(m)), g))));
""",
    ),
    # log(|p - x|.clamp_min(LOG_EPS)): the departure's log plane from its fp32 sources (the
    # departure plane itself is written in the head's dtype, so it is not read back for this)
    "logabs_sub": (
        "T p, T x",
        "return T(logf(fmaxf(fabsf(__fsub_rn(float(p), float(x))), 1e-6f)));",
    ),
    # bounds.log_grad_energy per element: the forward differences along rows and columns of
    # the two components, squared and summed per axis (``pow(2).sum(-1)`` of two terms), the
    # two axes added, clamped and logged -- the reference's ops in the reference's order. The
    # caller hands the field replicated one row and one column past its edge, so the last
    # row's and column's differences are the exact zeros the reference pads with.
    "grad_energy": (
        "T f0, T f0h, T f0w, T f1, T f1h, T f1w",
        """
    float a = __fsub_rn(float(f0h), float(f0)); float b = __fsub_rn(float(f1h), float(f1));
    float eh = __fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b));
    float c = __fsub_rn(float(f0w), float(f0)); float d = __fsub_rn(float(f1w), float(f1));
    float ew = __fadd_rn(__fmul_rn(c, c), __fmul_rn(d, d));
    return T(logf(fmaxf(__fadd_rn(eh, ew), 1e-6f)));
""",
    ),
}
_MULTI_SRC.update(_T1_SRC)
_SINGLE_SRC.update(_T1_SINGLE)


def t1_supported(model, *, tke_head, width_rule, lead_anchor, uses_tau) -> bool:
    """:class:`FastT1` reproduces ``make_predict``'s chunk for the Track-1 board bundle."""
    from .tke_head import N_BASE_FEATURES  # noqa: PLC0415

    if not supported(model) or lead_anchor is not None or model.state_frames or uses_tau:
        return False
    if width_rule is None or not getattr(width_rule, "is_uniform", False):
        return False
    if tke_head is None or tke_head.per_component:
        return False
    if int(tke_head.body[0].weight.shape[1]) != N_BASE_FEATURES:
        return False
    h = model.head
    if getattr(h, "film", 0) != 1 or getattr(h, "mode", "") != "logerr":
        return False
    body = list(h.body)
    if len(body) % 2 or not all(
        isinstance(c, torch.nn.Conv3d) and isinstance(g, torch.nn.GELU) and g.approximate == "none"
        for c, g in zip(body[::2], body[1::2], strict=True)
    ):
        return False
    if not all(_plain_conv(c) for c in [*body[::2], h.out]):
        return False
    if not (model.logs and model.coords and model.grad and model.use_cond):
        return False
    return not (model.noise or model.pool or model.fluc or model.lstd)


def _plain_conv(c) -> bool:
    """A biased, stride-1, dilation-1, ungrouped, zero-padded conv: what ``run`` spells out."""
    return (
        isinstance(c, torch.nn.Conv3d)
        and c.bias is not None
        and all(v == 1 for v in c.stride)
        and all(v == 1 for v in c.dilation)
        and c.groups == 1
        and c.padding_mode == "zeros"
    )


def _grad_energy(f: torch.Tensor) -> torch.Tensor:
    """``bounds.log_grad_energy`` of ``f[..., :2]`` in two kernels instead of thirteen: the
    field replicated one row and one column past its edge (so the forward differences there
    are exactly the zero differences the reference pads with), then one fused kernel per
    element -- the same subtractions, squares, sums, clamp and log in the reference's order."""
    h, w = f.shape[-3], f.shape[-2]
    fp = F.pad(f, (0, 0, 0, 1, 0, 1), mode="replicate")
    f0, f1 = fp[..., 0], fp[..., 1]
    return _kernel("grad_energy")(
        f0[..., :h, :w],
        f0[..., 1:, :w],
        f0[..., :h, 1:],
        f1[..., :h, :w],
        f1[..., 1:, :w],
        f1[..., :h, 1:],
    )


class FastT1:
    """One Track-1 chunk -- forward, width rule, guard, second-moment gain, zero pressure --
    with the same outputs as ``make_predict``'s ``chunk`` on its ``cond_device`` path, bit for
    bit, in fewer kernels. Every window has its own operating point and every call is a fresh
    process on the board, so nothing is cached across calls: the conditioning-derived values
    are computed per call and used in place, with no persistent buffers.

    ``centre_shrink`` (``BoundedModel.centre_shrink``, FINDINGS 196i): the interval head reads
    and the width rule centres on ``m + c * (pred - m)`` while the reported field stays
    ``pred``. Spelled with the same torch ops, on the same ``[..., :2]`` views, as
    ``bounds.shrunk_centre`` -- a mean over dim 1 and two element-wise ops -- so the chunk
    stays bit-identical; the layout matters because the head's ``std`` reduction reads the
    result, and a reduction's summation order follows the tensor's strides.
    """

    def __init__(
        self,
        model,
        *,
        width_rule,
        tke_head,
        blend,
        shrink,
        uses_tau,
        half_weights,
        centre_shrink: float = 1.0,
        stack_cond: bool | None = None,
        cl_head: bool | None = None,
        fuse_bias: bool | None = None,
    ) -> None:
        import numpy as np  # noqa: PLC0415

        from .bounds import LOG_WIDTH_MAX, LOG_WIDTH_MIN  # noqa: PLC0415
        from .fluctuations import DEFAULT_MAX_GAIN  # noqa: PLC0415

        if half_weights:
            # The model runs under autocast; the second-moment head does not (fp32 weights).
            model = _half_weight_copy(model)
        self.stack_cond = bool(half_weights if stack_cond is None else stack_cond)
        self.cl_head = bool(half_weights if cl_head is None else cl_head)
        self.fuse_bias = bool(half_weights if fuse_bias is None else fuse_bias)
        self.model = model
        self.st = FastSTFormer(model.backbone.backbone, stack_cond=self.stack_cond)
        self.head, self.tke = model.head, tke_head
        self.rule = width_rule
        self.uses_tau = uses_tau
        f32 = np.float32
        from .. import SIGMA_GLOBAL  # noqa: PLC0415

        edges = np.asarray(width_rule.rho_edges)
        self.width_k = {
            "inv_sigma": float(f32(1.0) / f32(float(SIGMA_GLOBAL))),
            "r0": float(f32(float(edges[0]))),
            "inv_step": float(f32(1.0) / f32(float(edges[1] - edges[0]))),
            "nre": float(len(edges)),
            "kk": float(width_rule.a_lo.shape[1] * width_rule.a_lo.shape[2]),
        }
        self.gain_k = {
            "shrink": float(shrink),
            "cb": float((1.0 - shrink) * blend),
            "gmax": float(DEFAULT_MAX_GAIN),
        }
        self.clamp_k = {"lo": float(LOG_WIDTH_MIN), "hi": float(LOG_WIDTH_MAX)}
        self._clamp_t: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.centre_shrink = float(centre_shrink)
        self.dev_tables: dict[Any, Any] = {}
        # The interval head's convolutions with their weights in the channels-last layout
        # cuDNN's fp16 kernels read natively (the head's features are written channel-last
        # too), so no layout copies surround each conv; the bias is added inside the fused
        # FiLM/GELU kernel. Same kernel, same sums (FINDINGS 236). Copies of the weights.
        hd = model.head
        mf = torch.channels_last_3d if self.cl_head else torch.contiguous_format
        with torch.no_grad():
            self.head_convs = [
                (c.weight.detach().contiguous(memory_format=mf), c.bias.detach(), c.padding)
                for c in hd.body
                if isinstance(c, torch.nn.Conv3d)
            ]
            self.head_out = (
                hd.out.weight.detach().contiguous(memory_format=mf),
                hd.out.bias.detach(),
                hd.out.padding,
            )
            # the FiLM projections stacked the same way as the backbone's (fp16 only, as there)
            self._w_fp = torch.cat([p.weight.detach() for p in hd.film_proj], 0)
            self._b_fp = torch.cat([p.bias.detach() for p in hd.film_proj], 0)
            self._n_fp = [p.out_features for p in hd.film_proj]

    def _clamp_tensors(self, dev) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(dev)
        if key not in self._clamp_t:
            # shape (1,), not 0-d: a 0-d tensor does not lift the promoted dtype to float32
            self._clamp_t[key] = tuple(
                torch.tensor([v], dtype=torch.float32, device=dev)
                for v in (self.clamp_k["lo"], self.clamp_k["hi"])
            )
        return self._clamp_t[key]

    def _tables(self, dev, t: int):
        key = (str(dev), t)
        tabs = self.dev_tables.get(key)
        if tabs is None:
            import numpy as np  # noqa: PLC0415

            r = self.rule
            n_ch, n_lead = r.a_lo.shape[1], r.a_lo.shape[2]
            lead = torch.arange(t, device=dev)
            edges = torch.as_tensor(np.asarray(r.lead_edges), device=dev, dtype=lead.dtype)
            i_lead = torch.bucketize(lead, edges).view(1, t, 1, 1, 1)
            i_ch = (torch.arange(n_ch, device=dev) * n_lead).view(1, 1, 1, 1, n_ch)
            tabs = self.dev_tables[key] = (
                (i_ch + i_lead).float(),
                torch.as_tensor(r.a_lo.reshape(-1), device=dev, dtype=torch.float32),
                torch.as_tensor(r.a_up.reshape(-1), device=dev, dtype=torch.float32),
            )
        return tabs

    @torch.no_grad()
    def run(self, xb, cb, ub, cond_np, autocast_dtype, out_dtype, round16=False):
        """``xb`` the freestream-normalised window, ``cb`` its conditioning, ``ub`` the
        freestream speeds, ``cond_np`` the same conditioning on the host (the second-moment
        head's static planes are built from it, as the reference builds them). Returns the
        three stored blocks in ``out_dtype`` with pressure zeroed, or ``None`` when the fp16
        forward is non-finite (the caller then recomputes in fp32 as the reference does).
        ``round16``: float32 blocks instead, holding exactly the values the fp16 blocks would
        hold (rounded to fp16 inside the last kernels), for a caller that wants the fp16
        results as float32 without a second pass over them."""
        from .tke_head import tke_static_planes  # noqa: PLC0415

        amp = autocast_dtype is not None
        ctx = torch.autocast("cuda", dtype=autocast_dtype) if amp else _nullctx()
        n, t, hh_, ww_, _ = xb.shape
        dev = xb.device
        with ctx:
            # the model (BoundedModel.forward on the residual STFormer)
            self.st.buf = self.st._cond_values(cb)
            out = self.st.forward(xb)
            pred = out + xb[:, -1:]
            pb2 = pred[..., :2]
            if self.centre_shrink != 1.0:
                # bounds.shrunk_centre, op for op and view for view (see the class docstring)
                m = pb2.mean(dim=1, keepdim=True)
                centre = pred.clone()
                centre[..., :2] = m + self.centre_shrink * (pb2 - m)
                cb2 = centre[..., :2]
            else:
                centre, cb2 = pred, pb2
            # The head's 20 feature planes, in the dtype the head's convs read under autocast:
            # every plane is computed in fp32 exactly as the reference computes it and rounds
            # once, on its write, as the reference's autocast cast of the fp32 volume rounds
            # it; a plane derived from another plane reads the fp32 sources, not the plane.
            fdt = autocast_dtype if amp else pred.dtype
            feat = torch.empty(n, 20, t, hh_, ww_, device=dev, dtype=fdt)
            fv = feat.permute(0, 2, 3, 4, 1)
            fv[..., 0:2].copy_(cb2)
            x_last = xb[:, -1:, ..., :2]
            torch.sub(cb2, x_last, out=fv[..., 2:4])
            psd = cb2.std(dim=1, keepdim=True)
            osd = xb[..., :2].std(dim=1, keepdim=True)
            fv[..., 4:6].copy_(psd)
            fv[..., 6:8].copy_(osd)
            lead = torch.linspace(0.0, 1.0, t, device=dev, dtype=pred.dtype)
            fv[..., 8] = lead.view(1, t, 1, 1)
            vsum = xb[..., :2].abs().sum(dim=(1, -1))
            valid = _kernel("positive")(vsum)
            fv[..., 9].copy_(valid[:, None])
            logabs = _kernel("logabs")
            fv[..., 10:12].copy_(_kernel("logabs_sub")(cb2, x_last))
            fv[..., 12:14].copy_(logabs(psd))
            fv[..., 14:16].copy_(logabs(osd))
            rows = torch.linspace(-1.0, 1.0, hh_, device=dev, dtype=pred.dtype)
            cols = torch.linspace(-1.0, 1.0, ww_, device=dev, dtype=pred.dtype)
            fv[..., 16] = rows.view(1, 1, hh_, 1)
            fv[..., 17] = cols.view(1, 1, 1, ww_)
            fv[..., 18].copy_(_grad_energy(centre))
            fv[..., 19].copy_(_grad_energy(xb[:, -1:]))
            # the interval head, FiLM from the (fp32) conditioning columns: the convs on the
            # channels-last features with the bias inside the FiLM/GELU kernel (``__init__``)
            cond = cb.to(pred.dtype)
            stack = self.stack_cond and n >= 2  # as in FastSTFormer._cond_values
            if stack:
                fp = F.linear(cond, self._w_fp, self._b_fp)
                fp1 = 1.0 + fp
            if self.cl_head:
                # one layout copy here instead of cuDNN's around every conv; written plane by
                # plane above (contiguous writes), read channel-last by the convs below
                feat = feat.contiguous(memory_format=torch.channels_last_3d)
            h = feat
            o = 0
            for i, ((w, bias, pad), n_out) in enumerate(
                zip(self.head_convs, self._n_fp, strict=True)
            ):
                half = n_out // 2
                if stack:
                    a = fp1[:, o : o + half, None, None, None]
                    sh = fp[:, o + half : o + n_out, None, None, None]
                    o += n_out
                else:
                    scale, shift = self.head.film_proj[i](cond).chunk(2, dim=-1)
                    a, sh = 1.0 + scale[:, :, None, None, None], shift[:, :, None, None, None]
                # under autocast the conv's output is fp16 whatever the features' dtype; the
                # fused kernels round their intermediates to fp16 exactly then
                if self.fuse_bias:
                    h = F.conv3d(h, w, None, padding=pad)
                    half16 = h.dtype == torch.float16
                    h = _kernel("bias_film_gelu", half16)(h, bias.view(1, -1, 1, 1, 1), a, sh)
                else:
                    h = F.conv3d(h, w, bias, padding=pad)
                    h = _kernel("film_gelu", h.dtype == torch.float16)(h, a, sh)
            w, bias, pad = self.head_out
            if self.fuse_bias:
                h = F.conv3d(h, w, None, padding=pad).permute(0, 2, 3, 4, 1)
                lo_t, hi_t = self._clamp_tensors(dev)
                half16 = h.dtype == torch.float16
                wdt = _kernel("bias_clamp_exp", half16)(h, bias.view(1, 1, 1, 1, -1), lo_t, hi_t)
            else:
                h = F.conv3d(h, w, bias, padding=pad).permute(0, 2, 3, 4, 1)
                if h.dtype == torch.float16:
                    wdt = h.clamp(self.clamp_k["lo"], self.clamp_k["hi"]).exp()
                else:
                    wdt = _kernel("clamp_exp", **self.clamp_k)(h, **self.clamp_k)
        # the fp16 guard: the model's three outputs finite, as ``chunk`` checks them
        if amp and not bool(
            torch.stack((torch.isfinite(pred).all(), torch.isfinite(wdt).all())).all()
        ):
            return None
        # back to fp32 and physical units, then the width rule -- about the interval's centre
        # (the shrunk field when a centre shrink is set, else the prediction)
        mul = ub.view(-1, 1, 1, 1, 1)
        predf = pred.float() * mul
        pf2 = predf[..., :2]
        cf2 = pf2 if centre is pred else (centre.float() * mul)[..., :2]
        icl, a_lo, a_up = self._tables(dev, t)
        below, above, flat = _kernel("width_in", **self.width_k)(
            cf2, cb2, wdt, mul, icl, **self.width_k
        )
        flat = flat.long()
        r16 = bool(round16)
        lo2, hi2 = _kernel("width_out_plain", round_out=r16)(
            cf2, below, above, a_lo[flat], a_up[flat]
        )
        # the second-moment head (static planes from the host conditioning, as the reference)
        xd = xb[..., :2]
        mean_t = pf2.mean(dim=-4, keepdim=True)
        m2p = _kernel("sqdev")(pf2, mean_t).mean(dim=-4)
        m2x = _kernel("sqdev")(xd, xd.mean(dim=-4, keepdim=True)).mean(dim=-4)
        ratio, level = _kernel("ke")(
            m2p[..., 0], m2p[..., 1], m2x[..., 0], m2x[..., 1], ub.view(-1, 1, 1)
        )
        sp = tke_static_planes(cond_np, hh_, ww_, dev, uses_tau=self.uses_tau)
        tf = torch.empty(n, 7, hh_, ww_, device=dev, dtype=torch.float32)
        tf[:, 0].copy_(ratio)
        torch.sub(level, level.mean(dim=(1, 2), keepdim=True), out=tf[:, 1])
        tf[:, 2].copy_(valid)
        tf[:, 3:].copy_(sp)
        rh = self.tke(tf)  # outside autocast, as ``rescale_reported`` runs it
        pg = _kernel("gain_plain", round_out=r16, **self.gain_k)(
            pf2, mean_t, rh[:, None, :, :, None], ratio[:, None, :, :, None], **self.gain_k
        )
        # the three stored blocks, pressure zero
        odt = torch.float32 if r16 else out_dtype
        outs = torch.zeros(3, n, t, hh_, ww_, 3, device=dev, dtype=odt)
        outs[0, ..., :2].copy_(pg)
        outs[1, ..., :2].copy_(lo2)
        outs[2, ..., :2].copy_(hi2)
        return outs[0], outs[1], outs[2]


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
