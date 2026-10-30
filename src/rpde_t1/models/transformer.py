"""Two space-time transformers: a naive baseline and a conditioned encoder-decoder.

Both consume the whole 20-frame window as one sequence of space-time patches, which
is the point: every other model here (CNO, FNO, U-Net) stacks time into channels or
convolves over it, so nothing in the benchmark has ever asked what happens when the
window is treated as a sequence. The task argues for it -- the system is partially
observed (no ``w``, no ``p``, unknown shedding phase), so a single frame is not a
Markovian state and the 20 observed frames are a state-estimation device.

Why two of them, sharing a tokenizer
------------------------------------
:class:`PatchTransformer` is the honest control: patchify, add a learned position
embedding, run a plain pre-LN encoder, project every token back to its patch. No
conditioning, no decoder, no anchor. Whatever it scores is what "a transformer" is
worth on this task before anyone tunes anything.

:class:`MOCPT` is the same tokenizer with the two ideas the SimFormer/NeuroPCT line
contributes, and nothing else:

1. **adaLN-Zero conditioning.** The operating point ``(log-Re, AoA, domain)`` drives
   per-block scale/shift/gate rather than being concatenated as three extra input
   channels. Under freestream non-dimensionalisation nothing else in the features
   carries physical scale, and the one place this project has already measured that
   argument -- conditioning the interval head on the same vector -- was worth +1.52
   SPS at 7.6x its own noise threshold. Gates start at zero, so the model begins
   life exactly as its unconditioned twin and conditioning has to earn its way in.
2. **Learned output queries with cross-attention.** The encoder summarises the
   observed window; a separate set of query tokens, one per output patch and carrying
   an explicit lead-time embedding, reads that summary. This is the grid analogue of
   NeuroPCT's sampled query points and pooled output heads, and it decouples "how
   many frames were observed" from "how many are predicted" -- the encoder is not
   forced to also be the decoder, which is what the naive model does by construction.

Keeping the tokenizer, width, depth budget and training recipe identical between the
two means a difference between them is attributable to those two ideas, and nothing
else. That is the whole reason they live in one file.

Sizing follows ``unet3d.py`` (6.02M) rather than the kit baselines, so the three
"ours" models are within a factor of ~1.5 of each other on parameters and the
comparison at matched wall-clock stays interpretable.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..features import COND_MODES
from ..features import cond_dim as cond_vec_dim

#: ``(time, height, width)`` patch. At the scored geometry (20, 32, 64) this gives a
#: 5 x 8 x 8 = 320-token sequence, against the 40,960 tokens Transolver keeps -- which
#: is why Transolver needs fp32 at micro-batch 2 and this does not.
DEFAULT_PATCH = (4, 4, 8)


def _grid(shape: tuple[int, int, int], patch: tuple[int, int, int]) -> tuple[int, int, int]:
    """Patch-grid extent, refusing a patch that does not tile the field."""
    for name, s, p in zip(("T", "H", "W"), shape, patch, strict=True):
        if s % p:
            raise ValueError(
                f"patch {patch} does not tile {shape}: {name}={s} is not a multiple of {p}"
            )
    return tuple(s // p for s, p in zip(shape, patch, strict=True))  # type: ignore[return-value]


def patchify(x: torch.Tensor, patch: tuple[int, int, int]) -> torch.Tensor:
    """``(B, T, H, W, C) -> (B, N, patch_volume * C)``, N in (t, h, w) raster order."""
    b, t, h, w, c = x.shape
    pt, ph, pw = patch
    gt, gh, gw = _grid((t, h, w), patch)
    x = x.reshape(b, gt, pt, gh, ph, gw, pw, c)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)  # (B, gt, gh, gw, pt, ph, pw, C)
    return x.reshape(b, gt * gh * gw, pt * ph * pw * c)


def unpatchify(
    tokens: torch.Tensor, patch: tuple[int, int, int], shape: tuple[int, int, int], channels: int
) -> torch.Tensor:
    """Inverse of :func:`patchify`, back to ``(B, T, H, W, C)``."""
    b = tokens.shape[0]
    pt, ph, pw = patch
    gt, gh, gw = _grid(shape, patch)
    x = tokens.reshape(b, gt, gh, gw, pt, ph, pw, channels)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)  # (B, gt, pt, gh, ph, gw, pw, C)
    return x.reshape(b, gt * pt, gh * ph, gw * pw, channels)


def sinusoidal_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal features for an integer position vector."""
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    angles = positions.float()[:, None] * freqs[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Attention(nn.Module):
    """Multi-head attention, optionally cross-attending to a separate memory.

    ``qk_norm`` unit-normalises the per-head query and key before the dot product,
    which bounds the attention logits and is the standard cure for the loss spike a
    growing query-key norm produces. It is **off by default** so that every number
    already in FINDINGS 9 stays reproducible from this class; the spike it fixes was
    measured here, on the ``transformer`` control at lr 1e-3 (field loss 0.0046 ->
    0.0377 at step 7,500, recovering over ~1,500 steps -- an eval that lands inside
    that window reads the architecture as far worse than it is).
    """

    def __init__(self, dim: int, heads: int, qk_norm: bool = False) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.qk_norm = qk_norm
        if qk_norm:
            # A learned per-head temperature restores the logit scale that normalising
            # to the unit sphere removes, starting where the unnormalised product sits.
            # Registered only when it is used, so a checkpoint written by the default
            # (unnormalised) class still loads into it key-for-key.
            self.logit_scale = nn.Parameter(
                torch.full((heads, 1, 1), math.log(math.sqrt(dim // heads)))
            )

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        b, n, d = t.shape
        return t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        q_scale: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``q_scale`` is an optional per-sample, per-head multiplier on the query.

        Scaling ``q`` scales the attention logits, so this is a **temperature** knob:
        it lets something outside the block decide how sharply each head attends,
        without touching what it attends to. Used by ``stformer`` to make that
        temperature a function of the operating point -- at a higher Reynolds number the
        wake is narrower and the useful spatial context is different, which a per-channel
        affine (what adaLN gives) cannot express. Shape ``(B, heads)``. Default ``None``
        leaves every existing caller bit-identical.

        ``attn_bias`` is an optional additive bias on the attention logits, broadcastable to
        ``(B, heads, N, N)`` -- the standard entry point for a relative-position bias. It is
        added *before* the softmax by the kernel's ``attn_mask`` argument, and ``None`` (the
        default) leaves every existing caller bit-identical.
        """
        kv_source = x if memory is None else memory
        q = self._heads(self.to_q(x))
        k, v = (self._heads(t) for t in self.to_kv(kv_source).chunk(2, dim=-1))
        if self.qk_norm:
            # ``scale`` on scaled_dot_product_attention is a python float, so the
            # per-head temperature is folded into q instead and the kernel's own
            # scaling is switched off.
            #
            # ``q_scale`` must be applied **after** the normalisation, not before:
            # ``F.normalize`` divides by the norm, so a scalar multiplier applied first
            # is exactly cancelled and the whole pathway is a silent no-op. Caught by
            # ``test_cond_attn_actually_changes_attention``, which failed on the first
            # implementation.
            q = F.normalize(q, dim=-1) * self.logit_scale.clamp(max=math.log(100.0)).exp()
            k = F.normalize(k, dim=-1)
            if q_scale is not None:
                q = q * q_scale[:, :, None, None]
            out = F.scaled_dot_product_attention(q, k, v, scale=1.0, attn_mask=attn_bias)
        else:
            if q_scale is not None:
                q = q * q_scale[:, :, None, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        return self.proj(out.transpose(1, 2).flatten(2))


class FeedForward(nn.Sequential):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))


class Block(nn.Module):
    """Pre-LN transformer block. The naive baseline's only building block."""

    def __init__(self, dim: int, heads: int, mlp_mult: int = 4) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = FeedForward(dim, mlp_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ConditionedBlock(nn.Module):
    """adaLN-Zero block: conditioning drives scale, shift and a residual gate.

    The gate projection is zero-initialised, so at step 0 the block is the identity on
    its residual stream and the network is *exactly* its unconditioned counterpart.
    Conditioning therefore cannot make the model worse at initialisation, only later
    and only if the gradient asks for it -- which is what makes an on/off A/B between
    :class:`MOCPT` and :class:`PatchTransformer` interpretable.
    """

    def __init__(self, dim: int, heads: int, cond_dim: int, cross: bool = False) -> None:
        super().__init__()
        # elementwise_affine=False: the scale and shift come from the conditioning, so
        # a learned per-channel affine here would be a second, redundant one.
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads)
        self.mlp = FeedForward(dim)
        self.cross = cross
        n_mod = 9 if cross else 6
        if cross:
            self.norm_x = nn.LayerNorm(dim, elementwise_affine=False)
            self.cross_attn = Attention(dim, heads)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * n_mod))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    @staticmethod
    def _mod(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, memory: torch.Tensor | None = None
    ) -> torch.Tensor:
        mods = self.modulation(cond).chunk(9 if self.cross else 6, dim=-1)
        sh1, sc1, g1, sh2, sc2, g2 = mods[:6]
        x = x + g1[:, None] * self.attn(self._mod(self.norm1(x), sh1, sc1))
        if self.cross:
            assert memory is not None, "a cross block needs an encoder memory"
            sh3, sc3, g3 = mods[6:]
            x = x + g3[:, None] * self.cross_attn(self._mod(self.norm_x(x), sh3, sc3), memory)
        return x + g2[:, None] * self.mlp(self._mod(self.norm2(x), sh2, sc2))


class PatchTransformer(nn.Module):
    """The naive baseline: patchify, encode, project back. ``(B,T,H,W,C) -> (B,T,H,W,C)``.

    Accepts ``cond`` and ignores it, exactly as :class:`~.wrappers.BaselineAdapter`
    does for the kit models, so it drops into the same training loop and the same
    inference path without either branching on model type.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        out_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        patch: tuple[int, int, int] = DEFAULT_PATCH,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.in_shape, self.out_shape, self.patch = in_shape, out_shape, patch
        t_in, h, w, c_in = in_shape
        t_out, h_out, w_out, c_out = out_shape
        if (h, w) != (h_out, w_out):
            raise ValueError("input and output must share the spatial grid")
        self.n_in = math.prod(_grid((t_in, h, w), patch))
        self.n_out = math.prod(_grid((t_out, h, w), patch))
        patch_vol = math.prod(patch)

        self.embed = nn.Linear(patch_vol * c_in, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.n_in, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_vol * c_out)
        # A zero-initialised head starts the model at the zero field rather than at
        # noise. With freestream normalisation the target has O(1) mean, so this is a
        # worse start than persistence but a far better one than random patches.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        del cond
        h = self.embed(patchify(x, self.patch)) + self.pos
        for block in self.blocks:
            h = block(h)
        tokens = self.head(self.norm(h))
        t_out, h_, w_, c_out = self.out_shape
        return unpatchify(tokens, self.patch, (t_out, h_, w_), c_out)


class MOCPT(nn.Module):
    """Conditioned encoder-decoder over space-time patches.

    ``cond_mode='none'`` degrades it to the same conditioning the U-Net's
    ``cond_mode='none'`` has -- a learned constant -- so the conditioning pathway can
    be ablated without changing anything else about the architecture.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        out_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        patch: tuple[int, int, int] = DEFAULT_PATCH,
        dim: int = 256,
        depth: int = 4,
        decoder_depth: int = 2,
        heads: int = 8,
        cond_mode: str = "operating_point",
        cond_features: int | None = None,
        pe_content: bool = False,
    ) -> None:
        super().__init__()
        if cond_mode not in COND_MODES:
            raise ValueError(f"cond_mode must be one of {COND_MODES}, got {cond_mode!r}")
        # Resolved here rather than defaulted to COND_DIM: the width follows from the
        # mode, and a mode-width mismatch is a checkpoint that loads and then reads a
        # conditioning vector one element short of what it was trained on.
        cond_features = cond_vec_dim(cond_mode) if cond_features is None else cond_features
        self.in_shape, self.out_shape, self.patch = in_shape, out_shape, patch
        self.cond_mode = cond_mode
        t_in, h, w, c_in = in_shape
        t_out, h_out, w_out, c_out = out_shape
        if (h, w) != (h_out, w_out):
            raise ValueError("input and output must share the spatial grid")
        self.grid_in = _grid((t_in, h, w), patch)
        self.grid_out = _grid((t_out, h, w), patch)
        self.n_in, self.n_out = math.prod(self.grid_in), math.prod(self.grid_out)
        patch_vol = math.prod(patch)

        cond_dim = dim
        self.cond_features = cond_features
        if cond_mode != "none":
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_features, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
            )
        else:
            # A learned constant, matching UNet3d's cond_mode='none'. Built as a
            # parameter rather than a dead MLP so an ablation costs dim parameters
            # instead of an unused pathway in every checkpoint.
            self.cond_proj = None
            self.cond_const = nn.Parameter(torch.zeros(cond_dim))

        self.embed = nn.Linear(patch_vol * c_in, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.n_in, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.encoder = nn.ModuleList(ConditionedBlock(dim, heads, cond_dim) for _ in range(depth))

        # Output queries: one per output patch, plus an explicit lead-time feature so
        # the decoder knows how far ahead each query sits. Predictability falls from
        # ~86% of the variance at lead 1 to ~33% at lead 20, and a query that cannot
        # tell the two apart has to learn one compromise for both.
        self.query = nn.Parameter(torch.zeros(1, self.n_out, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.register_buffer("lead", self._lead_index(), persistent=False)
        self.lead_proj = nn.Linear(dim, dim)
        self.decoder = nn.ModuleList(
            ConditionedBlock(dim, heads, cond_dim, cross=True) for _ in range(decoder_depth)
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_vol * c_out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _lead_index(self) -> torch.Tensor:
        """Time-patch index of each output token, in raster order."""
        gt, gh, gw = self.grid_out
        return torch.arange(gt).repeat_interleave(gh * gw)

    def _conditioning(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        b = x.shape[0]
        if self.cond_proj is None:
            return self.cond_const[None].expand(b, -1)
        if cond is None:
            cond = x.new_zeros(b, self.cond_features)
        return self.cond_proj(cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        c = self._conditioning(x, cond)

        memory = self.embed(patchify(x, self.patch)) + self.pos
        for block in self.encoder:
            memory = block(memory, c)

        lead = sinusoidal_embedding(self.lead, self.query.shape[-1]).to(memory.dtype)
        q = self.query + self.lead_proj(lead)[None]
        q = q.expand(x.shape[0], -1, -1)
        for block in self.decoder:
            q = block(q, c, memory=memory)

        t_out, h_, w_, c_out = self.out_shape
        return unpatchify(self.head(self.norm(q)), self.patch, (t_out, h_, w_), c_out)


# --------------------------------------------------------------------------- #
# ModPCT — a faithful port of SimFormer's production model
# --------------------------------------------------------------------------- #
#
# Ported from `simformer/models/modpct` (ModPCT, "Modulated Point-Cloud Transformer").
# The class above, :class:`MOCPT`, was written from a verbal description before that
# source was read and is **not** ModPCT: it added a cross-attending query decoder that
# ModPCT does not have. That addition broke input-to-output connectivity at
# initialisation (see the note on :class:`ModPCT.forward`), which is why it trails its
# own unconditioned control. It is kept only so its checkpoints still load and because
# it is the evidence for that diagnosis.
#
# What ModPCT is, from the source:
#
#   * **Encoder only.** `DiTransformer` is a stack of self-attention DiT blocks with
#     adaLN-Zero. There is no decoder and no cross-attention.
#   * **Per-token conditioning.** `cond = embedded_scalars.unsqueeze(1) + centroid_pe`,
#     so every token's modulation carries both the global operating point and a Fourier
#     encoding of *where that token is*. Position enters through the conditioning rather
#     than as an additive positional embedding.
#   * **Per-field decode.** Each output field gets its own decode branch
#     (`per_field_decode`), instead of one shared projection.
#
# One deliberate grid adaptation: ModPCT tokenises a point cloud with a learned
# within-cluster attention pooling (`LearnedClusterPooling`). On a regular grid the patch
# *is* the cluster, so tokens come from the same linear patch embedding
# :class:`PatchTransformer` uses. That keeps the tokenizer identical between the two, so
# the difference between them stays attributable to conditioning and the decode head --
# which is the entire point of running them as a pair.


class FourierPositionalEncoding(nn.Module):
    """sin/cos of continuous coordinates at fixed frequencies, projected to ``out_dim``.

    Ported verbatim in behaviour from ModPCT. Frequencies are ``pi * 2**[0..bands-1]``
    and assume roughly unit-variance inputs; beyond ~6 bands the encoding aliases the
    token grid, which is why ``bands`` defaults there.
    """

    freqs: torch.Tensor

    def __init__(self, in_dim: int, num_bands: int, out_dim: int) -> None:
        super().__init__()
        freqs = math.pi * 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(in_dim * num_bands * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(-2))


class DiTBlock(nn.Module):
    """One adaLN-Zero self-attention block, matching ModPCT's ``DiTLayer``.

    Accepts conditioning either per sample ``(B, D)`` or **per token** ``(B, S, D)``;
    ModPCT always uses the latter, so a token's modulation depends on where it is.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.dim() == 2:
            c = c.unsqueeze(1)
        sh1, sc1, g1, sh2, sc2, g2 = self.adaln_modulation(c).chunk(6, dim=-1)
        x = x + g1 * self.attn(self.modulate(self.norm1(x), sh1, sc1))
        return x + g2 * self.mlp(self.modulate(self.norm2(x), sh2, sc2))


class ModPCT(nn.Module):
    """SimFormer's ModPCT on the scored space-time grid.

    ``(B, T, H, W, C) -> (B, T, H, W, C_out)``. Depth, width, head count and tokenizer
    match :class:`PatchTransformer` exactly, so the pair isolates **adaLN-Zero per-token
    conditioning plus per-field decode** and nothing else.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        out_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        patch: tuple[int, int, int] = DEFAULT_PATCH,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        fourier_bands: int = 6,
        cond_mode: str = "operating_point",
        cond_features: int | None = None,
        pe_content: bool = False,
    ) -> None:
        super().__init__()
        if cond_mode not in COND_MODES:
            raise ValueError(f"cond_mode must be one of {COND_MODES}, got {cond_mode!r}")
        # Resolved here rather than defaulted to COND_DIM: the width follows from the
        # mode, and a mode-width mismatch is a checkpoint that loads and then reads a
        # conditioning vector one element short of what it was trained on.
        cond_features = cond_vec_dim(cond_mode) if cond_features is None else cond_features
        self.in_shape, self.out_shape, self.patch = in_shape, out_shape, patch
        self.cond_mode, self.cond_features = cond_mode, cond_features
        # Position reaches faithful ModPCT *only* through the modulation: adaLN can
        # scale and shift a token by where it is, but never adds positional content.
        # That is fine for a steady-state point cloud and looks costly here -- the
        # conditioned model converged by 2700 steps and then stopped improving, while
        # its control (which has a learned additive positional embedding) kept going and
        # overtook it. ModPCT's own source keeps the escape hatch: without global scalars
        # it "re-injects the centroid PE as content rather than feeding the adaLN cond".
        # ``pe_content`` does both at once, which their code never does.
        self.pe_content = pe_content
        t_in, h, w, c_in = in_shape
        t_out, h_out, w_out, c_out = out_shape
        if (h, w) != (h_out, w_out) or t_in != t_out:
            raise ValueError("ModPCT decodes on the input token grid, so shapes must match")
        self.grid = _grid((t_in, h, w), patch)
        self.n_tokens = math.prod(self.grid)
        patch_vol = math.prod(patch)
        self.c_out = c_out

        self.embed = nn.Linear(patch_vol * c_in, dim)
        self.centroid_pe = FourierPositionalEncoding(3, fourier_bands, dim)
        # A separate projection of the same Fourier features: content and modulation are
        # different jobs, and sharing one projection would force a single embedding to
        # serve both. ~9k parameters.
        self.content_pe = FourierPositionalEncoding(3, fourier_bands, dim) if pe_content else None
        self.register_buffer("centroids", self._centroids(), persistent=False)
        self.scalar_embed = (
            nn.Sequential(nn.Linear(cond_features, dim), nn.SiLU(), nn.Linear(dim, dim))
            if cond_mode != "none"
            else None
        )

        self.blocks = nn.ModuleList(DiTBlock(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # per_field_decode: one branch per output field, rather than one shared projection.
        self.field_branches = nn.ModuleList(nn.Linear(dim, patch_vol) for _ in range(c_out))
        self.initialize_weights()

    def _centroids(self) -> torch.Tensor:
        """Standard-scaled ``(t, h, w)`` centre of every token, ``(S, 3)``.

        Standardised rather than mapped to [0, 1] because ModPCT's Fourier bands assume
        roughly unit-variance coordinates; on [0, 1] the top bands would be wasted.
        """
        gt, gh, gw = self.grid
        idx = torch.stack(
            torch.meshgrid(torch.arange(gt), torch.arange(gh), torch.arange(gw), indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        c = (idx.float() + 0.5) / torch.tensor([gt, gh, gw], dtype=torch.float32)
        return (c - c.mean(0)) / c.std(0).clamp_min(1e-6)

    def initialize_weights(self) -> None:
        """ModPCT's scheme: xavier everywhere, then zero every adaLN and the readout."""

        def _basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic)
        for block in self.blocks:
            nn.init.constant_(block.adaln_modulation[-1].weight, 0)
            nn.init.constant_(block.adaln_modulation[-1].bias, 0)
        for branch in self.field_branches:
            nn.init.constant_(branch.weight, 0)
            nn.init.constant_(branch.bias, 0)

    def _conditioning(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        pe = self.centroid_pe(self.centroids).unsqueeze(0)  # (1, S, D)
        if self.scalar_embed is None:
            return pe.expand(x.shape[0], -1, -1)
        if cond is None:
            cond = x.new_zeros(x.shape[0], self.cond_features)
        return self.scalar_embed(cond).unsqueeze(1) + pe

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Encode the token grid under per-token modulation, decode each field separately.

        The property the query-decoder variant lost: **the residual stream carries the
        input**. adaLN-Zero starts every gate at zero, so at initialisation each block is
        the identity *on the embedded input* and the readout still sees ``x``. Nothing
        has to grow open before the encoder receives gradient.
        """
        c = self._conditioning(x, cond)
        h = self.embed(patchify(x, self.patch))
        if self.content_pe is not None:
            h = h + self.content_pe(self.centroids).unsqueeze(0)
        for block in self.blocks:
            h = block(h, c)
        h = self.norm(h)
        # (B, S, patch_vol, C_out) -> flatten with C fastest, matching unpatchify's order.
        fields = torch.stack([branch(h) for branch in self.field_branches], dim=-1)
        t_out, h_, w_, _ = self.out_shape
        return unpatchify(fields.flatten(-2), self.patch, (t_out, h_, w_), self.c_out)
