"""A space-time transformer built around the three things the 320-token model gives up.

``models/transformer.py``'s :class:`~.transformer.PatchTransformer` is the honest
control and it loses to U-Net on every accuracy axis (FINDINGS 9: rel_l2 94.63 against
95.50 on ``sim_val_extrap``, at 1/24th the inference cost). This module is the attempt
to keep the cost and close the accuracy gap, by changing exactly what the diagnosis
points at rather than by making the same model bigger.

What the control gives up, and what is done about it
----------------------------------------------------

**1. The decode is a linear map from one token to a whole space-time patch.**
``(4,4,8)`` patches mean a single 256-dimensional token must reconstruct
``4*4*8*3 = 384`` field values with no access to its neighbours, so patch seams are
unresolvable by construction and the highest spatial frequencies -- the ones
``tke_score`` reads -- are the first thing lost. Here the patch decode stays, but a
convolutional refinement runs at **full resolution** with a skip from the stem, which
is the one structural advantage U-Net has that no amount of attention supplies. The
refinement's last convolution is zero-initialised, so ``refine=False`` and an
untrained ``refine=True`` are the same model and the ablation is attributable.

**2. Tokens are built by a single linear projection of raw values.** A convolutional
stem before the patch embedding is the standard fix (it is why hybrid ViTs beat pure
ones at small resolution) and it costs ~0.5 GFLOP against the blocks' ~18.

**3. Time is a sequence axis it does not need to be.** The default patch spans the
*whole* 20-frame window, so every token carries a complete local history and attention
is spatial only: 512 tokens at ``(20,2,2)`` against the control's 320 at ``(4,4,8)``,
i.e. **4x the spatial resolution for fewer time-slices**. The window is 20 frames long
and the task is one-shot -- there is no rollout for a temporal axis to help with -- so
mixing time in the channel dimension buys spatial tokens with it. ``patch[0] < 20``
restores a genuine space-time token grid, so this is a swept axis rather than an
assumption (ROADMAP N12 is precisely the claim that it has never been swept).

The conditioning fix
--------------------
adaLN-Zero as ``models/transformer.py`` implements it (and as ModPCT does) gates each
residual branch **shut** at initialisation and drops the LayerNorm's own affine, on the
grounds that scale and shift should arrive from the conditioning. FINDINGS 9 measured
the consequence: ModPCT converged by 2700 steps and was then overtaken by its own
unconditioned control, and the leading explanation on record is that it "traded that
free affine for a modulation driven by nothing much".

Here the gate is initialised to **identity, not zero**: the modulation predicts a
*deviation* ``(1 + gate)`` from an ordinary pre-LN block that keeps its affine
LayerNorm. Both properties that made zero-init worth having survive -- at
initialisation the conditioned model is exactly its unconditioned twin
(``test_stformer_conditioning_is_identity_at_init``), so an on/off A/B is not an init
lottery -- while the attention and MLP parameters receive full gradient from step 0
instead of waiting for a gate to grow open.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..features import COND_MODES
from ..features import cond_dim as cond_vec_dim
from .transformer import Attention, FeedForward, _grid, unpatchify

#: ``(time, height, width)``. The default spans the whole window in time, giving a
#: purely spatial 16x32 = 512-token grid at the scored geometry.
DEFAULT_PATCH = (20, 2, 2)

#: Index of the domain bit inside the conditioning vector, i.e. ``is_sim``. Tied to the
#: order ``features.conditioning_features`` emits ``(log-Re, AoA, is_sim)``; asserted rather
#: than assumed when ``domain_interface`` is on, because a silently wrong index would route
#: every sample to one branch and still train.
#:
#: It survives ``operating_point_tau`` because that mode **appends** tau at index 3 rather
#: than inserting it, so the bit does not move. That is worth stating rather than relying on:
#: any future mode that reorders the vector has to update this constant, and the failure it
#: would cause is silent -- a model that routes every window to one interface and trains.
DOMAIN_FEATURE = 2

#: How the operating point reaches the trunk. ``adaln`` is the shipped per-block six-way
#: modulation; ``bias`` adds the projected conditioning vector to every token once, before
#: the first block, and leaves the blocks as plain pre-LN transformer blocks; ``shared`` is
#: adaLN-single (PixArt-alpha): ONE ``SiLU -> Linear(dim, 6*dim)`` at model level plus a
#: zero-initialised per-block offset, the design ``tubelet_former.py`` uses.
MODULATIONS = ("adaln", "bias", "shared")

#: The modules a ``domain_interface=True`` model duplicates per domain: exactly what the ``io``
#: rung of ``train.loop.UNFREEZE_LADDER`` adds over ``cond``, which is **0.8M parameters, 4% of
#: the model** (FINDINGS 31c has the decomposition -- the rung's headline 8.18M is 87% inherited
#: adaLN modulation, and an earlier revision mis-attributed that to the interface).
#:
#: 0.8M is the whole argument for putting the seam here: it is the increment that carried `io`'s
#: +0.11 rel_l2 / +0.14 mvpe on ``val_extrap`` over ``cond``, so it is the part of the network
#: the measurement says wants to be domain-specific -- and duplicating it costs 4 MB in a 256 MB
#: budget rather than the 33 MB the mis-read would have implied.
INTERFACE_MODULES = (
    "stem",
    "embed",
    "pos",
    "head",
    "head_mean",
    "to_map",
    "refine",
    "film_stem",
    "film_refine",
)


def _groups(channels: int, target: int = 8) -> int:
    """Largest divisor of ``channels`` no greater than ``target``, for GroupNorm."""
    return next(g for g in range(min(target, channels), 0, -1) if channels % g == 0)


class ConvGRUHistory(nn.Module):
    """Causal spatial recurrent encoder returning a zero-initialised history correction.

    A compact ConvGRU tracks a hidden state over the 20 observed frames at full spatial
    resolution. Its readout is zero-initialised, so enabling the encoder preserves the
    exact input and persistence initialization while training can expose motion state to
    the otherwise time-folded STFormer stem.
    """

    def __init__(self, channels: int, hidden: int = 12) -> None:
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(channels + hidden, 2 * hidden, 3, padding=1)
        self.candidate = nn.Conv2d(channels + hidden, hidden, 3, padding=1)
        self.readout = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, _ = x.shape
        state = x.new_zeros(b, self.hidden, h, w)
        corrections = []
        for frame in x.unbind(dim=1):
            frame = frame.permute(0, 3, 1, 2)
            reset, update = self.gates(torch.cat([frame, state], dim=1)).sigmoid().chunk(2, dim=1)
            proposal = torch.tanh(self.candidate(torch.cat([frame, reset * state], dim=1)))
            state = update * state + (1.0 - update) * proposal
            corrections.append(self.readout(state).permute(0, 2, 3, 1))
        return torch.stack(corrections, dim=1)


class TemporalRefine(nn.Module):
    """A refinement that mixes across the output-frame axis, drop-in for the 2D ``refine``.

    The default STFormer folds the whole 20-frame window into each token's channels, so the
    trunk has **no temporal axis** and the decode is one linear map from a token to all 20 of
    its output frames. The measured consequence is a ``tke`` deficit -- the relative L2 of the
    *temporal variance* field -- that is uniform across the three real splits (-0.40/-0.37/-0.42
    vs the U-Net), i.e. a property of the model class rather than of the data. :class:`STFormer3D`
    fixed it by making time an axis in the whole trunk, and paid 2.78x inference time for it,
    which is what made it unshippable -- the accuracy delta was positive.

    This puts the temporal axis **only in the full-resolution refinement**, where it is nearly
    free: the refine runs on a 32x64 grid at ~1.5% of the trunk's FLOPs, so a depthwise temporal
    convolution over the 20 frames adds negligible wall-clock while giving the decode's correction
    frame-to-frame coherence -- a vortex the model places can now move consistently across leads
    rather than being read out independently per frame.

    Same input/output contract as the ``refine`` :class:`~torch.nn.Sequential` it replaces:
    ``(B*gt, refine_channels + stem_channels, H, W) -> (B*gt, pt*C_out, H, W)`` in the frame-major
    channel order :meth:`STFormer._from_groups` expects. The output convolution is
    zero-initialised for the same reason as the coarse ``head`` and the plain refine: with it
    silent at step 0 the model starts at the coarse decode, so ``refine_temporal`` is a clean
    additive ablation and key-matches onto a checkpoint trained without it.
    """

    def __init__(
        self,
        in_channels: int,
        refine_channels: int,
        pt: int,
        c_out: int,
        t_hidden: int = 32,
        t_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.pt, self.c_out, self.t_hidden = pt, c_out, t_hidden
        # Spatial body: identical to the plain refine's first conv block -- a full-resolution
        # 3x3 over the token-map/stem-skip concatenation.
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, refine_channels, 3, padding=1),
            nn.GroupNorm(_groups(refine_channels), refine_channels),
            nn.SiLU(),
        )
        # Lift to per-frame features, then mix across frames with a *depthwise* temporal conv
        # (one 1D kernel per channel over the 20 leads). Depthwise keeps the temporal mixing at
        # ``pt * t_hidden`` taps rather than the ``pt * t_hidden^2`` a dense conv would cost, which
        # is the whole reason this is affordable where a 3D trunk was not.
        self.to_frames = nn.Conv2d(refine_channels, pt * t_hidden, 3, padding=1)
        self.temporal = nn.Sequential(
            nn.Conv3d(
                t_hidden, t_hidden, (t_kernel, 1, 1), padding=(t_kernel // 2, 0, 0), groups=t_hidden
            ),
            nn.GroupNorm(_groups(t_hidden), t_hidden),
            nn.SiLU(),
        )
        self.out = nn.Conv3d(t_hidden, c_out, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bg, _, h, w = x.shape
        z = self.to_frames(self.spatial(x))  # (B*gt, pt*t_hidden, H, W)
        # (B*gt, pt, t_hidden, H, W) -> (B*gt, t_hidden, pt, H, W): the temporal axis contiguous
        # for the depthwise Conv3d, exactly as STFormer3D lays time out for its refinement.
        z = z.view(bg, self.pt, self.t_hidden, h, w).permute(0, 2, 1, 3, 4)
        z = self.out(self.temporal(z))  # (B*gt, C_out, pt, H, W)
        # Back to frame-major channels [f0c0 f0c1 f0c2 f1c0 ...] for ``_from_groups``.
        return z.permute(0, 2, 1, 3, 4).reshape(bg, self.pt * self.c_out, h, w)


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    """Full-resolution 3x3 conv + GroupNorm + SiLU -- the plain refine's unit of work."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(_groups(cout), cout),
        nn.SiLU(),
    )


class MultiScaleRefine(nn.Module):
    """A two-scale (mini-U-Net) refinement, drop-in for the 2D ``refine``.

    The plain refine is a single full-resolution conv block. FINDINGS records that STFormer's
    ~0.1 rel_l2 deficit to the U-Net is structural and that the U-Net's one advantage "no amount
    of attention supplies" is its **multi-scale full-resolution convolutional trunk** -- yet every
    capacity experiment on record widened the existing convs (stem 96->160, refine 64->128) or the
    token grid and moved rel_l2 <0.05. A *deeper, multi-scale* refine is the untried axis: a
    stride-2 encoder gives the correction a 2x-larger receptive field and a coarse-scale pathway,
    then an upsample with a skip from the full-resolution features restores detail -- exactly the
    down/up/skip structure of the model it loses to, imported into the cheap 2D refine (both scales
    are on a <=32x64 grid, ~a few percent of the trunk's FLOPs).

    Same input/output contract as the ``refine`` :class:`~torch.nn.Sequential` it replaces, and the
    output conv is zero-initialised for the same reason: ``refine_multiscale`` is then a clean
    additive ablation that starts at the coarse decode and key-matches onto a plain-refine trunk.
    """

    def __init__(self, in_channels: int, refine_channels: int, pt: int, c_out: int) -> None:
        super().__init__()
        rc = refine_channels
        self.e0 = _conv_block(in_channels, rc)  # full resolution
        self.down = nn.AvgPool2d(2)
        self.e1 = _conv_block(rc, 2 * rc)  # half resolution
        self.mid = _conv_block(2 * rc, 2 * rc)
        self.up = _conv_block(2 * rc + rc, rc)  # upsampled + full-res skip
        self.out = nn.Conv2d(rc, pt * c_out, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.e0(x)  # (B*gt, rc, H, W)
        m = self.mid(self.e1(self.down(e0)))  # (B*gt, 2rc, H/2, W/2)
        u = nn.functional.interpolate(m, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.up(torch.cat([u, e0], dim=1)))  # (B*gt, pt*C_out, H, W)


#: Token-mixing mechanisms selectable per block via the ``mixer`` arch flag. ``softmax`` is the
#: ordinary self-attention trunk; the rest are genuinely different mechanisms swapped in for the
#: attention sublayer while the adaLN conditioning, MLP, pre-LN and residual structure are kept, so
#: an A/B isolates the token mixer. They change the *keys* of the trunk, so a model using one cannot
#: ``--init-from`` a softmax checkpoint -- these are trained from scratch and compared as
#: architectures, not as perturbations of a shared init.
MIXERS = ("softmax", "xca", "galerkin", "spectral", "swin")


class XCA(nn.Module):
    """Cross-covariance attention (XCiT): attention over feature *channels*, not tokens.

    A ``d x d`` attention map from the token-normalised query/key covariance, so cost is
    ``O(N d^2)`` rather than ``O(N^2 d)``, and it mixes *what* each channel carries rather than
    *where* each token looks -- a different inductive bias for a field whose channels (u, v, p and
    derived structure) are strongly coupled.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.temp = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h, dh = self.heads, d // self.heads
        q, k, v = (t.reshape(b, n, h, dh).permute(0, 2, 3, 1) for t in self.to_qkv(x).chunk(3, -1))
        # normalise over the token axis, then a channel-by-channel attention map (b, h, dh, dh)
        q, k = nn.functional.normalize(q, dim=-1), nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temp
        out = attn.softmax(dim=-1) @ v  # (b, h, dh, n)
        return self.proj(out.permute(0, 3, 1, 2).reshape(b, n, d))


class GalerkinAttention(nn.Module):
    """Softmax-free Galerkin-type linear attention (Cao 2021, "Fourier or Galerkin").

    Drops the softmax and instance-normalises the keys/values, then computes ``Q (K^T V) / n`` --
    a linear-complexity operator-learning attention derived from a Petrov-Galerkin projection, i.e.
    the attention variant purpose-built for the exact problem class here (learning a solution
    operator), rather than for language.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.ln_k, self.ln_v = nn.LayerNorm(dim // heads), nn.LayerNorm(dim // heads)
        self.temp = nn.Parameter(torch.zeros(heads, 1, 1))  # learned query temperature (exp -> 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        b, n, d = x.shape
        h, dh = self.heads, d // self.heads
        q, k, v = (t.reshape(b, n, h, dh).transpose(1, 2) for t in self.to_qkv(x).chunk(3, -1))
        # Two-part stabilisation, both learned the hard way in the arch campaign:
        # (1) fp32 core -- without a softmax the ``K^T V`` accumulation over n tokens overflows
        #     under fp16 autocast (galerkin_s0 diverged);
        # (2) unit-normalise the query -- the query is otherwise unbounded and grows during
        #     training until ``Q(K^T V)`` blows up late (galerkin_s1 went NaN at ft step ~3950
        #     even in fp32). Normalising q bounds the product; the temperature restores scale.
        q, k, v = q.float(), k.float(), v.float()
        q = nn.functional.normalize(q, dim=-1) * self.temp.exp()
        k, v = self.ln_k(k), self.ln_v(v)
        kv = k.transpose(-2, -1) @ v / n  # (b, h, dh, dh), normalised by token count
        out = (q @ kv).transpose(1, 2).reshape(b, n, d)  # (b, n, d)
        return self.proj(out.to(dt))


class SpectralMixer(nn.Module):
    """Global Fourier token mixing (FNO-style), a drop-in for the attention sublayer.

    Mixes tokens by a 2D real FFT over the spatial grid, a shared complex channel-mixing linear per
    frequency mode, and an inverse FFT. Global and translation-equivariant by construction -- the
    natural mechanism for a PDE solution operator -- and a fundamentally different way of moving
    information than pairwise attention. Only defined on a purely spatial token grid
    (``patch[0]==T``, ``gt==1``); the FFT runs in fp32 (complex ops are unsafe under fp16 autocast).
    """

    def __init__(self, dim: int, gh: int, gw: int) -> None:
        super().__init__()
        self.gh, self.gw = gh, gw
        scale = 1.0 / dim
        self.wr = nn.Parameter(torch.randn(dim, dim) * scale)
        self.wi = nn.Parameter(torch.randn(dim, dim) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        b, n, d = x.shape
        xf = torch.fft.rfft2(x.float().reshape(b, self.gh, self.gw, d), dim=(1, 2), norm="ortho")
        yf = xf @ torch.complex(self.wr, self.wi)  # channel mix, shared across modes
        y = torch.fft.irfft2(yf, s=(self.gh, self.gw), dim=(1, 2), norm="ortho")
        return y.reshape(b, n, d).to(dt)


class SwinAttention(nn.Module):
    """Shifted-window MHSA (Swin Transformer, Liu et al. 2021), drop-in for the attention sublayer.

    Partitions the spatial token grid into non-overlapping local windows and runs MHSA
    independently within each. With ``shift=True`` the grid is cyclically shifted by half
    a window before partitioning, and an additive mask prevents cross-boundary tokens from
    attending -- so information propagates across window boundaries via the alternating
    shift/non-shift pattern across blocks.

    Restricted to the purely spatial token grid (``grid[0] == 1``): the partition lives on the
    ``(gh, gw)`` plane and has no temporal analogue, matching the ``spectral`` mixer restriction.

    The primary motivation here is not efficiency (512 global tokens are already cheap) but
    architectural diversity and a local-first inductive bias: windows constrain each block to
    a fixed-size neighbourhood, which may help capture near-wake structures that dominate in
    the training region without noisily aggregating far-field tokens.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        gh: int,
        gw: int,
        window_h: int = 4,
        window_w: int = 8,
        shift: bool = False,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if gh % window_h != 0 or gw % window_w != 0:
            raise ValueError(
                f"Token grid ({gh},{gw}) must be divisible by window ({window_h},{window_w})"
            )
        self.heads = heads
        self.gh, self.gw = gh, gw
        self.wh, self.ww = window_h, window_w
        self.shift = shift
        dh = dim // heads
        self.scale = dh**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(dh) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(dh) if qk_norm else nn.Identity()
        # Learnable relative position bias: one scalar per (Δrow, Δcol) per head.
        # Shape (heads, (2*wh-1)*(2*ww-1)); indexed by a precomputed (ws, ws) int table.
        self.rpb = nn.Parameter(torch.zeros(heads, (2 * window_h - 1) * (2 * window_w - 1)))
        self.register_buffer(
            "rel_index", self._build_rel_index(window_h, window_w), persistent=False
        )
        if shift:
            self.register_buffer(
                "attn_mask",
                self._build_shift_mask(gh, gw, window_h, window_w),
                persistent=False,
            )
        else:
            self.attn_mask: torch.Tensor | None = None

    @staticmethod
    def _build_rel_index(wh: int, ww: int) -> torch.Tensor:
        """``(ws, ws)`` int index into the RPB table for each within-window token pair.

        For tokens at flat positions i=(ri,ci) and j=(rj,cj), the bias index is
        ``(ri-rj + wh-1) * (2*ww-1) + (ci-cj + ww-1)``, covering ``[0, (2wh-1)*(2ww-1))``.
        """
        coords = torch.stack(
            torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing="ij")
        ).reshape(2, -1)  # (2, ws)
        delta = coords[:, :, None] - coords[:, None, :]  # (2, ws, ws)
        delta[0] += wh - 1
        delta[1] += ww - 1
        return (delta[0] * (2 * ww - 1) + delta[1]).long()  # (ws, ws)

    @staticmethod
    def _build_shift_mask(gh: int, gw: int, wh: int, ww: int) -> torch.Tensor:
        """``(n_windows, wh*ww, wh*ww)`` additive mask: 0 inside a window, ``-1e4`` across.

        Tokens that land in the same window after the cyclic shift but originated from
        different spatial regions must not attend to each other. This is computed by
        labelling each position with a region id (one per (h-zone, w-zone) combination),
        rolling the label map the same way the data is rolled in ``forward``, partitioning
        into windows, and masking all cross-region pairs within each window.
        """
        sh, sw = wh // 2, ww // 2
        label = torch.zeros(gh, gw, dtype=torch.long)
        label[gh - sh :, :] += 1  # rows that wrap in h after rolling
        label[:, gw - sw :] += 2  # cols that wrap in w after rolling
        label = torch.roll(label, shifts=(-sh, -sw), dims=(0, 1))
        nwh, nww = gh // wh, gw // ww
        label = label.reshape(nwh, wh, nww, ww).permute(0, 2, 1, 3).reshape(nwh * nww, wh * ww)
        return (label[:, :, None] != label[:, None, :]).float() * -1e4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        dh = d // self.heads
        nwh, nww = self.gh // self.wh, self.gw // self.ww
        n_windows = nwh * nww
        ws = self.wh * self.ww

        x2d = x.reshape(b, self.gh, self.gw, d)
        if self.shift:
            x2d = torch.roll(x2d, shifts=(-self.wh // 2, -self.ww // 2), dims=(1, 2))

        # (b * n_windows, ws, d)
        x_win = x2d.reshape(b, nwh, self.wh, nww, self.ww, d)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).reshape(b * n_windows, ws, d)

        q, k, v = (
            t.reshape(b * n_windows, ws, self.heads, dh).transpose(1, 2)
            for t in self.to_qkv(x_win).chunk(3, -1)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (b*nw, heads, ws, ws)
        # Relative position bias: (heads, ws, ws) broadcast over b*n_windows
        attn = attn + self.rpb[:, self.rel_index][None]
        if self.attn_mask is not None:
            attn = attn.reshape(b, n_windows, self.heads, ws, ws)
            attn = attn + self.attn_mask[:, None, :, :]  # (n_windows, 1, ws, ws)
            attn = attn.reshape(b * n_windows, self.heads, ws, ws)
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b * n_windows, ws, d)
        out = self.proj(out)

        out = out.reshape(b, nwh, nww, self.wh, self.ww, d)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(b, self.gh, self.gw, d)
        if self.shift:
            out = torch.roll(out, shifts=(self.wh // 2, self.ww // 2), dims=(1, 2))
        return out.reshape(b, n, d)


def _build_mixer(
    mixer: str,
    dim: int,
    heads: int,
    qk_norm: bool,
    grid: tuple[int, int, int],
    block_idx: int = 0,
    window_h: int = 4,
    window_w: int = 8,
) -> nn.Module:
    """The token-mixing sublayer for a block. ``spectral`` and ``swin`` need a spatial-only grid."""
    if mixer == "softmax":
        return Attention(dim, heads, qk_norm=qk_norm)
    if mixer == "xca":
        return XCA(dim, heads)
    if mixer == "galerkin":
        return GalerkinAttention(dim, heads)
    if mixer == "spectral":
        if grid[0] != 1:
            raise ValueError("spectral mixer needs a purely spatial token grid (patch[0]==T_in)")
        return SpectralMixer(dim, grid[1], grid[2])
    if mixer == "swin":
        if grid[0] != 1:
            raise ValueError("swin mixer needs a purely spatial token grid (patch[0]==T_in)")
        return SwinAttention(
            dim,
            heads,
            grid[1],
            grid[2],
            window_h,
            window_w,
            shift=(block_idx % 2 == 1),
            qk_norm=qk_norm,
        )
    raise ValueError(f"mixer must be one of {MIXERS}, got {mixer!r}")


class ConditionedBlock(nn.Module):
    """Pre-LN block whose conditioning perturbs an already-working block.

    Contrast :class:`~.transformer.ConditionedBlock`, which drops the affine LayerNorm
    and multiplies each residual branch by a gate that starts at **zero**. Both
    modulations are zero-initialised here too, but they enter as ``1 + scale`` and
    ``1 + gate``, so at initialisation this is an ordinary transformer block and the
    conditioning starts as the identity rather than as an off switch.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        cond_attn: bool = False,
        mixer: str = "softmax",
        grid: tuple[int, int, int] = (1, 16, 32),
        block_idx: int = 0,
        window_h: int = 4,
        window_w: int = 8,
        modulation: str = "adaln",
    ) -> None:
        super().__init__()
        if modulation not in MODULATIONS:
            raise ValueError(f"modulation must be one of {MODULATIONS}, got {modulation!r}")
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mixer = mixer
        self.modulation_mode = modulation
        self.attn = _build_mixer(
            mixer,
            dim,
            heads,
            qk_norm,
            grid,
            block_idx=block_idx,
            window_h=window_h,
            window_w=window_w,
        )
        self.mlp = FeedForward(dim, mult=int(mlp_ratio))
        # ``bias``: no per-block modulation at all. The block is a plain pre-LN transformer
        # block and ignores ``cond``; the operating point reaches it only as the one vector
        # STFormer adds to every token before the first block. Six adaLN maps of
        # ``cond_dim -> 6*dim`` per block are 29% of the shipped 4.63M parameters, so this
        # is the largest single structural simplification available on the trunk.
        if modulation == "adaln":
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 6))
            nn.init.zeros_(self.modulation[1].weight)
            nn.init.zeros_(self.modulation[1].bias)
        elif modulation == "shared":
            # ``cond`` arrives as the model-level shared ``(B, 6*dim)`` modulation; this block
            # adds its own offset. Zero-initialised, so every block starts identical and plain.
            self.mod_offset = nn.Parameter(torch.zeros(dim * 6))
        # Conditioned attention temperature. adaLN can only rescale and shift a token's
        # channels; it cannot change *which* tokens a head reads. This can.
        self.cond_attn = cond_attn
        if cond_attn and modulation != "adaln":
            raise ValueError("cond_attn needs the per-block adaLN modulation path")
        if cond_attn:
            self.attn_temp = nn.Linear(cond_dim, heads)
            nn.init.zeros_(self.attn_temp.weight)
            nn.init.zeros_(self.attn_temp.bias)

    @staticmethod
    def _mod(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Broadcasts over tokens for a per-sample cond (B, D) and elementwise for a
        # per-token cond (B, N, D), so both conditioning granularities share one path.
        if shift.dim() == 2:
            shift, scale = shift[:, None], scale[:, None]
        return x * (1.0 + scale) + shift

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, attn_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.modulation_mode == "bias":
            xn = self.norm1(x)
            if self.mixer == "softmax":
                x = x + self.attn(xn, attn_bias=attn_bias)
            else:
                x = x + self.attn(xn)
            return x + self.mlp(self.norm2(x))
        if self.modulation_mode == "shared":
            mods = cond + self.mod_offset
        else:
            mods = self.modulation(cond)
        sh1, sc1, g1, sh2, sc2, g2 = mods.chunk(6, dim=-1)
        if g1.dim() == 2:
            g1, g2 = g1[:, None], g2[:, None]
        xn = self._mod(self.norm1(x), sh1, sc1)
        if self.mixer == "softmax":
            q_scale = None
            if self.cond_attn:
                # cond may be per-token; the temperature is a property of the head, so pool.
                c = cond.mean(dim=1) if cond.dim() == 3 else cond
                q_scale = 1.0 + self.attn_temp(c)
            h = self.attn(xn, q_scale=q_scale, attn_bias=attn_bias)
        else:
            # xca / galerkin / spectral / swin take only the modulated tokens; q_scale
            # (cond_attn) and attn_bias (rel_pos) are softmax-only and do not apply.
            h = self.attn(xn)
        x = x + (1.0 + g1) * h
        return x + (1.0 + g2) * self.mlp(self._mod(self.norm2(x), sh2, sc2))


class STFormer(nn.Module):
    """``(B, T, H, W, C_in) -> (B, T, H, W, C_out)`` over space-time patches.

    Parameters mirror the control's where they mean the same thing, so a matched-width
    comparison is a matter of passing the same ``dim``/``depth``/``heads``.

    ``patch[0] == T_in`` (the default) folds the whole window into each token's channel
    dimension and the token grid is purely spatial; a smaller value restores time as a
    sequence axis, with the stem applied per time-group and attention joint over all
    tokens.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        out_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        patch: tuple[int, int, int] = DEFAULT_PATCH,
        dim: int = 384,
        depth: int = 8,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        stem_channels: int = 96,
        refine: bool = True,
        refine_channels: int = 64,
        refine_stem_skip: bool = True,
        refine_temporal: bool = False,
        refine_t_hidden: int = 32,
        refine_t_kernel: int = 3,
        refine_multiscale: bool = False,
        reynolds: bool = False,
        rel_pos: bool = False,
        temporal_anchor: bool = False,
        temporal_gru: bool = False,
        temporal_gru_hidden: int = 12,
        lead_query_decoder: bool = False,
        head_fourier: bool = False,
        fourier_modes: int = 4,
        mixer: str = "softmax",
        window_h: int = 4,
        window_w: int = 8,
        qk_norm: bool = True,
        cond_conv: bool = False,
        cond_attn: bool = False,
        cond_token: bool = False,
        cond_content: bool = False,
        cond_mode: str = "operating_point",
        cond_features: int | None = None,
        cond_width: int | None = None,
        domain_interface: bool = False,
        modulation: str = "adaln",
        stem: bool = True,
        coarse_head: bool = True,
        recur: int = 0,
        recur_err: bool = True,
    ) -> None:
        super().__init__()
        if cond_mode not in COND_MODES:
            raise ValueError(f"cond_mode must be one of {COND_MODES}, got {cond_mode!r}")
        if modulation not in MODULATIONS:
            raise ValueError(f"modulation must be one of {MODULATIONS}, got {modulation!r}")
        if modulation == "shared" and cond_token:
            raise ValueError("cond_token (per-token modulation) needs the per-block adaLN path")
        # Width of the vector that carries the operating point, decoupled from the model width.
        # The operating point is three numbers, so nothing makes ``dim`` the right size for the
        # code that carries them -- it is ``dim`` here only because it was never a separate knob.
        # Every per-block ``Linear(cond_width, 6*dim)`` shrinks in proportion, and unlike
        # ``modulation='shared'`` each block keeps its own map. ``None`` means ``dim``, so every
        # checkpoint trained before this knob existed rebuilds byte-identically.
        cond_width = dim if cond_width is None else cond_width
        if cond_width < 1:
            raise ValueError(f"cond_width must be >= 1, got {cond_width}")
        if modulation == "bias" and cond_width != dim:
            # ``bias`` adds the conditioning vector straight to the tokens, so it has to be
            # token-width. Narrowing is only meaningful for the per-block and FiLM paths.
            raise ValueError(
                "modulation='bias' adds the conditioning vector to the tokens, so it needs "
                f"cond_width == dim; got cond_width={cond_width}, dim={dim}"
            )
        self.cond_width = cond_width
        if not coarse_head and not refine:
            raise ValueError("coarse_head=False leaves no decoder unless refine=True")
        if not coarse_head and (lead_query_decoder or head_fourier or reynolds):
            raise ValueError("coarse_head=False is incompatible with the head variants")
        if not stem and refine and refine_stem_skip:
            # The skip reads the stem's output; without a stem there is nothing to read.
            raise ValueError("stem=False requires refine_stem_skip=False (or refine=False)")
        if not stem and temporal_gru:
            raise ValueError("stem=False and temporal_gru are not combined")
        # Resolved here rather than defaulted to COND_DIM: the width follows from the
        # mode, and a mode-width mismatch is a checkpoint that loads and then reads a
        # conditioning vector one element short of what it was trained on.
        cond_features = cond_vec_dim(cond_mode) if cond_features is None else cond_features
        # Both checks come *after* that resolution, since the width is no longer known
        # from the signature alone.
        if domain_interface and cond_mode == "none":
            # With cond_mode='none' the model is handed a learned constant instead of the
            # conditioning vector, so every sample would route to the real branch and the
            # sim branch would train on nothing while still being saved.
            raise ValueError(
                "domain_interface needs a conditioning mode that carries is_sim; got "
                f"cond_mode='none'. Use one of {COND_MODES[1:]}"
            )
        if domain_interface and cond_features <= DOMAIN_FEATURE:
            raise ValueError(
                f"domain_interface routes on conditioning feature {DOMAIN_FEATURE} (is_sim) "
                f"but cond_features is {cond_features}"
            )
        t_in, h, w, c_in = in_shape
        t_out, h_out, w_out, c_out = out_shape
        if (h, w) != (h_out, w_out):
            raise ValueError("input and output must share the spatial grid")
        self.in_shape, self.out_shape, self.patch = in_shape, out_shape, patch
        self.cond_mode, self.cond_features = cond_mode, cond_features
        pt, ph, pw = patch
        # The token grid is the OUTPUT's: ``patch[0]`` output frames per temporal group. The
        # stem folds ``t_in // gt`` input frames into each group, so a longer input context
        # (``t_in`` 40 against ``t_out`` 20) only widens the stem's first convolution -- the
        # tokens, blocks, head and refinement are exactly the 20-frame model's. With
        # ``t_in == t_out`` (every checkpoint before 2026-09-09) this is the old layout.
        self.grid = _grid((t_out, h, w), patch)
        gt, gh, gw = self.grid
        if t_in % gt:
            raise ValueError(f"T_in={t_in} must be a multiple of the temporal groups gt={gt}")
        self.pt_in = t_in // gt
        self.n_tokens = math.prod(self.grid)
        self.c_out = c_out

        self.temporal_gru = temporal_gru
        if temporal_gru:
            self.history_encoder = ConvGRUHistory(c_in, temporal_gru_hidden)

        # Horizon-specific correction to the outer ResidualWrapper's fixed last-frame
        # anchor. Each lead learns a zero-sum linear combination of the observed
        # history through ``x_t - x_last``. Zero initialisation makes this exactly the
        # existing persistence skip at step 0 while exposing velocity/trend directly,
        # instead of asking the spatial trunk to reconstruct it through patch tokens.
        self.temporal_anchor = temporal_anchor
        if temporal_anchor:
            if c_in != c_out:
                raise ValueError("temporal_anchor requires matching input/output channels")
            self.anchor_weights = nn.Parameter(torch.zeros(t_out, t_in - 1))

        # Conditioning is a per-sample vector; ``none`` replaces it with a learned
        # constant so the pathway can be ablated without removing the machinery, which
        # is what makes an on/off A/B a one-variable change (models/transformer.py
        # takes the same approach for the same reason).
        self.cond_proj = (
            nn.Sequential(
                nn.Linear(cond_features, cond_width),
                nn.SiLU(),
                nn.Linear(cond_width, cond_width),
            )
            if cond_mode != "none"
            else None
        )
        if self.cond_proj is None:
            self.cond_const = nn.Parameter(torch.zeros(cond_width))
        self.modulation = modulation
        if modulation == "shared":
            self.shared_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_width, dim * 6))
            nn.init.zeros_(self.shared_modulation[1].weight)
            nn.init.zeros_(self.shared_modulation[1].bias)
        if modulation == "bias" and self.cond_proj is not None:
            # Same discipline as every other conditioning pathway in this file: the token
            # bias starts at zero, so the model is a plain unconditioned transformer at
            # step 0 and the ablation measures the pathway, not an initialisation.
            nn.init.zeros_(self.cond_proj[-1].weight)
            nn.init.zeros_(self.cond_proj[-1].bias)

        # Stem: two full-resolution convolutions before any downsampling. Its output is
        # also the skip the refinement reads, so the high-frequency content the patch
        # decode cannot represent never has to survive the token bottleneck.
        # cond_content: the operating point broadcast as constant input planes, so the
        # convolutional stem sees it as *content* it can filter, not only as a
        # per-channel affine. FiLM (cond_conv) and this are different operations -- one
        # rescales existing features, the other adds a field the stem can convolve with
        # -- and the geometry planes already work this way.
        self.cond_content = cond_content
        stem_in = self.pt_in * c_in + (cond_features if cond_content else 0)
        # stem=False: a plain ViT patch embedding straight off the input channels. The
        # full-resolution features the refinement skip would read do not exist, which the
        # constructor guard above enforces.
        self.has_stem = stem
        if stem:
            self.stem = nn.Sequential(
                nn.Conv2d(stem_in, stem_channels, 3, padding=1),
                nn.GroupNorm(_groups(stem_channels), stem_channels),
                nn.SiLU(),
                nn.Conv2d(stem_channels, stem_channels, 3, padding=1),
                nn.GroupNorm(_groups(stem_channels), stem_channels),
                nn.SiLU(),
            )
        embed_in = stem_channels if stem else stem_in
        self.embed = nn.Conv2d(embed_in, dim, kernel_size=(ph, pw), stride=(ph, pw))
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # cond_token: let the modulation depend on *where* a token is, not only on the
        # operating point. The block's shared SiLU->Linear then sees a nonlinear
        # interaction between position and (log-Re, AoA), so the wake region can be
        # conditioned differently from the freestream. ModPCT's per-token adaLN was
        # refuted in FINDINGS 9, but on a different architecture and with the LayerNorm's
        # own affine removed; here that affine is kept.
        self.cond_token = cond_token
        if cond_token:
            self.cond_pos = nn.Parameter(torch.zeros(1, self.n_tokens, cond_width))
            nn.init.trunc_normal_(self.cond_pos, std=0.02)
        self.blocks = nn.ModuleList(
            ConditionedBlock(
                dim,
                heads,
                cond_width,
                mlp_ratio,
                qk_norm=qk_norm,
                cond_attn=cond_attn,
                mixer=mixer,
                grid=self.grid,
                block_idx=i,
                window_h=window_h,
                window_w=window_w,
                modulation=modulation,
            )
            for i in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        # The Track-2 recurrent memory (models/recurrent.py): a per-token state carried across
        # the windows of a stream, injected before the first block and advanced from the final
        # token grid plus the revealed error. ``recur`` is its width; 0 builds nothing, so
        # every checkpoint without it rebuilds byte-identically.
        self.recur = None
        if recur > 0:
            raise ValueError("the recurrent token memory (models/recurrent.py) is not included in this package")

        # Spatial (space-time) relative-position bias. The trunk otherwise carries only a learned
        # *absolute* positional embedding; flow is approximately translation-covariant, so a
        # relative encoding is the better physical prior and should help the extrapolation split
        # most, where an unseen (Re, AoA) shifts *where* the wake structures sit. A per-head bias
        # table indexed by the token-grid displacement ``(dt, dy, dx)`` is added to the attention
        # logits (the same mechanism the U-Net uses on its temporal axis). Zero-initialised, so
        # ``rel_pos=True`` is the identity at step 0 and the ablation measures the pathway.
        self.rel_pos = rel_pos
        if rel_pos:
            ext = (2 * gt - 1) * (2 * gh - 1) * (2 * gw - 1)
            self.rel_bias_table = nn.Parameter(torch.zeros(heads, ext))
            coords = torch.stack(
                torch.meshgrid(torch.arange(gt), torch.arange(gh), torch.arange(gw), indexing="ij"),
                dim=-1,
            ).reshape(-1, 3)  # (N, 3) in the (t, h, w) raster order the tokens use
            rel = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
            rel[..., 0] += gt - 1
            rel[..., 1] += gh - 1
            rel[..., 2] += gw - 1
            index = (rel[..., 0] * (2 * gh - 1) + rel[..., 1]) * (2 * gw - 1) + rel[..., 2]
            self.register_buffer("rel_index", index, persistent=False)

        # Coarse path: the control's decode, one linear map per token. Zero-initialised,
        # so the model starts at the zero field (or, under ResidualWrapper, at
        # persistence) rather than at random patches.
        #
        # ``head_fourier`` reparameterises the *temporal* half of that decode: instead of pt free
        # values per (space, channel), the head emits ``1 + 2*fourier_modes`` coefficients of a
        # fixed temporal Fourier basis (DC + K sin/cos harmonics of the window), and the pt frames
        # are reconstructed from them. The shedding is quasi-periodic, so an oscillatory basis is a
        # natural prior that *sustains* fluctuation across all 20 leads rather than letting a free
        # map decay to the mean -- the failure mode behind the 0.556 energy ratio ``tke`` reads.
        # The output stays the full pt-frame patch, so every metric is unchanged.
        if lead_query_decoder and head_fourier:
            raise ValueError("lead_query_decoder and head_fourier are mutually exclusive")
        self.lead_query_decoder = lead_query_decoder
        self.head_fourier = head_fourier
        head_time = 1 if lead_query_decoder else ((1 + 2 * fourier_modes) if head_fourier else pt)
        # coarse_head=False: no linear patch decode at all; the full-resolution refinement is
        # the only decoder. Both paths are zero-initialised, so with or without the head the
        # model starts at the same zero field and the ablation measures the pathway.
        self.coarse_head = coarse_head
        if coarse_head:
            self.head = nn.Linear(dim, head_time * ph * pw * c_out)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        if lead_query_decoder:
            # One query per forecast horizon retrieves a global summary from the spatial
            # memory. Adding that summary to every spatial token gives a shared per-frame
            # readout without the monolithic ``dim -> 20*patch`` projection used by the
            # default decoder. The output head remains zero-initialised, so both decoders
            # begin at the same zero field (persistence under ``ResidualWrapper``).
            self.lead_queries = nn.Parameter(torch.zeros(1, pt, dim))
            nn.init.trunc_normal_(self.lead_queries, std=0.02)
            self.lead_query_norm = nn.LayerNorm(dim)
            self.lead_memory_norm = nn.LayerNorm(dim)
            self.lead_cross = Attention(dim, heads, qk_norm=qk_norm)
            self.lead_spatial_norm = nn.LayerNorm(dim)
        if head_fourier:
            t = torch.arange(pt, dtype=torch.float32)
            cols = [torch.ones(pt)]
            for k in range(1, fourier_modes + 1):
                cols += [torch.sin(2 * math.pi * k * t / pt), torch.cos(2 * math.pi * k * t / pt)]
            self.register_buffer("fourier_basis", torch.stack(cols, dim=1), persistent=False)

        # Reynolds-decomposition decode: the output ``u(t) = ubar + u'(t)`` is built from two
        # heads -- a spatial ``head_mean`` for the time-mean field ``ubar`` and the existing
        # ``head`` for the 20 frames, whose *own* time-mean is subtracted so it carries only the
        # zero-mean fluctuation ``u'(t)``. The model still emits the full recomposed 20-frame
        # field, so every scored metric applies unchanged; the split is an inductive bias, and a
        # metric-aligned one -- ``mvpe`` reads ``ubar`` (the time-averaged wake profile) and
        # ``tke`` reads ``var_t(u')`` (the fluctuation energy), so each head owns one axis. Both
        # heads are zero-initialised, so ``reynolds=True`` and ``reynolds=False`` are the same
        # function (the zero field) at step 0 and the ablation measures the parameterisation.
        self.reynolds = reynolds
        if reynolds:
            self.head_mean = nn.Linear(dim, ph * pw * c_out)
            nn.init.zeros_(self.head_mean.weight)
            nn.init.zeros_(self.head_mean.bias)

        # FiLM on the convolutional paths. Measured motivation: with `cond_conv=False`
        # the operating point reaches the transformer blocks and *nothing else*, so the
        # stem and the full-resolution refinement -- which is where the high spatial
        # frequencies live -- are blind to it. On `val_extrap` the same conditioning
        # vector is worth +0.35 / +1.19 / +0.36 to the U-Net, whose FiLM modulates its
        # whole convolutional trunk, and only +0.07 / +0.44 / -0.09 to STFormer
        # (FINDINGS 23). This closes that structural difference.
        #
        # Zero-initialised, so `cond_conv=True` and `cond_conv=False` are the same
        # function at step 0 and the ablation measures the pathway rather than an init.
        self.cond_conv = cond_conv
        if cond_conv:
            if stem:
                self.film_stem = nn.Linear(cond_width, 2 * stem_channels)
                nn.init.zeros_(self.film_stem.weight)
                nn.init.zeros_(self.film_stem.bias)
            if refine:
                self.film_refine = nn.Linear(cond_width, 2 * refine_channels)
                nn.init.zeros_(self.film_refine.weight)
                nn.init.zeros_(self.film_refine.bias)

        if refine_temporal and refine_multiscale:
            raise ValueError("refine_temporal and refine_multiscale are mutually exclusive")
        self.refine_enabled = refine
        self.refine_stem_skip = refine_stem_skip
        self.refine_temporal = refine_temporal
        if refine:
            self.to_map = nn.Linear(dim, ph * pw * refine_channels)
            refine_in_channels = refine_channels + (stem_channels if refine_stem_skip else 0)
            if refine_temporal:
                # Same input/output contract, but the correction is mixed across the 20 output
                # frames rather than read out one per channel. See :class:`TemporalRefine`.
                self.refine = TemporalRefine(
                    refine_in_channels,
                    refine_channels,
                    pt,
                    c_out,
                    t_hidden=refine_t_hidden,
                    t_kernel=refine_t_kernel,
                )
            elif refine_multiscale:
                # Same contract, but a two-scale down/up conv with a full-res skip -- the U-Net
                # structure STFormer lacks, in the cheap 2D refine. See :class:`MultiScaleRefine`.
                self.refine = MultiScaleRefine(refine_in_channels, refine_channels, pt, c_out)
            else:
                self.refine = nn.Sequential(
                    nn.Conv2d(refine_in_channels, refine_channels, 3, padding=1),
                    nn.GroupNorm(_groups(refine_channels), refine_channels),
                    nn.SiLU(),
                    nn.Conv2d(refine_channels, pt * c_out, 3, padding=1),
                )
                # Zero-initialised for the same reason as ``head``: with both paths silent at
                # step 0, ``refine=True`` and ``refine=False`` are the same function, so the
                # ablation measures the pathway and not a different initialisation.
                nn.init.zeros_(self.refine[-1].weight)
                nn.init.zeros_(self.refine[-1].bias)

        # A second copy of the interface, for simulated samples. Built *last* and under
        # ``<name>_sim`` rather than by nesting both copies in a ModuleList, so every
        # parameter of the real branch keeps the name it has in a shared-interface
        # checkpoint: an ``--init-from`` of an ordinary run key-matches the real branch
        # exactly and reports the sim branch as missing, which is true and is then filled by
        # ``sync_domain_interface``. Nesting would rename all 8.2M parameters and silently
        # train from scratch under ``strict=False`` -- the failure class FINDINGS 12 cost
        # 3.3 GPU-h to find.
        self.domain_interface = domain_interface
        if domain_interface:
            import copy  # noqa: PLC0415

            for name in INTERFACE_MODULES:
                mod = getattr(self, name, None)
                if mod is None:
                    continue  # refine/film modules are optional
                setattr(self, f"{name}_sim", copy.deepcopy(mod))

    def sync_domain_interface(self) -> int:
        """Copy the real interface onto the simulated one. Returns tensors copied.

        Called after an ``--init-from`` load so the two branches *start identical*: the split
        model is then exactly its shared-interface twin at step 0, and the A/B measures the
        split rather than an initialisation lottery. Same discipline as the zero-initialised
        ``refine`` and adaLN gates elsewhere in this file.
        """
        if not self.domain_interface:
            return 0
        copied = 0
        with torch.no_grad():
            for name in INTERFACE_MODULES:
                src, dst = getattr(self, name, None), getattr(self, f"{name}_sim", None)
                if src is None or dst is None:
                    continue
                if isinstance(src, nn.Parameter):
                    dst.copy_(src)
                    copied += 1
                else:
                    # strict=True: the two are deepcopies of one module, so a length
                    # mismatch would mean the copy is not the module it claims to be.
                    for a, b in zip(
                        src.state_dict().values(), dst.state_dict().values(), strict=True
                    ):
                        b.copy_(a)
                        copied += 1
        return copied

    def _domain(self, cond: torch.Tensor | None, b: int, device: torch.device) -> torch.Tensor:
        """``(B,)`` branch index per sample: 1 where the sample is simulated, else 0."""
        if not self.domain_interface or cond is None:
            return torch.zeros(b, dtype=torch.long, device=device)
        return (cond[:, DOMAIN_FEATURE] > 0.5).long()

    def _route(self, name: str, x: torch.Tensor, dom: torch.Tensor) -> torch.Tensor:
        """Apply the real or simulated copy of ``name`` per row of ``x``.

        With ``domain_interface=False`` this is a plain module call, so the unsplit model runs
        exactly the code it ran before. With the split on, each branch sees only its own rows
        -- a single-domain batch therefore costs nothing extra, which is the common case at
        submission and during a real-only stage.
        """
        real = getattr(self, name)
        if not self.domain_interface:
            return real(x)
        sim = getattr(self, f"{name}_sim")
        out: torch.Tensor | None = None
        for idx, mod in ((0, real), (1, sim)):
            sel = (dom == idx).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            y = mod(x[sel])
            if out is None:
                out = y.new_zeros((x.shape[0], *y.shape[1:]))
            out[sel] = y
        assert out is not None  # dom is non-empty, so one branch ran
        return out

    def _pos_table(self, dom: torch.Tensor) -> torch.Tensor:
        """``(B, n_tokens, dim)`` positional table, per-sample branch."""
        if not self.domain_interface:
            return self.pos
        return torch.stack([self.pos[0], self.pos_sim[0]])[dom]

    def _conditioning(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        b = x.shape[0]
        if self.cond_proj is None:
            return self.cond_const[None].expand(b, -1)
        if cond is None:
            cond = x.new_zeros(b, self.cond_features)
        return self.cond_proj(cond)

    def _to_groups(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, H, W, C) -> (B*gt, pt*C, H, W)``, channel-fastest within a group."""
        b, t, h, w, c = x.shape
        gt = self.grid[0]
        x = x.reshape(b, gt, t // gt, h, w, c).permute(0, 1, 2, 5, 3, 4)
        return x.reshape(b * gt, (t // gt) * c, h, w)

    def _from_groups(self, maps: torch.Tensor, b: int) -> torch.Tensor:
        """``(B*gt, pt*C_out, H, W) -> (B, T, H, W, C_out)``, the inverse layout."""
        gt = self.grid[0]
        _, _, h, w = maps.shape
        maps = maps.reshape(b, gt, -1, self.c_out, h, w).permute(0, 1, 2, 4, 5, 3)
        return maps.reshape(b, -1, h, w, self.c_out)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        gt, gh, gw = self.grid
        # Per sample, not per group: the groups are stacked into the batch dimension for
        # the stem alone and are folded back into one sequence before the blocks run.
        c = self._conditioning(x, cond)
        # The raw feature vector, not the projected embedding: the content planes are
        # meant to give the stem the operating point itself, in its own units.
        c_raw = cond if cond is not None else x.new_zeros(b, self.cond_features)

        # Which interface each sample uses, and the same index lifted to the group rows the
        # convolutional path is stacked into. A no-op when the split is off.
        dom = self._domain(cond, b, x.device)
        dom_g = dom.repeat_interleave(gt) if gt > 1 else dom

        model_input = x + self.history_encoder(x) if self.temporal_gru else x
        groups = self._to_groups(model_input)
        if self.cond_content:
            planes = c_raw[:, :, None, None].expand(-1, -1, *groups.shape[-2:])
            if gt > 1:
                planes = planes.repeat_interleave(gt, 0)
            groups = torch.cat([groups, planes], dim=1)
        if self.has_stem:
            feat = self._route("stem", groups, dom_g)  # (B*gt, stem_ch, H, W)
            if self.cond_conv:
                feat = self._film(feat, self._route("film_stem", c, dom), gt)
        else:
            feat = groups  # never read by a skip: stem=False forbids refine_stem_skip
        tokens = self._route("embed", feat, dom_g)  # (B*gt, dim, gh, gw)
        tokens = tokens.reshape(b, gt, -1, gh, gw).permute(0, 1, 3, 4, 2)
        h = tokens.reshape(b, self.n_tokens, -1) + self._pos_table(dom)
        if self.modulation == "bias":
            # The whole conditioning pathway of the trunk: one vector added to every token.
            h = h + c[:, None, :]

        # Per-token conditioning is built here rather than per block, so every block sees
        # the same position-aware vector and the cost is paid once.
        c_blocks = c[:, None, :] + self.cond_pos if self.cond_token else c
        if self.modulation == "shared":
            c_blocks = self.shared_modulation(c)  # (B, 6*dim), one projection for all blocks
        # ``(1, heads, N, N)`` relative-position bias, gathered once and shared by every block.
        attn_bias = self.rel_bias_table[:, self.rel_index][None] if self.rel_pos else None
        if self.recur is not None:
            h = self.recur.inject(h)
        for block in self.blocks:
            h = block(h, c_blocks, attn_bias)
        h = self.norm(h)
        if self.recur is not None:
            self.recur.observe(h)

        t_out, h_, w_, _ = self.out_shape
        pt, ph, pw = self.patch
        if not self.coarse_head:
            out = h.new_zeros(b, t_out, h_, w_, self.c_out)
        elif self.lead_query_decoder:
            q = self.lead_queries.expand(b, -1, -1)
            q = q + self.lead_cross(self.lead_query_norm(q), self.lead_memory_norm(h))
            z = self.lead_spatial_norm(h[:, None] + q[:, :, None])  # (B, pt, N, D)
            coeffs = self._route("head", z, dom).permute(0, 2, 1, 3).reshape(b, self.n_tokens, -1)
        elif self.coarse_head:
            coeffs = self._route("head", h, dom)
        if self.coarse_head and self.head_fourier:
            # (B, N, M*ph*pw*C) -> reconstruct pt frames from the temporal Fourier basis.
            m = self.fourier_basis.shape[1]
            coeffs = coeffs.reshape(b, self.n_tokens, m, ph * pw * self.c_out)
            frames = torch.einsum("tm,bnmc->bntc", self.fourier_basis, coeffs)
            coeffs = frames.reshape(b, self.n_tokens, pt * ph * pw * self.c_out)
        if self.coarse_head:
            out = unpatchify(coeffs, self.patch, (t_out, h_, w_), self.c_out)
        if self.reynolds:
            # ``head`` now carries only the fluctuation: subtract its own time-mean so ``ubar``
            # from ``head_mean`` is the sole owner of the mean, then recompose the 20-frame field.
            out = out - out.mean(dim=1, keepdim=True)
            mean = unpatchify(
                self._route("head_mean", h, dom), (1, ph, pw), (1, h_, w_), self.c_out
            )
            out = mean + out

        if self.refine_enabled:
            pt, ph, pw = self.patch
            # (B, N, ph*pw*rc) -> (B*gt, rc, H, W): the token grid upsampled back to full
            # resolution, laid out so that channel-fastest ordering matches ``unpatchify``.
            m = self._route("to_map", h, dom).reshape(b * gt, gh, gw, ph, pw, -1)
            m = m.permute(0, 5, 1, 3, 2, 4).reshape(b * gt, -1, gh * ph, gw * pw)
            if self.cond_conv:
                m = self._film(m, self._route("film_refine", c, dom), gt)
            refine_input = torch.cat([m, feat], dim=1) if self.refine_stem_skip else m
            out = out + self._from_groups(self._route("refine", refine_input, dom_g), b)
        if self.temporal_anchor:
            history_delta = x[:, :-1] - x[:, -1:]
            out = out + torch.einsum("ot,bthwc->bohwc", self.anchor_weights, history_delta)
        return out

    @staticmethod
    def _film(feat: torch.Tensor, params: torch.Tensor, gt: int) -> torch.Tensor:
        """``feat * (1 + scale) + shift`` from a per-sample parameter vector.

        ``feat`` carries the token-groups in its batch dimension while ``params`` is one
        row per *sample*, so the parameters are repeated across a sample's groups.
        """
        scale, shift = params.chunk(2, dim=-1)
        if gt > 1:
            scale, shift = scale.repeat_interleave(gt, 0), shift.repeat_interleave(gt, 0)
        return feat * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


# --------------------------------------------------------------------------- #
# STFormer3D -- the temporal axis STFormer never had
# --------------------------------------------------------------------------- #
#
# Every STFormer arm run so far used ``patch[0] == 20``, folding the whole window into
# each token's channels. That leaves the trunk with **no temporal axis at all**: no
# locality, no ordering, no mixing beyond one linear map from a token to its 20 output
# frames. The U-Net it loses to has a ``Conv3d`` stem with kernel 7 in all three axes,
# temporal attention with rotary embeddings, and a T5-style relative position bias on
# time.
#
# The measured deficit matches that difference precisely. Against ``unet_cond``,
# STFormer's ``tke`` -- the relative L2 of the *temporal variance field*, a purely
# temporal statistic -- is -0.40 / -0.37 / -0.42 on the three real splits: **uniform**
# across three regimes that differ in extrapolation, transfer and data volume. A
# split-independent deficit is a property of the model class, not of the data. Meanwhile
# rel_l2 loses only on ``val_extrap`` and mvpe is level.
#
# So this is ROADMAP N12 -- "the genuinely under-explored axis... nobody compares how to
# consume the 20-frame input" -- built at last.


class FactorizedBlock(nn.Module):
    """adaLN block with attention factorised into space then time.

    Joint attention over a space-time token grid costs ``(gt*S)^2``; at ``patch=(5,2,2)``
    that is 2048 tokens against the spatial-only model's 512, i.e. 16x. Factorising into
    spatial attention *within* each time-group and temporal attention *across* groups at
    each spatial position costs ``gt*S^2 + S*gt^2`` instead -- 4x here, not 16x -- which
    is what makes the temporal axis affordable at all. Same decomposition as ViViT's
    factorised encoder.

    Conditioning enters as ``1 + scale`` / ``1 + gate`` throughout, so the block is an
    ordinary transformer block at initialisation for the same reason
    :class:`ConditionedBlock` is.
    """

    def __init__(
        self, dim: int, heads: int, cond_dim: int, mlp_ratio: float = 4.0, qk_norm: bool = True
    ) -> None:
        super().__init__()
        self.norm_s, self.norm_t, self.norm_m = (nn.LayerNorm(dim) for _ in range(3))
        self.attn_s = Attention(dim, heads, qk_norm=qk_norm)
        self.attn_t = Attention(dim, heads, qk_norm=qk_norm)
        self.mlp = FeedForward(dim, mult=int(mlp_ratio))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 9))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    @staticmethod
    def _mod(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, x: torch.Tensor, cond: torch.Tensor, gt: int) -> torch.Tensor:
        """``x`` is ``(B, gt*S, D)`` in (time, space) raster order."""
        b, n, d = x.shape
        s = n // gt
        m = self.modulation(cond).chunk(9, dim=-1)
        sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = m

        # Spatial: every token attends within its own time slice.
        h = self._mod(self.norm_s(x), sh1, sc1).reshape(b * gt, s, d)
        x = x + (1.0 + g1[:, None]) * self.attn_s(h).reshape(b, n, d)

        if gt > 1:
            # Temporal: every spatial position attends across time. The transpose is what
            # makes the time axis contiguous; without it this would attend over space again.
            h = self._mod(self.norm_t(x), sh2, sc2).reshape(b, gt, s, d)
            h = h.transpose(1, 2).reshape(b * s, gt, d)
            h = self.attn_t(h).reshape(b, s, gt, d).transpose(1, 2).reshape(b, n, d)
            x = x + (1.0 + g2[:, None]) * h

        return x + (1.0 + g3[:, None]) * self.mlp(self._mod(self.norm_m(x), sh3, sc3))


class STFormer3D(nn.Module):
    """Space-time STFormer: ``Conv3d`` stem and refinement, factorised attention.

    Differs from :class:`STFormer` in exactly the thing that was never tested -- time is
    an axis rather than a channel dimension -- and keeps everything that was measured to
    work: the full-resolution refinement with a stem skip, the zero-initialised decode,
    identity-at-init conditioning, and FiLM on the convolutional paths.

    The convolutions are factorised into ``(1,3,3)`` spatial and ``(3,1,1)`` temporal
    rather than a dense ``(3,3,3)``: dense costs 27 taps against 12 for the pair, and a
    96-channel dense 3D stem at full resolution is ~200 GFLOP/sample, which is more than
    the entire attention trunk. This is why the U-Net costs 34 ms/window.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        out_shape: tuple[int, int, int, int] = (20, 32, 64, 3),
        patch: tuple[int, int, int] = (5, 2, 2),
        dim: int = 384,
        depth: int = 8,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        stem_channels: int = 48,
        refine: bool = True,
        refine_channels: int = 48,
        qk_norm: bool = True,
        cond_conv: bool = True,
        cond_mode: str = "operating_point",
        cond_features: int | None = None,
    ) -> None:
        super().__init__()
        if cond_mode not in COND_MODES:
            raise ValueError(f"cond_mode must be one of {COND_MODES}, got {cond_mode!r}")
        # Resolved from the mode rather than defaulted to a constant, matching STFormer:
        # a mode-width mismatch is a checkpoint that loads and then reads a conditioning
        # vector one element short of what it was trained on.
        cond_features = cond_vec_dim(cond_mode) if cond_features is None else cond_features
        t_in, h, w, c_in = in_shape
        t_out, h_out, w_out, c_out = out_shape
        if (h, w) != (h_out, w_out) or t_in != t_out:
            raise ValueError("STFormer3D decodes on the input token grid, so shapes must match")
        self.in_shape, self.out_shape, self.patch = in_shape, out_shape, patch
        self.grid = _grid((t_in, h, w), patch)
        self.gt, gh, gw = self.grid
        self.n_tokens = math.prod(self.grid)
        self.c_out, self.cond_features = c_out, cond_features
        pt, ph, pw = patch
        g = _groups(stem_channels)

        self.cond_proj = (
            nn.Sequential(nn.Linear(cond_features, dim), nn.SiLU(), nn.Linear(dim, dim))
            if cond_mode == "operating_point"
            else None
        )
        if self.cond_proj is None:
            self.cond_const = nn.Parameter(torch.zeros(dim))

        def conv_block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(cin, cout, (1, 3, 3), padding=(0, 1, 1)),
                nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0)),
                nn.GroupNorm(_groups(cout), cout),
                nn.SiLU(),
            )

        self.stem = nn.Sequential(
            conv_block(c_in, stem_channels), conv_block(stem_channels, stem_channels)
        )
        self.embed = nn.Conv3d(stem_channels, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            FactorizedBlock(dim, heads, dim, mlp_ratio, qk_norm=qk_norm) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

        self.head = nn.Linear(dim, pt * ph * pw * c_out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.cond_conv = cond_conv
        if cond_conv:
            self.film_stem = nn.Linear(dim, 2 * stem_channels)
            nn.init.zeros_(self.film_stem.weight)
            nn.init.zeros_(self.film_stem.bias)
        self.refine_enabled = refine
        if refine:
            self.to_map = nn.Linear(dim, pt * ph * pw * refine_channels)
            self.refine = nn.Sequential(
                nn.Conv3d(
                    refine_channels + stem_channels, refine_channels, (1, 3, 3), padding=(0, 1, 1)
                ),
                nn.Conv3d(refine_channels, refine_channels, (3, 1, 1), padding=(1, 0, 0)),
                nn.GroupNorm(_groups(refine_channels), refine_channels),
                nn.SiLU(),
                nn.Conv3d(refine_channels, c_out, 3, padding=1),
            )
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)
        del g

    def _conditioning(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        b = x.shape[0]
        if self.cond_proj is None:
            return self.cond_const[None].expand(b, -1)
        if cond is None:
            cond = x.new_zeros(b, self.cond_features)
        return self.cond_proj(cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        gt, gh, gw = self.grid
        pt, ph, pw = self.patch
        c = self._conditioning(x, cond)

        feat = self.stem(x.permute(0, 4, 1, 2, 3))  # (B, sc, T, H, W)
        if self.cond_conv:
            scale, shift = self.film_stem(c).chunk(2, dim=-1)
            feat = feat * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]

        tokens = self.embed(feat)  # (B, dim, gt, gh, gw)
        h = tokens.flatten(2).transpose(1, 2) + self.pos  # (B, N, dim), (t,h,w) raster
        for block in self.blocks:
            h = block(h, c, gt)
        h = self.norm(h)

        t_out, h_, w_, _ = self.out_shape
        out = unpatchify(self.head(h), self.patch, (t_out, h_, w_), self.c_out)

        if self.refine_enabled:
            m = self.to_map(h).reshape(b, gt, gh, gw, pt, ph, pw, -1)
            m = m.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, -1, gt * pt, gh * ph, gw * pw)
            det = self.refine(torch.cat([m, feat], dim=1))  # (B, c_out, T, H, W)
            out = out + det.permute(0, 2, 3, 4, 1)
        return out
