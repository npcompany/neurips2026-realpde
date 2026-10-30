"""A fourth output channel that carries the temporal-variance map, separately from the frames.

Why (FINDINGS 80f, 126, 127, 128). The shipped fine-tune's ``tke`` terms make the *frames* carry
the second moment: every lead is inflated a little and lead 20 carries a mostly uncorrelated
fluctuation (126). That costs ``rel_l2`` at every lead; turning the terms off buys +0.27 ``rel_l2``
and loses 5.4 ``tke``, of which the post-hoc head wins back 2.6 (80f). This module lets the frames
stay at the conditional mean and asks a separate output channel, trained end to end on the same
trunk, for the variance map -- which is then spent post hoc through the mean-preserving gain or
127's exact fill. A probe of the shipped trunk (128) found no readable map information beyond the
cached maps; this is the end-to-end test of the same question, requested despite that probe.

What the channel predicts. A per-pixel **log ratio** of the target's temporal-variance map over
the input window's own map, emitted as 20 frames like the other channels and read as their time
mean. The input map is known at inference, scale-free under the freestream normaliser and
unbiased in total (5e), and a zero-initialised head makes the estimate start *at* the input map
(tke_score ~74.9) rather than at zero -- the same reason the shipped head predicts a ratio.

Where it lives. ``TrainConfig.kmap_channel`` widens ``out_shape`` from 3 to 4 channels; the frames
are channels ``0..2`` exactly as before and channel ``3`` is the map. ``losses.kmap_loss`` grades
it with the scorer's own relative L2 on the map. ``ResidualWrapper`` anchors only the physical
channels. ``predictor.make_predict`` strips the channel before anything downstream sees the field
and hands its time mean back through ``kmap_out``. :func:`widen_output_channels` lets a 3-channel
checkpoint initialise a 4-channel run with the new slots at zero.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from .. import N_MEASURED, C

#: index of the map channel in a widened output
KMAP_INDEX = C

_HEAD = re.compile(r"(^|\.)head(_sim)?\.(weight|bias)$")
_REFINE_OUT = re.compile(r"(^|\.)refine(_sim)?\.3\.(weight|bias)$")
_UNSUPPORTED = re.compile(r"(^|\.)(head_mean|refine\.out|refine\.to_frames)(_sim)?\.")


def kmap_log_ratio(pred: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` time-mean of the map channel of a ``(B, T, H, W, >=4)`` output."""
    if pred.shape[-1] <= KMAP_INDEX:
        raise ValueError(f"no map channel: output has {pred.shape[-1]} channels")
    return pred[..., KMAP_INDEX].mean(dim=1)


def kmap_estimate(log_ratio: np.ndarray, k_input: np.ndarray) -> np.ndarray:
    """The channel's estimate of the target's variance map, ``k_input * exp(r)``."""
    return np.asarray(k_input, dtype=np.float64) * np.exp(np.asarray(log_ratio, dtype=np.float64))


def input_energy_map(x: np.ndarray) -> np.ndarray:
    """``(N, H, W)`` temporal fluctuation energy of a window's measured channels (numpy)."""
    u = x[..., 0].astype(np.float64)
    v = x[..., 1].astype(np.float64)
    return 0.5 * (u.var(axis=1) + v.var(axis=1))


def _widen_leading(t: torch.Tensor, c_old: int, c_new: int) -> torch.Tensor:
    """Pad a tensor whose leading dim is ``n * c_old`` (channel fastest) to ``n * c_new``."""
    if t.shape[0] % c_old:
        raise ValueError(f"leading dim {t.shape[0]} is not a multiple of {c_old}")
    n = t.shape[0] // c_old
    out = t.new_zeros((n, c_new, *t.shape[1:]))
    out[:, :c_old] = t.reshape(n, c_old, *t.shape[1:])
    return out.reshape(n * c_new, *t.shape[1:])


def widen_output_channels(
    sd: dict[str, torch.Tensor], want: dict[str, torch.Tensor], c_old: int = C, c_new: int = C + 1
) -> dict[str, torch.Tensor]:
    """Widen a ``c_old``-channel STFormer state dict to ``c_new`` output channels.

    Only the two output layers change shape: the coarse ``head`` (``Linear(dim, pt*ph*pw*c)``,
    channel fastest in the flattened patch, see ``transformer.unpatchify``) and the last refine
    conv (``Conv2d(rc, pt*c, 3)``, frame-major with channel fastest, the layout the refine's
    ``_from_groups`` reshapes). New slots are zero, so the widened model emits exactly the old
    frames plus a zero map channel. ``want`` is the target model's state dict: a key whose shape
    already matches is left alone (a 4-channel checkpoint resuming), and the other decoders this
    file does not know how to widen (``head_mean``, the temporal/multiscale refines) raise.
    """
    out = dict(sd)
    for k, t in sd.items():
        w = want.get(k)
        if w is None or tuple(w.shape) == tuple(t.shape):
            continue
        if _UNSUPPORTED.search(k):
            raise NotImplementedError(f"cannot widen {k} to {c_new} output channels")
        if _HEAD.search(k) or _REFINE_OUT.search(k):
            widened = _widen_leading(t, c_old, c_new)
            if tuple(widened.shape) != tuple(w.shape):
                raise ValueError(
                    f"{k}: widened to {tuple(widened.shape)}, model wants {tuple(w.shape)}"
                )
            out[k] = widened
    return out


__all__ = [
    "KMAP_INDEX",
    "N_MEASURED",
    "input_energy_map",
    "kmap_estimate",
    "kmap_log_ratio",
    "widen_output_channels",
]
