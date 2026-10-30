"""Frozen neural AoA estimator: a §122 anchor + learned residual, used as a conditioning source.

Trained by ``scripts/eval/supcon_estimator.py --anchor-aoa`` (real-only). This wraps the checkpoint
so both training (``data.dataset`` via ``features.estimate_operating_point``) and inference
(``features.estimate_batch``) can add its residual to §122's AoA when ``RPDE_AOA_NET`` points at a
checkpoint -- an A/B where the operating-point AoA is swapped in *both* stages. Torch is imported
only here, and only when the override is active, so the default estimate path stays torch-light.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from ..features import AOA_MAX, FREESTREAM_ROWS


class _Enc(nn.Module):  # identical to scripts/eval/supcon_estimator.py's Encoder
    def __init__(self, t_in: int = 20, c_in: int = 3, dim: int = 64):
        super().__init__()
        ch = t_in * c_in
        self.net = nn.Sequential(
            nn.Conv2d(ch, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.head = nn.Linear(128, dim)

    def forward(self, x):
        b, t, h, w, c = x.shape
        z = self.net(x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)).mean(dim=(2, 3))
        return self.head(z)


class AoANet:
    """Loads the checkpoint; ``residual_deg`` maps physical windows -> AoA residual in degrees."""

    def __init__(self, path: str, device: str = "cpu"):
        ck = torch.load(path, map_location=device)
        self.enc = _Enc(dim=ck["dim"]).to(device).eval()
        self.enc.load_state_dict(ck["encoder"])
        self.dec = nn.Sequential(
            nn.Linear(ck["dim"], 128), nn.GELU(), nn.Linear(128, 2)
        ).to(device).eval()
        self.dec.load_state_dict(ck["decode"])
        self.device = device

    @torch.no_grad()
    def residual_deg(self, windows: np.ndarray) -> np.ndarray:
        """``(N, T, H, W, C)`` physical windows -> ``(N,)`` AoA residual in degrees.

        Freestream-normalised exactly as the net was trained (window / mean|u| over the top rows).
        """
        x = np.asarray(windows, dtype=np.float32)
        ufs = np.abs(x[:, :, :FREESTREAM_ROWS, :, 0]).mean(axis=(1, 2, 3))
        xn = x / np.maximum(ufs, 1e-6).reshape(-1, 1, 1, 1, 1)
        out = self.dec(self.enc(torch.from_numpy(xn).to(self.device)))[:, 1].cpu().numpy()
        return out.astype(np.float64) * AOA_MAX


@lru_cache(maxsize=4)
def load_aoa_net(path: str, device: str = "cpu") -> AoANet:
    return AoANet(path, device)


def install_override(path: str, device: str = "cpu") -> None:
    """Install the neural-AoA override into ``features`` (adds the residual to §122's AoA)."""
    from .. import features  # noqa: PLC0415

    net = load_aoa_net(path, device)

    def _fn(windows, aoa):
        return np.clip(aoa + net.residual_deg(windows), *features.AOA_CLIP_DEG)

    features._aoa_override = _fn


def install_from_env(device: str = "cpu") -> bool:
    """Install estimator overrides from env: RPDE_AOA_NET (AoA) and RPDE_RE_CORNERS (Re).

    Returns whether anything installed.
    """
    import os  # noqa: PLC0415

    from .. import features  # noqa: PLC0415

    path = os.environ.get("RPDE_AOA_NET")
    if path:
        install_override(path, device)
    if os.environ.get("RPDE_RE_CORNERS"):
        features._re_override = features.corners3_reynolds
    return bool(path) or bool(os.environ.get("RPDE_RE_CORNERS"))
