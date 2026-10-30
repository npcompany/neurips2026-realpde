"""Torch twins of the scorer's temporal-energy maps, used by the second-moment head
(``models/tke_head.py``) at fit and inference time.

The flow-matching generator this module used to hold (``FlowVelocity``, the integrators and the
ensemble reduction) is not part of this package: no served model samples.
"""

from __future__ import annotations

import torch

from .. import N_MEASURED


def kinetic_energy_map(fields: torch.Tensor) -> torch.Tensor:
    """``(..., T, H, W, C) -> (..., H, W)``, the scorer's ``0.5 * (var_t u + var_t v)``.

    Torch twin of :func:`rpde_t1.models.predictor.kinetic_energy_map`, needed because the
    ensemble is reduced on the device: materialising ``n_samples`` windows in numpy to
    take one mean would move the whole point of the reduction to the host.
    """
    return kinetic_energy_map_components(fields).sum(dim=-1)


def kinetic_energy_map_components(fields: torch.Tensor) -> torch.Tensor:
    """``(..., T, H, W, C) -> (..., H, W, 2)``: each component's ``0.5 * var_t``, unsummed.

    Torch twin of :func:`rpde_t1.models.predictor.kinetic_energy_map_components`.
    """
    f = fields[..., :N_MEASURED]
    return 0.5 * ((f - f.mean(dim=-4, keepdim=True)) ** 2).mean(dim=-4)
