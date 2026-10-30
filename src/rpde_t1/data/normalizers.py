"""Deprecated location. The normalizers moved to :mod:`rpde_t1.features`.

Importing anything from ``rpde_t1.data`` executes that package's ``__init__``, which
imports ``h5py`` -- absent from the evaluation container. Since a gaussian-normalized
submission needs ``gaussian_stats`` at inference, the definitions had to leave this
package. Kept as a shim so existing callers keep working.
"""

from __future__ import annotations

from ..features import (
    REAL_MEAN_IN,
    REAL_MEAN_TARGET,
    REAL_STD_IN,
    REAL_STD_TARGET,
    SIM_MEAN,
    SIM_STD,
    GaussianStats,
    gaussian_stats,
    normalized_mse,
)

__all__ = [
    "REAL_MEAN_IN",
    "REAL_MEAN_TARGET",
    "REAL_STD_IN",
    "REAL_STD_TARGET",
    "SIM_MEAN",
    "SIM_STD",
    "GaussianStats",
    "gaussian_stats",
    "normalized_mse",
]
