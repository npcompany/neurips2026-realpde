"""Mean-preserving rescaling of temporal fluctuations.

Three places need to scale a prediction's fluctuations toward some other variance map:
the closed-form geometric blend (:func:`rpde_t1.models.predictor.rescale_fluctuations`),
the learned second-moment head (:func:`rpde_t1.models.tke_head.apply_gain`), and the TKE
ceiling ladder (:func:`rpde_t1.score.diagnostics.rescale_fluctuations`). They differ in
*how they compute the gain* -- a closed form, a network, an oracle map -- and used to
each carry their own copy of what to do with it.

That copy is the load-bearing part, because two properties of ``tke_score`` follow from
it rather than from any gain:

* **``mvpe_score`` is invariant.** The gain is constant in time, so the time mean is
  untouched, and ``mvpe`` time-averages before comparing. Measured unchanged to every
  printed digit.
* **The ladder measures what the submission ships.** A ceiling computed by a slightly
  different operation than the deployed one is not a ceiling. One applier makes that
  true by construction instead of by inspection.

Numpy only, and no imports beyond the package root: ``score/`` must keep working on a
machine with no torch installed, and ``models/`` ships inside the submission bundle.
"""

from __future__ import annotations

import numpy as np

from .. import N_MEASURED

#: Bound on the multiplicative correction. The ratio explodes where the prediction is
#: nearly static -- chiefly inside the PIV measurement hole, where the target is
#: identically zero and the scorer ignores the element anyway.
DEFAULT_MAX_GAIN = 10.0


def apply_fluctuation_gain(
    prediction: np.ndarray,
    gain: np.ndarray,
    max_gain: float = DEFAULT_MAX_GAIN,
    n_channels: int = N_MEASURED,
) -> np.ndarray:
    """Scale each pixel's fluctuations about its own time mean by ``gain``.

    ``prediction`` is ``(N, T, H, W, C)`` and ``gain`` is ``(N, H, W)`` -- one factor per
    pixel, constant over time, which is what leaves the time mean (and therefore ``mvpe``)
    exactly where it was. A ``(N, H, W, n_channels)`` gain scales each velocity component
    by its own factor (the per-component head); the time mean is preserved either way.

    Only the leading ``n_channels`` are touched. The default is the two measured
    velocity components; a caller scoring a set whose pressure plane is also non-zero
    passes the width the scorer is actually using, or it would rescale two channels
    against an energy map built from three.

    A *uniform* gain is known to be harmful -- 1.4 moves tke 71.1 -> 59.5 -- because the
    predicted TKE map is mis-localised (spatial correlation 0.758) rather than merely
    under-amplitude, so amplifying it amplifies its error. Scaling per pixel toward a map
    that is actually supported is the opposite operation.
    """
    g = np.clip(np.asarray(gain, dtype=np.float32), 0.0, max_gain)
    g = g[:, None, :, :, None] if g.ndim == 3 else g[:, None]
    out = prediction.copy()
    sl = out[..., :n_channels]
    mean_t = sl.mean(axis=1, keepdims=True)
    out[..., :n_channels] = mean_t + (sl - mean_t) * g
    return out


def apply_fluctuation_gain_torch(
    prediction, gain, max_gain: float = DEFAULT_MAX_GAIN, n_channels: int = N_MEASURED
):
    """Torch twin of :func:`apply_fluctuation_gain`, for a device-resident prediction.

    Exists because this is the single most expensive step in the deployed second-moment path
    and it is pure elementwise arithmetic: measured **1.797 ms/window** in numpy against
    **0.011** here, 158x, on a tensor that the chunk loop already holds on the device
    (FINDINGS 63m). Four full passes over a 103 MB array is simply not work a host should do.

    Same contract, same clip, same mean-preservation -- ``mvpe`` stays invariant -- and
    ``tests/test_tke_rescale.py`` asserts the two agree to fp32 tolerance on real windows
    rather than on random noise, since the masks and the wake are structural.
    """
    import torch  # noqa: PLC0415  (numpy-only callers must not pay this import)

    g = torch.clamp(gain, 0.0, max_gain)
    g = g[:, None, :, :, None] if g.dim() == 3 else g[:, None]
    out = prediction.clone()
    sl = out[..., :n_channels]
    mean_t = sl.mean(dim=-4, keepdim=True)
    out[..., :n_channels] = mean_t + (sl - mean_t) * g
    return out
