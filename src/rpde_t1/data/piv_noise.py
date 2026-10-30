"""Synthetic PIV measurement noise, at the level measured on the delivered real data.

Calibrated, not invented. ``FINDINGS`` section 3 splits the in-plane continuity residual
by its spatial autocorrelation and puts ~16% of its variance in a pixel-decorrelated
component, implying a per-component velocity noise of ``sigma_n`` ~ 2.5-2.7 mm/s. That is
close to a *fixed* floor while the freestream varies 6.7x across the Reynolds ladder, so in
relative terms it runs from **2.4% of freestream at Re=3750 to 1.0% at Re=26700** -- the two
endpoints this module interpolates between.

**Why white.** Section 6f measured the spectra above the coarse Nyquist: real carries a
high-k shelf 7x above sim's, and that shelf *is* the sensor floor -- it survives the 2x
stride to the scored grid, which a spatially correlated field would not. So the
pixel-decorrelated component is the part worth injecting, and a correlation length is
offered (``corr_px``) but defaults to zero.

**Why relative, and applied after normalisation.** Training runs in non-dimensional space,
where the freestream is 1 by construction, so a noise level expressed as a fraction of
freestream is the same number before and after the division. Injecting post-normalisation
keeps the ``u_fs`` estimator on the clean field; the estimator averages over 20 frames and
a whole upstream band, so noise at this level moves it by <0.1% either way and the choice
is not measurable -- but it is stated because "which side of the normaliser" is exactly the
kind of detail that silently makes two arms incomparable.
"""

from __future__ import annotations

import numpy as np

#: The two measured endpoints: ``(Re, sigma_n / u_freestream)``. FINDINGS section 3.
PIV_SIGMA_ENDPOINTS = ((3750.0, 0.024), (26700.0, 0.010))


def piv_sigma_rel(re: float) -> float:
    """Per-component velocity noise as a fraction of freestream, at Reynolds ``re``.

    Log-linear in ``Re`` between the two measured endpoints, and **clamped** outside them:
    the fit is an interpolation of two numbers, and continuing its slope down to Re=0 would
    manufacture a noise level nothing measured. Clamping makes the extrapolated cells
    (which is what ``sim_val_extrap`` is made of) carry the nearest measured level instead.
    """
    (re_lo, s_lo), (re_hi, s_hi) = PIV_SIGMA_ENDPOINTS
    if re <= re_lo:
        return s_lo
    if re >= re_hi:
        return s_hi
    frac = np.log(re / re_lo) / np.log(re_hi / re_lo)
    return float(np.exp(np.log(s_lo) + frac * np.log(s_hi / s_lo)))


def add_piv_noise(
    fields: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    n_measured: int = 2,
    corr_px: float = 0.0,
) -> np.ndarray:
    """``fields`` + additive Gaussian noise of std ``sigma`` on the measured channels.

    ``(T, H, W, C)`` in, same shape out, a fresh array. Only the first ``n_measured``
    channels are touched: pressure is not a PIV measurement, so noising it would model
    nothing (and in this study it is zeroed anyway).

    ``corr_px > 0`` low-pass filters the noise with a separable box of that width and
    renormalises the variance, giving a correlated field of the same total energy. Left at
    zero by default -- see the module docstring.
    """
    if sigma <= 0.0:
        return fields
    out = np.array(fields, dtype=np.float32, copy=True)
    noise = rng.standard_normal((*fields.shape[:-1], n_measured)).astype(np.float32)
    if corr_px > 0:
        k = max(1, int(round(corr_px)))
        if k > 1:
            kernel = np.ones(k, dtype=np.float32) / k
            for axis in (-3, -2):  # H then W; time is left alone
                noise = np.apply_along_axis(
                    lambda m, _k=kernel: np.convolve(m, _k, mode="same"), axis, noise
                )
            # A box filter divides the variance by k per axis; restore it so ``sigma``
            # means the same thing whether or not the noise is correlated.
            noise *= float(k)
    out[..., :n_measured] += noise * np.float32(sigma)
    return out
