"""Loss terms, written to mirror the scored metrics rather than to proxy them.

The scoring function is fully known, so each term below is defined against the
scorer's own arithmetic:

* ``field_loss`` tracks ``rel_l2_score`` -- accuracy on ``u, v`` (and ``p`` on
  simulated samples only, where it exists as free extra supervision).
* ``tke_loss`` tracks ``tke_score`` exactly: the relative L2 of the *temporal
  variance field*. This is the term a plain MSE objective cannot supply, because
  an MSE-optimal predictor converges to the conditional mean, whose variance is
  only the predictable fraction of the truth (measured: ~33% at lead 20). That
  variance collapse is why the CNO baseline sits at a TKE error of 0.71.
* ``spectral_loss`` matches how fluctuation energy is distributed across
  timescales, which ``tke_loss`` alone does not constrain -- a prediction can have
  the right total variance with the wrong temporal structure.
* ``mvpe_loss`` tracks ``mvpe_score`` exactly: the relative L2 of the *time-averaged*
  ``(u, v)`` profile at the scorer's 9x4 probe stations, each column normalised on its
  own. Unlike the two above it constrains the **first** moment, which the field term
  already supervises everywhere -- the difference is that it is normalised per column,
  so a wake station with a small velocity counts as much as the freestream.
* ``sps_loss`` is a differentiable relaxation of the SPS functional itself, so the
  interval heads optimise the scored quantity instead of a likelihood proxy.

One subtlety runs through all of this: the model works in **non-dimensional**
space (each sample divided by its freestream speed), but ``SIGMA_GLOBAL`` in the
scorer is a **physical** constant. Any term involving sigma therefore has to
convert widths back to physical units first; :func:`sps_loss` takes ``u_fs`` for
exactly that reason.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Re-exported for the callers that import them from here (and via ``train/__init__``).
# The definitions live in ``rpde_t1/__init__`` because ``models.interval`` and
# ``models.bounds`` need the same two numbers and cannot import this module.
from .. import N_MEASURED, SIGMA_GLOBAL, C, mvpe_probe_indices


def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    huber_delta: float = 0.0,
) -> torch.Tensor:
    """Per-sample relative L2, matching the scorer's ``rel_l2_per_sample``.

    ``huber_delta > 0`` replaces the numerator's L2 norm with the **Huber pseudo-norm**
    ``sqrt(2*beta*sum_i rho_beta(e_i))``, where ``rho_beta`` is smooth-L1. That constant is not
    cosmetic: ``2*beta*rho_beta(e) == e**2`` exactly whenever ``|e| < beta``, so the pseudo-norm
    **is** the L2 norm on the body of the error distribution and only departs from it in the tail,
    where it grows like ``sqrt(|e|)`` instead of ``|e|``. ``huber_delta = 0`` is therefore not an
    approximation of the plain form but literally it, which is what makes an A/B on this argument
    a clean one-factor comparison (the same property ``field_loss``'s ``huber_mix = 0`` has, and
    ``tests/test_train.py`` pins both).

    Why it belongs on a *relative* norm at all. The denominator is a constant per sample, so the
    per-element gradient of the plain form is ``e_i / (||e|| ||t||)`` -- still linear in ``e_i``,
    i.e. within a window the rare large-error pixels dominate exactly as they do under an absolute
    MSE. FINDINGS 28 measured that tail dominance to be the single largest lever on this task
    (``huber_delta 0.005``, +0.49 SPS), and 55a records the lever being unavailable to every
    relative/ceiling arm ever run, because those set ``--w-field 0`` and the tail-cap lived only
    on ``field_loss``. This is the argument that removes that confound.
    """
    p, t = _flatten_per_sample(pred), _flatten_per_sample(target)
    denom = torch.linalg.vector_norm(t, dim=1).clamp_min(eps)
    if huber_delta <= 0:
        return torch.linalg.vector_norm(p - t, dim=1) / denom
    rho = F.smooth_l1_loss(p, t, reduction="none", beta=huber_delta)
    return (2.0 * huber_delta * rho.sum(dim=1)).clamp_min(0.0).sqrt() / denom


#: Smallest fluctuation RMS the amplitude-free loss will normalise by, in freestream-normalised
#: units. See the third paragraph of ``amplitude_free_field_loss``: it bounds the gradient
#: amplification at 1/FLUCT_FLOOR and is what keeps near-laminar simulated windows from producing
#: a NaN 3,000 steps into a pretrain.
FLUCT_FLOOR = 0.02


def amplitude_free_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    has_p: torch.Tensor | None = None,
    huber_delta: float = 0.0,
    huber_mix: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Field loss that teaches the mean flow and the fluctuation SHAPE, never its amplitude.

    For the **simulated** stage only, and the reason is measured rather than aesthetic. Sim
    over-fluctuates real by **1.8-20x** in energy (FINDINGS 13, README), and 37e split the
    pretraining-budget effect along exactly that seam: ``rel_l2`` saturates in ~2,000 steps while
    ``tke`` keeps moving to ~8,000, i.e. what accumulates past 2k is the second moment -- the part
    that does not transfer. 37d then measured zero-shot real ``tke`` degrading *monotonically* in
    budget, falling below the 66.67 smooth-prediction floor past ~8k. So a longer pretrain buys
    mean-flow specialisation and pays for it in an inflated fluctuation amplitude that the fine-tune
    only partly removes.

    This closes that channel at the objective instead of correcting it in data space. The window is
    split into its temporal mean and its fluctuation; the mean keeps the ordinary loss, and the
    fluctuation is compared **after each side is divided by its own RMS**, so a global rescaling of
    the predicted fluctuation leaves the loss unchanged and sim carries no information about how
    much fluctuation to produce. Real, which does carry the right amount, sets it in stage iii.

    Strictly more general than ``rescale_fluct`` (FINDINGS 13, "the largest sim2real gain
    measured", closing 78-85% of the tke gap): that rescales sim's *data* toward real's measured
    intensity, which still commits to an amplitude estimate and only exists for the 81 paired
    cells. Here there is no amplitude to get wrong, and it applies to the 19 sim-only cells too.

    **The normaliser is detached, and that is load-bearing.** A truly scale-invariant loss leaves
    the predicted amplitude free to drift anywhere, including to zero, which would hand the
    fine-tune a badly scaled initialisation rather than a neutral one. Detaching makes the
    normaliser a per-step constant: the loss is invariant to the scale it *arrives* with, while the
    gradient still points at the shape. If the amplitude drifts anyway it is visible as a worse
    post-fine-tune ``rel_l2``, which is measurable rather than silent.

    **Falsifiable prediction.** If amplitude inflation is really why more pretraining costs
    ``val_extrap`` tke, then under this objective the budget ladder should FLATTEN -- 30k should
    stop losing to 8k on that axis. A ladder that still tilts says the mechanism is something else.
    """
    mean_p = pred.mean(dim=1, keepdim=True)
    mean_t = target.mean(dim=1, keepdim=True)
    fluct_p = pred - mean_p
    fluct_t = target - mean_t

    # Per window and channel, over time and space: the amplitude this loss must not grade.
    #
    # **In float32 and with a RELATIVE floor, both for one reason found by smoke test.** At
    # initialisation the model's own fluctuation is near zero, so an absolute ``clamp_min(1e-6)``
    # amplifies the gradient through ``1 / rms_p`` by ~1e6, the fp16 GradScaler overflows and the
    # first step reports ``|g| = nan``. Flooring ``rms_p`` at a fraction of the target's own RMS
    # bounds that amplification at ``1 / floor_frac`` -- 20x here, which still spans sim's measured
    # 1.8-20x over-fluctuation, so the invariance survives over the range it exists for. Under
    # ``--zero-p`` the unmeasured pressure plane is identically zero and would hit the floor on
    # every window; it never reaches the loss (``field_loss`` slices ``:N_MEASURED``), and the
    # relative floor keeps it finite regardless.
    # Normalisation applies to the MEASURED channels only, and the pressure plane is passed
    # through untouched. Under ``--zero-p`` the pressure fluctuation is identically zero, so
    # normalising it divides by the floor on every window; `field_loss`'s auxiliary pressure term
    # then sees ~1e7 values and the first backward reports `|g| = inf`. Found by smoke test, twice
    # -- the first fix (a relative floor) was necessary and not sufficient, because the floor is
    # relative to a target RMS that is itself zero there. Pressure is auxiliary physics and its
    # amplitude was never the transfer problem, so it keeps the ordinary loss.
    dims = (1, 2, 3)
    mp32 = fluct_p[..., :N_MEASURED].float()
    mt32 = fluct_t[..., :N_MEASURED].float()
    # An ABSOLUTE floor on both normalisers, and the third attempt at this line. Dividing by a
    # per-window RMS is unbounded when that RMS is small, and `sim_all` contains near-laminar
    # windows -- low Reynolds at AoA 0, where the true fluctuation is essentially zero. A floor
    # expressed as a FRACTION of the target's own RMS therefore collapses with it: at
    # rms_t = 1e-6 the floor was 5e-8 and the gradient amplification 2e7. The first 8k pretrain
    # trained cleanly for 3,000 steps (loss 0.95 -> 0.54) and then went NaN, which no
    # initialisation test could catch.
    #
    # `FLUCT_FLOOR` is in freestream-normalised units, where fields are O(1) and real
    # fluctuations are O(0.01-0.3). So the loss is scale-invariant over the range where
    # fluctuation actually exists -- which is where sim's measured 1.8-20x inflation lives -- and
    # degrades to a plain reweighted difference below it, where "shape" has no meaning anyway.
    # Amplification is bounded by 1 / FLUCT_FLOOR = 50 whatever the data does.
    rms_t = mt32.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(FLUCT_FLOOR)
    rms_p = mp32.pow(2).mean(dim=dims, keepdim=True).sqrt().detach().clamp_min(FLUCT_FLOOR)
    fluct_p = torch.cat([mp32 / rms_p, fluct_p[..., N_MEASURED:].float()], dim=-1)
    fluct_t = torch.cat([mt32 / rms_t, fluct_t[..., N_MEASURED:].float()], dim=-1)

    mean_term = field_loss(
        mean_p, mean_t, has_p=has_p, huber_delta=huber_delta, huber_mix=huber_mix
    )
    shape_term = field_loss(
        fluct_p,
        fluct_t,
        has_p=has_p,
        huber_delta=huber_delta,
        huber_mix=huber_mix,
    )
    return mean_term + shape_term


def field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    has_p: torch.Tensor | None = None,
    huber_delta: float = 0.0,
    huber_mix: float = 1.0,
    p_valid: torch.Tensor | None = None,
    w_p: float = 1.0,
) -> torch.Tensor:
    """Accuracy on the measured channels, plus pressure where it exists.

    ``pred``/``target`` are ``(B, T, H, W, 3)``. Pressure is supervised only on
    samples with ``has_p == 1`` (simulated): it is unscored, but it is free physics
    from the CFD side and a useful auxiliary target during pretraining.

    ``huber_delta > 0`` switches to smooth-L1, which is less dominated by the rare
    large-error pixels in the wake (the per-element error distribution is heavy
    tailed: mean 0.0059, std 0.0136). **It is the largest single lever FINDINGS 28
    found**, worth +0.49 SPS on ``val_extrap``, and essentially all of that arrives
    through interval *efficiency* rather than through the accuracy axes.

    ``huber_mix`` blends the two rather than choosing: ``1.0`` (the default, and the
    behaviour before it existed) is pure smooth-L1, ``0.0`` is pure MSE, and a value
    between is ``(1-m)*MSE + m*smoothL1``. It exists to *discriminate the mechanism*,
    not as a tuning knob. Smooth-L1's gradient is ``e/delta`` below ``delta`` and a
    constant ``±1`` above, so **every finite delta caps the tail's influence** while
    ``delta -> inf`` recovers scaled MSE and ``delta -> 0`` recovers L1: the delta
    sweep therefore interpolates MSE and L1 and every point on it caps the tail. A
    blend does not — it keeps MSE's *unbounded* tail gradient and adds the body
    boost — so it lies outside that span and separates "the win is capping the tail"
    from "the win is up-weighting the body".

    ``p_valid`` is the measurement mask, broadcast against the pressure error, and ``w_p``
    the term's weight. Both exist because the pressure term arrived with neither: it was
    normalised by every cell of the grid, including the ones ``mask_sim`` blanks, and it
    entered at a fixed weight of 1 -- so an *unscored* channel carried as much of the field
    loss as ``u`` and ``v`` together, at whatever amplitude the release happened to store.
    ``w_p=1.0`` with ``p_valid=None`` reproduces the old value exactly (pinned in
    ``tests/test_phys_losses.py``); the masking matters because at AoA=10 the 1.5% of cells
    inside the solid body hold 76% of the gauge-removed squared-pressure mass, at 177x the
    surrounding flow's variance (see ``data/dataset.py``'s ``p_target`` block).
    """

    loss = _field_elem(
        pred[..., :N_MEASURED], target[..., :N_MEASURED], huber_delta, huber_mix
    ).mean()
    p_term = _pressure_term(pred, target, has_p, huber_delta, huber_mix, p_valid, w_p)
    return loss if p_term is None else loss + p_term


def _field_elem(
    a: torch.Tensor, b: torch.Tensor, huber_delta: float, huber_mix: float
) -> torch.Tensor:
    """Elementwise field error: MSE, smooth-L1 at ``huber_delta``, or their ``huber_mix`` blend."""
    if huber_delta <= 0:
        return (a - b) ** 2
    h = F.smooth_l1_loss(a, b, reduction="none", beta=huber_delta)
    if huber_mix >= 1.0:
        return h
    return (1.0 - huber_mix) * (a - b) ** 2 + huber_mix * h


def _pressure_term(
    pred: torch.Tensor,
    target: torch.Tensor,
    has_p: torch.Tensor | None,
    huber_delta: float,
    huber_mix: float,
    p_valid: torch.Tensor | None,
    w_p: float,
) -> torch.Tensor | None:
    """The pressure half of :func:`field_loss`, shared with :func:`frame_normalized_field_loss`."""
    if has_p is None or w_p <= 0 or float(has_p.sum()) <= 0:
        return None
    p_elem = _field_elem(pred[..., N_MEASURED:], target[..., N_MEASURED:], huber_delta, huber_mix)
    w = has_p.reshape(-1, *([1] * (p_elem.ndim - 1)))
    if p_valid is not None:
        w = w * p_valid
    # Normalise by the number of contributing *elements*, not the batch size, so the
    # pressure term neither shrinks in a mixed real/sim batch nor pays for the cells the
    # measurement mask blanks.
    w = w.expand_as(p_elem)
    return w_p * (p_elem * w).sum() / w.sum().clamp_min(1.0)


#: Smallest per-frame error RMS ``frame_normalized_field_loss`` divides by, in
#: freestream-normalised units (fields are O(1), errors O(1e-2)).
FRAME_RMS_FLOOR = 1e-6


def frame_normalized_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    has_p: torch.Tensor | None = None,
    huber_delta: float = 0.0,
    huber_mix: float = 1.0,
    p_valid: torch.Tensor | None = None,
    w_p: float = 1.0,
) -> torch.Tensor:
    """:func:`field_loss` with every output frame normalised by its own error and masked.

    Bruno's loss for the `unet2d` arm (2026-09-17). For each sample ``b`` and lead ``t`` the
    measured-channel error is restricted to the PIV-valid cells and divided by that frame's
    **detached** error RMS over the same cells::

        loss = mean_{b,t}  mean_{valid cells, u v} elem(e_bt)  /  rms(e_bt).detach()

    With smooth-L1 ``elem`` this is exactly smooth-L1 at ``delta / rms`` on the standardised
    residual ``e / rms``, so the shipped tail-capping is preserved. The normaliser is detached.
    In the L1 regime (``|e| > delta``, which at the shipped ``delta = 0.005`` is nearly every
    element) the per-frame value is ``mean|e| / rms``, a scale-free number, and the gradient of
    frame ``t`` is ``sign(e) / rms_bt``: every lead contributes equal gradient instead of the
    late frames, whose errors are largest, carrying most of it. ``valid`` is ``(B, H, W)``.
    Pressure, where present, is supervised exactly as in :func:`field_loss`.
    """
    e = (pred[..., :N_MEASURED] - target[..., :N_MEASURED]).float()
    m = valid.to(e.dtype)[:, None, :, :, None]  # (B, 1, H, W, 1)
    count = (m.sum(dim=(2, 3, 4)) * N_MEASURED).clamp_min(1.0)  # (B, 1) valid elements per frame
    rms = ((e.square() * m).sum(dim=(2, 3, 4)) / count).sqrt().detach().clamp_min(FRAME_RMS_FLOOR)
    elem = _field_elem(
        pred[..., :N_MEASURED].float(), target[..., :N_MEASURED].float(), huber_delta, huber_mix
    )
    per_frame = (elem * m).sum(dim=(2, 3, 4)) / count  # (B, T)
    loss = (per_frame / rms).mean()
    p_term = _pressure_term(pred, target, has_p, huber_delta, huber_mix, p_valid, w_p)
    return loss if p_term is None else loss + p_term


def mean_fluct_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fluct_weight: float,
    has_p: torch.Tensor | None = None,
    huber_delta: float = 0.0,
    huber_mix: float = 1.0,
    p_valid: torch.Tensor | None = None,
    w_p: float = 1.0,
    robust_fluct: bool = False,
) -> torch.Tensor:
    """Robust temporal-mean error plus quadratic temporal-fluctuation error.

    Replaces the field term; TKE objectives remain separate. Means are per window,
    pixel and channel. Pressure retains field_loss's gating and masking.
    With pure smooth-L1, fluct_weight=1/(2*huber_delta) matches the original
    small-residual curvature by the mean/fluctuation MSE decomposition. Large
    fluctuation errors retain quadratic cost instead of entering Huber's linear
    tail. Temporal reductions use at least float32 under AMP.

    ``robust_fluct=True`` grades the fluctuation half with the SAME elementwise error as the
    mean half (smooth-L1 at ``huber_delta``, ``huber_mix``) instead of MSE, so the split changes
    only the relative PRICE of fluctuation error, not its shape: at ``fluct_weight=1`` the two
    halves are the shipped robust field term applied to the mean and to the fluctuation
    separately. It exists because FINDINGS 144's arms changed both at once -- any quadratic
    fluctuation tail removes the lead-16 dump, which is what cost them their tke -- and the
    question this answers is the price alone: along the second-moment head's own correction
    (a time-constant per-pixel gain, mean-invariant) the shipped loss charges ~2.4x what its
    tke terms return, while the board pays for it.
    """
    if not math.isfinite(fluct_weight) or fluct_weight < 0:
        raise ValueError("fluct_weight must be finite and nonnegative")
    if pred.dtype in (torch.float16, torch.bfloat16):
        pred = pred.float()
    if target.dtype in (torch.float16, torch.bfloat16):
        target = target.float()
    mean_p, mean_t = pred.mean(dim=1, keepdim=True), target.mean(dim=1, keepdim=True)
    kwargs = dict(has_p=has_p, p_valid=p_valid, w_p=w_p)
    mean_loss = field_loss(mean_p, mean_t, huber_delta=huber_delta, huber_mix=huber_mix, **kwargs)
    fluct_kwargs = dict(huber_delta=huber_delta, huber_mix=huber_mix) if robust_fluct else {}
    fluct_loss = field_loss(pred - mean_p, target - mean_t, **fluct_kwargs, **kwargs)
    return mean_loss + fluct_weight * fluct_loss


def lead_weights(t_out: int, decay: float, device=None, dtype=None) -> torch.Tensor:
    """Per-lead weights ``exp(-decay * t/T)``, renormalised to mean 1. ``decay=0`` is uniform.

    **Why this exists, and it is the one lever §42's measurement actually motivates.** That section
    measured, per lead on `val_extrap`, the correlation between the model's fluctuation and the
    target's: it falls **0.846 at lead 1 to 0.319 at lead 20**, while the amplitude the metric wants
    rises 1.13x to 2.13x. So the two scored axes want opposite things, and *how strongly* depends on
    the lead:

    * at short leads the forecast is genuinely informative (rho ~0.85) and accuracy is cheap to buy;
    * at long leads it is mostly unpredictable (rho ~0.32), the MSE-optimal fluctuation gain has
      already fallen *below* 1, and the tke deficit is almost entirely there.

    Renormalising to mean 1 keeps the term's overall scale fixed, so ``decay`` moves *where* the
    accuracy pressure goes without changing how much of it there is -- which is what makes a sweep
    over it interpretable rather than confounded with a dose change.

    `NEW_ITEMS TR` lists uniform frame weighting as **settled** ("renormalize per-frame only if
    early frames prove disproportionate"). §42 is the evidence that reopens it: early frames *are*
    disproportionate, by a factor of 2.6 in fluctuation correlation, and that was not measured when
    the spec was written.
    """
    t = torch.arange(t_out, device=device, dtype=dtype or torch.float32)
    w = torch.exp(-decay * t / max(t_out - 1, 1))
    return w / w.mean()


def _apply_lead_weights(err: torch.Tensor, decay: float) -> torch.Tensor:
    """Weight a ``(B, T, ...)`` per-element error by :func:`lead_weights` along the time axis."""
    if decay == 0.0:
        return err
    w = lead_weights(err.shape[1], decay, device=err.device, dtype=err.dtype)
    return err * w.reshape(1, -1, *([1] * (err.ndim - 2)))


def lead_weighted_field_loss(
    pred: torch.Tensor, target: torch.Tensor, decay: float = 0.0
) -> torch.Tensor:
    """Absolute MSE on the measured channels, with accuracy pressure tilted toward short leads.

    Paired with an unweighted ``tke`` term this asks the model to be accurate where accuracy is
    achievable and leaves the long-lead frames -- where §42 measured rho at 0.32 and the MSE-optimal
    gain already below 1 -- freer to carry the fluctuation ``tke_score`` wants. The bet is that
    ``rel_l2``, which pools over every lead, loses little by relaxing the leads it was never going
    to predict, while ``tke``, whose deficit is concentrated exactly there, gains.

    If it fails, the interesting failure is *which* way: a rel_l2 loss with no tke gain says the
    long leads were carrying more of rel_l2 than their predictability suggests, which would itself
    be worth knowing.
    """
    err = (pred[..., :N_MEASURED] - target[..., :N_MEASURED]) ** 2
    return _apply_lead_weights(err, decay).mean()


def _var_t(x: torch.Tensor) -> torch.Tensor:
    """Variance about the window's own time mean, over ``dim=1``. Biased, as the scorer's is."""
    return (x - x.mean(dim=1, keepdim=True)).pow(2).mean(dim=1)


def temporal_tke(fields: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` temporal fluctuation energy, exactly as the scorer defines it.

    ``0.5 * (var_t(u) + var_t(v))`` where the variance is taken about the *time*
    mean of the window. A prediction that is constant in time has zero TKE, which
    is why persistence scores a TKE relative error of exactly 1.0.
    """
    return 0.5 * (_var_t(fields[..., 0]) + _var_t(fields[..., 1]))


@torch.no_grad()
def unpredictability_gate(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-12,
    invert: bool = False,
) -> torch.Tensor:
    """``(B, H, W)`` per-pixel weight for the dense second-moment terms, detached.

    The raw gate is ``var_t(pred - target) / var_t(target)`` on the measured channels,
    clamped to 1: it reads how much of a pixel's target fluctuation the prediction fails
    to track. Where the model sits at the conditional mean the residual fluctuation *is*
    the target fluctuation, so the gate is 1 -- and there adding variance is nearly free
    in ``rel_l2``, whose residual is already at its ceiling (FINDINGS 126's pricing
    mechanism). Where the phase is tracked the gate falls toward 0 -- and there variance
    pressure is what buys ``tke`` at ``rel_l2``'s expense, the give-back FINDINGS 98 and
    141 both measured. Wrong fluctuation (residual above the target's own energy) clamps
    to full pressure rather than above it, so the relative terms still correct
    over-prediction.

    The gate is renormalised to **mean 1 over valid pixels**, so a gated term is the same
    dose as its ungated twin (55a's matched-contribution rule) with only the spatial
    placement moved -- the same discipline ``lead_weights`` records for the lead axis.
    ``no_grad``: the gate must never pull on the residual it reads, or the term would pay
    the model to make pixels less predictable so it may fluctuate there.

    ``invert=True`` weights by **predictability** (``1 -`` the clamped ratio) instead: full
    pressure where the phase is tracked, none where the fluctuation is unpredicted. The first
    campaign's decomposition is the argument for this direction (FINDINGS 162): removing
    pressure from tracked pixels lost 0.44-0.78 ``tke`` while the extra pressure it moved onto
    unpredicted pixels returned only 0.02-0.03 ``rel_l2``, so the terms' purchase lives at the
    tracked pixels and the untracked ones are cheap but ineffective.
    """
    g = temporal_tke(pred - target) / temporal_tke(target).clamp_min(eps)
    g = g.clamp(max=1.0)
    if invert:
        g = 1.0 - g
    n = g.shape[0]
    if valid is not None:
        g = g * valid
        count = valid.expand_as(g).reshape(n, -1).sum(dim=1).clamp_min(1.0)
        mean = g.reshape(n, -1).sum(dim=1) / count
    else:
        mean = g.reshape(n, -1).mean(dim=1)
    return g / mean.clamp_min(eps).reshape(-1, 1, 1)


def _tke_lead_subset(
    pred: torch.Tensor, target: torch.Tensor, dropout: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the same random subset of output leads for both TKE maps."""
    if dropout <= 0.0:
        return pred, target
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"tke lead dropout must be in [0, 1), got {dropout}")
    keep = max(2, int(round(pred.shape[1] * (1.0 - dropout))))
    if keep >= pred.shape[1]:
        return pred, target
    indices = torch.rand(pred.shape[0], pred.shape[1], device=pred.device).topk(
        keep, dim=1, largest=False
    ).indices
    gather = indices[(...,) + (None,) * (pred.ndim - 2)]
    return pred.gather(1, gather.expand(-1, -1, *pred.shape[2:])), target.gather(
        1, gather.expand(-1, -1, *target.shape[2:])
    )


def tke_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    lead_dropout: float = 0.0,
) -> torch.Tensor:
    """Relative L2 between predicted and true temporal-variance fields.

    This is ``tke_score``'s error term, differentiated. Because it is *relative*,
    it penalises over-prediction as hard as under-prediction -- which matches the
    measured behaviour of the metric (scaling a prediction's fluctuations to 2x
    cost 20 TKE points) and rules out "inject noise until the variance matches".

    ``valid`` excludes the PIV measurement mask, where the target variance is
    identically zero and would otherwise dilute the statistic.
    """
    tp, tt = temporal_tke(pred), temporal_tke(target)
    if valid is not None:
        tp, tt = tp * valid, tt * valid
    return relative_l2(tp, tt).mean()


def kmap_loss(
    pred: torch.Tensor,
    x: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    base: str = "input",
) -> torch.Tensor:
    """Grade the map channel: relative L2 of ``k_input * exp(r)`` against the target's map.

    ``pred`` is the full ``(B, T, H, W, 4)`` output; ``r`` is the time mean of channel 3
    (``models/kmap.py``). ``x`` is the model input in the same units as ``target``, so the ratio
    is unit-free; only its measured channels are read (geometry and flow planes may follow them).
    This is ``tke_loss``'s functional with the channel's estimate in place of the frames' own
    variance, i.e. exactly what the scorer will charge the estimate with once it is spent
    through a mean-preserving match (63e's identity). ``valid`` excludes the PIV mask.

    ``base="frames"`` anchors the ratio to the predicted frames' own variance map instead of the
    input's, **detached** so the term never pulls on the frames (``TrainConfig.kmap_base``).
    """
    if pred.shape[-1] <= C:
        raise ValueError(f"kmap_loss needs a map channel; output has {pred.shape[-1]} channels")
    r = pred[..., C].mean(dim=1)
    if base == "input":
        k_base = temporal_tke(x[..., :N_MEASURED])
    elif base == "frames":
        k_base = temporal_tke(pred[..., :N_MEASURED]).detach()
    else:
        raise ValueError(f"kmap_loss base must be 'input' or 'frames', got {base!r}")
    k_t = temporal_tke(target[..., :N_MEASURED])
    est = k_base * torch.exp(r)
    if valid is not None:
        est, k_t = est * valid, k_t * valid
    return relative_l2(est, k_t).mean()


def kmap_e2e_log_ratio(
    pred_all: torch.Tensor, x: torch.Tensor, base: str = "input", eps: float = 1e-12
) -> torch.Tensor:
    """The log-ratio that moves the FRAMES' variance map onto the channel's estimate, in-graph.

    ``TrainConfig.kmap_e2e``: FINDINGS 131 trained the channel end to end but spent its map
    post hoc, and every post-hoc spend paid ~4-5x the ``rel_l2`` the network pays in training
    (127.1). This puts the spend itself in the graph: the mean-preserving gain
    ``exp(r/2)`` with this ratio realises the channel's estimate as the corrected frames' own
    map (63e's identity), and the stage objective graded on the corrected output reaches the
    channel, the frames, and their interaction.

    ``base`` follows ``kmap_loss``: the estimate is ``k_base * exp(r_chan)``. With
    ``base="frames"`` (detached, like the loss) the ratio is numerically ``r_chan`` alone, so a
    zero-init channel is the **identity** and the arm starts at its control; with
    ``base="input"`` a zero-init channel starts at the geometric blend toward the input map
    instead — a sensible but non-control start, so prefer ``frames`` for matched arms.
    """
    if pred_all.shape[-1] <= C:
        raise ValueError(f"kmap_e2e needs a map channel; output has {pred_all.shape[-1]} channels")
    r_chan = pred_all[..., C].mean(dim=1)
    ke_pred = temporal_tke(pred_all[..., :N_MEASURED])
    if base == "input":
        k_base = temporal_tke(x[..., :N_MEASURED])
    elif base == "frames":
        k_base = ke_pred.detach()
    else:
        raise ValueError(f"kmap_e2e base must be 'input' or 'frames', got {base!r}")
    return r_chan + torch.log(k_base.clamp_min(eps)) - torch.log(ke_pred.clamp_min(eps))


def tke_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-12,
    gate: torch.Tensor | None = None,
    lead_dropout: float = 0.0,
) -> torch.Tensor:
    """``tke_loss`` with the amplitude removed: match *where* the fluctuations are, not how many.

    Both variance maps are divided by their own spatial mean before comparison, so the term is
    invariant to any global rescaling of the prediction's map and therefore has **exactly zero
    gradient toward inflating variance**.

    Why that is the interesting half. ``rel_l2`` is minimised by the conditional mean, and a
    deterministic predictor can only raise its temporal variance by moving away from it -- any
    variance it adds that is uncorrelated with the truth costs ``rel_l2`` precisely its own
    energy (NEW_ITEMS ``UQ.1``). So the amplitude half of ``tke_loss`` is *intrinsically* paid
    for in ``rel_l2``. The shape half is not: putting variance in the right places is what a
    better forecast does anyway.

    And shape is the larger half. Decomposing the shipped fine-tune's ``val_extrap`` tke error of
    0.6550 (``runs/aux/tke_structural.json``): 0.5962 of it survives an oracle per-window rescale,
    i.e. is shape, leaving ~0.27 for amplitude. Correspondingly, a perfect-shape prediction left
    at that model's measured energy ratio of 0.556 would score ``|a-1| = 0.44`` -> tke_score ~81.8,
    against 77.0 for the best possible amplitude-only fix and 77.5 for ``--w-tke 0.005``.

    ``valid`` excludes the PIV hole, where the target variance is identically zero -- it would
    otherwise dominate the normalisation and drag both maps toward each other for free.

    ``gate`` is an optional ``(B, H, W)`` mean-1 weight (:func:`unpredictability_gate`); it
    scales the numerator's elementwise residual (by ``sqrt(gate)``, so squared error is weighted
    by the gate itself) while the denominator keeps the whole target map -- a redistribution of
    the same pressure, zero everywhere reproducing zero loss and ones reproducing the plain term.
    """
    pred, target = _tke_lead_subset(pred, target, lead_dropout)
    tp, tt = temporal_tke(pred), temporal_tke(target)
    if valid is not None:
        tp, tt = tp * valid, tt * valid
    n = tp.shape[0]
    tp = tp / tp.reshape(n, -1).mean(dim=1).clamp_min(eps).reshape(-1, 1, 1)
    tt = tt / tt.reshape(n, -1).mean(dim=1).clamp_min(eps).reshape(-1, 1, 1)
    if gate is None:
        return relative_l2(tp, tt).mean()
    num = torch.linalg.vector_norm(((tp - tt) * gate.sqrt()).reshape(n, -1), dim=1)
    den = torch.linalg.vector_norm(tt.reshape(n, -1), dim=1).clamp_min(1e-8)
    return (num / den).mean()


def tke_temporal_profile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Match the SHAPE of the per-lead fluctuation-energy profile, amplitude-invariant.

    The temporal complement of ``tke_shape_loss``. Where ``tke_shape`` constrains the SPATIAL
    map of temporal variance without touching amplitude, this constrains the temporal PROFILE --
    how the fluctuation energy is distributed across the ``T`` output leads -- also without
    touching amplitude, by normalising each window's per-lead energy by its own lead-mean.

    Motivation is FINDINGS 143/140: the model over-concentrates fluctuation at one interior lead
    (lead 16, amplitude ~1.48x target), which costs ``rel_l2`` there because the energy is
    mis-*placed* in time, while ``tke_score`` -- which reads only the per-pixel temporal *total*
    -- rewards the magnitude and so locks the dump in. Forcing the amplitude down (140's
    mean/fluct loss) removes the dump but pays the ``rel_l2``/``tke`` give-back at ~5:1. This term
    penalises the temporal mis-placement while, being shape-only over the lead axis, adding zero
    gradient toward inflating or deflating the per-window total variance -- the same property that
    makes ``tke_shape``'s amplitude half free. It should move the dumped energy to the leads the
    target actually carries it on, rather than deleting it.

    ``valid`` excludes the PIV hole, where the target fluctuation is identically zero.
    """
    c = 2  # u, v only, matching temporal_tke / the scorer
    fp = pred[..., :c] - pred[..., :c].mean(dim=1, keepdim=True)
    ft = target[..., :c] - target[..., :c].mean(dim=1, keepdim=True)
    ep, et = fp.pow(2), ft.pow(2)
    if valid is not None:
        v = valid.reshape(valid.shape[0], 1, *valid.shape[1:], 1)
        ep, et = ep * v, et * v
    ep = ep.sum(dim=(2, 3, 4))  # (B, T) per-lead fluctuation energy
    et = et.sum(dim=(2, 3, 4))
    ep = ep / ep.mean(dim=1, keepdim=True).clamp_min(eps)  # shape only, over leads
    et = et / et.mean(dim=1, keepdim=True).clamp_min(eps)
    return relative_l2(ep, et).mean()


def tke_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    delta: float = 1.0,
    eps: float = 1e-12,
    gate: torch.Tensor | None = None,
    lead_dropout: float = 0.0,
    normalize: bool = True,
) -> torch.Tensor:
    """``tke_loss`` with a robust penalty on the variance map instead of a plain L2.

    Residuals are divided by the target map's per-sample RMS -- so every window weighs equally,
    exactly as the scorer's per-window relative error does -- and then passed through smooth-L1.
    (In the loop's freestream-normalised units the map RMS spans only ~1.8x across the Reynolds
    ladder but ~8x window to window, p5-p95; the 57x/78x figure is physical m/s, FINDINGS 196j.)
    Cells beyond ``delta`` standard residuals contribute linearly rather than quadratically.

    **What this is and is not for.** It does NOT spread pressure away from the wake, and it must
    not: ``tke_score`` is itself a relative L2 on the raw map, so it is wake-dominated by
    construction and a loss that equalised across the field would optimise something the metric
    does not measure. (A log-space penalty would do exactly that, which is why it is not used
    here.) What this changes is only the *within-window* dominance of the few worst cells, on the
    optimisation-dynamics argument FINDINGS already makes for ``huber_delta`` on the field term:
    the per-element error distribution is heavy tailed, and a handful of cells monopolising the
    gradient can starve the rest.

    Read the result sceptically. The measured heavy tail was in the per-window *optimal scalar*
    (median 1.12, mean 1.83, range 0.69-9.84), and the relative form already normalises each
    window by its own ``||ke_target||``, so windows are equalised before this term sees them.
    The remaining case for robustness is within-window only, and is weaker than the across-window
    figure first suggested.

    ``gate`` is an optional ``(B, H, W)`` mean-1 weight (:func:`unpredictability_gate`) on the
    per-cell smooth-L1 residual; ones reproduce the plain term exactly.

    ``normalize=False`` drops the per-window RMS division: the residual is the raw map difference
    in the loop's (freestream) units and ``delta`` is then an ABSOLUTE crossover (the training
    windows' median target-map RMS is ~0.005). Windows then weigh by their own energy instead of
    equally, so the high-Reynolds windows -- where the energy deficit is -- carry more of the term.
    """
    pred, target = _tke_lead_subset(pred, target, lead_dropout)
    tp, tt = temporal_tke(pred), temporal_tke(target)
    if valid is not None:
        tp, tt = tp * valid, tt * valid
    n = tp.shape[0]
    if normalize:
        scale = tt.reshape(n, -1).pow(2).mean(dim=1).sqrt().clamp_min(eps).reshape(-1, 1, 1)
        r = (tp - tt) / scale
    else:
        r = tp - tt
    sl = F.smooth_l1_loss(r, torch.zeros_like(r), reduction="none", beta=delta)
    if gate is not None:
        sl = sl * gate
    return sl.mean()


def tke_amplitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    delta: float = 0.5,
    eps: float = 1e-12,
    lead_dropout: float = 0.0,
) -> torch.Tensor:
    """Grade the fluctuation energy's *total*, and nothing about where it sits.

    One scalar per window -- ``log(sum k_pred / sum k_target)`` through a smooth-L1 -- so the term
    is blind to placement by construction, exactly as :func:`tke_shape_loss` is blind to amplitude.

    **Why this exists** (FINDINGS 183b). The scored energy error splits orthogonally, per window,
    at the least-squares scale ``a``, into a placement part and an amplitude part. Measured on the
    shipped recipe: amplitude is 0.026 at Re 3750 and 0.150-0.189 at Re 26700, one-sided -- the
    model emits 60-70% of the target's energy at the top of the ladder and is at the right scale at
    the bottom. Amplitude's share of the SQUARED error is 0.5% there and 8.5% here, so
    :func:`tke_huber_loss`, which grades the raw map difference, hands it at most a twelfth of its
    gradient, and :func:`tke_shape_loss` hands it none. The recipe presses placement twice and the
    amplitude almost not at all, on the one axis where the deficit grows across the ladder.

    **Why a log, when FINDINGS 27/§162 rule a log-space penalty out on the MAP.** That objection is
    about flattening pressure across the field: ``tke_score`` is a relative L2 on the raw map and is
    wake-dominated by construction, so a per-pixel log would optimise something the metric does not
    measure, and the predictability gate measured that damage directly. Here there is no map left to
    flatten -- the sum is taken first. On a scalar ratio the log buys three things a linear residual
    does not: over- and under-prediction by the same factor cost the same, the term is free of the
    10x energy span across operating points, and its gradient does not shrink just because the
    window's placement error is large.

    **It needs no per-cell weighting to reach the corner.** Where the amplitude is already right the
    ratio is 1 and the term contributes no gradient, so a single global dose concentrates itself
    where the deficit is -- the property FINDINGS 182's per-cell weighting idea was reaching for,
    obtained by isolating the quantity rather than by scheduling a weight.

    ``delta`` caps the tail in LOG units: 0.5 is a factor 1.65, outside every deficit measured, so
    it guards degenerate windows without touching the body of the distribution. ``valid`` excludes
    the PIV measurement hole, where the target variance is identically zero.
    """
    pred, target = _tke_lead_subset(pred, target, lead_dropout)
    tp, tt = temporal_tke(pred), temporal_tke(target)
    if valid is not None:
        tp, tt = tp * valid, tt * valid
    n = tp.shape[0]
    sp = tp.reshape(n, -1).sum(dim=1).clamp_min(eps)
    st = tt.reshape(n, -1).sum(dim=1).clamp_min(eps)
    r = (sp / st).log()
    return F.smooth_l1_loss(r, torch.zeros_like(r), reduction="none", beta=delta).mean()


def error_homogeneity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    k: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalise the LOCAL dispersion of the field error, so the residual is predictable from its
    own neighbourhood -- the property SPS interval efficiency is bounded by.

    Motivation (FINDINGS 206/207). The board leader's edge is interval efficiency, and 207 measured
    it to be a property of the backbone's ERROR SHAPE that no post-hoc interval trick reaches: the
    fitted stack sits at the local-LEVEL oracle (tightness 0.739), while an oracle that knew each
    element's own error would reach 0.879. That gap is exactly the amount by which per-element
    ``|error|`` varies *within* a neighbourhood beyond what its local level predicts. The interval
    head sizes by local level (coords + gradients + a 3x3-ish reduction); it cannot bracket the
    excess per-element scatter. So the only way to close it is to make the excess smaller at the
    source -- train the backbone to emit errors whose magnitude is locally homogeneous.

    Construction: ``e = |pred - target|`` on the scored velocity channels, per (window, lead,
    channel); a KxK spatial box gives each pixel its neighbourhood mean and variance of ``e``; the
    loss is the mean local standard deviation of ``e`` divided by the GLOBAL mean of ``e`` over
    scored pixels. Global (not local) normalisation keeps it scale-free without exploding where the
    local mean error is ~0, and weights the wake -- where error and interval width both live --
    rather than the quiet free stream. It is minimised (=0) by an error field that is piecewise
    constant at box scale, which is precisely the field a local-level sizer brackets with no waste.
    ``valid`` excludes the PIV measurement hole. Composes with ``w_field`` (which drives ``e``
    down); this term only shapes what is left.
    """
    p = pred[..., :N_MEASURED]
    t = target[..., :N_MEASURED]
    e = (p - t).abs()  # (B, T, H, W, N)
    b, tt_, h, w, c = e.shape
    x = e.permute(0, 1, 4, 2, 3).reshape(b * tt_ * c, 1, h, w).float()
    pad = k // 2
    m = F.avg_pool2d(x, k, stride=1, padding=pad, count_include_pad=False)
    m2 = F.avg_pool2d(x * x, k, stride=1, padding=pad, count_include_pad=False)
    local_std = (m2 - m * m).clamp_min(0.0).sqrt()  # (B*T*N, 1, H, W)
    local_std = local_std.reshape(b, tt_, c, h, w).permute(0, 1, 3, 4, 2)  # (B, T, H, W, N)
    e_f = e.float()
    if valid is not None:
        v = valid.reshape(b, 1, h, w, 1).to(local_std.dtype)
        denom = (e_f * v).sum().clamp_min(eps)
        return (local_std * v).sum() / denom
    return local_std.sum() / e_f.sum().clamp_min(eps)


def pvar_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    has_p: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """``tke_loss``'s form, on the *pressure* channel: match where the pressure fluctuates.

    Simulated samples only, gated on ``has_p`` exactly as the field loss's pressure term is, and
    meaningless without ``p_target='gauge'`` -- an ungauged channel's temporal variance is
    dominated by the drift of an arbitrary datum rather than by the flow.

    **Why the second moment and not just the first.** Two measurements point here.
    ``tke_score`` is the weakest axis and §40's decomposition puts most of its error in *shape*
    -- fluctuations in the wrong places -- not amplitude. And the low-pressure core is the
    classical marker of a vortex, so ``var_t(p)`` is a localiser of the same structures, but an
    elliptic (hence spatially smooth) one rather than a differential one.

    That is measured, not asserted. Over 300 windows spanning the 100 simulated cells, the
    correlation between ``var_t(p')`` and the target's scored ``tke`` map is **+0.545**, on a par
    with the best predictor a p-free model already holds (the *input* window's own tke map,
    +0.548) -- while being much less correlated *with* it (+0.376). Regressing the target tke map
    on both p-free predictors (input tke and enstrophy fluctuation) gives R^2 0.463; adding
    ``var_t(p')`` gives **0.547**. So pressure fluctuation explains 8.4 points of the scored
    quantity's spatial variance that the model's own p-free features do not.

    Relative, and for ``tke_loss``'s reason: fluctuation energy spans orders of magnitude over
    the Reynolds ladder, so an absolute form would be a Reynolds weighting in disguise. ``valid``
    excludes the measurement mask, where the gauge-removed target is identically zero.
    """
    if has_p is not None and float(has_p.sum()) == 0:
        return target.new_zeros(())
    vp, vt = _var_t(pred[..., N_MEASURED]), _var_t(target[..., N_MEASURED])
    if valid is not None:
        vp, vt = vp * valid, vt * valid
    per_sample = relative_l2(vp, vt)
    if has_p is None:
        return per_sample.mean()
    w = has_p.reshape(-1)
    return (per_sample * w).sum() / w.sum().clamp_min(eps)


def vorticity(fields: torch.Tensor) -> torch.Tensor:
    """``(B, T, H-2, W-2)`` out-of-plane vorticity ``dv/dx - du/dy``, in **grid units**.

    Central differences on the interior only, so there is no invented boundary rule: a
    one-sided difference at the inlet or the tunnel wall would be a different operator on
    the edge ring than in the bulk, and the ring is where the mean shear is strongest.

    **Grid units are exact here, not an approximation.** ``geometry.GRID_X_M`` /
    ``GRID_Y_M`` give ``dx = 1.71102 mm`` and ``dy = 1.71111 mm`` on the native grid --
    equal to 5e-5 -- so the cell is square and the two derivative terms share one spacing.
    The physical field is therefore this one divided by a single constant ``h`` (3.422 mm at
    the scored 32x64), and every term below is a *relative* L2, in which that constant
    cancels. Nothing here needs calibrating against a length scale, which is the reason to
    write it this way rather than carrying ``h`` through the objective.

    ``w`` (the third velocity component) is not measured by planar PIV, so this is the only
    component of the vorticity vector the data can support.

    **fp16-safe, measured rather than assumed.** A derivative is where a half-precision
    subtraction of two nearby values would show, and this project has paid for two fp16 traps
    already (``loop.FP16_UNSAFE``, and ``spectral_loss``'s cuFFT crash). Rounding a real batch
    to fp16 moves ``vorticity_loss`` by **2.0e-4** relative and ``enstrophy_loss`` by 2.2e-5,
    against a seed floor of order 1e-3 on the scored axes -- so no float32 cast is warranted
    here, unlike in :func:`mvpe_loss` where the reduction is over 72 numbers.
    """
    u, v = fields[..., 0], fields[..., 1]
    dv_dx = 0.5 * (v[..., 1:-1, 2:] - v[..., 1:-1, :-2])
    du_dy = 0.5 * (u[..., 2:, 1:-1] - u[..., :-2, 1:-1])
    return dv_dx - du_dy


def stencil_valid(valid: torch.Tensor) -> torch.Tensor:
    """``(B, H-2, W-2)``: 1 where the whole 3x3 neighbourhood is measured.

    A derivative term needs a *stricter* mask than a pointwise one, and this is the part
    that would silently corrupt the objective if it were skipped. The target's ``u, v`` are
    identically zero inside the PIV hole (airfoil body, laser shadow), so a central
    difference that straddles the boundary reads the full freestream velocity across two
    cells and reports a vorticity several times the physical maximum -- an artefact of the
    mask, present in the *target*, which a fitted vorticity term would then reward the model
    for reproducing.

    Implemented as a min-filter (``-max_pool2d(-valid)``) over 3x3 with no padding, so the
    output is aligned with :func:`vorticity`'s interior and the edge ring drops out too.
    The 3x3 is one cell stricter than the 5-point stencil actually used, which costs the
    diagonal cells adjacent to the hole and is the safe direction.
    """
    return -F.max_pool2d(-valid[:, None], kernel_size=3, stride=1)[:, 0]


def vorticity_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-sample relative L2 on the instantaneous vorticity field.

    ``curl`` is *linear* in the velocity, which is what separates this term from every
    second-moment term in this module: composed with an L2 it is still a quadratic form on
    the field error, so its population minimiser is still the conditional mean and it does
    **not** buy into the rel_l2/tke exchange FINDINGS 27c describes. What it changes is the
    *weighting* of the field error -- a discrete curl has transfer function ~``|k|`` in
    space, so error at the resolved small scales counts for more and error in the mean flow
    for less. Concretely it grades the placement and sharpness of the shear layers and the
    shed vortices, which is where both ``tke`` and ``mvpe`` live.

    Two reasons to expect less from it than the mechanism suggests, both already measured
    elsewhere in this file and neither fatal:

    * **the high-k content of real PIV is partly sensor floor.** Section 6f measured real
      carrying a high-k shelf 7x above sim's and identified it *as* the noise floor;
      ``data.piv_noise`` puts the per-component level at 1.0-2.4% of freestream, white. For
      a white field the curl above has variance exactly ``sigma^2`` in grid units, so this
      term's target is noisier than the field term's by construction. ``scripts/
      vorticity_diagnostic.py`` measures what fraction that is before any weight is chosen.
    * **section 43 found rel_l2 to be a hard plateau for this architecture.** A reweighting
      of the same residual is not obviously a way through it.

    ``valid`` is the ``(B, H, W)`` measurement mask, eroded by :func:`stencil_valid`.
    """
    wp, wt = vorticity(pred), vorticity(target)
    if valid is not None:
        m = stencil_valid(valid)[:, None]
        wp, wt = wp * m, wt * m
    return relative_l2(wp, wt).mean()


def temporal_enstrophy(fields: torch.Tensor) -> torch.Tensor:
    """``(B, H-2, W-2)`` temporal variance of the vorticity -- ``tke``'s vorticity twin.

    The name is used in the fluctuating sense: ``var_t(omega)``, the vorticity analogue of
    :func:`temporal_tke`'s ``0.5 * (var_t(u) + var_t(v))``, and **not** the classical
    enstrophy ``integral of omega^2``. Written to mirror ``temporal_tke`` because that is
    the quantity the scorer grades and this term's whole case is that it is the same
    statistic taken one derivative up.
    """
    return _var_t(vorticity(fields))


def enstrophy_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """``tke_loss``'s form on ``var_t(omega)`` instead of ``var_t(u), var_t(v)``.

    **Why this is the more interesting of the two vorticity terms.** FINDINGS 27c isolated
    *why* ``w_tke`` is the one objective in this project that pays: forcing the total
    variance to match leaves the model no way to satisfy the term except to put fluctuation
    energy where the target has it, because under a relative-L2 map penalty energy in the
    wrong place raises the error. Remove the amplitude constraint and the shape pressure has
    nothing to push against -- which is exactly what ``w_tke_shape`` measured, and why it
    failed. This term keeps that working mechanism and sharpens the localisation it acts
    through: ``var_t(u,v)`` is broad over the wake, ``var_t(omega)`` concentrates on the
    shear layers that bound it, so the same "put it in the right place" gradient is asked
    for at the scale of a vortex rather than of the wake.

    It is a strictly finer statistic than ``tke_score``, and FINDINGS 24c is the warning
    that comes with that: the spectral term was also finer, was driven 3-4x down, and bought
    nothing the composite pays for. The difference claimed here is only that this one shares
    ``tke``'s *form* (a relative L2 on a spatial map of temporal variance) rather than
    merely its subject.

    Relative and per-sample for the same reason ``tke_loss`` is: the scorer weighs every window
    equally, and window-to-window energy spans ~8x even in freestream units (57x across
    ``val_extrap``'s Reynolds rows only in physical m/s, FINDINGS 196j); one derivative widens
    that, not narrows it. If this term pays at all, ``tke_huber_loss``'s robust form is the
    follow-up (27d), not a heavier weight.
    """
    ep, et = temporal_enstrophy(pred), temporal_enstrophy(target)
    if valid is not None:
        m = stencil_valid(valid)
        ep, et = ep * m, et * m
    return relative_l2(ep, et).mean()


#: The scorer's own SPS-ceiling arithmetic, mirrored here so the objective and the metric cannot
#: drift. ``score.report`` owns these numbers; they are re-stated rather than imported because
#: ``score`` is deliberately torch-free and must keep working without a GPU stack installed.
CEILING_WEIGHTS = {"rel_l2": 0.5, "tke": 0.3, "mvpe": 0.2}
CEILING_NF = 0.5


def ceiling_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
    nf: float = CEILING_NF,
    huber_delta: float = 0.0,
) -> torch.Tensor:
    """``1 - ceiling/100``: the scored composite itself, differentiated.

    ``score.report.sps_ceiling`` is ``100 * sum_k w_k * nf/(nf + e_k)`` over the three accuracy
    branches at weights ``.5/.3/.2``, where ``e_k`` is that axis's **relative** error. This
    returns ``sum_k w_k * e_k/(nf + e_k)``, which is that expression up to an additive constant
    and a sign, so minimising it maximises the ceiling exactly.

    **Why this is not another hand-weighted auxiliary term, and is the point of the exercise.**
    Every objective arm in §22-§40 set ``w_tke`` against ``w_field`` by calibrating a *dose* --
    the term's value relative to a field loss that is itself only a proxy for ``rel_l2_score``.
    FINDINGS 33i records that heuristic crippling a working term tenfold and 34c records the
    resulting valuations being upper bounds. This objective removes the free parameter: the
    relative weighting of the three axes is fixed by the scorer, and each axis's *marginal*
    pressure is the derivative ``w_k * nf/(nf + e_k)**2``, which is large where an axis is
    already accurate and small where it is far off. So it self-balances at the current operating
    point instead of at a dose chosen months ago on a different backbone.

    **And it makes a falsifiable prediction that runs against the campaign's instinct.** At the
    baseline's operating point (``e_rel`` 0.087, ``e_tke`` 0.63, ``e_mvpe`` 0.054) the marginal
    pressures are rel_l2 **0.73**, mvpe **0.33**, tke **0.12**. The ceiling therefore wants
    *less* tke pressure than any arm in §27-§40 applied, because a large relative error sits in
    the flat part of ``nf/(nf+e)``. If the ceiling is what pays, a term that chases tke is
    over-spending -- and if instead the tke-weighted arms score better on the ceiling than this
    does, the ceiling is not the right functional and that is worth knowing too.

    Each ``e_k`` uses the differentiable mirror of its own metric: :func:`relative_l2` on the
    measured channels, :func:`tke_loss`, :func:`mvpe_loss`. All three are already per-sample
    relative and batch-averaged, matching how the scorer forms its errors.

    ``huber_delta > 0`` caps the tail of the **rel_l2 branch only**, through
    :func:`relative_l2`'s Huber pseudo-norm. It exists because 55a records the confound that
    invalidates every replacement arm ever run: ``huber_delta`` lived on ``field_loss``, so
    ``--w-field 0 --w-ceiling 1`` silently discarded the recipe's largest lever and measured
    "composite vs huber'd field" rather than the composite itself. Only the rel_l2 branch takes
    it -- ``tke`` has its own robust form in :func:`tke_huber_loss` and ``mvpe`` reduces over 72
    numbers, where a tail-cap has nothing to cap. The default is ``0.0``, so every recorded
    ``--w-ceiling`` arm reproduces bit-exactly from this code.
    """
    w = CEILING_WEIGHTS if weights is None else weights
    errs = {
        "rel_l2": relative_l2(
            pred[..., :N_MEASURED], target[..., :N_MEASURED], huber_delta=huber_delta
        ).mean(),
        "tke": tke_loss(pred, target, valid=valid),
        "mvpe": mvpe_loss(pred, target),
    }
    out = pred.new_zeros(())
    for name, weight in w.items():
        e = errs[name]
        out = out + weight * e / (nf + e)
    return out


def relative_field_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-sample relative L2 on the measured channels -- ``rel_l2_score``'s error exactly.

    The accuracy term this project has always used is :func:`field_loss`, a plain **absolute**
    MSE. The metric is a **per-sample relative** L2. That mismatch is `NEW_ITEMS TR.1`, open and
    unmeasured since the spec was written, and its direction is known a priori: an absolute MSE
    weights each window by its own energy, so the high-Reynolds rows dominate the gradient and
    the low-energy ones are underfit. ``val_extrap`` contains Re=3750, the **lowest**-energy row
    on the ladder, so this is not a neutral bookkeeping difference -- it is a systematic
    misweighting against the split PROTOCOL 2 argues matters most.

    ``valid`` is accepted for interface symmetry and deliberately unused: the scorer's own
    ``rel_l2`` includes the masked cells (their target is zero, so they enter both numerator and
    denominator), and matching the metric means matching that too.
    """
    del valid
    return relative_l2(pred[..., :N_MEASURED], target[..., :N_MEASURED]).mean()


def mean_field_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 on the window's 20-frame **time mean**, measured channels only.

    Every accuracy term in this file supervises the *instantaneous* field, so the first moment
    is only ever fitted as a by-product. Decomposing the shipped model's error about its own
    time mean says that is a mistake in a specific, measurable way:

    * The relative error of the mean field runs **0.030 at AoA=0 to 0.069 at AoA=20** while the
      relative error of the fluctuation about it is **flat, and slightly better at high angle**
      (0.82 -> 0.70). So the angle gradient in ``rel_l2`` -- the largest structured error on the
      grid -- is a **first-moment** failure, and ``mvpe`` (a time-averaged probe metric, and the
      axis an unseen angle costs most) agrees: its absolute probe error is 2.1-2.2x larger at
      high angle at fixed Reynolds.
    * Across Reynolds the same quantity is a **U** -- 0.051 / 0.036 / 0.057 at 3750 / 16500 /
      26700 -- i.e. both extrapolated rows are 40-60% worse than the interpolated one. That is
      a genuine extrapolation cost on the one part of the signal that is deterministic.
    * The fluctuation, by contrast, is near its predictability limit: at the top of the ladder
      the flow's own 20-frame memory is 0.03-0.06, so no term can buy phase there.

    Relative rather than absolute, and on the mean rather than the frames, because both are
    what the scored axes are: the mean field's *absolute* error grows ~6x across the ladder
    while its relative error does not, so an absolute term would spend the ladder's top on
    what a relative metric charges evenly.

    **What this is not.** It is not an addition of pressure -- it is a redistribution, and it
    should be dosed as one (FINDINGS 55a: every published null in this family was measured at
    ~0.2x a matched contribution). And it is not orthogonal to ``field_loss``: the time mean is
    a linear functional of the frames, so this reweights rather than supervises something new.
    The failure mode to watch is ``tke`` falling, which is the fluctuation giving up the
    pressure this term takes.
    """
    return relative_l2(
        pred[..., :N_MEASURED].mean(dim=1), target[..., :N_MEASURED].mean(dim=1)
    ).mean()


def spectral_loss(
    pred: torch.Tensor, target: torch.Tensor, n_bands: int = 4, eps: float = 1e-8
) -> torch.Tensor:
    """Match how fluctuation energy is spread across temporal frequencies.

    ``tke_loss`` constrains the total variance; this constrains its distribution in
    time. Energy is pooled into ``n_bands`` logarithmic-ish frequency bands and
    compared in relative terms, which is far more stable than matching the raw
    spectrum bin by bin over a 20-frame window.
    """
    # float32 for the transform, and not as a precision nicety: under fp16 autocast cuFFT
    # refuses any signal size that is not a power of two, and the temporal axis is 20. The arm
    # that first used this term crashed at step 0 with
    # "cuFFT only supports dimensions whose sizes are powers of two ... got a signal size of
    # [20]" -- the same constraint already recorded in ``loop.FP16_UNSAFE`` for FNO, arriving
    # through a loss term rather than a backbone. Casting here keeps the fix local to the one
    # operation that needs it instead of disabling autocast for the whole step.
    pred, target = pred.float(), target.float()
    out = pred.new_zeros(())
    for ch in range(N_MEASURED):
        p = pred[..., ch] - pred[..., ch].mean(dim=1, keepdim=True)
        t = target[..., ch] - target[..., ch].mean(dim=1, keepdim=True)
        # (B, F, H, W) power, averaged over space -> (B, F)
        pp = torch.fft.rfft(p, dim=1).abs().pow(2).mean(dim=(2, 3))
        tt = torch.fft.rfft(t, dim=1).abs().pow(2).mean(dim=(2, 3))
        # Drop bin 0: the mean was already removed, so it carries no information.
        pp, tt = pp[:, 1:], tt[:, 1:]
        edges = torch.linspace(0, pp.shape[1], n_bands + 1).long().tolist()
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            if hi <= lo:
                continue
            pb, tb = pp[:, lo:hi].sum(1), tt[:, lo:hi].sum(1)
            out = out + ((pb - tb).abs() / tb.clamp_min(eps)).mean()
    return out / (N_MEASURED * n_bands)


def reynolds_stress(fields: torch.Tensor) -> torch.Tensor:
    """``(B, H, W, 3)`` per-pixel Reynolds stress ``<u'u'>, <v'v'>, <u'v'>``.

    Fluctuations are about the *window's own* time mean, which is the definition
    ``tke_score`` uses. ``0.5 * (R_uu + R_vv)`` is therefore exactly :func:`temporal_tke`,
    and the tensor's remaining two degrees of freedom are what the metric cannot see.
    """
    u, v = fields[..., 0], fields[..., 1]
    du = u - u.mean(dim=1, keepdim=True)
    dv = v - v.mean(dim=1, keepdim=True)
    return torch.stack(
        [(du * du).mean(dim=1), (dv * dv).mean(dim=1), (du * dv).mean(dim=1)], dim=-1
    )


def rstress_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Match the **anisotropy** of the Reynolds stress: its shape, with the trace divided out.

    The per-pixel normalised anisotropy of a 2x2 symmetric second moment has exactly two
    degrees of freedom,

        b_1 = (R_uu - R_vv) / (R_uu + R_vv)        b_2 = 2 R_uv / (R_uu + R_vv)

    and **neither is visible to any scored subscore.** ``tke_score`` grades the relative L2
    of ``0.5 (R_uu + R_vv)`` -- the trace alone -- so this term is orthogonal to
    :func:`tke_loss` by construction rather than by tuning, which matters because ``w_tke``
    is measured and cut (FINDINGS 22h): a full-tensor loss would silently re-run it.

    **Why the shape is worth constraining even though it is unscored.** Measured on the
    shipped STFormer (``scripts/eval/rstress_audit.py``, ``val_extrap``): the model retains 0.364
    of the target's ``<u'u'>`` but 0.619 of its ``<v'v'>``, so the deficit is strongly
    anisotropic rather than a uniform variance collapse, and the shear stress ``<u'v'>`` is
    the least controlled component of all -- captured at 0.661 on ``val_extrap`` and
    **1.698** on ``val_interp``, i.e. over-predicted by 70% on one split while
    under-predicted on another. A quantity that unstable is not being constrained by
    anything the objective currently contains.

    Normalised by the trace and weighted by it. The normalisation is what makes it a *shape*
    term and keeps it finite in the freestream, where both components go to zero and the raw
    ratio is 0/0; the weighting is what stops it from grading the freestream as heavily as
    the wake, which would optimise a region no subscore looks at (the same failure that made
    a log-space TKE penalty the wrong choice in :func:`tke_huber_loss`).

    ``valid`` excludes the PIV measurement hole, where the target's fluctuations are
    identically zero and the shape is undefined.
    """
    # float32, and this one is NOT a precision nicety -- it is the difference between the
    # term working and the run silently not training. Under fp16 autocast the per-pixel
    # variances underflow toward zero in the freestream, ``1/tr`` then overflows fp16's 65504
    # ceiling, the step's gradient goes non-finite, and ``GradScaler`` skips it. Skipped
    # every step: measured as `|g|=nan` once and then `|g|=0.00` for 400 steps with the field
    # loss pinned at 2.4x the control's, which is PROTOCOL 5's Transolver failure arriving
    # through a loss term. The same cast is already on ``spectral_loss`` and ``mvpe_loss``.
    rp, rt = reynolds_stress(pred.float()), reynolds_stress(target.float())
    tr_p = (rp[..., 0] + rp[..., 1]).clamp_min(eps)
    tr_t = (rt[..., 0] + rt[..., 1]).clamp_min(eps)
    b_p = torch.stack([(rp[..., 0] - rp[..., 1]) / tr_p, 2 * rp[..., 2] / tr_p], dim=-1)
    b_t = torch.stack([(rt[..., 0] - rt[..., 1]) / tr_t, 2 * rt[..., 2] / tr_t], dim=-1)

    w = rt[..., 0] + rt[..., 1]
    if valid is not None:
        w = w * valid
    # Normalise by the weight, not by the element count, so the term means the same thing
    # across windows whose trace spans ~8x in freestream units (57x physical, FINDINGS 196j).
    num = ((b_p - b_t).pow(2).sum(dim=-1) * w).sum()
    return num / w.sum().clamp_min(eps)


def _radial_bins(h: int, w: int, n_bands: int, device: torch.device) -> torch.Tensor:
    """Band index per ``rfft2`` mode. Band 0 is the DC mode alone; cached per shape."""
    key = (h, w, n_bands, str(device))
    hit = _RADIAL_CACHE.get(key)
    if hit is not None:
        return hit
    ky = torch.fft.fftfreq(h, device=device)[:, None]
    kx = torch.fft.rfftfreq(w, device=device)[None, :]
    kmag = torch.sqrt(ky**2 + kx**2)
    idx = torch.zeros_like(kmag, dtype=torch.long)
    nz = kmag > 0
    idx[nz] = 1 + torch.clamp((kmag[nz] / kmag.max() * n_bands).long(), max=n_bands - 1)
    _RADIAL_CACHE[key] = idx
    return idx


_RADIAL_CACHE: dict[tuple, torch.Tensor] = {}


def energy_spectrum_loss(
    pred: torch.Tensor, target: torch.Tensor, n_bands: int = 8, eps: float = 1e-8
) -> torch.Tensor:
    """Match how fluctuation energy is spread across **spatial** scales.

    Not to be confused with :func:`spectral_loss`, which bins *temporal* frequencies -- that
    is the term measured and cut as ``TR.2`` (FINDINGS 24). This one bins the radial spatial
    wavenumber ``|k|``, which is the classical turbulence spectrum ``E(k)`` and a different
    quantity: a prediction can have the right temporal spectrum at every pixel and still be
    smooth in space.

    The grid is isotropic -- ``|dx| = |dy| = 1.7108 mm`` at 64x128, hence 3.4217 mm on the
    scored 32x64 grid -- which is what makes one radial magnitude well defined. Bands are
    equal-width rings over ``(0, |k|_max]``, with the DC mode alone in band 0 so a spatial
    mean is never traded against a fluctuation scale.

    **What it has to move, measured** (``scripts/eval/phys_audit.py``, ``val_extrap``): the
    prediction's radial spectrum runs 0.85-0.90 of the target's through the energy-bearing
    mid bands and falls to 0.16-0.24 in the top three. So the deficit is real and the term is
    not asking for something already achieved.

    **And what the same measurement says against it.** The least-squares optimal *gain* per
    radial band is 0.77-1.00 -- within 1% of unity everywhere that carries energy -- because
    the missing energy is uncorrelated with the target (per-band coherence 0.31-0.99). The
    squared-error floor of the whole per-band gain family is **0.08% below the current
    error**. So spectral energy cannot be added in a way ``rel_l2`` rewards, and this term is
    a request to move *away* from the L2 optimum in exchange for a second-moment statistic.
    That is a legitimate trade -- ``tke_score`` is a variance metric -- but it is a trade,
    and the arm has to be judged on both axes.

    Relative per band, matching how the metric grades: absolute band powers span orders of
    magnitude across the ladder and a plain difference would grade only the largest scale.
    """
    # float32 for the transform for the same reason ``spectral_loss`` does it: an fp16 FFT
    # here is rounding error and no speed, and cuFFT's autocast path is size-fussy.
    p = pred[..., :N_MEASURED].float()
    t = target[..., :N_MEASURED].float()
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)

    h, w = p.shape[2], p.shape[3]
    idx = _radial_bins(h, w, n_bands, p.device)
    # rfft2 stores one half-plane; interior columns each stand for two conjugate modes.
    mult = torch.ones(w // 2 + 1, device=p.device)
    mult[1 : (w + 1) // 2] = 2.0

    def bands(x: torch.Tensor) -> torch.Tensor:
        # (B, T, H, W, C) -> (B, n_bands+1): power per band, pooled over time and channels.
        z = torch.fft.rfft2(x.permute(0, 1, 4, 2, 3), dim=(-2, -1))
        power = (z.real**2 + z.imag**2) * mult
        flat = power.reshape(power.shape[0], -1, h * (w // 2 + 1)).sum(dim=1)
        out = torch.zeros(flat.shape[0], n_bands + 1, device=x.device, dtype=flat.dtype)
        return out.index_add_(1, idx.reshape(-1), flat)

    pb, tb = bands(p), bands(t)
    # Drop the DC band: the time mean was already removed, so it carries the window's
    # *spatial* mean fluctuation, which is not a scale.
    return ((pb[:, 1:] - tb[:, 1:]).abs() / tb[:, 1:].clamp_min(eps)).mean()


def helmholtz_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = 1.0,
    huber_delta: float = 0.0,
) -> torch.Tensor:
    """Field loss with the error's **divergent** part weighted by ``lam``.

    In-plane incompressibility is *false* on a planar slice of a 3D flow -- ``d_x u + d_y v =
    -d_z w``, and FINDINGS 3 measures ``rho_div = 0.857`` on real PIV where a perfectly
    2D-incompressible field would give 0 -- so ``||div u||^2`` is not a residual to drive to
    zero. It is a real component of the data. What *is* available is a reweighting: split the
    error field at each wavevector into the part parallel to ``k`` (longitudinal, the only
    part carrying divergence) and the part perpendicular to it (transverse, the only part
    carrying vorticity), and grade them differently.

    ``lam = 1`` is exactly MSE, so the arm nests its own control. ``lam < 1`` trains
    preferentially on the solenoidal error -- "fit the incompressible part of the flow" --
    and ``lam > 1`` does the reverse.

    **Why both directions have to be run.** Measured on the shipped STFormer
    (``scripts/eval/phys_audit.py``, ``val_extrap``): the longitudinal component holds **7.4% of
    the target's energy and 40.8% of the squared error**, a relative error 8.7x the
    solenoidal part's. That is a large misallocation *if* the component is unpredictable --
    but its coherence with the target is 0.971, so it is not noise and the model does track
    it. MSE already gives it 41% of the gradient in proportion to its error, and the
    least-squares optimal post-hoc reweighting is 0.995 against 0.999, i.e. nothing. So the
    sign of any gain here is not determined by the diagnostic and must be measured.

    The transform is over the two spatial axes of the *error*, which is linear, so this is a
    reweighting of MSE in an orthogonal basis and Parseval makes ``lam = 1`` exact rather
    than approximate. ``huber_delta`` is not applied here: smooth-L1 is not diagonal in the
    Fourier basis, so combining the two would silently change what either means -- an arm
    wanting both should stack this on the field term rather than replace it.
    """
    if huber_delta > 0:
        raise ValueError("helmholtz_field_loss is an L2 reweighting; huber_delta cannot apply")
    err = (pred[..., :N_MEASURED] - target[..., :N_MEASURED]).float()
    h, w = err.shape[2], err.shape[3]
    z = torch.fft.rfft2(err.permute(0, 1, 4, 2, 3), dim=(-2, -1))  # (B, T, C, H, W')
    ky = torch.fft.fftfreq(h, device=err.device)[:, None]
    kx = torch.fft.rfftfreq(w, device=err.device)[None, :]
    k2 = (ky**2 + kx**2).clamp_min(1e-30)
    # The stored axis-0 order is physically reversed (y decreases with the index), so the
    # sign of ky must flip with it or the longitudinal and transverse parts are exchanged --
    # scripts/eval/divergence_diagnostic.py records this exact mistake, which is invisible because
    # both quantities are O(1).
    dot = kx * z[:, :, 0] - ky * z[:, :, 1]
    long_energy = (dot.real**2 + dot.imag**2) / k2
    total = (z.real**2 + z.imag**2).sum(dim=2)
    mult = torch.ones(w // 2 + 1, device=err.device)
    mult[1 : (w + 1) // 2] = 2.0
    # Parseval: sum |Z|^2 * mult / (H*W) is the mean square of the real field.
    weighted = ((total - long_energy) + lam * long_energy) * mult
    # Parseval carries one factor of (h*w) from the transform and a second from averaging
    # over the pixels of each field, which is why it appears squared here. The `lam = 1`
    # unit test pins this: with a single factor the term is (h*w)x the MSE it must equal.
    n_elem = err.shape[0] * err.shape[1] * N_MEASURED * h * w
    return weighted.sum() / (h * w * n_elem)


def mvpe_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """``mvpe_score``'s error term, differentiated.

    The metric time-averages ``(u, v)``, samples 9 rows at each of 4 downstream columns,
    and takes a relative L2 **per column** before averaging the columns -- so each column
    is normalised by its own mean-flow magnitude. That per-column normalisation is the
    whole reason this is not already covered by ``field_loss``: the wake stations carry
    small velocities and are weighted to nothing by an absolute MSE, while the metric
    grades them equally with the freestream.

    Stations come from :func:`rpde_t1.mvpe_probe_indices`, shared with the numpy mirror in
    ``score.sps_lab`` so the objective cannot drift from the metric.

    72 values out of the time-averaged field's 4,096 is **gameable by construction** --
    the probes are a measure-zero slice of what ``rel_l2_score`` grades, and they say
    nothing at all about any individual frame. Weight it as an auxiliary and watch
    ``rel_l2``.
    """
    h, w = pred.shape[2], pred.shape[3]
    rows, cols = mvpe_probe_indices(h, w)
    if not rows or not cols:
        return pred.new_zeros(())

    # float32 for the reduction: this is 72 numbers per sample against a relative
    # normaliser, so an fp16 norm here is all rounding error and no speed.
    r = torch.as_tensor(rows, device=pred.device)
    c = torch.as_tensor(cols, device=pred.device)
    p = pred.float()[:, :, r][:, :, :, c, :N_MEASURED]
    t = target.float()[:, :, r][:, :, :, c, :N_MEASURED]

    # (B, T, rows, cols, C) -> mean over time -> (B, cols, rows * C), one row per column.
    p = p.mean(dim=1).permute(0, 2, 1, 3).flatten(2)
    t = t.mean(dim=1).permute(0, 2, 1, 3).flatten(2)
    num = torch.linalg.vector_norm(p - t, dim=2)
    den = torch.linalg.vector_norm(t, dim=2).clamp_min(eps)
    return (num / den).mean()


def weighted_mse(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """MSE over the elements ``weight`` selects, normalised by their number.

    The normalisation is what makes an auxiliary weight mean the same thing across arms: a
    ``sum / numel`` would shrink a masked term in proportion to how little was masked, so
    ``--w-recon 1.0`` at mask ratio 0.25 and at 0.5 would be two different weights wearing
    one name. Zero-weight everywhere returns zero rather than NaN.
    """
    err = (pred - target) ** 2
    if weight is None:
        return err.mean()
    w = weight.expand_as(err) if weight.shape != err.shape else weight
    return (err * w).sum() / w.sum().clamp_min(1.0)


def subgrid_variance(fine: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Per-coarse-cell variance of a ``(B, T, fH, fW, C)`` native field.

    Returns ``(B, T, H, W, C)``: within each ``factor x factor`` block, the variance of the
    native values about their own block mean. This is the part of the sub-grid field that a
    coarse input *does* determine -- unlike the field itself, whose phase it does not -- so
    it is a well-posed target rather than a conditional mean dressed as one.
    """
    b, t, fh, fw, c = fine.shape
    blocks = fine.reshape(b, t, fh // factor, factor, fw // factor, factor, c)
    blocks = blocks.permute(0, 1, 2, 4, 6, 3, 5).reshape(b, t, fh // factor, fw // factor, c, -1)
    return blocks.var(dim=-1, unbiased=False)


def sps_scored_mask(target: torch.Tensor) -> torch.Tensor:
    """Elements v9 actually scores for SPS: those with a non-zero target.

    Straight from the v9 scorer (``scored = t != 0.0``). Targets outside the PIV
    field of view or inside the airfoil body are exactly zero and leave the SPS
    average, its numerator and its coverage.

    This matters more than it sounds: under v8 those elements were *cheap credit*
    (target 0, prediction ~0, so any narrow band trivially covered them), which
    inflated SPS. Excluding them is most of why our measured SPS fell from an
    equivalent ~34 to ~20 when we moved to v9.
    """
    return target[..., :N_MEASURED] != 0.0


def sps_loss(
    lower: torch.Tensor,
    upper: torch.Tensor,
    target: torch.Tensor,
    u_fs: torch.Tensor,
    tau: float = 0.02,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative of a differentiable relaxation of the v9 SPS credit.

    The scorer credits, per scored element, ``exp(-(upper - lower) / sigma)`` where
    the target lies inside the interval, and zero where it does not. The indicator
    is replaced by a product of sigmoids with temperature ``tau`` so gradients flow;
    everything else is the scorer's own expression, including which elements count.

    Two details that are easy to get wrong and both cost real points:

    * ``u_fs`` converts the width from the model's non-dimensional space back to
      physical units, because ``SIGMA_GLOBAL`` is physical. Skipping it mis-scales
      the width penalty by up to 17x across the Reynolds ladder.
    * only elements with a non-zero target are averaged over, matching
      :func:`sps_scored_mask`. Averaging over the full field would let the head buy
      credit in the measurement hole, which the scorer no longer pays for.
    """
    lo, hi = lower[..., :N_MEASURED], upper[..., :N_MEASURED]
    t = target[..., :N_MEASURED]

    scale = u_fs.reshape(-1, *([1] * (lo.ndim - 1))).clamp_min(eps)
    width_physical = (hi - lo) * scale

    inside = torch.sigmoid((t - lo) / tau) * torch.sigmoid((hi - t) / tau)
    credit = inside * torch.exp(-width_physical / SIGMA_GLOBAL)

    scored = sps_scored_mask(target)
    n = scored.sum().clamp_min(1.0)
    return -(credit * scored).sum() / n


def interval_credit_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    u_fs: torch.Tensor,
    k: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative mean ``exp(-k |err|_physical / sigma)``: the credit the *best* interval could earn.

    This is a loss on the **prediction**, not on a head, and it is the one term here that
    targets SPS's interval factor rather than one of the accuracy factors. Its value is
    ``score.sps_lab.oracle_interval_credit`` averaged over the scored elements -- the credit an
    interval sized at the realised error would collect, and therefore an upper bound on what any
    head can reach on that element. Maximising it asks the backbone to produce a field that is
    *cheap to bracket*, which is a different request from "small on average".

    Two properties of the SPS arithmetic make it the right shape, and neither is expressible
    through ``field_loss``:

    * **The penalty is physical while the model is not.** ``u_fs`` restores the scale, so a
      window at the top of the Reynolds ladder -- where the same *relative* error buys three
      times the width penalty (see ``models.bounds.bound_features``, where SPS falls 54.5 to
      32.5 across freestream quartiles at flat coverage) -- gets proportionally more gradient.
      A non-dimensional field term weights every operating point equally, which is not what is
      being scored.
    * **The credit is convex in the error, so sparsity pays.** ``exp(-k|e|)`` is
      redescending: an element already predicted well contributes the largest gradient, and one
      far outside any sensible interval contributes almost none. At fixed mean squared error the
      metric therefore prefers *many near-exact elements and a few hopeless ones* over a
      uniformly mediocre field, and this term is the only one that says so. ``huber_delta``
      moves in the same direction and measured +0.005 efficiency (FINDINGS 14); this goes all
      the way to the scorer's own functional.

    ``k`` is the width-per-error exponent: ``k=2`` is the symmetric oracle interval
    ``[pred - |err|, pred + |err|]`` and hence the literal bound. Smaller values relax toward an
    L1 term, larger ones sharpen the sparsity preference; it is a knob because the *achieved*
    width is a fitted multiple of the error scale rather than exactly 1. On the STFormer backbone
    of FINDINGS 28 that multiple is 1.28 -- mean half-width 0.132 ``sigma`` against a mean
    ``|err|`` of 0.103 -- so ``k=2`` evaluates the exponential within 30% of where the shipped
    interval actually sits and needed no sweep.

    Not a substitute for ``field_loss``. Alone it would abandon the wake, where the exponential
    is flat -- and the wake is most of ``rel_l2``. It is meant to be added at a weight measured
    against the field term on the backbone in question (FINDINGS 24a): its gradient is ~150x the
    field term's at ``k=2``, so parity is ``w ~ 0.007``, which is where the measured gain
    saturates.

    **Measured (FINDINGS 28), STFormer real fine-tune, `val_extrap`:** +0.05 / +0.35 / +0.37 SPS
    at 0.3x / 1x / 3x parity, paired CI [+0.269, +0.378] at parity, with ``rel_l2`` and ``mvpe``
    *improving* and ``tke`` giving back 0.07. The gain is efficiency (0.712 -> 0.717) rather than
    the ceiling, which is the point: it is the only term here that moves the interval factor.
    """
    err = (pred[..., :N_MEASURED] - target[..., :N_MEASURED]).abs()
    scale = u_fs.reshape(-1, *([1] * (err.ndim - 1))).clamp_min(eps)
    credit = torch.exp(-k * err * scale / SIGMA_GLOBAL)
    scored = sps_scored_mask(target)
    return -(credit * scored).sum() / scored.sum().clamp_min(1.0)


def interval_calibration_loss(
    lower: torch.Tensor, upper: torch.Tensor, pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Warm-up objective: make the half-width track the actual error magnitude.

    Optimising the SPS relaxation from a random initialisation is awkward -- the
    sigmoid gate gives almost no gradient when the interval sits far from the
    target. Regressing the half-width onto ``|pred - target|`` first puts the head
    in the right neighbourhood, after which the SPS term can shape it.
    """
    half = 0.5 * (upper[..., :N_MEASURED] - lower[..., :N_MEASURED])
    err = (pred[..., :N_MEASURED] - target[..., :N_MEASURED]).abs().detach()
    scored = sps_scored_mask(target)
    n = scored.sum().clamp_min(1.0)
    return (((half - err) ** 2) * scored).sum() / n


def ordering_penalty(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Penalise ``lower > upper``, which the scorer rejects outright.

    A submission with a single reversed interval scores zero on *every* subscore,
    not just SPS, so this is a hard constraint worth enforcing in training as well
    as clamping at inference.
    """
    return F.relu(lower - upper).mean()


def opreg_loss(pred: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """MSE between :class:`~..models.aux.OpHead`'s prediction and the true operating point.

    Reads only ``cond[:, :2]`` -- ``(log-Re, AoA)`` -- and never index 2, the domain bit: the
    target is the physical operating point sim and real share, not the thing that tells them
    apart. See ``TrainConfig.w_opreg`` for how this differs from :func:`domain_moment_loss`.
    """
    return F.mse_loss(pred, cond[:, :2])


def domain_moment_loss(
    feat: torch.Tensor, is_sim: torch.Tensor, channel_dim: int = 1
) -> torch.Tensor:
    """Per-channel first- and second-moment gap between a batch's sim and real halves.

    A latent-alignment term for joint sim/real fine-tuning: it pushes the encoder's deepest
    representation to have the same per-channel mean and spread on both domains, so what the
    decoder reads is a shared code rather than two.

    **Why moments and not CORAL.** Full CORAL matches the C x C feature covariance, which at
    this batch size cannot be estimated: batch 8 with a 50/50 mixture gives ~4 samples per
    domain against a 128-channel covariance. Pooling over batch x time x space instead gives
    ~10^4 tokens per domain, so per-channel mean and std are well determined. Same intent,
    computable at the batch size a T4 allows.

    **Why the encoder only, and weakly.** Sim and real differ *physically* -- sim
    over-fluctuates by 1.8x-20x in energy -- so a term strong enough to erase the gap erases
    what fine-tuning exists to learn (NEW_ITEMS section IV). Worse here than in the general
    case: with ``cond_mode="none"`` the U-Net discards ``cond``, so no domain bit reaches the
    decoder and this term can only remove information, never route it.

    Returns exactly zero for a single-domain batch, which is not a corner case to ignore:
    the sampler draws domains independently, so a small batch legitimately comes out all
    real, and a mean over an empty half would be NaN.

    ``channel_dim`` says which axis the channels are on, and it is a required input rather
    than an assumption: a convolutional bottleneck hands over ``(B, C, T, H, W)`` while a
    transformer's post-block LayerNorm hands over ``(B, tokens, C)``. Defaulting to 1 for
    both would silently align *per-token-position* statistics across the feature axis on the
    transformer -- a different objective that still returns a plausible number, which is
    PROTOCOL 5's "two definitions of the same thing" in the shape of a tensor layout.
    ``loop.FeatureTap`` reports it from the module it matched.
    """
    b = feat.shape[0]
    # (B, ..., C, ...) -> (B, N, C): channels to the end, everything else pooled as tokens.
    # Float32 because a std over 10^4 fp16 tokens loses most of its digits.
    n_channels = feat.shape[channel_dim]
    tokens = feat.float().movedim(channel_dim, -1).reshape(b, -1, n_channels)
    sim = is_sim.reshape(b) > 0.5
    if not (bool(sim.any()) and bool((~sim).any())):
        return feat.new_zeros((), dtype=torch.float32)
    a = tokens[sim].reshape(-1, tokens.shape[-1])
    c = tokens[~sim].reshape(-1, tokens.shape[-1])
    return ((a.mean(0) - c.mean(0)) ** 2 + (a.std(0) - c.std(0)) ** 2).mean()


def domain_correlation_loss(
    feat: torch.Tensor, is_sim: torch.Tensor, channel_dim: int = 1
) -> torch.Tensor:
    """Match cross-channel correlation structure between simulated and real features.

    Tokens are pooled within each domain exactly as in :func:`domain_moment_loss`. Correlation,
    rather than covariance, prevents high-variance channels from setting the dose and makes this
    complementary to the existing diagonal mean/std objective. The diagonal is excluded because
    it is one by construction and would only dilute the useful off-diagonal signal.
    """
    b = feat.shape[0]
    n_channels = feat.shape[channel_dim]
    tokens = feat.float().movedim(channel_dim, -1).reshape(b, -1, n_channels)
    sim = is_sim.reshape(b) > 0.5
    if not (bool(sim.any()) and bool((~sim).any())):
        return feat.new_zeros((), dtype=torch.float32)

    def correlation(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, n_channels)
        x = x - x.mean(0)
        x = x / x.square().mean(0).add(1e-6).sqrt()
        return x.T @ x / x.shape[0]

    gap = correlation(tokens[sim]) - correlation(tokens[~sim])
    off_diag = ~torch.eye(n_channels, dtype=torch.bool, device=gap.device)
    return gap[off_diag].square().mean()
