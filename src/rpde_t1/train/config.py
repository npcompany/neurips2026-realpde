"""Training configuration.

A plain dataclass serialised to JSON next to every checkpoint, so a run is
reproducible from its output directory alone and two runs can be diffed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """Everything that defines a run."""

    # --- what to train -------------------------------------------------------
    model: str = "cno"
    """``cno`` | ``fno`` | ``transolver`` for the retrained baselines, or one of our own."""

    cond_mode: str = "none"
    """``none`` reproduces the reference U-Net (its FiLM conditioning is a learned
    constant, a diffusion leftover); ``operating_point`` feeds (log-Re, AoA, domain)
    into that same slot; ``operating_point_tau`` adds the window's decorrelation time;
    ``operating_point_noise`` adds the realized ``real_noise_scale`` dose instead, so the
    backbone is told how corrupted its input is (DPOT-style denoising conditioning) rather
    than being trained against one fixed, unconditioned noise level."""

    t_in: int = 20
    """Input frames the model reads. The scored task hands 20; ``40`` (the Track-2 stream
    exposes every past frame) folds the previous 20 into the stem as well, with the output
    horizon, tokens and every other module unchanged. Datasets enumerate ``t_in + 20``-frame
    windows, so a longer context loses the first ``t_in - 20`` frames of every trajectory. A
    checkpoint trained at 20 warm-starts a 40-frame run bit-exactly (``loop.widen_stem_history``).
    Every operating-point / freestream estimate reads the LAST 20 frames, so the conditioning
    and normalisation are identical to the 20-frame model's."""

    cond_source: str = "label"
    """Where a **real** window's ``(Re, AoA)`` in the conditioning vector come from:
    ``label`` reads the cell's metadata, ``estimate`` runs ``features.estimate_operating_point``
    on the raw window, exactly what inference does (metadata is empty on scored calls).
    Simulated windows always use their labels: sim is non-dimensional, so the freestream
    estimator cannot read a Reynolds number there. ``label`` is what every checkpoint before
    2026-09-09 trained on; the mismatch it leaves is bounded by FINDINGS 121."""

    residual: bool = False
    """Predict a delta from the last observed frame. Off for the baselines, so the
    retrained reference matches the published architecture; on for our models."""

    residual_mode: str = "last"
    """What ``residual`` predicts a delta from: ``last`` (the last observed frame, the shipped
    anchor) or ``damped`` (``exp(-t/residual_tau)``-weighted blend of the last frame and the
    window's confidence-weighted mean -- the tubelet model's reference, FINDINGS 114). Only
    read when ``residual`` is on."""

    residual_tau: float = 3.0
    """Decay time in frames of the ``damped`` reference's weights."""

    bounds: bool = False
    """Emit interval bounds and train them against the SPS functional."""

    freeze_backbone: bool = True
    #: Exponent tilting the training sampler toward high Reynolds number
    #: (``P(window) ~ (Re / Re_max) ** re_tilt``). 0 keeps plain shuffling.
    re_tilt: float = 0.0
    """With ``bounds``, train only the interval head. The cheap path: it banks the
    SPS gain without touching the other four subscores, and isolates the effect."""

    head_lr_mult: float = 1.0
    """Learning-rate multiplier for the bound head relative to the backbone.

    Only meaningful when the backbone is *not* frozen: the head is 13k randomly
    initialised parameters wanting ~3e-3, while an already fine-tuned backbone wants
    ~3e-4. Left at 1.0 the optimiser behaves exactly as before, so frozen runs are
    unaffected."""

    w_jepa: float = 0.0
    """Weight on a JEPA-style latent-prediction term, applied DURING supervised training.

    The co-objective form of self-supervision, as opposed to a separate pretraining stage. Each
    step, the live model sees the **masked** input (``aux_mask``, so this requires it) while an EMA
    copy of the same model sees the **clean** input; the term is the MSE between their deepest
    shared representations (``loop.BOTTLENECK_MODULES``, the same tap ``align_weight`` uses). So the
    model is asked to infer from a corrupted view what it would have encoded from the whole one.

    Why this shape rather than a pretraining stage. FINDINGS 67k measured a *sequential* JEPA
    pretrain losing 0.3-0.8 ceiling to the supervised sim pretrain, with the damage almost entirely
    `tke` -- an amplitude the latent objective never has to calibrate. Run as a co-objective the
    supervised term keeps calibrating amplitude every step while the latent term shapes the
    representation, so the two failure modes do not have to trade.

    ``jepa_ema`` sets the target's decay. Needs ``aux_mask``: with an unmasked input the term is
    the identity map between a model and its own EMA, which is near-zero and teaches nothing."""

    jepa_ema: float = 0.996
    """EMA decay for the ``w_jepa`` target encoder. 0.996 is `models.jepa`'s own default."""

    dropout_sites: str = ""
    """Which submodules ``dropout`` is injected after; empty uses ``uq.dropout_targets``.

    Comma-separated module-name components, e.g. ``"mlp,proj"``. The default resolves to
    ``("mlp",)`` on every transformer, which is **half of the standard transformer dropout**:
    ViT/DeiT/Swin place it after the attention output projection *and* after the MLP, and this
    lineage has only ever had the MLP half. `attn.proj` is the missing site.

    ``modulation`` stays excludable and is not in any default: FINDINGS 23z-ter measured the
    conditioning path as this architecture's one conditioning win, so dropping it is an ablation
    rather than a regulariser."""

    dropout_kind: str = "element"
    """``element`` (the default, ``F.dropout``) or ``channel`` (``F.dropout2d``: whole feature
    maps of a 4-D conv output, the conv-net form; see ``uq.DROPOUT_KINDS``). Added 2026-09-18 for
    the `unet2d` regularisation batch (FINDINGS 153)."""

    drop_path: float = 0.0
    """Stochastic depth: drop whole residual branches per sample (see models.uq).

    A different prior from ``dropout``, not a stronger dose of it. Elementwise dropout removes
    units inside a sublayer, so every sample still traverses every block; stochastic depth
    removes the sublayer for that sample, training the network as an ensemble over depths. It is
    the regulariser transformers actually ship with -- ViT/DeiT/Swin set elementwise dropout to 0
    and tune this instead -- so a dropout result says nothing about it in either direction.

    Applies to ``attn`` and ``mlp`` branches, never ``modulation``: FINDINGS 23z-ter measured the
    conditioning path as this architecture's one conditioning win, and dropping it is an ablation
    rather than a regulariser. Composes with ``dropout``; both are forward hooks on the bare
    backbone, so the interval head is never sampled with them."""

    drop_path_hi: float | None = None
    """Operating-point-dependent dose for ``drop_path`` (2026-09-24), the ``re_dose`` idea reused
    from ``fluct_aug_hi`` on the regulariser rather than an augmentation. When set, each sample's
    stochastic-depth rate is linear in log Re from ``drop_path`` at Re 3750 to this value at Re
    26700 (clamped outside), so the model is regularised harder on the data-rich low-Re cells it
    over-fits and left deeper at the high-Re corner where §168 leaves headroom and capacity helps.
    ``None`` keeps the flat ``drop_path`` dose. Real windows only; simulated windows keep the flat
    dose (they carry no real operating point to condition the trade on). Inference is unchanged --
    the hooks are identity at eval, exactly as the flat dose is."""

    dropout: float = 0.0
    """Inject dropout into the backbone via forward hooks (see models.uq).

    Needed for a *valid* MC-dropout comparison: sampling dropout at test time on a
    network trained without it measures layer sensitivity to random masking, not
    predictive uncertainty. Hooks carry no parameters, so a dropout-trained checkpoint
    still loads into the plain architecture."""

    log_domain_split: bool = False
    """Diagnostic only: on log steps, also record the train-batch relative-L2 split by domain
    (``rel_real``/``rel_sim``, keyed on the ``is_sim`` flag ``cond[:, 2]``). Detached, never
    enters ``loss`` or the RNG, so a run with it on optimises byte-identically to one without.
    For the joint stage the pooled ``field``/``total`` mixes the 50/50 real+sim batch; this
    separates the two halves so the real fit can be watched apart from the (easy, abundant)
    simulated fit. Off by default."""

    amplitude_free: bool = False
    """Grade the SIM stage on mean flow and fluctuation *shape*, never amplitude.

    See ``losses.amplitude_free_field_loss``. Sim over-fluctuates real 1.8-20x (FINDINGS 13) and
    37e localises the pretraining-budget cost to the second moment, so this removes the channel
    through which a longer pretrain installs the wrong amplitude, instead of correcting it in data
    space as ``rescale_fluct`` does. Real sets the amplitude in stage iii.

    Refused on real data by ``train`` -- there the amplitude is the signal."""

    moment_of: str = ""
    """Train a MOMENT NETWORK instead of a field model (2026-09-19, FINDINGS 153n): the path of a
    frozen mean-optimal checkpoint whose per-element squared residuals ``(y - mu)^2`` on u and v
    are the target. The network then outputs ``log sigma^2`` with two channels per lead and is
    fitted with the Gaussian negative log-likelihood ``0.5 * (s + r^2 exp(-s))`` over the valid
    cells; the field and tke terms are not applied, and ``residual`` is forced off (the output is
    not a field). Empty (default) is ordinary training."""

    field_norm: str = "none"
    """``frame``: divide each output frame's field error by that frame's own detached error RMS.

    See ``losses.frame_normalized_field_loss``. Every lead then contributes equal gradient
    instead of the late frames, whose errors are largest, carrying most of it; the loss is also
    restricted to the PIV-valid cells. Replaces the field term (Bruno's design, 2026-09-17), so it
    is exclusive with the other replacements (``helm_lambda``, ``lead_decay``, ``amplitude_free``).
    ``none`` is the shipped ``field_loss``."""

    fluct_aug: float = 0.0
    """Fluctuation-amplitude augmentation of REAL training windows (regulariser, 2026-09-19).

    Each window's 40-frame block is rescaled about its own per-pixel time mean by one factor
    ``exp(N(0, fluct_aug))``, input and target together, so the trajectory stays consistent
    and only its fluctuation level moves. Aimed at the mechanism §153 measured -- the network
    memorising each training cell's fluctuation statistics -- by making the level something it
    has to read from the window rather than infer from the cell. 0 is off; sim windows are
    never touched (they have their own ``rescale_fluct``)."""

    aoa_jitter_deg: float = 0.0
    """AoA-conditioning jitter for angle generalisation (§52d, the surviving mitigation for the
    -1.5..-1.8 unseen-angle hole §63g). Each REAL window's conditioning (and geometry-plane) angle
    is perturbed by ``U(-aoa_jitter_deg, +aoa_jitter_deg)`` degrees; the velocity/pressure pixels
    and the forecast target are UNTOUCHED (§52d keeps the pixels alone; the sign of AoA breaks
    spatial symmetry, so no pixel resampling). The model then sees a CONTINUOUS AoA axis and learns
    to interpolate to unseen intermediate angles instead of memorising the five discrete labels.
    ~half the 5-degree spacing (2.5) makes adjacent angles' jitter ranges overlap, which is what
    turns discrete labels into continuous coverage; too large collapses the angle signal. 0 is off;
    sim windows (which already span the five angles) and every eval pass are never jittered."""

    fluct_aug_hi: float | None = None
    """Operating-point-dependent dose for ``fluct_aug`` (2026-09-23). When set, the log-normal
    sigma is linear in log Re from ``fluct_aug`` at Re 3750 to this value at Re 26700, read off
    the cell label (``data.dataset.re_dose``). Motivated by FINDINGS 184i/195: the flat dose's
    gain is at the bottom of the ladder and its `val_time`/high-Re cost grows with the dose.
    None (default) keeps the flat sigma, bit for bit."""

    tgt_amp_aug: float = 0.0
    """One-sided TARGET-only fluctuation boost at Re 26700 (2026-09-23); ``tgt_amp_aug_lo`` is
    the same at Re 3750, linear in log Re between. Per real window the target frames'
    fluctuation about their own time mean is multiplied by ``exp(U(0, dose))`` >= 1 while the
    input is left alone. Unlike ``fluct_aug`` this is a bias, not an invariance: it moves the
    input-to-target amplitude relation, aimed at FINDINGS 183b's one-sided 14-30% energy
    deficit at the top of the ladder (which the scalar loss term of §188 could not reach). 0 off."""

    tgt_amp_aug_lo: float = 0.0
    """``tgt_amp_aug``'s dose at Re 3750 (default 0: nothing where the amplitude is right)."""

    tgt_fluct_aug: float = 0.0
    """Symmetric TARGET-only fluctuation augmentation (mechanism probe, 2026-09-23 night): the
    target frames' fluctuation about their own time mean is multiplied by ``exp(N(0, sigma))``,
    input untouched; ``tgt_fluct_aug_hi`` tapers sigma by log Re exactly as ``fluct_aug_hi``
    does for ``fluct_aug``. Zero-mean in log, so no bias. Separates the two halves of
    ``fluct_aug`` (FINDINGS 209a): if this half alone reproduces the Re 3750 gain, the mechanism
    is label noise on the target amplitude; if not, invariance to the input's amplitude. 0 off."""

    tgt_fluct_aug_hi: float | None = None
    """``tgt_fluct_aug``'s sigma at Re 26700 (None = flat), see ``fluct_aug_hi``."""

    mean_aug: float = 0.0
    """Mean-flow perturbation of real windows toward an adjacent-Reynolds training cell
    (2026-09-23). Per window a donor is drawn among the same-angle training cells one rung
    up or down, and ``alpha ~ U(0, mean_aug)`` of the climatology difference (freestream
    units, zero in either cell's holes) is added to the whole block, input and target,
    fluctuations untouched. The operating-point estimate reads the shifted window, so the
    conditioning follows: the window is presented as a nearby unseen operating point
    (FINDINGS 46 R1: the mean is the moment that is smooth in the operating point). 0 off."""

    cond_drop: float = 0.0
    """Conditioning shuffle (regulariser, 2026-09-19): with this probability per sample, the
    operating-point vector is replaced by another sample's from the same batch during
    training. In-distribution (unlike zeroing), and it makes the operating point an unreliable
    witness of the fluctuation level, so the window has to carry it. 0 is off."""

    moment_joint: bool = False
    """One network for frames AND their posterior variance (Bruno, 2026-09-19 evening).

    The backbone emits ``C + 2`` channels per frame: the three frames, then ``log sigma^2`` for
    ``u`` and ``v``. The residual anchor and every field / tke term read the frames only (the
    ``kmap_channel`` precedent); the two extra channels are graded by the Gaussian NLL against
    the DETACHED residual of the frames, weighted by ``w_moment``, so the variance head cannot
    pull the frames toward a heteroscedastic fit. At inference the predictor keeps the frames
    and the variance is read from the same forward. Exclusive with ``moment_of``."""

    w_moment: float = 0.1
    """Weight of the joint model's NLL term (``moment_joint``). The NLL's per-element gradient
    is ``0.5 (1 - r^2 / sigma^2)``, O(1); the shipped field term's is O(1/delta); 0.1 keeps the
    variance head trained without letting its loss dominate the reported total."""

    bound_cond: bool = True
    """Give the interval head the conditioning vector (normalised log-Re, AoA, is_sim).

    Under freestream normalisation the head is otherwise blind to Reynolds while its
    objective is physical, which costs ~22 SPS points between the lowest and highest
    freestream quartile. Defaults on; checkpoints written before this field existed
    read back as False so they still load."""

    bound_noise: bool = False
    """Give the interval head two estimates of the PIV measurement-noise level.

    Part of ``|pred - target|`` is measurement noise in the *target*, which no dynamics
    model can predict away -- an irreducible floor on the error and therefore on the width
    worth paying for. None of the head's other ten features says how noisy this particular
    window is. See ``models.bounds.piv_noise_features``."""

    bound_mode: str = "offsets"
    """How the interval head parameterises its output.

    * ``offsets`` -- the head chooses the width directly.
    * ``scale`` -- the head predicts an error scale and the width follows the *Gaussian*
      closed-form optimum (``models/interval.py``). Measured worth +0.00 (FINDINGS 10), and
      FINDINGS 14 explains why: the Gaussian over-widens at every rho.
    * ``logerr`` -- the head predicts an error scale and the width is left to a fitted
      :class:`~rpde_t1.models.width_rule.WidthRule`. This is the configuration that took
      ``val_extrap`` SPS from 47.49 to 50.67, but **only in combination** with that rule:
      the scale alone is worth -0.38 and the rule alone -1.65 (FINDINGS 14). Trained by
      least squares on ``log|err|``, so ``--w-sps`` should be 0 with it."""

    bound_logs: bool = False
    """Give the interval head log-magnitude channels alongside the linear ones.

    The structure that predicts the local error is multiplicative, and a GELU conv stack on
    linear magnitudes has to approximate a log-additive function over three decades. The one
    feature change that measured: -0.36 SPS without it, against +-0.1 for capacity, depth,
    steps, PIV-noise features and coordinate planes (FINDINGS 14)."""

    bound_coords: bool = False
    """Give the interval head normalised row/column planes. Measured no effect (50.42 against
    50.55); kept because the error field is strongly positional and the negative result is
    worth being able to reproduce."""

    bound_grad: bool = False
    """Give the interval head ``log|grad pred|^2`` and ``log|grad x_last|^2`` (FINDINGS §78c-bis):
    where the field is steep, not merely where it is large. Substitutes for capacity at width 32
    and composes with it at width 64. Missing from this config once (FINDINGS §82): dropped
    silently by ``load_checkpoint``'s unknown-field filter, so a checkpoint assembled with it set
    rebuilt one channel narrow and failed to load its own state dict."""

    bound_pool: bool = False
    """Give the interval head a pooled (global + per-lead + per-pixel-over-time) reduction of the
    departure field, an infinite receptive field for the cost of a mean. Measured a null
    (FINDINGS §78h) and kept only because it is now measured rather than assumed."""

    bound_fluc: bool = False
    """Give the interval head the input window's fluctuation-energy map."""

    bound_lstd: int = 0
    """Lead-std planes the interval head reads (0 = none)."""

    bound_film: int = 0
    """Route the operating point into the interval head as FiLM modulation instead of (1) or as
    well as (2) input planes. Row 23's shipped head sets 1.

    These three landed on ``ScaleFitConfig`` and in ``factory.wrap_model``'s dict reader but not
    here, which is the THIRD time this omission has bitten (see ``bound_grad`` for the second,
    FINDINGS §82). ``load_checkpoint`` rebuilds through ``TrainConfig``, so an unknown key is
    dropped and the head comes back without its modulation and four channels wide -- while
    ``build_for_inference``, which reads the raw dict, builds it correctly. The two paths
    disagreeing is what broke `train tke-head` on any assembled FiLM bundle (2026-09-18):
    `make ship-t1` itself would have hit it, since row 23 reused row 22's TKE head rather than
    fitting one on its new bundle. Defaults are the pre-FiLM head, so every older checkpoint
    rebuilds exactly as before. ``scale_fit.merge_scale_head`` explains what each one changes."""

    bound_width: int = 16
    bound_depth: int = 3
    bound_kt: int = 3
    """Kernel size of the interval head's convolutions along the lead axis (3 shipped; 1 convolves
    each lead on its own). See ``BoundHead``; ``scale_fit.merge_scale_head`` writes it."""
    w_calib: float = 1.0
    """Weight on the width-vs-|error| warm-up term, which puts the head near the
    right scale before the SPS relaxation shapes it."""

    calib_warmup_frac: float = 0.3
    """Fraction of steps during which the warm-up term is active."""

    # --- data ----------------------------------------------------------------
    train_split: str = "train"

    train_kind: str = "real"
    """``real`` | ``sim`` | ``sim_resampled``.

    ``sim_resampled`` is the simulated grid with its time axis rescaled per cell so a
    simulated frame advances the flow as much as a real frame does at the same operating
    point (FINDINGS 3). Build it with ``scripts/gen/build_resampled_sim.py``; it is a
    pretraining source only, and fine-tuning still uses ``real``."""

    normalizer: str = "freestream"
    """``freestream`` (ours) | ``gaussian`` (the organizers') | ``none``.

    ``gaussian`` exists to reproduce their released checkpoints: every
    ``configs/foil/*.yaml`` sets ``normalizer: "gaussian"``. It weights high-Re
    windows far more heavily than ours does, which is why the two schemes can rank
    models in opposite orders."""

    mask_prob: float = 0.0
    """Probability of zeroing the simulated pressure channel. Their configs use 0.1,
    a sim2real bridge that shows the model the real data's missing-p case."""

    noise_scale: float = 0.0
    """Multiplicative Gaussian noise on simulated data. Their configs use 0.1."""

    real_noise_scale: float = 0.0
    """Multiplicative noise on the *real* input window (not the target, not sim).

    The counterpart to ``noise_scale``, which the organizers apply to clean CFD. Here the
    model is asked to predict the measured future from a noisier measured past, which is
    input-noise regularisation rather than a sim2real bridge. Multiplicative, so exact
    zeros stay zero and the PIV holes are not filled in.

    Note this *widens* the sim/real measurement gap rather than closing it: to close it,
    put the noise on the simulated half instead (``noise_scale``, already available).

    Under ``cond_mode="operating_point_noise"`` this stops being a fixed dose and becomes
    the *ceiling* a per-window dose is drawn uniformly from, with the realized fraction fed
    to the backbone as a conditioning feature -- the fixed-dose form gives a conditioning
    slot nothing to vary over. Under every other ``cond_mode`` the dose stays fixed, exactly
    as before this mode existed."""

    mask_swap_prob: float = 0.0
    """Probability of unioning another training cell's measurement mask into a sample.

    Real PIV loses 5-11% of the field to the airfoil, its laser shadow and rejected
    vectors, and **that region is scored**: the target reads zero there and the scorer
    keeps those elements, so the model must learn to output zero exactly where the input
    is missing. With 69 training cells it can do that by memorising 69 hole patterns; the
    private test set has patterns it has never seen.

    Union, never replacement -- see ``WindowDataset.__getitem__``. Only the training
    loader is augmented; validation keeps each cell's own mask.

    Not expected to recover scoring loss directly: FINDINGS section 7 measures dropped
    elements at 5.68% of the tensor but 0.01% of its squared norm, worth ~+0.04 rel_l2.
    The case is generalisation of the wake geometry, which is what ``mvpe`` grades."""

    zero_p: bool = False
    """Zero the pressure plane of simulated samples, in the input and the target.

    Makes a simulated window structurally identical to a real one -- real PIV has no
    pressure at all -- which is what lets an architecture benchmarked on sim be read
    against the real arms. It also keeps the official scorer at two measured channels:
    sim's ``p`` has std 12.6 against ``u``'s 0.44, so a target that carries it turns
    ``rel_l2_score`` into a pressure metric. Off by default, so sim *pretraining*
    keeps the auxiliary pressure signal it has always had."""

    p_target: str = ""
    """Supervise simulated pressure on the backbone's own third output channel: ``"gauge"``.

    The channel already exists (``out_shape`` is three wide), ``losses.field_loss`` already
    has a term for it, and ``models.predictor`` already zeroes it at inference -- so this
    costs no parameters and no inference time. What it adds is the three corrections without
    which the channel is not supervisable: the per-frame spatial gauge is removed (pressure
    is defined up to ``p + f(t)``, and one release cell's datum drifts by 38.3 against a
    field spread of 0.15), the freestream divisor is squared (``p ~ rho U**2``), and the
    measurement mask is honoured (at AoA=10 the solid section holds 76% of the squared
    mass). See ``data/dataset.py``'s ``p_target`` block for the measurements.

    Orthogonal to ``zero_p``, which blanks the plane in the *input* as well: ``p_target``
    always blanks the input, because a simulated window carrying ``p`` in its input is a
    different task from the real one. So the two compose as "pressure as privileged
    supervision" rather than "pressure as an extra input channel".

    Sim-only, and inherited by a joint fine-tune's simulated half the way ``zero_p`` is."""

    w_p: float = 1.0
    """Weight on the pressure term of the field loss, relative to ``u`` and ``v``.

    Declared because it was previously a hard-coded 1.0: the term is a per-element mean over
    the contributing samples, so an *unscored* channel carried as much of the field loss as
    both scored channels together. ``1.0`` reproduces that; the dose is what a sweep varies."""

    w_pvar: float = 0.0
    """Weight on the temporal variance of the pressure channel (``losses.pvar_loss``).

    The second moment of what ``p_target`` supervises pointwise, and the same relation to it
    that ``w_tke_huber`` has to the field term. Needs ``p_target='gauge'``: the temporal
    variance of an ungauged channel is the drift of an arbitrary datum.

    Motivated by a measurement rather than by analogy -- ``var_t(p')`` explains 8.4 points of
    the scored ``tke`` map's spatial variance beyond the p-free features the model already
    holds (see ``losses.pvar_loss``). Relative-form like ``w_tke``, so it plateaus at O(0.1-1)
    and the dose is not comparable with an absolute term's.

    **Not usable alone.** ``var_t`` has gradient ``2(x - mean_t x)``, identically zero where the
    channel is constant -- and the backbone's pressure rows are zero-initialised and receive no
    gradient under ``zero_p``, so they are exactly constant. A pure second-moment term therefore
    opens at a *stationary point*: measured, ``w_p 0`` with ``w_pvar 1.0`` holds ``pvar`` at
    1.0000 for 150 steps with the channel untouched. Pair it with a small ``w_p`` to break the
    symmetry (FINDINGS 52i)."""

    joint_rescale_fluct: bool | None = None
    """Override the simulated half's ``rescale_fluct`` in a joint fine-tune. ``None`` inherits.

    Exists because inheritance had no escape hatch and failed silently. ``joint_sim_real`` builds
    the simulated half's shaping as ``bool(pre.get(key, cfg_value))`` -- ``dict.get`` with the run's
    own flag as the *default* -- so any key the pretrain records at all wins unconditionally, and
    every run's ``config.json`` records ``rescale_fluct`` explicitly. Net effect:
    ``--rescale-fluct`` on a joint fine-tune was accepted, written to ``config.json``, and applied
    to nothing (FINDINGS 52j).

    The inheritance itself is right and is kept as the default -- an anchor to the pretraining
    distribution is only an anchor if it *is* that distribution. What was missing is the ability to
    say otherwise, which needs a per-field sentinel because a bool cannot distinguish "unset" from
    "False" the way ``p_target``'s empty string can.

    Why it is worth having: §30b names sim's imported first-moment offset the largest gap in the
    release, §32-bis prices joint training's ``val_extrap`` cost at rel_l2 -0.34 / mvpe -0.81, and
    this is the correction built for that family -- median gain 0.488 on ``sim_resampled``, i.e. sim
    carries ~4x real's fluctuation energy. It had never been testable on the stage that uses sim."""

    rescale_fluct: bool = False
    """Rescale simulated fluctuations per cell toward real's turbulence intensity.

    ``r = sqrt(TKE_real / TKE_sim)`` measured on the 81 paired cells and fitted over
    (log Re, AoA) for the 19 sim-only ones (``configs/sim_fluctuation_scale.json``).
    Non-dimensionalising by ``u_fs`` fixes the mean-flow scale and leaves ``u'/U``
    untouched; sim over-fluctuates by 1.8x-20x in energy at every cell, which is the
    residual this corrects."""

    phase_aug: bool = False
    """Also train on the three unscored ``[::2, ::2]`` phases of the recorded 64x128 grid.

    The raw PIV is 64x128 and Track 1 scores the ``(0, 0)`` decimation; the other three
    offsets are equally valid 32x64 recordings of the same 82 trajectories, so this is 4x
    the training windows at no cost in inference or in held-out validity -- the test input
    is phase (0, 0) and the model is never told which phase it is looking at.

    Only the *training* loader is augmented. Validation and scoring stay on phase (0, 0),
    which is the phase the competition scores.

    Needs the caches: ``make phase-cache`` (~7 min, one pass over the real HDF5)."""

    cond_planes: bool = False
    """Feed the conditioning vector to a vendored baseline as constant input planes.

    CNO, FNO and Transolver have no conditioning slot and ``vendor/`` is never edited, so
    every baseline arm this project has run was **unconditioned** while the recipe it was
    being compared against was not -- a handicap FINDINGS 70l prices at ~+0.12 and 72e
    records as the reason a baseline loss was never clean evidence. This broadcasts
    (log-Re, AoA, is_sim) to ``H x W`` planes and concatenates them onto the input channel
    axis, so the baseline's own first layer learns to read the operating point.

    Same device as ``geom_channels`` -- a field the model convolves with rather than a
    per-channel affine -- and accounted for in the same place, so the first layer is sized
    for it at construction and the checkpoint rebuilds from its own config. Refused on our
    own models, which take ``cond`` directly (``factory.n_cond_planes``).

    Defaults False so every checkpoint written before it existed builds the same width."""

    geom_channels: str = ""
    """Comma-separated geometry planes appended to the input, e.g. ``"sdf,inlet"``.

    Empty reproduces the three-channel input exactly. See
    ``geometry.GEOMETRY_CHANNELS`` for the groups and their widths.

    Why this exists: an unseen angle of attack costs about **-4 final points**, worst on
    ``mvpe`` (-5.52), a wake-position metric -- and AoA currently reaches the model as the
    scalar ``AoA/20`` inside the conditioning vector, which the vendored baselines *discard
    entirely*. So for FNO the angle is not an input at all. A scalar could not express
    where the body sits in any case; a signed-distance field can, at any angle including
    unseen ones.

    The angle used is the one **estimated from the input window** (1.77 deg MAE), not the
    label, in training as well as at inference. Same discipline as ``u_fs``: if the two
    paths used different estimates the model would meet a distribution at test time that
    it never saw in training."""

    flow_channels: str = ""
    """Comma-separated planes derived from the observed window, e.g. ``"vort,tke"``.

    Empty reproduces the three-channel input exactly. See ``features.FLOW_CHANNELS`` for the
    groups; each is one plane, scaled to unit rms by a measured constant.

    Why it is a separate field from ``geom_channels`` rather than another group in it: a
    geometry plane is a function of the *angle* and is looked up, while these are functions
    of the *window* and are computed from what the model is being handed. That difference is
    also what makes the experiment worth running in two halves --

    * ``vort`` is linear in the input, so a 3x3 stem already represents it and the plane can
      only change the conditioning of the optimisation, never the information available.
      FINDINGS 43 is the precedent for reading a null here as "the stem was already doing
      it".
    * ``enstrophy`` and ``tke`` are quadratic and **no** linear stem produces them at any
      width. ``tke`` in particular is the quantity the post-hoc fluctuation blend uses --
      the input window's own fluctuation energy is unbiased for the target's where the
      model's own prediction is not (FINDINGS 5c) -- so that plane is N3's correction handed
      to the trunk as a feature instead of applied afterwards.

    An STFormer pretrain is transferred into a widened stem by
    ``train.loop.widen_grouped_stem``, which zeroes the new columns; the arm therefore starts
    bit-identical to its control."""

    pe_content: bool = False
    """ModPCT only: also add the centroid Fourier encoding to the tokens as *content*.

    Faithful ModPCT injects position only into the adaLN conditioning, which measured as
    a strong early inductive bias with a lower ceiling -- it converged by 2700 steps and
    was then overtaken by its own unconditioned control. Defaults False so the faithful
    port stays the reference."""

    arch: str = ""
    """Architecture hyper-parameters as ``key=value`` pairs, e.g. ``"dim=512,depth=10"``.

    Passed to the model constructor by ``models.factory.parse_arch``. One field rather
    than one per knob: every architecture column added here has to be defaulted
    correctly forever so that checkpoints written before it existed still rebuild, and
    ``evaluate.POST_HOC_DEFAULTS`` is what that costs when it is got wrong. Empty means
    the architecture's own defaults, so this changes nothing for any existing model."""

    stride: int = 1
    recent_dropout: float = 0.0
    """Probability that a training window's MOST RECENT frames are replaced by its time mean.

    The opposite end of the window from ``history_dropout``, and a different question.
    ``scripts/eval/input_memory_probe.py`` measured the shipped backbone's effective input
    memory at about TWO frames: replacing the other eighteen with the window's time mean costs
    ~2% of block ``rel_l2`` and nothing on ``tke``, uniformly across operating points -- including
    the high-incidence cells where the wake sheds on a ~20-frame cycle. So the model reads a
    twenty-frame window and uses two of them.

    That is either the flow's limit (its fluctuation decorrelates in 2-3 frames) or a learned
    shortcut. This flag tests it: hide the frames the model relies on, with probability
    ``recent_dropout`` and depth uniform in ``1..recent_dropout_max``, and it must find whatever
    is left in the older frames or do worse. Replacement is the window's own time mean rather
    than zeros, matching the probe's ablation so the arm and the instrument measure on the same
    scale. The conditioning is untouched -- every estimate has already been taken."""

    recent_dropout_max: int = 3
    """Deepest recent-frame dropout, in frames. Uniform in ``1..this`` when it fires."""
    stream_state: str = ""
    """Causal trajectory-state pseudo-frames prepended to the input (Track-2 stream arm,
    runs/t2_backbone/PLAN.md). ``"meanvar"`` prepends TWO frames built from every frame the
    stream has revealed through the current input (``arr[: start + 20]``, so the first window's
    state is its own statistics and there is no cold-start case): ``[running mean u, running
    mean v, n_revealed/820]`` and ``[running temporal std u, running temporal std v, 0]``, both
    divided by the window's own freestream scale. The model's stem reads ``t_in + 2`` frames
    (``loop`` widens the in-shape and zero-pads a warm start via ``widen_stem_history``, so at
    step 0 the model is bit-identical to its ``--init-from``); the dataset still enumerates
    plain ``t_in + 20``-frame windows — the state costs no trajectory frames. Every
    operating-point / freestream estimate keeps reading the LAST 20 real frames."""

    stream_corr_lam: float = 0.0
    """Correction-in-the-loop (Track-2 stream arm, runs/t2_backbone/PLAN.md): take the training
    loss on ``pred + stopgrad(lam * bias_traj)`` where ``bias_traj`` is a per-trajectory EMA
    (``stream_corr_ema``) of the lead-averaged residual ``mean_t(target - pred)`` — a
    training-time image of the shipped Track-2 mean-flow correction in its frozen regime
    (serving: strength 0.75, freeze 16 → effective shrink 0.75 * 16/18 = 0.667). The backbone
    stops re-learning the first moment the serving correction owns. Real windows only; 0
    disables. Distinct from §144's mean/fluct re-weighting: the mean supervision is shifted by
    the correction's own running estimate, not down-weighted."""

    stream_corr_ema: float = 0.99
    """EMA decay of the per-trajectory residual-mean estimate ``stream_corr_lam`` applies."""

    recur_k: int = 0
    """Truncated BPTT over consecutive Track-2 stream windows (``train/recurrent.py``): each
    training item is a chain of ``recur_burn`` gradient-free windows then ``recur_k`` graded
    ones, 20 frames apart, run in order so a ``TokenMemory`` (``--arch ...,recur=<width>``,
    ``models/recurrent.py``) carries state across them. ``batch_size`` counts graded windows.
    Without a memory in the arch it is the chain-sampling control. 0 disables."""

    recur_burn: int = 8
    """Gradient-free windows run before each chain's graded part to build the memory's state
    (slots before the trajectory's first frame are masked). Ignored unless ``recur_k``."""

    train_compile: str = ""
    """``torch.compile`` mode for the chain step's forwards ("default" or "reduce-overhead");
    empty is eager. The chain step is launch-bound (12 sequential forwards at batch 4) and
    reduce-overhead measured 3.5x on the T4 (FINDINGS 202). Only with ``recur_k`` for now."""

    nbr_prior: bool = False
    """Append the climatology of the nearest OTHER training operating points (time-mean and
    temporal std of u, v, plus their distance) as five input planes, looked up from the estimated
    operating point (``models.wrappers.NbrPriorWrapper``, FINDINGS 224). The table is the training
    split's cells (``nbr_cells``, filled by the loop) and rides in the checkpoint."""

    nbr_prior_mode: str = "all"
    """``all``: the neighbours' time-mean and std of u, v plus distance (5 planes); ``std``: the std
    maps and distance only (3 planes) -- FINDINGS 224b: the mean planes cost mvpe."""

    nbr_cells: list = field(default_factory=list)
    """``[[Re, AoA], ...]`` behind ``nbr_prior``'s table; written by the loop, read to rebuild."""

    state_planes: bool = False
    """Feed the state as extra CHANNELS on every frame rather than as two extra frames, so it
    reaches the full-resolution refinement at native resolution instead of only the stem
    (``models.wrappers.StateRefineWrapper``). Needs ``stream_state``."""

    state_anchor: bool = False
    """Blend the residual anchor per lead between the last frame and the trajectory mean
    (``models.wrappers.StateAnchorWrapper``). The per-lead weight is LEARNED and initialised at
    1.0, so step 0 is today's model and any departure is earned. Needs ``stream_state``."""

    state_norm: bool = False
    """Standardise the window by the ``stream_state`` statistics instead of feeding them.

    ``models.wrappers.StateNormWrapper``: ``x' = (x - m)/clamp(s, floor)`` in, prediction
    restored by the same affine out. Needs ``stream_state``; the backbone then sees a plain
    20-frame window (no widened stem) and the whole serving path downstream is unchanged."""

    state_norm_mode: str = "meanstd"
    """``meanstd`` standardises by mean and std; ``mean`` removes only the first moment.
    Measured 2026-09-20: the full standardisation costs 0.63-0.66 `tke` and is refuted; the
    damage is entirely the scale half, so ``mean`` is the variant worth keeping."""

    state_norm_floor: float = 0.05
    """Floor on the state std before dividing, in freestream-normalised units. Guards the PIV
    dead region, where ``s`` is genuinely tiny and an unfloored division amplifies noise."""

    state_lr_mult: float = 1.0
    """Gradient multiplier on the stem columns that read the ``stream_state`` pseudo-frames.

    Those columns are zero-initialised by ``loop.widen_stem_history`` while the rest of the
    network arrives pretrained (8k sim + 4k joint), so they are the undertrained part of the
    model at any fine-tune length -- which is the best available explanation for why 24k steps
    beat 12k on a state backbone when FINDINGS 119e measured longer schedules as a LOSS without
    it. This scales only their gradient, leaving every other parameter's step untouched, so it
    buys the state more learning without lengthening the fine-tune or disturbing the weights
    that are already converged. 1.0 is a no-op and the default."""

    history_dropout: float = 0.0
    """Probability that a training window's EARLIER frames are zeroed (``t_in > 20`` only).

    Track 2's stream has no history on the first window of a trajectory, so a long-context
    backbone is served zeros there that a fine-tune on always-complete windows never showed
    it. This presents that case during training. The conditioning is unaffected: every
    estimate reads the most recent 20 frames."""

    state_dropout: float = 0.0
    """Cold-state dropout (Track-2 stream arm). Probability that a training window's
    ``stream_state`` pseudo-frames are rebuilt from only its FIRST ``k`` reveals instead of the
    full revealed history (``k`` uniform in ``1..state_dropout_max``, clamped to the window's
    true reveal count). The stream's state is coldest and noisiest at a trajectory's start --
    estimated from ~20 frames against ~800 later -- yet the always-late training windows only
    ever show the model a well-averaged state. This presents the genuine early-stream state,
    paired with the small ``n_revealed/820`` count plane that indexes it, so the backbone learns
    to lean on the raw window when the count is low rather than overtrust a cold state. Needs
    ``stream_state``; 0 disables. The mean-flow analogue of ``history_dropout`` for the state
    arm (which the state arm cannot use: ``stream_state`` fixes ``t_in == 20``)."""

    state_dropout_max: int = 4
    """Deepest cold state, in 20-frame reveals: ``k`` is drawn uniform in ``1..this`` when
    ``state_dropout`` fires, clamped to the window's true reveal count. 4 => at most 80 frames
    of history stand in for the full trajectory."""

    dt_aug: str = ""
    """All-to-all temporal sampling: comma-separated strides *within* a window, e.g.
    ``"1,1,2,3"`` (repeats weight a value, so that one is 50% scored-clock).

    Not to be confused with ``stride``, which spaces window *starts* and only changes how
    many windows exist. Here a stride-``s`` item observes frames ``(t, t+s, ... t+19s)``
    and is supervised on ``(t+20s, ... t+39s)``: still 20->20, still one clock, so the
    model can read ``s`` off the input's own decorrelation rate and the task stays
    well-posed without a conditioning slot. Nothing reaches inference, and the validation
    loaders stay at ``s=1`` because that is the scored task.

    Why it should pay: ``dt* = dt.U/c`` scales with Re at fixed ``dt``, so an unseen high
    Reynolds number is partly an unseen forecast *horizon* (ROADMAP family D, the
    all-to-all row). Strided windows of a training cell supply that horizon without new
    data. Empty disables it, and the code path is then bit-identical to before.

    FINDINGS 28 measured it without a conditioning slot: the pretrain half was a null and
    the fine-tune half cost ``tke``, in proportion to the share of windows off the scored
    clock. With ``cond_mode="operating_point_dt"`` the stride is also written into the
    conditioning vector (``features.dt_cond_value``), which is the 2026-09-14 design."""

    dt_ratio_aug: str = ""
    """Time-step augmentation of SIMULATED windows: ``"lo,hi"``, the range of the window's
    frame interval as a multiple of the real frame interval at its Reynolds number.

    Each window of a native ``sim`` cell is resampled on the fly at ``ratio * step(Re)``
    source frames per frame, ``step(Re) = features.real_step_sim_frames`` (0.77 at Re 3750,
    5.5 at 26700 -- the units cache's step), by the same linear blend of neighbouring frames
    the cache uses. ``ratio`` is drawn log-uniformly in ``[lo, hi]`` with two guards: the
    effective step never drops below one source frame (no frames are invented between
    frames; cells whose real step is already below 1 only ever go faster), never exceeds
    ``dataset.DT_RATIO_MAX_STEP`` (8 frames, above real's fastest clock), and never reaches
    past the trajectory or a reserved tail. A ratio-1 draw is the units cache's own
    series. ``dt_ratio_p1`` of the draws are pinned at ratio 1. Requires ``train_kind="sim"``
    (the native cache) and is ignored for real windows, whose interval is fixed by the PIV
    rate: use ``dt_aug`` (whole steps) for those. Pair with ``cond_mode="operating_point_dt"``
    so the model is told the interval; at inference the slot is 0 on every window.

    Why: after freestream normalisation the main visible sign of Re in a window is how far
    structures move per frame, so real data locks speed to Re. This breaks the lock during
    training and lets the model learn speed and structure as separate things (Bruno,
    2026-09-14). Empty disables it and the code path is bit-identical to before."""

    dt_ratio_p1: float = 0.5
    """Share of ``dt_ratio_aug`` draws pinned at ratio 1 (the real clock), so the deployment
    distribution stays dominant. FINDINGS 28e found the fine-tune's cost monotone in how much
    of the budget leaves the scored clock."""

    val_splits: tuple[str, ...] = ("val_time", "val_interp", "val_extrap")

    # --- optimisation --------------------------------------------------------
    steps: int = 5000
    batch_size: int = 16
    """Micro-batch actually put on the GPU."""

    grad_accum: int = 1
    """Gradient accumulation steps. Effective batch is ``batch_size * grad_accum``.

    Needed for Transolver: it keeps one token per space-time grid point (40,960 of
    them at 256 hidden) and cannot use fp16 (Physics-Attention overflows), so fp32
    at batch 8 exhausts a 15 GB T4. Accumulating lets the effective batch stay
    comparable to the other baselines, which is what makes them comparable at all."""
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 200
    grad_clip: float = 1.0
    amp: bool = True
    """fp16 autocast. The eval GPU is A800 (sm_80) and ours is T4 (sm_75); fp16
    works on both, whereas bf16 would silently be untestable locally."""

    num_workers: int = 4
    seed: int = 0

    deterministic: bool = True
    """Pin the CUDA kernels so a seed reproduces a run bit for bit.

    On by default because without it a seed does not reproduce a run: two ``stformer``
    runs at one seed differed on 130 of 144 tensors, and one *fixed* checkpoint scored
    three different numbers on three evaluations. See ``seeds.seed_everything`` for the
    measurements and for why ``cudnn.deterministic`` alone is not enough.

    Costs ~3-5% training wall clock and nothing that is scored -- the platform-timed
    ``make_predict`` path is untouched. Recorded here so ``config.json`` says which
    arithmetic a checkpoint was produced under."""

    # --- auxiliary objectives ------------------------------------------------
    #
    # All of these are training-time only: the heads are skipped in eval mode and
    # ``models.aux.strip_aux`` reduces a checkpoint to the plain backbone, so an arm that
    # uses them is scored on the same architecture, at the same inference cost, as one that
    # does not. That is what makes the comparison a comparison of objectives.

    piv_noise: float = 0.0
    """Synthetic PIV noise on the **input**, as a multiple of the measured level.

    1.0 is the noise the delivered real data carries: 2.4% of freestream at Re=3750 falling
    to 1.0% at Re=26700 (``data/piv_noise.py``, from FINDINGS 3). With
    ``piv_noise_target=0`` this is a *denoising* objective -- corrupt input, clean target --
    which asks the model to invert the measurement rather than merely tolerate it."""

    piv_noise_target: float = 0.0
    """The same, on the target. ``piv_noise=piv_noise_target`` is the test-time task; a
    smaller value here than on the input is measurement super-resolution."""

    aux_mask: str = ""
    """Input mask geometry for masked spatio-temporal modelling: ``tube`` | ``frame`` |
    ``cube`` (``data/masking.py``), empty to disable.

    Adds one visibility channel to the input, because zero-fill alone is ambiguous against
    the PIV measurement hole, which is also zero."""

    aux_mask_ratio: float = 0.5
    """Fraction of patches (or frames, for ``frame``) hidden. Ignored when ``aux_mask``
    is empty."""

    aux_mask_plane: bool = True
    """Append a visibility plane to masked inputs and retain it at inference.

    Disable only with two-view training, where the production forward is clean and masking is an
    auxiliary corruption. This keeps the shipped backbone at its original input width."""

    w_recon: float = 0.0
    """Weight on reconstructing the clean input where ``aux_mask`` hid it. Gated on
    ``aux_mask``: reconstructing a visible input is the identity map."""

    w_recon_tke: float = 0.0
    """Weight on recovering hidden input tubes' per-pixel temporal velocity energy.

    This uses the reconstruction decoder only as a training-time parameterisation. It does not
    score pixel phase: the decoded window is reduced to variance over time before comparison, so
    the pretext explicitly preserves the second moment that pixel MAE damaged."""

    recon_tke_scope: str = "hidden"
    """Frames used to form the amplitude target: ``hidden`` or the full observed window.

    Full-window targets are less noisy for causal tail masking while the auxiliary view still has
    to infer them from the visible prefix."""

    recon_tke_target: str = "amplitude"
    """Direct decoder target: RMS ``amplitude`` or temporal ``energy``.

    Both retain a live gradient at zero because the decoder predicts the statistic directly;
    energy is more tightly aligned with the scored TKE map."""

    recon_tke_spatial: str = "map"
    """Supervise the dense per-pixel moment ``map`` or only its per-window ``global`` mean.

    The global form isolates amplitude calibration from spatial TKE-shape supervision."""

    w_recon_acf: float = 0.0
    """Weight on predicting hidden frames' per-pixel lag-1 velocity autocorrelation.

    This is the temporal off-diagonal second moment: it teaches persistence/timescale rather than
    amplitude and uses the second decoder channel only during training."""

    recon_clean_field: bool = False
    """Use two views for masked auxiliary objectives during task training.

    The forecasting loss sees the clean window; a second forward pass sees ``aux_mask`` and
    supplies only the reconstruction/energy loss. This isolates SSL from the robustness penalty of
    asking the task itself to forecast from missing observations. Training-time only."""

    recon_stop_frac: float = 1.0
    """Stop the masked auxiliary forward at this fraction of training.

    The remaining task-only steps consolidate the production forecast without changing the total
    fine-tune budget. One keeps the original concurrent behaviour."""

    w_fine: float = 0.0
    """Weight on decoding the native ``64x128`` field over the predicted frames.

    Input stays ``32x64`` -- no fine input exists at submission time. Ill-posed by
    construction, so read the ``fine_vs_upsample`` diagnostic before believing it: a head
    that merely matches bilinear upsampling has taught the trunk nothing."""

    w_subgrid: float = 0.0
    """Weight on per-coarse-cell sub-grid variance of the native field. The well-posed
    version of ``w_fine`` on the same data, at an eighth of the output width, aimed at the
    second moment."""

    w_paux: float = 0.0
    """Weight on predicting pressure over the output frames from a p-free input.

    Needs simulated data (real PIV has no pressure) and leaves ``target`` p-free, so
    ``rel_l2`` still measures exactly what it measured before."""

    w_opreg: float = 0.0
    """Weight on regressing the operating point ``(log-Re, AoA)`` off the pooled trunk
    embedding (``models.aux.OpHead``, the ``opreg`` head).

    Complementary to ``align_weight``, not a duplicate of it: that term matches the
    sim/real *marginal* statistics of the embedding (subtractive -- erase what tells the
    domains apart), while this one supervises the embedding to keep *decoding* the
    physical operating point from either domain through one shared head (additive --
    reward what the domains share). Meaningful only where a domain-mixed batch exists, so
    a joint stage (``joint_sim_frac > 0``) is the natural place to set it, though nothing
    here enforces that -- a real-only run just regresses a real-only target."""

    ema_decay: float = 0.0
    """Keep an exponential moving average of the weights, saved as ``ema.pt``.

    Both checkpoints are written, so one run yields the EMA and non-EMA arms of the
    comparison rather than two -- and they are exactly paired, sharing every batch and
    every random draw. 0 disables it and leaves ``last.pt`` the only artefact.

    Why it belongs in a fine-tune: 4000 steps over 69 real trajectories is ~25 epochs of
    *independent* data (the 51,732 windows overlap 20-fold at stride 1), so the last
    iterate sits in the noise of a small-sample objective. Averaging costs nothing at
    inference -- it is one set of weights, not an ensemble."""

    # --- loss weights --------------------------------------------------------
    w_field: float = 1.0
    kmap_channel: bool = False
    """Emit a fourth output channel carrying the temporal-variance map (``models/kmap.py``).

    The frames stay channels 0..2; channel 3 is a per-pixel log ratio of the target's variance
    map over the input window's, read as its time mean. Widening a 3-channel ``--init-from``
    checkpoint zero-fills the new slots, so the run starts at the source's frames plus the input
    map as its estimate. Graded by ``w_kmap``; stripped by ``make_predict`` before scoring."""
    kmap_base: str = "input"
    """What the map channel's log ratio is anchored to: ``input`` (the input window's own
    variance map -- stationary, scale-free, tke_score ~74.9 on its own) or ``frames`` (the
    predicted frames' variance map, detached -- the shipped head's form, a far better base once
    the frames are trained but noise at step 0 of a pretrain and a moving target throughout)."""
    w_kmap: float = 0.0
    """Weight on ``losses.kmap_loss``. Raw magnitude at step 0 is ~0.76 (the input map's own
    relative error against the target's), so a dose matched to the shipped ``tke`` terms'
    contribution (55a: 0.00239) is ~0.003."""
    w_tke: float = 0.0
    w_tke_tprofile: float = 0.0
    """Weight on the per-lead temporal fluctuation-energy PROFILE term
    (``losses.tke_temporal_profile_loss``). Shape-only over the lead axis, the temporal
    complement of ``w_tke_shape``; targets FINDINGS 143's lead-16 dump without the 144
    amplitude give-back. 0 disables."""
    w_tke_shape: float = 0.0
    """Weight on the scale-invariant TKE term (``losses.tke_shape_loss``).

    Grades *where* the fluctuations are and is blind to how many. **Measured to fail** -- see
    FINDINGS 27: it makes ``tke_score`` worse at both weights tried, monotonically in the weight,
    because with amplitude left free the model satisfies it by rearranging an already-deficient
    map. Kept because the negative result is the load-bearing part of 27's mechanism argument, and
    a term that is documented as harmful is more useful than one nobody can find."""

    tke_huber_delta: float = 1.0

    tke_huber_normalize: bool = True
    """False drops ``tke_huber``'s per-window RMS normalisation (``losses.tke_huber_loss``
    ``normalize=False``): raw map residual, ``tke_huber_delta`` read as an absolute crossover in
    freestream units. True reproduces every recorded ``w_tke_huber`` arm."""

    w_tke_amp: float = 0.0
    """Weight on the fluctuation energy's TOTAL, graded alone (``losses.tke_amplitude_loss``).

    The third second-moment term, added 2026-09-22. ``w_tke_huber`` grades the raw variance map,
    where placement is 91-99.5% of the squared error, and ``w_tke_shape`` removes amplitude by
    construction -- so before this the recipe pressed placement twice and the total almost not at
    all. FINDINGS 183b measures the consequence: at Re 26700 the model emits 60-70% of the target's
    energy while at Re 3750 it is at the right scale. 0 is off and reproduces the previous loss
    exactly."""

    tke_amp_delta: float = 0.5
    """Tail cap of ``w_tke_amp``'s smooth-L1, in LOG units: 0.5 is a factor 1.65, outside every
    per-cell deficit measured (1.25-1.67 on the total), so it guards degenerate windows without
    reshaping the body of the distribution."""

    w_errhomog: float = 0.0
    """Weight on the local-error-homogeneity term (``losses.error_homogeneity_loss``, 2026-09-24).

    Trains the backbone to emit a field error whose magnitude is locally homogeneous, so it is
    predictable from its own neighbourhood -- the property SPS interval efficiency is bounded by
    (FINDINGS 207: the fitted stack is at the local-level oracle 0.739, the per-element oracle is
    0.879, and no post-hoc trick closes the gap because it is the backbone's error shape). Penalises
    the mean local std of ``|pred-target|`` over the GLOBAL mean error, on scored velocity channels.
    0 is off. Composes with ``w_field``; only shapes the residual it leaves. See ``errhomog_k``."""

    errhomog_k: int = 3
    """Spatial box size (odd) of ``w_errhomog``'s neighbourhood, matching the interval head's local
    reduction. 3 is the head's own scale; larger boxes ask for homogeneity over a wider region than
    the sizer resolves."""

    """smooth-L1 crossover for ``w_tke_huber``, in units of the sample's own k-map RMS.

    A **different quantity** from ``huber_delta``, which is 0.005 because it grades raw
    velocity errors (mean 0.0059). This grades ``(k_pred - k_true) / rms_pixels(k_true)``,
    which is dimensionless and runs mean 0.18 / median 0.04 on the shipped backbone, so its
    natural scale is O(1) and 1.0 is the historical default rather than a tuned value.

    It was hardcoded at the function default until 2026-09-04: ``loop.py``'s single call site
    passed no ``delta``, so the knob existed in ``losses.tke_huber_loss`` and was unreachable
    from a config or the CLI -- FINDINGS 58's failure mode, where an unreachable flag turned
    out to matter once plumbed. Default kept at 1.0, so every recorded result is unchanged.

    Not inert, and not obviously right either. Measured on ``runs/g/ftlr30_s0_ft/ema.pt``:
    at 1.0, **3.9%** of training pixels sit above the crossover and carry **38%** of the
    term's loss mass, so it does clip a real tail. But the tail starts far earlier -- at 0.25
    it is 17.7% of pixels carrying 78% of the mass -- so the informative sweep range is
    roughly 0.1 to 2.0 and nobody has run it. Sweeping it changes the model's own variance
    map, so it interacts with the second-moment head exactly as ``w_tke_huber`` does: refit
    the head per arm."""

    w_tke_huber: float = 0.0

    tke_lead_dropout: float = 0.0
    """Training-only fraction of output leads randomly omitted from TKE-family maps."""
    """Weight on the robust (smooth-L1) TKE term (``losses.tke_huber_loss``).

    An **alternative** to ``w_tke``, not an addition -- both grade the same residual. Equivalent to
    ``w_tke`` on ``val_time``/``val_interp`` and better on ``val_extrap``. The original reason given
    -- a 57x fluctuation-energy range between Re 3750 and 26700 letting high-Re cells monopolise a
    plain L2 -- is physical-unit; in the loop's freestream units that range is ~1.8x
    (FINDINGS 196j), so the robust form's edge is within-window tails, not a Reynolds-scale
    imbalance.

    **Use 0.02.** FINDINGS 27d recommended 0.04 from a U-Net; the 3-point dose curve on STFormer
    (36i) puts the optimum at **0.02**, worth +2.32 tke, +0.65 SPS ceiling and +0.205 final on
    ``val_extrap`` -- against 27e's +0.118 projection. 0.04 buys more tke and loses final on every
    split, and 0.002 (the 0.1x "auxiliary band" of 22/24) costs three quarters of the gain.

    **The dose is a property of the term, not of the band.** Being a *bounded* penalty is why this
    one tolerates ~1x the field loss where the L2-form ``w_enstrophy`` knees at 0.1x: residuals past
    ``delta`` contribute linearly, so no handful of variance cells runs away with the gradient. Any
    new term needs its own curve -- 36j records the band heuristic misleading in both directions."""

    tke_gate: str = "none"
    """Per-pixel predictability gate on the dense second-moment terms
    (``losses.unpredictability_gate``); STATUS item 6's mechanism, from FINDINGS 96-98.

    ``none`` (default) reproduces the shipped terms exactly. ``huber`` gates ``w_tke_huber``
    only; ``both`` gates ``w_tke_shape`` too (``w_tke`` is never gated -- it is not part of the
    shipped recipe and one mechanism per arm). The gate is ``var_t(pred - target) / var_t(target)``
    clamped to 1, detached, renormalised to mean 1 over valid pixels, so the term's dose is
    unchanged and only its spatial placement moves: full variance pressure where the window's
    fluctuation is unpredicted (there the ``rel_l2`` residual is already at its ceiling, so added
    variance is nearly free -- FINDINGS 126's pricing mechanism), none where the phase is tracked
    (there amplitude pressure is what buys ``tke`` at ``rel_l2``'s expense -- the give-back
    FINDINGS 98 and 141 measured).

    **``huber`` and ``both`` are measured negative on every held-out column** (FINDINGS 162,
    n=3: -0.15 to -0.33 ceiling, tke -0.44 to -0.78 against rel_l2 +0.02): the terms' purchase
    lives at the phase-tracked pixels, not the unpredicted ones. ``pred`` is that decomposition's
    follow-up -- the INVERTED gate (weight = predictability, ``huber`` term only), pressing where
    tracking makes amplitude effective and releasing the incoherent background."""

    kmap_e2e: bool = False
    """Spend the variance-map channel's estimate on the frames INSIDE the step.

    Needs ``kmap_channel``. The mean-preserving gain toward the channel's map
    (``losses.kmap_e2e_log_ratio``, anchored per ``kmap_base``; ``frames`` = identity at a
    zero-init channel) is applied before the loss, so the whole stage objective grades the
    corrected output — FINDINGS 131's channel with its spend trained jointly instead of post
    hoc, which is the half every §131 arm lost on. ``w_kmap`` still grades the channel against
    the true map directly (the anchor that stops frames and channel agreeing on a wrong map).
    Exclusive with ``tke_head_e2e`` — two trained gain sources in one step would attribute
    nothing to either. Serving/pricing instrument: ``scripts/eval/kmap_report.py`` (the gain
    rows are exactly this correction applied post hoc to the trained pair)."""

    tke_head_e2e: bool = False
    """Train the w8/d2 second-moment head END TO END with the backbone in this stage.

    The head has only ever been fitted sequentially on cached maps from a frozen backbone
    (``train tke-head``); this is the backbone-in-the-loop version its docstring priced and
    nobody ran. A zero-initialised ``TkeHead`` (identity gain at step 0) reads the same seven
    feature planes off the live prediction (``tke_head.training_log_ratio``), its mean-preserving
    gain is applied to the frames, and the WHOLE stage objective — field loss and the dense
    ``tke`` terms alike — is evaluated on the corrected output, so gradients reach both modules
    and the head prices its own ``rel_l2`` (no shrink; serve with ``--tke-shrink 1.0
    --tke-blend 0``). The head rides the same EMA as the backbone and is saved beside the
    checkpoints as ``tke_head_e2e.pt`` in the ``train tke-head`` payload layout, so
    ``train score --tke-head`` and the packaging path serve it unchanged.

    What it can and cannot settle: FINDINGS 131.4 predicts joint training lands on the same
    rel_l2-for-tke trade the sequential fit sits on (buying variance costs the same either
    route); the untested part is co-adaptation — a backbone free to stay nearer the conditional
    mean because the head carries the amplitude. Incompatible with ``bounds`` (the correction
    would sit between the frames and the widths they calibrate)."""

    w_spectral: float = 0.0

    lead_decay: float = 0.0
    """Tilt the accuracy term toward short leads: weights ``exp(-lead_decay * t/T)``, mean 1.

    0 is uniform and reproduces every prior run. Motivated by §42's per-lead measurement -- the
    model's fluctuation correlation falls 0.846 to 0.319 over the horizon, so the two scored axes
    want opposite things and the strength of the conflict depends on the lead. `NEW_ITEMS TR` lists
    uniform frame weighting as settled; §42 is the evidence that reopens it.

    Applies to ``w_field`` only. The relative and ceiling forms normalise per sample over the whole
    window, so a per-lead weight inside them would no longer match the metric they mirror."""

    w_absorb: float = 0.0
    """Weight on the absorb term (``train.absorb``): pull the prediction toward a FROZEN
    second-moment head's proposal for it, stop-gradient, graded with the field term's own
    elementwise error. Needs ``absorb_head``. Zero preserves the existing loss exactly.

    MEASURED TO RUN AWAY (FINDINGS 196c): the head's proposal does not fall as the student adds
    energy. Use ``distill_head`` (fixed target on a frozen teacher) instead."""

    absorb_head: str = ""
    """Path to the frozen head ``scripts/eval/fit_absorb_head.py`` saved."""

    absorb_shrink: float = 0.5
    """Exponent on the head's log-gain in the proposal (0.5 = the shipped serving shrink)."""

    tke_loop_head: str = ""
    """A FROZEN second-moment head (``scripts/eval/fit_absorb_head.py``) put in front of the dense
    ``tke`` terms only: ``w_tke``/``w_tke_shape``/``w_tke_huber`` grade ``m + g (p - m)`` with
    ``g = exp(tke_loop_head_shrink * r / 2)`` read off the detached prediction, while the field
    term keeps grading the bare ``p``. The backbone is charged for energy only net of what the
    head will add at serving, and the interval (sized on bare ``p``) never sees it. Differs from
    ``tke_head_e2e`` (FINDINGS 163: head trained, whole objective on the corrected output) and
    ``w_absorb`` (196c: pull toward the head's proposal). Empty preserves the loss exactly.
    FINDINGS 196k."""

    tke_loop_head_shrink: float = 0.5
    """Exponent on ``tke_loop_head``'s log-gain (0.5 = the shipped serving shrink)."""

    tke_loop_head_field: bool = False
    """With ``tke_loop_head``, the field term grades the head-corrected prediction too, so the
    whole accuracy objective sees ``m + g (p - m)`` as the scorer's accuracy axes do at serving.
    The band credit keeps grading bare ``p`` (the serving interval is centred on it). Differs
    from ``tke_head_e2e`` (FINDINGS 163) in that the head is frozen. False preserves 196k's
    ``loophead`` exactly."""

    w_ceiling: float = 0.0
    """Weight on the differentiable SPS ceiling (``losses.ceiling_loss``) -- the scored composite.

    Set ``--w-field 0 --w-ceiling 1`` for a pure swap: the objective becomes the metric, with the
    three accuracy axes at the scorer's own .5/.3/.2 and each axis's marginal pressure set by
    where it currently sits rather than by a hand-tuned dose. FINDINGS 36i/37c are the argument
    for wanting that: the dose heuristic crippled a working term tenfold, and every valuation
    built on it is an upper bound.

    Note the term is O(0.1) where an absolute field loss converges to O(1e-3), so mixing the two
    at ``w_field 1`` makes this the objective, not an auxiliary. Mix deliberately or not at all."""

    ceiling_huber: float = 0.0
    """Tail-cap on ``w_ceiling``'s rel_l2 branch (``losses.ceiling_loss``'s ``huber_delta``).

    Separate from ``huber_delta``, which applies to ``w_field`` only, and defaulting to 0 so every
    recorded ``--w-ceiling`` arm reproduces from this code unchanged. It exists to remove FINDINGS
    55a's confound: a pure swap sets ``--w-field 0``, which discarded ``huber_delta 0.005`` -- the
    recipe's largest single lever (FINDINGS 28, +0.49 SPS) -- so every composite-objective arm ever
    run measured the composite *without* the tail-cap against a control *with* it. Set it to the
    lineage's own ``huber_delta`` to make the swap a one-factor comparison."""

    w_relfield: float = 0.0
    """Weight on the per-sample RELATIVE field loss (``losses.relative_field_loss``).

    `NEW_ITEMS TR.1`, open since the spec: the metric is a per-sample relative L2 and the
    objective has always been an absolute MSE, which weights each window by its own energy and
    therefore underfits the low-energy rows. ``val_extrap`` holds Re=3750, the lowest-energy row
    on the ladder, so the mismatch is biased against the split that matters most.

    An **alternative** to ``w_field``, not an addition -- both grade the same residual."""

    w_meanfield: float = 0.0
    """Weight on the relative L2 of the window's 20-frame TIME MEAN
    (``losses.mean_field_loss``).

    Every other accuracy term grades the instantaneous field, so the first moment is fitted
    only as a by-product -- and decomposing the shipped model's error about its own time mean
    says that is where the largest structured error on the grid lives: the mean field's relative
    error runs 0.030 at AoA=0 to 0.069 at AoA=20 while the fluctuation's is flat, and it is a U
    across Reynolds (0.051 / 0.036 / 0.057) so both extrapolated rows pay. Unlike the
    fluctuation, a 20-frame mean has no predictability barrier.

    A **redistribution** of pressure, not an addition -- the time mean is a linear functional
    of the frames -- so dose it at a matched contribution (FINDINGS 55a) and watch ``tke``,
    which is what gives the pressure up."""

    w_vorticity: float = 0.0
    """Weight on the instantaneous-vorticity term (``losses.vorticity_loss``).

    A high-pass reweighting of the field residual rather than a new quantity: the curl is
    linear, so this term's minimiser is still the conditional mean and it does not buy into
    27c's rel_l2/tke trade. It grades where the shear layers and shed vortices *are*.

    Calibrate the weight against the backbone it will run on -- this is a per-sample
    RELATIVE term (O(0.1-1)) against an absolute field term that converges to O(1e-3), the
    ratio is a property of the pair, and FINDINGS 24a is what happens when it is inherited.
    ``scripts/eval/vorticity_diagnostic.py`` measures it, and measures how much of the target
    vorticity is PIV sensor floor while it is there."""

    w_enstrophy: float = 0.0
    """Weight on the fluctuating-enstrophy term (``losses.enstrophy_loss``): ``tke_loss``'s
    exact form on ``var_t(omega)``.

    The one term in this family with 27c's *working* mechanism behind it -- amplitude
    matching forcing relocation -- asked for at the scale of a vortex instead of the wake.
    Same calibration rule as ``w_vorticity``, and read it against ``w_tke``/``w_tke_huber``
    rather than against zero: those already take this axis, and the question is whether a
    derivative-domain version of the same constraint adds anything they do not."""

    w_mvpe: float = 0.0
    """Weight on the mean-velocity-profile term (`ROADMAP N5`).

    ``mvpe`` is worth 0.188 final per point (FINDINGS 6c-ter) and **nothing has ever
    targeted it**: it is the one scored axis with no term of its own. It is also the
    narrowest -- 9 rows x 4 columns x 2 channels is 72 of the 4,096 values in the
    time-averaged field -- so a heavy weight optimises those and abandons the rest.
    Auxiliary scale only, and ``rel_l2`` is the axis to watch.

    Like ``w_tke``, this is a per-sample RELATIVE term: it plateaus at O(0.1-1) while the
    absolute field term converges to O(1e-3), so the weight must be set from a measured
    ratio on the backbone in question, not borrowed (FINDINGS 24a)."""

    w_rstress: float = 0.0
    """Weight on the Reynolds-stress **shape** term (``losses.rstress_shape_loss``).

    Grades the two degrees of freedom of the per-pixel second moment that no subscore sees:
    the anisotropy ``(R_uu - R_vv)/tr`` and the shear ``2 R_uv/tr``. ``tke_score`` grades the
    *trace* only, so this is orthogonal to ``w_tke`` by construction -- which is the point,
    since ``w_tke`` is measured and cut (FINDINGS 22h) and a full-tensor loss would re-run it.

    Motivated by ``scripts/eval/rstress_audit.py``: on ``val_extrap`` the shipped STFormer retains
    0.364 of ``<u'u'>`` against 0.619 of ``<v'v'>``, and ``<u'v'>`` comes in at 0.661 there
    against **1.698** on ``val_interp`` -- over-predicted by 70% on one split, under on
    another. Nothing in the objective constrains it.

    Relative and trace-normalised, so it plateaus at O(0.1-1) against the field term's
    O(1e-3): set the weight from a measured gradient ratio, never borrowed (FINDINGS 24a)."""

    w_espec: float = 0.0
    """Weight on the **spatial** energy-spectrum term (``losses.energy_spectrum_loss``).

    Distinct from ``w_spectral``, which bins *temporal* frequencies and is the term measured
    and cut as ``TR.2`` (FINDINGS 24). This bins the radial spatial wavenumber -- the
    classical ``E(k)`` -- and a field can match the temporal spectrum at every pixel while
    being smooth in space.

    ``scripts/eval/phys_audit.py`` sizes both sides of it: the prediction's radial spectrum runs
    0.85-0.90 of the target's in the energy-bearing bands and 0.16-0.24 in the top three, so
    the term has something real to move -- but the least-squares optimal per-band *gain* is
    0.99, because the missing energy is uncorrelated with the truth. So this necessarily
    trades ``rel_l2`` for a second-moment statistic rather than improving both."""

    helm_lambda: float = 1.0
    """Weight on the **divergent** part of the field error (``losses.helmholtz_field_loss``).

    1.0 is exactly MSE (Parseval, not an approximation), so the arm nests its own control and
    the flag is inert by default. Below 1 the objective trains preferentially on the
    solenoidal error -- "fit the incompressible part" -- and above 1 on the divergent part.

    In-plane incompressibility is **false** on a planar slice of a 3D flow (``rho_div`` 0.857
    on real PIV, FINDINGS 3), so a ``||div u||^2`` penalty is not available; a reweighting is.
    ``scripts/eval/phys_audit.py`` measures the longitudinal component at **7.4% of the target's
    energy and 40.8% of the squared error**, relative error 8.7x the solenoidal part's, at
    coherence 0.971 -- a large misallocation, but not noise, and the optimal post-hoc
    reweighting is 0.995 against 0.999. **The sign of any gain is therefore unmeasured and
    both directions have to be run.** Mutually exclusive with ``huber_delta``, which is not
    diagonal in the Fourier basis."""

    w_sps: float = 0.0
    w_credit: float = 0.0
    """Weight on the oracle-interval credit term (``losses.interval_credit_loss``).

    The only term that targets SPS's *interval* factor from the prediction side. Unlike
    ``w_sps`` -- which grades a head's interval and is inert under ``freeze_backbone`` -- this
    grades the field the interval will have to bracket, so it belongs in a fine-tune with the
    backbone unfrozen and no head present.

    Same measured-not-borrowed caution as ``w_tke``/``w_mvpe``: it is a mean of values in
    ``[0, 1]`` while the field term converges to O(1e-3), and its gradient is **150x** the field
    term's at ``credit_k=2``, so parity is ``0.0067``. **0.007 is the measured best and the gain
    saturates there** -- +0.05 / +0.35 / +0.37 SPS on ``val_extrap`` at 0.3x / 1x / 3x parity
    (FINDINGS 28). ``credit_k`` sets the exponent."""

    credit_k: float = 2.0
    """Width-per-error exponent inside ``w_credit``'s term. 2 is the literal oracle interval."""

    w_headcredit: float = 0.0
    """Weight on the band credit through a FROZEN interval head (``train.head_credit``).

    Unlike ``w_credit`` (the oracle band sized at the realised error), this grades the prediction
    against the band a real fitted head draws, with that width detached, so the only gradient is
    on the centre and it pulls errors just outside the band inside. FINDINGS 213/214."""

    headcredit_arm: str = ""
    """``sps_apply.py --arm-dir`` holding ``head.pt`` + ``width_rule.json`` for ``w_headcredit``."""

    headcredit_tau: float = 0.25
    """Sigmoid temperature of the soft "inside", as a fraction of the half-width."""

    headcredit_track_lr: float = 0.0
    """> 0: the ``w_headcredit`` head keeps learning during the fine-tune -- one step of its own
    fitting objective per training step on the current backbone's detached errors, at this
    constant learning rate (``FrozenHead.track_step``, FINDINGS 223). 0 keeps it frozen."""

    rollout_w: float = 0.0
    """> 0: rollout consistency (FINDINGS 224d). The training windows carry 2 x T_OUT target
    frames; the model's forecast of the first T_OUT is fed back as the next input window, and the
    forecast from it is scored against the following real frames with this weight, the gradient
    running through both calls. A forecast that damps the fluctuations continues badly, so the
    term presses on the conditional-mean energy deficit (FINDINGS 183) from the dynamics rather
    than from a statistic. Plain step only; validation still scores one window."""

    rollout_mode: str = "full"
    """``full``: the second window gets the whole loss; ``tke``: only its energy terms (field loss
    and band credit off) -- whether ``rollout_w``'s rel_l2 cost comes from scoring the fed-back
    forecast's pixels (FINDINGS 224f)."""

    rollout_aoa_min: float = 0.0
    """> 0: the rollout term only for windows whose LABEL angle is at least this many degrees
    (FINDINGS 224h: the energy-only rollout pays at 10-20 deg, where the wake sheds, and costs
    at 0-5 deg, where it is nearly steady). The gated-out windows get weight zero: the second
    call runs on the selected windows only and its loss is scaled by their share of the batch,
    so each selected window weighs what it weighs in the ungated term. 0 = every window."""

    rollout_re_min: float = 0.0
    """> 0: also require the window's LABEL Reynolds number to be at least this (the same
    zero-weight gate as ``rollout_aoa_min``; FINDINGS 224h's flat and angle-gated doses both cost
    tke at the unseen Re 3750, and this asks whether the low-Re training windows carry it)."""

    sam_rho: float = 0.0
    """> 0: sharpness-aware minimisation (Foret et al. 2021) with this neighbourhood radius. Each
    optimiser step takes the gradient at ``w + rho * g / ||g||`` instead of at ``w`` (one extra
    forward/backward on the same batch), steering the fine-tune toward flat minima -- a
    regulariser aimed at the held-out operating points, alongside drop-path. Plain step only:
    refused with the auxiliary forward paths (recon, recurrent chain, jepa, alignment taps,
    e2e heads, l2sp, the stream correction, the head-tracking step). FINDINGS 224."""

    rdrop_alpha: float = 0.0
    """> 0: R-drop consistency regularisation. Each step runs two forward passes with independent
    drop-path masks; the loss is the mean of the two task losses plus ``rdrop_alpha`` times the MSE
    between the two u,v predictions. Drives the stochastic sub-networks to agree (flatter minimum,
    smaller train/inference dropout-scaling gap). Plain forward path only. Untested-upstream
    lever."""

    headcredit_notch_lo: float = 1.0
    """Reynolds notch of ``w_headcredit`` (``head_credit.re_notch``): its multiplier at
    ``headcredit_notch_re``, rising linearly in ``|log(Re / centre)|`` to 1 at
    ``headcredit_notch_width``. Multiplies ``headcredit_lo``'s taper. 1.0 is off. FINDINGS 223:
    the band credit's tke cost sits on the Re 16500 row."""

    headcredit_notch_re: float = 16500.0
    headcredit_notch_width: float = 0.5

    headcredit_lo: float = 1.0
    """Reynolds taper of ``w_headcredit``: its multiplier at Re 3750, rising linearly in log Re to
    1 at Re 26700 (``data.dataset.re_dose``, on the cell's label Re). 1.0 is flat."""

    huber_delta: float = 0.0
    """> 0 switches the field term to smooth-L1, which the heavy-tailed per-element
    error distribution (mean 0.0059, std 0.0136) argues for.

    **The largest single lever FINDINGS 28 found**: +0.49 SPS on `val_extrap` at 0.02, positive on
    all four splits, and ~96% of it arrives through interval *efficiency* rather than the accuracy
    axes. Place it against the *non-dimensional* error the loss sees -- 0.02 is p68 of it on the
    STFormer real fine-tune -- and refit it per backbone: §15 measured the same flag as a wash on
    the U-Net, where its `tke` cost cancelled its efficiency gain."""

    huber_mix: float = 1.0
    """Blend between MSE and smooth-L1 when ``huber_delta > 0``: 1.0 is pure smooth-L1 (the
    behaviour before this existed), 0.0 is pure MSE.

    A mechanism probe, not a tuning knob. Every finite ``huber_delta`` **caps** the tail gradient
    at ±1, and ``delta -> inf`` / ``delta -> 0`` recover MSE / L1, so a delta sweep interpolates
    those two with the tail always capped. A blend keeps MSE's *unbounded* tail and adds the body
    boost, so it lies outside that span and separates "capping the tail" from "up-weighting the
    body" as the reason the flag pays."""

    mean_fluct_lambda: float = 0.0
    """Positive replaces the field term with robust temporal-mean error plus this
    weight times fluctuation MSE. Zero preserves the existing loss exactly.
    At huber_mix=1, 1/(2*huber_delta) matches small-error curvature (100 at .005).
    Exclusive with amplitude_free, lead_decay and helm_lambda != 1. TKE weights
    are unchanged. This is an experimental alternative, not a shipping default."""

    mean_fluct_robust: bool = False
    """With ``mean_fluct_lambda > 0``, grade the fluctuation half with the same smooth-L1 as the
    mean half instead of MSE (``losses.mean_fluct_field_loss(robust_fluct=True)``). The lambda
    is then a pure price on fluctuation error relative to the shipped robust field term, with
    the tail shape unchanged. False reproduces every recorded ``mean_fluct_lambda`` arm."""

    # --- transfer ------------------------------------------------------------
    init_from: str | None = None
    """Checkpoint to initialise from -- the sim-pretrained model when fine-tuning."""

    distill_from: str = ""
    """Comma-separated teacher checkpoints for ensemble distillation.

    The field target becomes a blend of the teachers' mean prediction and the ground
    truth (see ``distill_alpha``), so a single student inherits a diverse ensemble's
    accuracy at one model's inference cost -- the point being that a cross-architecture
    ensemble cannot be weight-souped. Combine with ``joint_sim_frac`` +
    ``joint_sim_split=sim_all`` to distil on sim inputs too, which is how the ensemble's
    extreme-Reynolds behaviour (absent from real train) reaches the student. Teachers may
    be heterogeneous architectures; each is rebuilt from its own config and frozen."""

    distill_alpha: float = 1.0
    """Weight on the teacher target in distillation: ``target = a*teacher + (1-a)*truth``.

    1.0 is pure distillation; ``<1`` keeps some ground-truth signal (and the exact PIV
    zeros, which the teacher only approximates). Ignored when ``distill_from`` is empty."""

    distill_part: str = ""
    """Reynolds split of the distillation target: which HALF of the teacher to take.

    ``""`` (default) distils the whole predicted field. ``"mean"`` takes the teacher's temporal
    mean and the *truth's* fluctuation about its own mean; ``"fluct"`` takes the truth's temporal
    mean and the teacher's fluctuation. Both recompose to a full window, so nothing downstream
    changes shape.

    Why this exists. The teacher's measured edge is split across two axes that a single field
    target fuses: ``rel_l2`` is a mean-seeking metric and ``tke`` is a variance one, and the
    campaign in ``scripts/train/trt3d_opt_distill.sh`` transferred them at very different rates --
    80%
    of the teacher's ``rel_l2`` headroom against 40% of its ``tke``. Splitting the target is the
    direct test of *which half of the teacher is worth distilling*, and it is the same
    decomposition ``distill_tke_truth`` applies to the loss rather than to the target."""

    distill_head: str = ""
    """A frozen second-moment head (``scripts/eval/fit_absorb_head.py``) applied to the TEACHERS'
    mean prediction before it becomes the distillation target, so the student learns the
    teacher's head-CORRECTED field and serves it without a head. The correction is computed on
    the frozen teacher, not on the student, so the target is fixed: the self-referential form
    (``w_absorb``) ran away because the head barely reads the energy level it is correcting
    (2026-09-22, ``runs/fluctprice0922/ab1_s0_RUNAWAY_killed_step12k``)."""

    distill_head_shrink: float = 0.5
    """Exponent on ``distill_head``'s log-gain (0.5 = the shipped serving shrink)."""

    distill_tke_truth: bool = False
    """Anchor the tke-family terms (``w_tke``/``w_tke_shape``/``w_tke_huber``) to the GROUND
    TRUTH during distillation, while the field term still learns the teacher's prediction.

    Why this exists: under distillation the field target is overwritten by the teacher, so a
    tke term would otherwise match the *teacher's* temporal-variance field -- and a deterministic
    teacher under-predicts variance (regression to the mean), so that is nearly redundant with the
    field loss and cannot raise tke toward reality. This routes the tke terms at the preserved
    truth instead: a Reynolds split of the objective -- the teacher supplies the mean/structure
    (field), the ground truth supplies the fluctuation energy (tke). No effect without a teacher,
    and default False so existing runs are unchanged."""

    backbone_lr_scale: float = 1.0
    """LR multiplier on the backbone during fine-tuning. The reference pipeline does
    plain full-model transfer; scoping the backbone lower is a cheap differentiator.

    **Was declared but never read until 2026-08-14** -- every run that set it trained at the
    unscaled rate and its `config.json` recorded a scaling that did not happen. Now applied as
    the backbone param group's `lr_scale`. Default 1.0, so no past run's behaviour is
    reinterpreted, only its (unset) field."""

    llrd: float = 1.0
    """Layer-wise learning-rate decay: a parameter at depth ``d`` of ``n`` trains at
    ``lr * llrd ** (n - 1 - d)``, so the stem moves slowest and the head at the full rate. Needs a
    backbone exposing ``lr_depth(name)`` and ``n_depths`` (``unet2d`` does); refused otherwise.
    Added 2026-09-18 for the `unet2d` "protect the pretrain" arms (FINDINGS 153): the CNN's held-out
    deficit is acquired in the real stages by a stronger fitter. 1.0 (default) is off."""

    freeze_patterns: str = ""
    """Comma-separated name substrings whose parameters are frozen for the whole run, e.g.
    ``"stem,encoder,down"`` freezes a U-Net's encoder in the fine-tune. Generic counterpart of
    the STFormer-shaped ``unfreeze`` ladder; refuses to match nothing. Empty (default) is off."""

    trunk_lr_scale: float = 1.0
    """LR multiplier on the **trunk** -- everything `unfreeze="io"` would freeze.

    The graded form of `unfreeze`, and the two name the same seam by construction: the split
    is `loop.unfreeze_patterns("io")`, so `trunk_lr_scale=0` and `unfreeze="io"` train exactly
    the same parameters and only differ in that the first still carries the trunk's optimiser
    state. Values in between are the continuum the ladder cannot express -- FINDINGS 31a
    measures the rungs as *not* monotone in trainable fraction, which is only interpretable
    against a lever that moves the same axis continuously.

    Distinct from `backbone_lr_scale`, which scopes everything except the 0.092M output head
    and is therefore a near-uniform rate cut rather than a differential one. Composes with it:
    the trunk ends at `backbone_lr_scale * trunk_lr_scale`.

    Default 1.0, and the param groups are only split when it is set, so no past run's
    behaviour is reinterpreted."""

    unfreeze: str = ""
    """Which parameters the fine-tune may move (`FT.2`'s ladder): `cond` | `io` | `blocks:N`
    | `all`. Empty and `all` are full fine-tuning, i.e. unchanged behaviour.

    The stages are cumulative and defined by `train.loop.UNFREEZE_LADDER`: `cond` is the
    conditioning MLP, the per-block modulation, the FiLM projections and every norm affine;
    `io` adds the conv stem, patch embedding, positional table and the decode/refinement
    heads; `blocks:N` adds the last N transformer blocks. Requires `init_from` -- freezing
    part of a randomly-initialised model trains an adapter over noise.

    Why it is worth a flag: FINDINGS 28g measures a full-parameter fine-tune erasing a
    6.47-point pretrain difference inside 100 steps, and how much of the trunk may move is
    the only lever that changes that rate instead of fighting it. The second reason is
    plainer -- 22.2M parameters against 69 real operating points."""

    joint_sim_frac_end: float = -1.0
    """Sim fraction to anneal *to* over training, linearly from ``joint_sim_frac``. Negative
    disables it, so the default is the constant mixture and no existing run changes.

    Why it exists: FINDINGS 32-bis measured joint sim+real training as a **free `tke` gain in
    distribution** (+1.37 `val_interp`, +0.96 `val_time`, rel_l2 and mvpe inside their bars)
    whose entire cost sits on `val_extrap` (rel_l2 -0.34, mvpe -0.81). 30b attributes that cost
    to sim's imported first-moment offset -- 6-7x its block floor, the largest gap in the release
    (6f) -- which nothing corrects at the two withheld Reynolds rows. Annealing bets that the
    second moment is learned early while the mean flow is set by whatever the batches end on, so
    a run that finishes on real-dominated batches keeps the gain and drops the cost.

    The schedule is a **staircase, not a ramp**: ``WeightedRandomSampler`` draws all of an
    epoch's indices at once, so the fraction can only change when the loader's iterator is
    rebuilt -- about every 3,200 steps at the standard mixture, i.e. ~9 updates over a 30k run.
    The realised value is logged at each update rather than assumed."""

    joint_sim_frac: float = 0.0
    """Fraction of each fine-tuning batch drawn from simulation instead of real (`FT.1`).

    Real-only fine-tuning overwrites the parametric structure the pretrain learned from
    100 simulated cells with a gradient that only ever sees 69 real ones; keeping a sim
    minority in every batch makes the sim half a regulariser anchoring that structure
    while real supplies the offset. 0 reproduces plain real-only transfer.

    The sim half is drawn from ``sim_train`` -- the same cells the pretrain used, so the
    held-out real Reynolds rows stay held out on both sides of the mixture."""

    synth_path: str = ""
    """Synthetic block set to mix into the training batches (``train gen-sample`` output).

    The generative arm of family F. A conditional flow model trained on real + rescaled sim
    (``train gen``) is sampled **off the release's grid** -- half-steps in angle, midpoints of the
    Reynolds ladder -- and what it emits is served as ordinary windows by
    :class:`~rpde_t1.data.synthetic.SyntheticBlockDataset`.

    It is aimed at the constraint ROADMAP 0 measures, and only at that one: **+25% more real
    windows moved `val_extrap` rel_l2 by +0.04**, so more data at cells we already have is worth
    nothing and the sampled operating points have to be ones no file in the release contains. The
    control that separates the two is ``--schedule ongrid``, which supplies the same volume at the
    training cells."""

    synth_frac: float = 0.0
    """Fraction of each batch drawn from ``synth_path`` instead of real.

    Weighted over a ``ConcatDataset`` exactly as ``joint_sim_frac`` is, so a step is one optimiser
    step at any fraction and a synthetic arm is comparable to its control at matched ``steps``."""

    gen_translate: bool = False
    """``train gen`` fits a sim->real **translator** rather than a generator.

    The generator draws a block from noise given only ``(log-Re, AoA)``, so its output is not a
    variant of any released sample. The translator starts the same rectified flow at a released
    *simulated* block instead, which is what the eligibility FAQ asks for -- "anything that takes
    a released sample and produces a variant of it", with every training sample tracing back to
    the release through a transform you can point at.

    The two share one interpolant and one integrator; see ``gen_source_sigma``, which is what
    selects between them numerically."""

    gen_source_sigma: float = 0.0
    """Noise mixed into the translator's source block, ``x_0 = sqrt(1-s^2)*y_sim + s*z``.

    The single knob of the translator, and it spans the whole design space: ``0.0`` is a
    deterministic map from the simulated block, ``1.0`` discards it and reproduces the pure
    generator bit for bit. Values between are what make the map one-to-many, so one simulated
    block yields as many distinct training samples as it is visited -- which matters because
    ``WindowDataset`` enumerates windows exhaustively and the only other randomness is the
    shuffled epoch order.

    It is a trade, not free diversity: as ``s`` rises the output depends less on the input, which
    weakens both the provenance argument above and the mechanism that distinguishes this from the
    generator FINDINGS 46 measured negative. Expect the useful setting to be small; sweep it and
    run the distribution gates at each value."""

    gen_keep_every: int = 0
    """``train gen``: also keep a step-tagged copy ``gen_step{step:06d}.pt`` (and
    ``gen_ema_step...`` when EMA is on) at every multiple of this many steps.

    ``gen.pt`` is overwritten at every eval with no best-of, so without this the steps axis
    of a sweep is lost the moment the run finishes. 0 keeps only ``gen.pt``."""

    gen_ema_decay: float = 0.0
    """``train gen``: exponential moving average of the velocity net's weights, written as
    ``gen_ema.pt`` beside ``gen.pt``. Diffusion and flow models are normally sampled from the
    averaged weights; the forecaster's own ``Ema`` helper is reused. 0 disables it."""

    gen_time_mlp: bool = False
    """``train gen``: give the interpolant time its own embedding MLP, added to the FiLM vector
    through ``UNet3d.forward``'s ``cond_extra`` hook, instead of sharing one projection input
    with the operating point. Recorded on the checkpoint; a net built one way does not load
    the other."""

    gen_source_context: bool = False
    """``train gen --translate``: feed the standardised simulated source block as two extra
    context planes per frame, so the flow is conditioned on the sim block explicitly rather
    than only starting its path there. ``sigma`` then buys diversity without the net having
    to infer the source from a noised start. Recorded on the checkpoint as ``context_channels``."""

    gen_latent_gamma: float = 0.0
    """``train gen``: amplitude of the two-sided interpolant's latent term,
    ``x_t = (1-t) x_0 + t x_1 + gamma(t) zeta`` with ``gamma(t) = g sin(pi t)``.

    0 keeps the one-sided interpolant every result on file was measured with. Above 0 the net
    grows a second output head (the denoiser, i.e. the score) and must be sampled with
    ``--gen-sde-eps`` and ``method="sde"``; the probability-flow ODE is still valid but is a
    deterministic map from its start, so it does not deliver the diversity the term exists for.

    The term separates two things ``gen_source_sigma`` was doing at once. Sigma is both the
    diversity budget and the amount by which the source block is discarded, which is why no
    checkpoint in the FINDINGS 110 sweep reached real-like energy and sim's variance-map
    correlation together: low sigma keeps the source and collapses to the conditional mean,
    high sigma restores the spread by throwing the source away. With a latent term the source
    can be kept whole (``sigma=0``) and the entropy comes from ``zeta`` instead. Expect the
    useful setting to be order the standardised field's own scale, i.e. near 1; sweep it and
    run the same distribution gates."""

    gen_denoiser_weight: float = 1.0
    """Weight on the two-sided interpolant's denoiser term, ``w * (eta - zeta)^2``.

    Only meaningful with ``gen_latent_gamma``. 1.0 weights the two heads equally, which is what
    the first fit used and what a report defended on the grounds that both targets are order one
    in the standardised space -- true, but beside the point: ``zeta`` is unit-variance noise and
    the denoiser's error has an irreducible floor near 1, so at equal weight a large share of the
    gradient goes to a term that cannot be driven down, while the velocity head -- the one that
    sets the output's fluctuation energy -- may be under-served. The measured symptom is a
    learnt marginal at 0.20 of real's energy, four times under the gate, with the sampler
    verified correct. Lowering this is the cheapest test of that explanation.

    It changes training only. The denoiser still trains (the SDE needs its score), and sampling
    is untouched, so unlike ``gen_latent_gamma`` this cannot desynchronise the fit and the
    sampler and is therefore an ordinary argument rather than something stored on the net."""

    gen_sde_eps: float = 0.0
    """Sampler noise level for the two-sided interpolant, ``eps(t) = gen_sde_eps * gamma(t)^2``.

    Used by the ``sde`` integrator only, and meaningless without ``gen_latent_gamma``. 0 is the
    probability-flow ODE. The schedule makes both the score term and the diffusion vanish at
    ``t=0`` and ``t=1``, so no clipping is needed at the ends; raising it trades sample
    diversity against how tightly the output tracks the source block, which is the trade the
    quality table is there to price."""

    gen_mask_dropout: bool = False
    """``train gen --translate``: treat the real target's per-frame rejected vectors (exact
    zeros in ``u`` and ``v``, ~7% of pixels per frame on top of the ~3% static mask) as
    missing rather than as flow: the block's time mean is taken over measured frames only and
    the flow-matching loss is masked there. The flow then learns the clean field; the dropout
    is put back at sampling time by ``rpde_t1.data.dropout_model``."""

    align_weight: float = 0.0
    """Weight on the sim/real latent-alignment term (``losses.domain_moment_loss``).

    Matches the per-channel mean and spread of the encoder's deepest features across the two
    domains, so the decoder reads one shared code instead of two. Needs
    ``joint_sim_frac > 0``: with a single-domain batch there is nothing to align.

    Expected to be small or negative, and it is run to find out. Sim and real differ
    *physically* (sim over-fluctuates 1.8x-20x in energy), the per-cell fluctuation rescaling
    has already closed 78-85% of the tke gap in data space, and NEW_ITEMS section IV argues
    an alignment strong enough to erase the rest erases the discrepancy fine-tuning exists to
    learn. Nobody has measured it here, which is the reason to."""

    coral_weight: float = 0.0
    """Weight on cross-channel sim/real correlation alignment at the deepest representation.

    Unlike ``align_weight``, this ignores marginal mean and scale and matches only relationships
    between channels. It is training-only and requires a mixed joint batch."""

    joint_sim_kind: str = ""
    """Override which simulated grid the joint half draws from (``sim`` | ``sim_resampled``).

    Empty inherits from the checkpoint being fine-tuned, which is the right default: an
    anchor to the pretraining distribution must *be* that distribution. Set it when the
    question is instead "which simulated data is the best auxiliary source", where the
    answer need not be what the backbone was pretrained on -- ``sim_resampled`` matches
    real's wake timescale (FINDINGS section 11) even though it was measured as a null for
    pretraining."""

    joint_sim_split: str = "sim_train"
    """Which *split* of the simulated grid the joint half draws from.

    ``sim_train`` is the safe default and the only one that keeps ``val_interp`` /
    ``val_extrap`` clean: the sim grid covers the Reynolds rows real holds out, so
    ``sim_all`` here shows the model **simulated** windows at the operating points those
    two splits exist to measure. That is not target leakage -- no real frame the scorer
    reads is in the batch -- but it does mean a model trained this way has seen the
    withheld ``(Re, AoA)`` cells in the other domain, so its Reynolds
    interpolation/extrapolation numbers are optimistic and no longer measure what the
    split names claim. Set it to ``sim_all`` when the auxiliary anchor is deliberately
    the whole simulated grid, and read the two OOD splits as in-sample when you do.

    **FINDINGS 66 bounds how optimistic that is, and it is less than the warning implies.**
    ``sim_train`` already contains an exact ``(Re, AoA)`` twin for all 5 ``val_interp`` cells and
    for the 26700 half of ``val_extrap``, so 8 of those 12 cells are sim-covered under EITHER
    setting. What ``sim_all`` uniquely adds is ``val_extrap``'s **Re 3750** end -- the one held-out
    row neither domain has otherwise seen. And the 30k ``sim_all`` pretrain every chain starts from
    covers all 12 regardless, so this knob governs how strongly, not whether. A campaign that holds
    the setting fixed across every arm therefore has valid DELTAS even on those splits; only the
    absolute levels are optimistic.

    ``sim_all_but_real_val`` is the third choice: the grid minus real's validation Reynolds
    rows (3750, 16500, 26700). Used with a pretrain on the same split it gives a model that has
    never seen ``val_interp`` / ``val_extrap``'s operating points in either domain, which is the
    control for "does simulated coverage of an operating point help the real prediction there".
    """

    l2sp: float = 0.0
    """Weight on ``||w - w_init||^2``, the L2-SP penalty toward the pretrained weights.

    The explicit form of what ``joint_sim_frac`` does implicitly: instead of re-showing
    the pretraining data, pull the parameters back toward the point that fit it. Cheaper
    (no second loader) and it constrains every weight rather than only those the sim
    windows exercise, but it cannot express *which* directions were load-bearing.
    Requires ``init_from``; 0 disables it."""

    snapshot_every: int = 0
    """Also write ``snap_<step>.pt`` every N steps, as soup ingredients.

    For averaging *along the trajectory* (SWA) rather than across runs. Motivated by a
    measurement on the naive baseline: ``val_extrap`` bottoms out at step 2000 and then
    degrades for the rest of the run while both in-distribution splits keep improving, so
    the last iterate is not the best model for the regime the private test set lives in.
    A two-point average of steps 3000 and 4000 interpolated between them rather than
    dominating -- expected, since two iterates 1000 steps apart at a decayed learning rate
    are nearly collinear. Averaging needs many points to do what it is known for.

    24 MB per snapshot, so ``snapshot_every=500`` over 4000 steps costs ~190 MB. 0 disables."""

    # --- bookkeeping ---------------------------------------------------------
    eval_every: int = 500
    log_every: int = 50
    score_points: int = 0
    """Number of evenly-spaced *official*-scorer checkpoints over the run, ``0`` to disable.

    Expressed as a count rather than a step interval because the natural unit is the
    fraction of the budget: ``20`` scores every ``steps / 20`` -- i.e. every 5% of training
    -- so the score trajectory has the same shape whatever ``steps`` is, and lengthening a
    run does not silently thin its scoring. The last iterate is always scored on top,
    whether or not it lands on one of the ``score_points``.

    The in-loop :func:`~rpde_t1.train.loop.quick_eval` metric is a non-dimensional
    diagnostic over a window subset, so it is not comparable to a leaderboard number (see
    that function's docstring). This instead runs the vendored ``scoring.py`` through the
    same submission path :mod:`~rpde_t1.train.evaluate` uses and logs its subscores
    (``rel_l2``, ``tke``, ``mvpe``, ``time``, ``sps``) so their evolution is visible during
    training, not only at the end; the trajectory is also written to ``scores.json``.

    Far more expensive than ``quick_eval`` -- a full forward over each eval set plus a
    scorer subprocess per split, and independent of ``eval_every`` -- which is why it is a
    handful of points, not every step. Needs the eval sets built (``make eval-set``); a
    missing set or a scorer error degrades to a note and disables in-loop scoring rather
    than ending the run, exactly as W&B logging does."""
    out_dir: str = "runs/default"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    # --- run logging (Weights & Biases) --------------------------------------
    wandb: bool = False
    """Mirror the loop's logging to W&B (see ``train.wandb_log``).

    Off by default: ``history.json`` remains the record of a run, so a run trained
    without this is not missing anything. Needs the extra: ``uv sync --extra wandb``."""

    wandb_project: str = "realpde-t1"
    wandb_entity: str = ""
    """W&B team. Empty uses the account default, which is what a single-user login has."""

    wandb_name: str = ""
    """Run name. Empty uses the output directory's basename, so the W&B run and the
    run directory are called the same thing -- that is the only join key between a
    checkpoint on disk and a curve on the dashboard."""

    wandb_group: str = ""
    """W&B group. ``train seeds`` sets it to the sweep prefix, which is what puts an
    arms x seeds sweep under one heading instead of scattering it over N runs."""

    wandb_tags: str = ""
    """Comma-separated W&B tags, e.g. ``"sim_pretrain,unet"``."""

    def save(self, path: Path | None = None) -> Path:
        path = path or (Path(self.out_dir) / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["val_splits"] = tuple(payload.get("val_splits", ()))
        return cls(**payload)

    @property
    def uses_bound_loss(self) -> bool:
        return self.bounds and self.w_sps > 0
