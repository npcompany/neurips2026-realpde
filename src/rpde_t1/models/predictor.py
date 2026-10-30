"""The one inference path: physical fields in, physical fields out.

Training, local validation and the submitted ``predict()`` all route through this,
so there is no way for the three to disagree about normalisation, conditioning or
bound handling -- historically the most common source of a model that validates
well and then scores badly.

The contract matches the competition's:

    (N, T_IN, H, W, 3) physical  ->  (N, T_OUT, H, W, 3) physical
                                     or dict with lower/upper

Inside, per batch: estimate the freestream speed from the input window, divide,
run the model, multiply back, zero the unscored pressure channel.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .. import T_IN, C, H, W
from ..features import (
    AOA_MAX,
    FREESTREAM_ROWS,
    batch_freestream_speed,
    conditioning_features_batch,
    decorrelation_time_batch,
    estimate_batch,
)
from ..features import flow_channels as flow_planes
from ..geometry import geometry_channels
from .fluctuations import apply_fluctuation_gain
from .wrappers import call_model, split_output

#: Streamwise velocity written into the synthetic warm-up window. Not arbitrary: an
#: all-zero window makes the freestream estimate zero and the normalisation divides by it,
#: so the warm-up would run on NaNs and warm the wrong kernels. 0.15 m/s is mid-ladder and
#: is what the packaging verifier already uses.
WARMUP_U = 0.15

#: Eager steps to run before a manual ``torch.cuda.CUDAGraph`` capture of the GPU-native tail
#: (``compile_mode="cudagraph"``). Unlike ``torch.compile(mode="reduce-overhead")``, this path
#: needs NO C compiler / Triton / Inductor -- it captures the *eager* pipeline and replays it,
#: so it works in the offline eval image (which has no ``gcc``), removing the batch-1 CPU
#: launch overhead that dominates the stream step. The warm-up lets cuDNN pick its algorithm,
#: the caching allocator settle, and (state path) the ``acc`` buffers be allocated and
#: static-addressed before the capture records their addresses.
CUDAGRAPH_WARMUP = 5


def estimate_freestream(x: np.ndarray, rows: int = FREESTREAM_ROWS) -> np.ndarray:
    """Per-sample freestream speed from ``(N, T, H, W, C)`` inputs.

    The same estimator the dataset uses at training time. It reads only the input
    window, so nothing here needs the Reynolds label -- which matters because the
    competition's ``metadata`` is not documented to carry it.
    """
    return batch_freestream_speed(x, rows=rows)


def _metadata_has_labels(metadata: dict | None) -> bool:
    """Whether ``metadata`` carries a usable ``(re, aoa)`` pair.

    Mirrors the test :func:`infer_conditioning` makes; a device path must defer to labels for
    the same reason the numpy one does, and must not re-implement the condition.
    """
    if not metadata:
        return False
    re = metadata.get("re", metadata.get("reynolds"))
    aoa = metadata.get("aoa", metadata.get("angle_of_attack"))
    return re is not None and aoa is not None


def infer_conditioning(
    x: np.ndarray,
    metadata: dict | None = None,
    domain: str = "real",
    cond_mode: str = "operating_point",
    noise_level: float = 0.0,
) -> np.ndarray:
    """Conditioning features for a batch, estimated from the input windows.

    Since the August 2026 restart ``metadata`` is **an empty dict on scored calls**,
    so the operating point has to be recovered from the 20 observed frames. Both
    quantities are recoverable (see ``data/operating_point.py``): Re from the
    freestream speed to 1.2% mean error, AoA from the mask geometry and wake depth
    to 1.77 deg MAE on held-out cells.

    ``metadata`` is still honoured when it happens to carry ``re``/``aoa`` -- useful
    for offline experiments against ground-truth labels -- but nothing depends on it.

    ``domain`` selects the ``is_sim`` flag *and* decides whether estimation is even
    legitimate: :func:`estimate_reynolds` reads the freestream speed, which is
    physical in real data and identically ~1.09 in simulated data. So a simulated
    evaluation must supply labels through ``metadata``; estimating there would hand
    every sim window the same Reynolds number.

    ``noise_level`` is honoured only under ``cond_mode="operating_point_noise"``, and
    unlike every other component here it is **not** recovered from the window -- training
    injects a synthetic dose that has no counterpart in a real evaluation window, so there
    is nothing to estimate. It is a pinned constant instead, broadcast to the whole batch:
    0.0 (the default) asks for the clean-forecast operating point the model saw least
    corrupted; a submission that wants the denoiser's other trained regime pins it to the
    real PIV noise level instead (``data/piv_noise.py``), the same "pin ``theta_out`` at
    submission" choice item CN.1 describes for the design this generalizes (the retired
    spec's entry is preserved in ``docs/closed_directions.md``).
    """
    n = x.shape[0]
    re_arr = aoa_arr = None
    if metadata:
        re = metadata.get("re", metadata.get("reynolds"))
        aoa = metadata.get("aoa", metadata.get("angle_of_attack"))
        if re is not None and aoa is not None:
            re_arr = np.broadcast_to(np.asarray(re, dtype=np.float64).reshape(-1), (n,))
            aoa_arr = np.broadcast_to(np.asarray(aoa, dtype=np.float64).reshape(-1), (n,))

    if re_arr is None or aoa_arr is None:
        if domain == "sim":
            raise ValueError(
                "Simulated windows carry no recoverable Reynolds number (the data is "
                "already non-dimensionalised, so u_fs ~ 1.09 everywhere). Pass "
                "metadata={'re': ..., 'aoa': ...} for a sim evaluation."
            )
        estimated = estimate_batch(x)
        re_arr, aoa_arr = estimated[:, 0], estimated[:, 1]

    # ``tau`` is the one component that is never a label: it is measured from the observed
    # frames in both domains, so unlike Re it needs no metadata escape hatch on sim.
    tau = decorrelation_time_batch(x) if cond_mode == "operating_point_tau" else None
    noise = (
        np.full(n, noise_level, dtype=np.float64) if cond_mode == "operating_point_noise" else None
    )
    # Every scored window is on the real clock, so the time-step slot reads its reference
    # value, ``dt_cond_value(1.0) == 0``: the augmentation is a training-time device only.
    dt = np.zeros(n, dtype=np.float64) if cond_mode == "operating_point_dt" else None
    return conditioning_features_batch(re_arr, aoa_arr, domain, tau, noise, dt)  # type: ignore[arg-type]


def kinetic_energy_map(fields: np.ndarray) -> np.ndarray:
    """``(N, H, W)`` temporal fluctuation energy, exactly as ``scoring.py`` defines it.

    ``0.5 * (var_t u + var_t v)`` about the window's own time mean. Duplicated here in
    numpy rather than imported from ``train.losses`` because this module has to run
    inside a submission, where torch is available but our training package is not
    shipped.
    """
    return kinetic_energy_map_components(fields).sum(axis=-1)


def kinetic_energy_map_components(fields: np.ndarray) -> np.ndarray:
    """``(N, H, W, 2)``: the two terms of :func:`kinetic_energy_map`, one per component.

    ``0.5 * var_t u`` and ``0.5 * var_t v`` separately, so that summing the last axis gives
    the scorer's map exactly. The per-component second-moment head reads and corrects these
    two maps rather than their sum: the scorer only sees the sum, which leaves it free to
    put the fluctuation energy into whichever component agrees with the truth.
    """
    f = np.asarray(fields, dtype=np.float32)
    out = np.zeros(f.shape[:1] + f.shape[2:4] + (2,), dtype=np.float32)
    for ch in (0, 1):
        c = f[..., ch]
        out[..., ch] = 0.5 * ((c - c.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)
    return out


def rescale_fluctuations(
    prediction: np.ndarray,
    input_window: np.ndarray,
    blend: float = 0.0,
    max_gain: float = 10.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Rescale the prediction's temporal fluctuations toward a blended TKE target.

    An L2-trained model converges to the conditional mean, so it is unbiased in the
    *first* moment and badly biased in the *second*: ours retains only 0.36-0.44 of the
    target's fluctuation energy. Meanwhile the **input** window's total fluctuation
    energy is an unbiased estimate of the target's (measured 0.99 / 1.03 / 1.01 across
    the three splits), so the missing signal is present in the 20 observed frames and
    L2 training simply discards it.

    The target variance map is a geometric blend of what the input shows and what the
    prediction produced, ``T_hat = ke(input)**blend * ke(pred)**(1 - blend)``, and each
    pixel's fluctuations are scaled by ``sqrt(T_hat / ke(pred))``. Written that way the
    gain collapses to ``(ke(input) / ke(pred)) ** (blend / 2)``, so ``blend=0`` is
    exactly the identity and no branch is needed to disable it.

    Two properties make this safe to apply post hoc:

    * **``mvpe_score`` is invariant by construction.** The gain is constant in time, so
      the rescaled field has the same time mean as the prediction, and ``mvpe``
      time-averages before comparing. Measured unchanged to every printed digit.
    * **It is local.** A *uniform* gain is known to be harmful -- 1.4 moves tke 71.1 ->
      59.5 -- because the predicted TKE map is mis-localised (spatial correlation 0.758)
      rather than merely under-amplitude, so amplifying it amplifies its error. Scaling
      per pixel toward a map the input actually supports is the opposite operation.

    ``max_gain`` bounds the correction where the prediction is nearly static and the
    ratio would otherwise explode -- chiefly inside the PIV measurement hole, where the
    target is identically zero and the scorer ignores the element anyway.

    Only the gain is computed here; :func:`~rpde_t1.models.fluctuations.apply_fluctuation_gain`
    applies it, so the blend, the learned head and the diagnostics ladder all rescale the
    same way and differ only in the answer they arrive at.
    """
    if blend == 0.0:
        return prediction

    ke_pred = kinetic_energy_map(prediction)
    ke_in = kinetic_energy_map(input_window)
    gain = np.power(
        np.clip(ke_in, eps, None) / np.clip(ke_pred, eps, None), 0.5 * blend, dtype=np.float32
    )
    return apply_fluctuation_gain(prediction, gain, max_gain=max_gain)


def _blend_of(tke_blend: float | None, tke_head) -> float:
    """The closed-form blend exponent to shrink a head toward.

    ``None`` -- the default -- means "the measured one when a head is present", which is what
    the ``tke_blend or DEFAULT_BLEND`` idiom this replaces was written for. An explicit
    ``0.0`` now means **no blend**, which that idiom silently turned into 0.5. It has to be
    reachable: on the current lineage the blend is negative at every exponent on both
    held-out splits (FINDINGS 63f), so an unblended head shrunk toward *zero* is the
    configuration that ships, and it was unrequestable.
    """
    from .tke_head import DEFAULT_BLEND  # noqa: PLC0415

    if tke_blend is not None:
        return float(tke_blend)
    return DEFAULT_BLEND if tke_head is not None else 0.0


def make_predict(
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    return_bounds: bool = False,
    bound_floor: float = 0.0,
    bound_scale: float = 1.0,
    width_rule: Any | None = None,
    lead_anchor: Any | None = None,
    normalizer: str = "freestream",
    domain: str = "real",
    cond_mode: str = "operating_point",
    stats_domain: str | None = None,
    zero_pressure: bool = True,
    tke_blend: float | None = None,
    tke_head: torch.nn.Module | None = None,
    tke_shrink: float | None = None,
    tke_head_uses_tau: bool = False,
    geom_channels: tuple[str, ...] = (),
    flow_channels: tuple[str, ...] = (),
    aux_mask: str = "",
    aux_mask_plane: bool = True,
    precision: str = "fp32",
    pad_batches: bool = False,
    compile_mode: str | None = None,
    out_transform: tuple[Any, Any] | None = None,
    out_dtype: str | None = None,
    kmap_out: dict[str, np.ndarray] | None = None,
    cond_device: bool = False,
    zero_p_device: bool = False,
    state_frames: int = 0,
    centre_shrink: float = 1.0,
) -> Callable[..., Any]:
    """Build a ``predict(input_array, metadata=None)`` closure around a model.

    ``bound_floor`` adds a constant physical half-width under the model's own
    bounds. Useful as a safety net for a bound head that has collapsed: the SPS
    credit is zero wherever the target falls outside the interval, so a
    pathologically narrow band is worse than a merely imprecise one.

    ``bound_scale`` multiplies the head's asymmetric half-widths about the prediction --
    the deployment side of :func:`rpde_t1.train.calibrate.scan_factor`, and *the same
    transform*, so a factor fitted there can be shipped without re-deriving anything. It
    exists because every head trained by this recipe comes out systematically too wide
    (FINDINGS 10 fits 0.60 independently on U-Net, FNO and CNO). Note that the recorded
    *reason* for that -- in-sample backbone errors met by larger held-out ones -- is refuted:
    0.60 is equally optimal on in-sample windows (FINDINGS 14). Applied *before*
    ``bound_floor`` so the floor stays a floor.

    ``width_rule`` is a :class:`~rpde_t1.models.width_rule.WidthRule`, the generalisation of
    that scalar: a multiplier per bin of the head's own predicted ``log rho``, per channel
    and per lead, each bin fitted by maximising the scorer's credit. It supersedes
    ``bound_scale`` when given -- passing both is an error rather than a silent composition,
    since two width calibrations multiplying each other is precisely the kind of duplicated
    definition PROTOCOL 5 records going wrong. On ``val_extrap`` the rule is worth +1.7 SPS
    over the best single multiplier, but *only* on a head whose scale is a calibrated
    log-error regression; on the older width-trained head it was worth +0.03.

    Second-moment correction, which is one code path with two settings:

    * ``tke_blend`` alone applies the closed-form geometric blend; ``0.0`` is the identity.
    * ``tke_head_uses_tau`` must match how the head was FITTED -- ``models.tke_head.head_uses_tau``
      reads it off the checkpoint. It cannot be inferred from the weights, and getting it wrong
      serves a channel the head never saw (FINDINGS 63m).
    * ``tke_head`` additionally mixes in a trained second-moment head, shrunk toward that
      blend by ``tke_shrink`` (default :data:`~rpde_t1.models.tke_head.DEFAULT_SHRINK`).
      **The shrinkage is not optional tuning** -- unshrunk, the head is worse than the free
      blend on Reynolds extrapolation.

    Either way the correction is applied to the prediction *before* the bounds are offset
    from it, so an interval head trained against the unrescaled prediction still brackets
    what is returned.

    ``aux_mask`` states that the model was trained with masked-modelling and therefore
    expects a visibility plane on its input. Nothing is ever hidden here -- masking is a
    training device, not the task -- so the plane is all ones, which is the same thing the
    validation loaders feed (``aux_mask_ratio=0``). It must still be *present*: without it
    the first convolution is handed one channel too few and the model cannot run at all.

    ``lead_anchor`` is a :class:`~rpde_t1.models.lead_anchor.LeadAnchor`: per-lead
    shrinkage toward the last observed frame, closed form and free at inference. It travels
    the same displacement path, for the same reason.

    ``geom_channels`` appends the geometry planes the model was trained with. They are
    built from the angle **estimated from the raw input window**, which is exactly what
    ``WindowDataset`` does at training time -- the estimate has 1.77 deg MAE, and a model
    trained on estimates must be served estimates.

    ``flow_channels`` appends the planes derived from the window itself -- vorticity and the
    two second moments (``features.FLOW_CHANNELS``). They need no estimator and no label at
    all, so unlike the geometry planes there is nothing here that can be right at training
    time and wrong at serving time *except the order and the scaling*, and both come from
    the one function the dataset also calls.

    ``precision`` runs the forward under ``torch.autocast``: ``"bf16"`` or ``"fp16"`` against
    the default ``"fp32"``. This is a *time* optimisation and only on CUDA -- the evaluation
    GPU is an A800 (A100 silicon), where fp32 peaks at 19.5 TFLOPS against 312 for both
    bf16 and fp16, and our measured T4->A800 factor of only 2.91x (against CNO's 18.8x with
    fp16) says the 3D convolutions are leaving the tensor cores idle. bf16 is preferred over
    fp16 for its fp32-sized exponent, which removes the overflow risk in the attention
    logits that made fp16 unsafe for Transolver. Autocast, rather than casting the module,
    because its per-op policy keeps softmax and the norms in fp32.

    **The reduced dtype changes the outputs**, so a build using it must be re-scored before
    it ships -- including `sps`, whose interval widths come through the same path.

    ``pad_batches`` repeats the final sample so every forward sees the same shape, and slices
    the padding off afterwards. It buys nothing alone -- 210 samples at batch 32 pays 6.7%
    extra compute for 14 phantom samples -- and exists because ``cudnn.benchmark`` and
    ``compile_mode`` both charge per *distinct* shape: without it the tail call pays their
    setup inside the timed region.

    ``compile_mode`` passes ``torch.compile(model, mode=...)``. ``"reduce-overhead"`` adds
    CUDA graphs, which need one shape, hence ``pad_batches``. Compilation happens here, at
    construction, which the competition leaves untimed. **A failure is caught and degraded to
    eager**: a compile error inside ``predict`` would zero all five subscores.

    ``out_dtype`` and ``out_transform`` describe what the *consumer* wants rather than what
    the forward computes, and both are paid for on the device instead of the host.
    ``out_dtype="fp16"`` stages the returned arrays in half width, halving the device->host
    transfer; the submission harness re-casts on write, so no subscore moves. It is separate
    from ``precision`` on purpose -- at batch 1 fp16 *compute* costs more in cast kernels than
    it saves, while the transfer saving is unconditional. ``out_transform`` is a per-channel
    ``(scale, shift)`` applied to the prediction and both bounds just before that transfer,
    for a consumer that grades in some other affine coordinate system -- which otherwise costs
    a full host round trip. It is refused when any host-side step would run after it; see the
    guard below the chunk loop.

    ``centre_shrink`` (FINDINGS 196i) builds the interval about ``m + c * (pred - m)`` -- ``m``
    the window's time mean of ``u``/``v`` -- instead of about the prediction: the interval head
    reads the shrunk field, and the width rule, the ordering guard and any floor are applied
    about it, while the REPORTED prediction stays the backbone's own (and the second-moment
    correction still applies to that reported field only). Post-hoc and time-free; the head
    and rule that ship with it must have been fitted on the same shrunk field
    (``scripts/sps/shrink_cache.py`` then ``sps_apply.py``). 1.0 -- the default -- is exactly
    the previous behaviour. Requires the outermost module to be a ``BoundedModel`` and refuses
    a lead anchor (the anchor's per-lead step differs between the two fields, and 196i's
    measurement ran without one).
    """
    if cond_device:
        # Refused at CONSTRUCTION rather than silently ignored at serve time. Each clause is
        # a case the device twin does not implement (models/op_estimate.py says so in its own
        # docstring); serving one of them through this path would quietly hand the model a
        # different conditioning vector than the recipe was validated on.
        if normalizer != "freestream":
            raise ValueError("cond_device needs the freestream normalizer")
        if cond_mode != "operating_point":
            raise ValueError(f"cond_device does not implement cond_mode={cond_mode!r}")
        if geom_channels or flow_channels or aux_mask:
            raise ValueError("cond_device does not implement geometry/flow/mask planes")
        if str(device).startswith("cpu"):
            raise ValueError("cond_device is a GPU path")

    if width_rule is not None and bound_scale != 1.0:
        raise ValueError(
            "pass either width_rule or bound_scale, not both: they are two calibrations of "
            "the same width and would multiply."
        )
    centre_shrink = float(centre_shrink)
    if centre_shrink != 1.0:
        from .bounds import BoundedModel  # noqa: PLC0415

        if not isinstance(model, BoundedModel):
            raise ValueError(
                "centre_shrink needs the outermost module to be a BoundedModel: the shrunk "
                "centre is formed inside its forward and returned beside the bounds."
            )
        if lead_anchor is not None:
            raise ValueError("centre_shrink and lead_anchor are not composed; ship one.")
        if not (0.0 < centre_shrink <= 1.0):
            raise ValueError(f"centre_shrink must be in (0, 1], got {centre_shrink}")
    if hasattr(model, "centre_shrink"):
        # Set (or reset to 1.0) on the model itself: BoundedModel.forward reads it, so the
        # attribute is the single switch both the reference chunk and FastT1 follow. It is
        # state on the MODEL: two closures over one model object serve the shrink of whichever
        # was built last (one predictor per process on every shipping path).
        model.centre_shrink = centre_shrink
    if width_rule is not None and not width_rule.is_uniform:
        # Resample the fitted step function onto an evenly spaced grid, here rather than in
        # the caller: model construction is outside the platform's timed region (FINDINGS 13),
        # and on quantile edges every element pays a binary search -- 9.4 ms/window against
        # 3.5 uniform, on a budget of ~21. Doing it automatically means the slow path cannot
        # be shipped by forgetting a call. Exact to 0.001 SPS at the default bin count.
        width_rule = width_rule.to_uniform()
    model = model.to(device).eval()

    # cuDNN autotunes a convolution algorithm per *input shape* and caches the choice, so the
    # trial is paid on the first call at each shape. FINDINGS 15 deferred it for that reason:
    # with 210 samples at batch 32 the tail is 18, an unwarmed shape, and the trial would land
    # inside a scored call. ``pad_batches`` removes that -- every forward sees exactly one
    # shape, warmed by ``warm_up_predict`` outside the timed region -- so the deferral's
    # premise expires here and nowhere else. Measured on the shipped artefact: **46.32 ->
    # 44.37 ms/window, -4.2%**, with all four accuracy/SPS axes unchanged.
    #
    # Gated on ``deterministic`` as well because the two are in direct conflict and one caller
    # depends on it: ``scripts/sps/sps_cache.py`` pins the algorithm choice so a fitted artefact is
    # reproducible (FINDINGS 17). It caches unpadded, so the first clause already spares it --
    # the second makes that a guarantee rather than a coincidence.
    # Excluded under ``compile_mode``, and that exclusion is the fix for a measured -8.6
    # ``time_score`` regression (FINDINGS 36). The autotune needs *workspace*; when free VRAM is
    # tight at construction the search fails, ``warm_up_predict`` returns False, and the search
    # re-runs inside the first *scored* call. Measured on a T4 squeezed to 1.3 GB free, compiled
    # build, first call: **22.50 ms/sample with benchmark, 6.71 without** -- +3.5 s over 210
    # samples, the size of the board's unexplained penalty. Unstarved it costs nothing to drop
    # (4.11 vs 4.18 ms/sample), because Inductor already fixes the shape that benchmark exists
    # to amortise, so this is a strict improvement rather than a trade.
    #
    # **RETIRED 2026-08-19, and the exclusion above was too narrow.** Day 19 shipped an EAGER
    # b210 bundle -- no compile, warm_up=True, 78 GB free on the board, so none of the
    # starvation conditions above -- and came back at **27.5 ms/sample against a local 4.8**,
    # `time_score` 83.73, -8.48 against the same-cost `stf_sps_b210`. Measured on the shipped
    # zip, the flag being the only difference:
    #
    #   | benchmark | import block | of which warm-up | steady state |
    #   | True      | 5.95 s       | 5.44 s           | 4.82 ms/sample |
    #   | False     | 2.33 s       | 1.82 s           | 4.80 ms/sample |
    #
    # The autotune costs **3.6 s of one-off search** and returns **nothing** -- 4.80 is if
    # anything the faster of the two, which also fails to reproduce the -4.2% this comment
    # claimed. Over ~210 scored samples a 3.6 s one-off is +17 ms/sample, i.e. the whole of the
    # observed gap. 36's own numbers already said the flag was free to drop; it removed it only
    # under `compile_mode`, which left every eager build paying the search. There is no
    # configuration in which it is now enabled: the search is either charged (catastrophic) or
    # untimed (worth 0.02 ms/sample), so the bet has no upside.

    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    if autocast_dtype is not None and not str(device).startswith("cuda"):
        autocast_dtype = None
    # Insurance, and it is not hypothetical: ``torch.autocast`` *raises* for bf16 on a device
    # without hardware support (measured on a T4, sm_75) rather than falling back, and a
    # raise inside predict() zeroes all five subscores. The evaluation A800 supports bf16, so
    # this branch should never fire there -- but the cost of being wrong is the whole run.
    if autocast_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        autocast_dtype = None

    # The host buffers the output is copied into, and an optional affine applied on the
    # device just before that copy. Both exist because the *reported* dtype and the
    # *reported* coordinate system are properties of the consumer, not of the forward.
    #
    # ``out_dtype`` decouples the two. Staging the output in half width halves the
    # device->host transfer (1.19 -> 0.60 ms/window over three arrays) and the submission
    # harness re-casts to fp32 on write, so every subscore is unchanged -- but until now it
    # was reachable only as a side effect of ``precision="fp16"``, which at batch 1 costs
    # more in cast kernels than the transfer saves. Naming it separately takes the transfer
    # saving without the compute penalty.
    #
    # ``out_transform`` is a per-channel ``(scale, shift)`` applied to the prediction and
    # both bounds. A consumer that wants them in some other affine coordinate system would
    # otherwise pay a full host round trip -- three arrays down, three arrays back up, the
    # arithmetic, and down again. Doing it here costs one broadcast multiply-add on a device
    # that is idle anyway, and removes the round trip entirely. The map must be applied
    # *after* the ordering guard above, which a positive scale preserves.
    store_dtype = {None: None, "fp32": np.float32, "fp16": np.float16}[out_dtype]
    if store_dtype is None:
        store_dtype = np.float16 if autocast_dtype is torch.float16 else np.float32
    out_scale = out_shift = None
    if out_transform is not None:
        out_scale = torch.as_tensor(np.asarray(out_transform[0], dtype=np.float32), device=device)
        out_shift = torch.as_tensor(np.asarray(out_transform[1], dtype=np.float32), device=device)

    # ``zero_p_device``: set the unscored pressure channel to zero on the resident tensor rather
    # than in three strided passes over the host buffers after the copy down. The host form is
    # the last step still running there on a shipped Track-1 artefact, and it is dearer than it
    # looks: a 2-byte element every 6 bytes touches every cache line, so zeroing 52 MB of
    # pressure reads and writes all 155 MB of output. Measured at batch 210 on this A10G,
    # 20 calls: **11.68 ms host against 1.43 ms on the device**, i.e. 0.0556 -> 0.0068 ms/window.
    #
    # Refused rather than silently skipped when ``out_transform`` would map the zero somewhere
    # else: the device write happens BEFORE that map in ``store``, so it is only equivalent when
    # the transform leaves pressure at zero. That is exactly the condition the host-tail guard
    # below already enforces, so the two agree by construction.
    zp_device = bool(zero_pressure and zero_p_device) and (
        out_shift is None or float(out_shift.reshape(-1)[-1]) == 0.0
    )

    eager_model, compiled_model = model, None
    if compile_mode and not compile_backend_is_usable():
        # Refuse to build a compiled wrapper at all when the backend cannot compile here. The
        # existing try/except below is not enough, and FINDINGS 187 measured why: `torch.compile`
        # returns a lazy wrapper that raises on the first forward at EACH NEW SHAPE, so in an
        # environment without a C compiler the failure is re-paid per batch size -- 19-21 s inside
        # a scored call, measured in the Track-1 evaluation image. Declining up front means the
        # worst case really is the eager time, which is what FINDINGS 148 claimed and did not have.
        compile_mode = ""
    if compile_mode:
        try:
            compiled_model = torch.compile(model, mode=compile_mode)
        except Exception:  # noqa: BLE001 - eager is always correct; a raise here is not
            compiled_model = None
    # Sticky, because the failure mode is measured rather than hypothetical: with
    # ``mode="reduce-overhead"`` the CUDA-graph path raises `Triton Error [CUDA]: invalid
    # argument` the first time it sees a *second* input shape, and it raises inside the
    # forward -- i.e. inside a scored ``predict`` call, which would zero all five subscores.
    # So a compiled forward that throws demotes the run to eager and never retries.
    use_compiled = {"on": compiled_model is not None}

    # A model trained under one normalization must be run under the same one, or the
    # inputs land on a different scale than it ever saw.
    stats = None
    if normalizer == "gaussian":
        from ..features import gaussian_stats

        # Separated from ``domain`` on purpose. ``domain`` says whose *conditioning* to
        # build; ``stats_domain`` says whose *normalisation table* to apply, and the two
        # are different questions for a sim-pretrained model evaluated on real data.
        #
        # It also makes the information budget explicit. Freestream normalisation uses
        # nothing about the target dataset -- the divisor is measured from each input
        # window. Gaussian needs six fitted constants, and taking them from ``real``
        # hands a sim-trained model a summary of the target distribution it would not
        # have under strict zero-shot transfer. Which is right depends on the question:
        # `real` is deployment-realistic (train_real exists and the organizers publish
        # its statistics), `sim` is the honest domain-transfer measurement.
        stats = gaussian_stats(stats_domain or domain)

    # Track 1's chunk served by ``fast_serve.FastT1``: the same forward, width rule, guard,
    # second-moment gain and pressure zeroing in fewer kernels, BIT-IDENTICAL to the chunk below
    # (FINDINGS 220). Only on the board bundle's device path (``cond_device``, device pressure
    # zeroing, the freestream normaliser, a uniform width rule, the second-moment head, no
    # anchor, geometry, flow or mask planes, no compile, no padding), and only after the first
    # chunk -- the import-time warm-up, untimed on the board -- has matched the reference
    # exactly; a mismatch or any error serves the reference.
    _t1f: dict[str, Any] = {"fast": None, "checked": False}
    if (
        str(device).startswith("cuda")
        and cond_device
        and zp_device
        and stats is None
        and return_bounds
        and width_rule is not None
        and tke_head is not None
        and lead_anchor is None
        and not (geom_channels or flow_channels or aux_mask)
        and compiled_model is None
        and not pad_batches
        and out_scale is None
        and kmap_out is None
        and not state_frames
        and bound_floor == 0.0
        and bound_scale == 1.0
        and autocast_dtype in (None, torch.float16)
    ):
        from . import fast_serve  # noqa: PLC0415

        if fast_serve.t1_supported(
            model,
            tke_head=tke_head,
            width_rule=width_rule,
            lead_anchor=lead_anchor,
            uses_tau=tke_head_uses_tau,
        ):
            _t1f["fast"] = fast_serve.FastT1(
                model,
                width_rule=width_rule,
                tke_head=tke_head,
                blend=_blend_of(tke_blend, tke_head),
                shrink=tke_shrink,
                uses_tau=tke_head_uses_tau,
                half_weights=autocast_dtype is torch.float16,
                centre_shrink=centre_shrink,
            )

    def predict(input_array, metadata=None):
        x = np.asarray(input_array, dtype=np.float32)
        # Estimates read the LAST T_IN frames, so a 40-frame-context model is conditioned and
        # normalised exactly as the 20-frame one (and as ``WindowDataset`` trained it).
        recent = x[:, -T_IN:]
        # ``cond_device``: the operating point, the freestream and the normalisation all run
        # on the device, from ONE upload of the raw window, instead of three numpy passes over
        # a 103 MB array before the GPU is touched. Measured on the shipped bundle at batch
        # 210: the numpy estimator alone is 54.8 ms (0.261 ms/sample, 58% of it the hole-mask
        # reduction) against 2.30 ms on the device, and it is 11.6% of the COMPILED path
        # because speeding the forward up makes the host half a larger share.
        #
        # ``metadata`` still wins when it carries labels -- that is the offline/oracle route
        # (FINDINGS 121) and it must not be silently overridden by an estimate.
        x_dev = None
        if cond_device and not _metadata_has_labels(metadata):
            from .op_estimate import (  # noqa: PLC0415
                conditioning_features_batch_torch,
                estimate_batch_torch,
            )

            x_dev = torch.from_numpy(np.ascontiguousarray(x)).to(device)
            recent_dev = x_dev[:, -T_IN:]
            cond_dev = conditioning_features_batch_torch(
                estimate_batch_torch(recent_dev), domain=domain
            )
            scale_dev = (
                recent_dev[:, :, :FREESTREAM_ROWS, :, 0]
                .abs()
                .mean(dim=(1, 2, 3))
                .clamp_min(1e-6)
            )
            scale = scale_dev.double().cpu().numpy()
            cond = cond_dev.cpu().numpy()
        else:
            scale = (
                np.ones(x.shape[0], dtype=np.float64) if stats else estimate_freestream(recent)
            )
            cond = infer_conditioning(recent, metadata, domain=domain, cond_mode=cond_mode)

        # The three outputs, written chunk-by-chunk into their final buffers rather than
        # appended and concatenated at the end. Measured on the shipped STFormer artefact,
        # 210 samples: ``concatenate(...).astype(float32)`` x3 costs **0.961 ms/window**
        # against **0.297** for slice assignment -- the concatenate allocates and copies
        # 310 MB it does not need to, and ``astype`` defaults to ``copy=True`` so it then
        # copies a float32 array to float32 again. That is 0.66 ms/window of a 2.60 ms
        # host-side budget, on an axis where the host half is the only remaining lever
        # (FINDINGS 30h). Allocated lazily on the first chunk, because the output's frame
        # count is the model's to decide, and indexed by ``start`` so the OOM retry's
        # adaptive chunk size needs no special case.
        out: dict[str, np.ndarray | None] = {
            "prediction": None,
            "lower": None,
            "upper": None,
            # The shrunk interval centre, copied down only when a host-side step still needs
            # it (see ``_chunk_ref``'s tail); None on every artefact without a centre shrink.
            "centre": None,
        }
        # A chunk's centre awaiting ``store``: the chunk loop copies it after the three outputs.
        side: dict[str, torch.Tensor | None] = {"centre": None}
        kmaps: list[np.ndarray] = []
        # Whether the width rule ran on the device, so the host-side pass below knows not to
        # apply it a second time. Set by ``chunk``; a rule that declines there leaves it False
        # and the numpy path runs exactly as before.
        on_device = {"rule": False, "anchor": False, "rescale": False, "zero_p": False}

        def _chunk_ref(sl: slice) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
            """One forward over ``sl``, denormalised back to physical units."""
            s = scale[sl].reshape(-1, 1, 1, 1, 1)
            # On the device path the normalisation happens below, on the resident tensor;
            # building the numpy copy here would pay exactly the cost this path removes.
            if x_dev is not None:
                xn = None
            elif stats:
                xn = (x[sl] - stats.mean_in) / stats.std_in
            else:
                xn = x[sl] / s
            if aux_mask and aux_mask_plane:
                # All ones: nothing is hidden at inference. Appended *before* the geometry
                # planes because ``WindowDataset.__getitem__`` concatenates them in that
                # order, and a permuted input channel order is a silent accuracy loss
                # rather than an error.
                xn = np.concatenate([xn, np.ones((*xn.shape[:4], 1), dtype=xn.dtype)], axis=-1)
            if geom_channels:
                # The angle comes from ``cond``, which already holds the estimate
                # ``infer_conditioning`` derived from the raw window. Recomputing it
                # here would duplicate the work and, worse, let the conditioning vector
                # and the geometry planes describe *different* angles if the two ever
                # diverged. Appended after normalisation because an SDF divided by a
                # freestream speed would be meaningless, and built from the raw window
                # because the gaussian normalizer is affine and would shift the wake
                # ratio the angle estimate reads.
                planes = np.stack(
                    [
                        geometry_channels(
                            float(a) * AOA_MAX, tuple(geom_channels), x.shape[2], x.shape[3]
                        )
                        for a in cond[sl, 1]
                    ]
                )
                if flow_channels:
                    # Flow planes come after the geometry planes and are computed on the host,
                    # so keep the whole layout on the host for that (unshipped) combination.
                    xn = np.concatenate(
                        [xn, np.broadcast_to(planes[:, None], (*xn.shape[:4], planes.shape[-1]))],
                        axis=-1,
                    )
                    planes = None
            else:
                planes = None
            if flow_channels:
                # Last, matching ``WindowDataset.__getitem__``, and computed from the first
                # ``C`` channels of the *normalised* window -- the identical call the dataset
                # makes, on the identical array, so training and serving cannot disagree
                # about a plane that is derived rather than looked up.
                xn = np.concatenate(
                    [xn, np.stack([flow_planes(w[..., :C], tuple(flow_channels)) for w in xn])],
                    axis=-1,
                )
            if x_dev is not None:
                # Already resident: normalise here rather than re-uploading the numpy copy.
                # (``cond_device`` refuses geometry/flow/mask planes, so ``planes`` is None here.)
                ub = scale_dev[sl].float()
                xb = x_dev[sl] / ub.view(-1, 1, 1, 1, 1)
                cb = cond_dev[sl]
            else:
                xb = torch.from_numpy(np.ascontiguousarray(xn)).to(device)
                cb = torch.from_numpy(np.ascontiguousarray(cond[sl])).to(device)
                ub = torch.from_numpy(s.reshape(-1).astype(np.float32)).to(device)
            if planes is not None:
                # Broadcast the once-per-window planes over the frames ON THE DEVICE: the host
                # version copied the whole batch through numpy (0.43 ms/window at batch 210,
                # more than the whole `unet2d` forward -- FINDINGS 153k).
                gb = torch.from_numpy(np.ascontiguousarray(planes)).to(device)
                xb = torch.cat([xb, gb[:, None].expand(-1, xb.shape[1], -1, -1, -1)], dim=-1)

            def run(m, dtype: torch.dtype | None):
                if dtype is None:
                    raw = call_model(m, xb, cb, ub)
                else:
                    with torch.autocast("cuda", dtype=dtype):
                        raw = call_model(m, xb, cb, ub)
                # ``centre`` is present only under a centre shrink (BoundedModel.forward).
                ctr = raw.get("centre") if isinstance(raw, dict) else None
                return (*split_output(raw), ctr)

            def forward(dtype: torch.dtype | None):
                if use_compiled["on"]:
                    try:
                        return run(compiled_model, dtype)
                    except Exception:  # noqa: BLE001 - see ``use_compiled`` above
                        use_compiled["on"] = False
                return run(eager_model, dtype)

            n_real = xb.shape[0]
            if pad_batches and n_real < batch_size:
                reps = batch_size - n_real
                xb = torch.cat([xb, xb[-1:].repeat(reps, *([1] * (xb.ndim - 1)))])
                cb = torch.cat([cb, cb[-1:].repeat(reps, *([1] * (cb.ndim - 1)))])
                ub = torch.cat([ub, ub[-1:].repeat(reps)])

            pred, lo, hi, ctr = forward(autocast_dtype)
            # fp16 spans 6e-5 to 65504 against fp32's 1e-38 to 3e38, so a reduced-precision
            # forward can in principle overflow where the reference would not. Measured, it
            # does not: freestream normalisation means the network always sees O(1) inputs
            # whatever the Reynolds number, and autocast keeps softmax and the norms in fp32.
            # But "measured on the cells we hold" is not "true on the private set", and a
            # single NaN zeroes **all five** subscores -- so a non-finite chunk is recomputed
            # in fp32 rather than shipped. One reduction per chunk, ~0.1 ms against ~675 ms.
            if autocast_dtype is not None and not all(
                bool(torch.isfinite(t).all()) for t in (pred, lo, hi, ctr) if t is not None
            ):
                pred, lo, hi, ctr = forward(None)
            # Back to fp32 before anything else touches it: autocast leaves the model's
            # outputs in the reduced dtype, and the denormalisation, the bound arithmetic
            # and the metrics are all fp32 quantities.
            pred, lo, hi, ctr = (t.float() if t is not None else None for t in (pred, lo, hi, ctr))
            if pred.shape[0] != n_real:  # drop the padding before denormalisation
                pred = pred[:n_real]
                lo = lo[:n_real] if lo is not None else None
                hi = hi[:n_real] if hi is not None else None
                ctr = ctr[:n_real] if ctr is not None else None
            if pred.shape[-1] > C:
                # The variance-map channel (models/kmap.py): its time mean goes back to the
                # caller through ``kmap_out``; the field every step below sees is the frames.
                if kmap_out is not None:
                    kmaps.append(pred[..., C].mean(dim=1).cpu().numpy())
                pred = pred[..., :C]
                lo = lo[..., :C] if lo is not None else None
                hi = hi[..., :C] if hi is not None else None
                ctr = ctr[..., :C] if ctr is not None else None
            if stats:
                mul = torch.from_numpy(stats.std_target).to(device)
                add = torch.from_numpy(stats.mean_target).to(device)
            else:
                mul, add = torch.from_numpy(s.astype(np.float32)).to(device), None

            def to_physical(t: torch.Tensor) -> torch.Tensor:
                scaled = t * mul
                return (scaled if add is None else scaled + add).float()

            paired = lo is not None and hi is not None
            pred_p = to_physical(pred)
            lo_p = to_physical(lo) if paired else None
            hi_p = to_physical(hi) if paired else None

            # Anchor here rather than on the host, for the reason ``apply_torch`` records:
            # measured paired, the numpy transform costs +2.86 ms/window, which cancels the
            # SPS it buys. Only the last observed frame is uploaded, so the extra bus traffic
            # is 0.5 MB at batch 32 rather than the whole 20-frame window's 10.5 MB.
            #
            # A displacement of all three fields, exactly as the host path treats it, so the
            # bounds keep bracketing what is returned -- and it runs *before* the width rule
            # here as it does there. The order is in fact free: the rule reads only
            # ``pred - lower`` and ``upper - pred``, both invariant under a common shift.
            if lead_anchor is not None:
                last = torch.from_numpy(np.ascontiguousarray(x[sl][:, -1:])).to(device)
                anchored = lead_anchor.apply_torch(pred_p, last)
                step = anchored - pred_p
                pred_p = anchored
                if paired:
                    lo_p, hi_p = lo_p + step, hi_p + step
                on_device["anchor"] = True

            def rescale_reported(t: torch.Tensor) -> torch.Tensor:
                """The second-moment correction, on the device, reported field only.

                Called on both return paths and always LAST, so the width rule and the ordering
                guard read the unrescaled field -- which is the field the interval head was
                fitted against, and the reason 63j's split exists. Two energy maps and the gain
                application are 2.4 ms/window of numpy on a 103 MB array against 0.03 on a
                resident tensor (63m); the numpy path in the host tail remains the definition
                and runs whenever this one is not reached.
                """
                if tke_head is None and not tke_blend:  # None or 0.0 -- nothing to apply
                    return t
                from .tke_head import (  # noqa: PLC0415
                    apply_gain_torch,
                    inference_log_ratio_torch,
                )

                # ``state_frames`` (TrainConfig.stream_state): the second-moment head's input
                # features are statistics of the MEASURED window; the pseudo-frames a stream-
                # state model prepends are for the backbone only, and the head was fitted on
                # 20-frame maps. 0 for every existing checkpoint -- identical behaviour.
                r = inference_log_ratio_torch(
                    t,
                    x[sl][:, state_frames:],
                    cond[sl],
                    head=tke_head,
                    blend=_blend_of(tke_blend, tke_head),
                    shrink=tke_shrink,
                    uses_tau=tke_head_uses_tau,
                    # Only under the freestream normaliser, where ``xn`` is ``x / s`` exactly.
                    # The gaussian one is affine, and an affine map does not commute with the
                    # energy ratio the head reads -- so it keeps the host path rather than
                    # getting a silently different feature.
                    input_device=None if stats else (xb[:, state_frames:][..., :C], ub),
                )
                on_device["rescale"] = True
                return apply_gain_torch(t, r)

            def zero_p(*fields: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
                """Pressure to zero here, on the resident tensors, if this path owns that step.

                Last, after the rescale, the anchor and the width rule, so it sees exactly the
                fields the host tail would have seen. Nothing downstream reads pressure -- the
                rule reads ``pred - lower`` per element and the guard is elementwise -- so the
                position is free and only the device matters.
                """
                if not zp_device:
                    return fields
                for f in fields:
                    if f is not None:
                        f[..., 2] = 0.0
                on_device["zero_p"] = True
                return fields

            if not paired:
                return (*zero_p(rescale_reported(pred_p)), None, None)

            # Size the interval here, on the device, rather than after the copy to host: the
            # rule is elementwise plus a gather, and on the host it measured +5.62 ms/window
            # against a ~46 ms budget -- 100% of this artefact's latency regression. It has to
            # run *after* denormalisation, because ``log rho`` is a ratio against the physical
            # SIGMA_GLOBAL, and it declines (returning None) on a non-uniform rule, which then
            # falls through to the numpy path below.
            #
            # Running here, before the host-side second-moment rescale, is not merely
            # equivalent -- it is the *required* order (FINDINGS 63j). The rule must read the
            # unrescaled field, because that is the field whose realised error it was fitted
            # against, and a rescale moves the prediction away from the target rather than
            # toward it. The host path below reads ``centre`` for exactly this reason; on this
            # path the ordering already guarantees it.
            # The interval's centre in physical units: the shrunk field under ``centre_shrink``
            # (formed in normalised units inside the model, once -- the shrink is linear, so
            # it commutes with the freestream scale up to rounding, and FastT1 does the same),
            # else the prediction itself. Bound HERE, after the anchor above has moved
            # ``pred_p``: the anchored field is the centre when no shrink is set (a shrink
            # refuses the anchor at construction).
            centre_p = pred_p if ctr is None else to_physical(ctr)
            if width_rule is not None:
                sized = width_rule.apply_torch(centre_p, lo_p, hi_p)
                if sized is not None:
                    lo_p, hi_p = sized
                    on_device["rule"] = True
            if centre_p is not pred_p and (
                bound_floor > 0 or bound_scale != 1.0 or not on_device["rule"]
            ):
                # A host-side step still reads the centre (the floor, the scalar multiplier,
                # or a rule that declined the device): hand it the SHRUNK centre, not the
                # prediction it would otherwise default to. The chunk loop stores it.
                side["centre"] = centre_p
            # Enforce ``lower <= upper`` here, on the device, rather than in two numpy passes
            # at the end. This is the same guard for the same reason -- the scorer zeroes
            # **all five** subscores on a reversed interval, so the head is not trusted -- but
            # it is applied at the point where a reversal can first appear, i.e. directly
            # after the head, the anchor and the width rule, and it costs nothing on a device
            # that is otherwise idle. The host version allocated two more 103 MB arrays and
            # measured **0.373 ms/window**.
            lo_p, hi_p = torch.minimum(lo_p, hi_p), torch.maximum(lo_p, hi_p)
            return zero_p(rescale_reported(pred_p), lo_p, hi_p)

        def chunk(sl: slice):
            fast = _t1f["fast"]
            if fast is None or x_dev is None:
                return _chunk_ref(sl)
            out_dt = torch.float16 if store_dtype == np.float16 else torch.float32
            try:
                ub = scale_dev[sl].float()
                xb = x_dev[sl] / ub.view(-1, 1, 1, 1, 1)
                got = fast.run(xb, cond_dev[sl], ub, cond[sl], autocast_dtype, out_dt)
            except Exception:  # noqa: BLE001 - the reference chunk is always correct
                _t1f["fast"] = None
                return _chunk_ref(sl)
            if not _t1f["checked"]:
                _t1f["checked"] = True
                ref = _chunk_ref(sl)
                if got is None or not all(
                    torch.equal(a.to(out_dt), b) for a, b in zip(ref, got, strict=True)
                ):
                    _t1f["fast"] = None
                return ref
            if got is None:  # a non-finite fp16 forward: the reference recomputes in fp32
                return _chunk_ref(sl)
            on_device["rule"] = on_device["rescale"] = on_device["zero_p"] = True
            return got

        def store(key: str, block: torch.Tensor, start: int) -> None:
            """Copy one chunk off the device straight into its slice of the final buffer.

            ``block.cpu().numpy()`` followed by a slice assignment does the same work twice:
            it allocates a fresh host array, copies 103 MB into it, then copies it again into
            the output. Copying into a numpy view of the destination is one transfer to its
            final address -- **0.592 -> 0.326 ms/window** over 310 MB of output.

            Pinning the destination was measured and **rejected**: the transfer drops further
            to 0.224, but ``pin_memory=True`` on 310 MB costs 0.929 ms/window to allocate, and
            it has to happen inside the timed call because the sample count is not known until
            then. Net 1.154 against 0.326, i.e. 3.5x worse.
            """
            buf = out[key]
            if buf is None:
                # Reduced-precision runs stage the output in that dtype: the block was computed
                # in fp16/bf16 and only cast up to fp32 for this copy, so the fp32 form carries
                # no extra information. A half-width buffer halves the device->host transfer
                # (**1.19 -> 0.60 ms/window** over the three arrays) while keeping the
                # copy-into-view path below -- so there is no host-side widening pass to pay it
                # back. The widening happens once, lazily, in ``finalize`` at the very end.
                buf = np.empty((x.shape[0], *block.shape[1:]), dtype=store_dtype)
                out[key] = buf
            n = block.shape[0]
            if out_scale is not None:
                block = block * out_scale + out_shift
            src = block.to(torch.float16) if buf.dtype == np.float16 else block
            torch.from_numpy(buf[start : start + n]).copy_(src)

        with torch.inference_mode():
            # Chunk size is adaptive downward only. An OOM inside predict() raises to the
            # platform, which zeroes **all five** subscores -- so a batch that does not fit
            # the evaluation GPU must degrade to a slower run, never to no run. Halving
            # rather than stepping keeps the retry count logarithmic.
            start, size = 0, batch_size
            while start < x.shape[0]:
                try:
                    p, lo_t, hi_t = chunk(slice(start, min(start + size, x.shape[0])))
                    # Inside the try: ``store`` is where the device->host copy happens, and
                    # allocating the destination is itself a place the run can fail. Copying
                    # outside it would put that allocation beyond the OOM retry's reach.
                    store("prediction", p, start)
                    if lo_t is not None and hi_t is not None:
                        store("lower", lo_t, start)
                        store("upper", hi_t, start)
                    if side["centre"] is not None:
                        store("centre", side["centre"], start)
                        side["centre"] = None
                except torch.cuda.OutOfMemoryError:
                    if size == 1:
                        raise
                    size = max(1, size // 2)
                    torch.cuda.empty_cache()
                    continue
                start += p.shape[0]

        prediction = out["prediction"]
        assert prediction is not None  # x is non-empty, so at least one chunk ran
        if kmap_out is not None:
            kmap_out["log_ratio"] = np.concatenate(kmaps) if kmaps else None

        # ``out_transform`` was applied on the device, before the copy above. Everything in
        # the host tail below therefore reads fields in the CONSUMER's coordinates, not
        # physical ones -- which is correct for the steps that already ran on the device and
        # wrong for any that did not. Rather than carry two spellings of the tail (PROTOCOL 5's
        # standing trap), refuse the combination: the shipped Track-2 path runs the anchor, the
        # width rule and the ordering guard on the device and leaves nothing here but
        # ``zero_pressure``, which is exact under any map whose pressure shift is zero.
        if out_scale is not None:
            host_tail = {
                "lead_anchor": lead_anchor is not None and not on_device["anchor"],
                "width_rule": width_rule is not None and not on_device["rule"],
                "tke": (tke_head is not None or bool(tke_blend)) and not on_device["rescale"],
                "bound_scale": bound_scale != 1.0,
                "bound_floor": bound_floor > 0,
                "zero_pressure": zero_pressure
                and not on_device["zero_p"]
                and float(out_shift.reshape(-1)[-1]) != 0.0,
            }
            live = [k for k, v in host_tail.items() if v]
            if live:
                raise RuntimeError(
                    "out_transform maps the output off the device, so every later step must "
                    f"already have run there; these did not: {', '.join(live)}. Either drop "
                    "out_transform and map on the host, or move them on-device."
                )

        # ``prediction`` is what we REPORT; ``centre`` is the field the interval is built
        # around. They diverge only when a second-moment correction is active, and the two
        # moments want opposite treatment (FINDINGS 63j):
        #
        # * the **lead anchor** is a first-moment shift TOWARD the target, so the bounds follow
        #   it -- they end up centred on a better estimate;
        # * a **second-moment rescale** adds fluctuation *uncorrelated* with the target, so it
        #   moves the prediction AWAY from it. Carrying the bounds along drags them off the
        #   thing they are graded against, and `aggregate_sps` tests ``inside`` against the
        #   target, not against the prediction -- nothing requires the interval to contain what
        #   we report.
        #
        # Measured on the shipped recipe, val_extrap, one fixed configuration: recentring costs
        # **-3.87 sps** (coverage 0.8393 -> 0.7872) where leaving the interval in place banks
        # **+0.295** at efficiency 0.6904 against 0.6906. ``centre is prediction`` whenever no
        # rescale runs, so this is an exact no-op -- and one fewer array -- on every artefact
        # shipped to date (all carry ``TKE_BLEND=0.0`` and no head).
        #
        # Under ``centre_shrink`` the chunk copied the shrunk centre down whenever a step
        # below still reads it (``out["centre"]``); otherwise every centre-reading step
        # already ran on the device about it and ``prediction`` is never consulted as a centre.
        centre = prediction if out["centre"] is None else out["centre"]
        if on_device["rescale"]:
            # Already applied per chunk, on the device, with the bounds deliberately left on
            # the unrescaled centre. Nothing to do here, and in particular ``centre`` must stay
            # ``prediction``: the interval was already sized on the device against the
            # unrescaled field, so no host-side step may re-centre it.
            pass
        elif tke_head is not None:
            from .tke_head import (
                DEFAULT_SHRINK,
                apply_gain,
                inference_log_ratio,
            )

            r = inference_log_ratio(
                prediction,
                x[:, state_frames:],
                cond,
                head=tke_head,
                blend=_blend_of(tke_blend, tke_head),
                shrink=DEFAULT_SHRINK if tke_shrink is None else tke_shrink,
                device=device,
                uses_tau=tke_head_uses_tau,
            )
            prediction = apply_gain(prediction, r)
        elif tke_blend:
            prediction = rescale_fluctuations(prediction, x, blend=float(tke_blend))
        # ``centre is prediction`` unless a rescale ran above, which is what makes every
        # branch below a no-op on an artefact that ships no second-moment correction.
        rescaled = centre is not prediction

        anchor_shift = None
        if lead_anchor is not None and not on_device["anchor"]:
            if rescaled:
                # Applied to the two fields separately rather than reused as one displacement:
                # the anchor is a per-lead reweighting toward the last observed frame, so the
                # step it produces on the rescaled field is not the step it produces on the
                # unrescaled one, and it is the latter the interval must follow.
                anchored_centre = lead_anchor.apply(centre, x)
                anchor_shift = anchored_centre - centre
                centre = anchored_centre
                prediction = lead_anchor.apply(prediction, x)
            else:
                anchored = lead_anchor.apply(prediction, x)
                anchor_shift = anchored - prediction
                prediction = centre = anchored

        if zero_pressure and not on_device["zero_p"]:
            # Pressure is unmeasured in real data and never scored. Left on only for a
            # simulated evaluation that deliberately keeps p as a supervised channel.
            prediction[..., 2] = 0.0
            if rescaled:
                centre[..., 2] = 0.0

        if not return_bounds or out["lower"] is None:
            return prediction

        lower, upper = out["lower"], out["upper"]
        assert upper is not None  # written in the same branch as `lower`
        # Only the FIRST-moment displacement reaches the bounds. A second-moment rescale's
        # does not, by construction: it never enters ``anchor_shift``.
        if anchor_shift is not None:
            lower, upper = lower + anchor_shift, upper + anchor_shift
        if zero_pressure and not on_device["zero_p"]:
            lower[..., 2] = upper[..., 2] = 0.0
        # Every interval-shaping step reads ``centre``, not ``prediction``: the width rule was
        # fitted against the unrescaled field's realised error, and the floor and the scale are
        # offsets about the interval's own centre.
        if width_rule is not None and not on_device["rule"]:
            lower, upper = width_rule.apply(centre, lower, upper)
        elif bound_scale != 1.0:
            lower = centre - (centre - lower) * bound_scale
            upper = centre + (upper - centre) * bound_scale
        if bound_floor > 0:
            lower = np.minimum(lower, centre - bound_floor)
            upper = np.maximum(upper, centre + bound_floor)
        # Ordering is already enforced on the device (see ``chunk``), and every host-side step
        # between there and here preserves it: ``anchor_shift`` displaces both bounds equally,
        # ``zero_pressure`` sets both to zero, and ``bound_floor`` puts them on opposite sides
        # of ``centre`` by construction. Two steps do *not*, so they re-enforce it here rather
        # than at the cost of two 103 MB passes on every call: the host width rule, which is
        # free to return anything, and ``bound_scale``, which reflects a bound through the
        # centre if the head ever put it on the wrong side to begin with.
        if (width_rule is not None and not on_device["rule"]) or bound_scale != 1.0:
            lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        return {"prediction": prediction, "lower": lower, "upper": upper}

    # Exposed so a *caller* can demote too, not only a raising forward. ``warm_up_predict``
    # returning False means the compiled path's setup cost is still unpaid and would otherwise
    # land in the first scored call; dropping to eager there is cheaper than paying it timed.
    predict.demote_to_eager = lambda: use_compiled.__setitem__("on", False)  # type: ignore[attr-defined]
    predict.uses_compiled = lambda: bool(use_compiled["on"])  # type: ignore[attr-defined]
    # The inverse, so a caller can demote, *measure*, and put it back. Cannot enable a path that
    # does not exist: a failed ``torch.compile`` leaves ``compiled_model`` None and this is a
    # no-op, which is the same guarantee the sticky demotion gives.
    predict.restore_compiled = lambda: use_compiled.__setitem__(  # type: ignore[attr-defined]
        "on", compiled_model is not None
    )
    return predict


#: ``compile_mode`` value that captures the EAGER pipeline in a plain CUDA graph instead of
#: calling ``torch.compile``. No Inductor, no Triton, so no C compiler: FINDINGS 190 measured
#: that neither evaluation image has one, and that the Track-2 bundle's ``reduce-overhead``
#: compile raises on its first call there and serves eager.
CUDA_GRAPH_MODE = "cudagraph"


def cuda_graph_callable(fn: Callable[..., Any], warmup: int = 3) -> Callable[..., Any]:
    """Wrap ``fn(*args)`` -- tensors, ``None`` or tuples of tensors -- as a replayed CUDA graph.

    One graph per argument signature (shape, dtype, structure), captured on the first call with
    that signature after ``warmup`` side-stream runs, as ``torch.cuda.graphs`` requires. Every
    later call copies its arguments into the graph's static inputs, replays, and returns CLONES
    of the static outputs, because the caller keeps outputs across steps (the mean-flow residual)
    and the next replay would overwrite them. ``fn`` must be pure tensor work with no host sync;
    capture raises otherwise, and the caller's sticky demotion then serves eager.
    """
    graphs: dict[Any, Any] = {}

    def _flat(args):
        out, spec = [], []
        for a in args:
            if a is None:
                spec.append(None)
            elif isinstance(a, tuple):
                spec.append(tuple((t.shape, t.dtype) for t in a))
                out.extend(a)
            else:
                spec.append((a.shape, a.dtype))
                out.append(a)
        return out, tuple(spec)

    def _rebuild(args, flat):
        it = iter(flat)
        return tuple(
            None if a is None else tuple(next(it) for _ in a) if isinstance(a, tuple) else next(it)
            for a in args
        )

    def run(*args):
        flat, spec = _flat(args)
        g = graphs.get(spec)
        if g is None:
            static_in = [t.clone() for t in flat]
            static_args = _rebuild(args, static_in)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(max(1, warmup)):
                    fn(*static_args)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = fn(*static_args)
            g = graphs[spec] = (graph, static_in, static_out)
        graph, static_in, static_out = g
        for dst, src in zip(static_in, flat, strict=True):
            dst.copy_(src)
        graph.replay()
        return tuple(None if o is None else o.clone() for o in static_out)

    return run


def make_gpu_predict(
    model: torch.nn.Module,
    device: str | torch.device,
    width_rule: Any,
    lead_anchor: Any | None,
    cond_mode: str = "operating_point",
    compile_mode: str = "reduce-overhead",
    zero_pressure: bool = True,
    mean_bias_strength: float = 0.0,
    bias_warmup_n0: float = 2.0,
    bias_freeze_n: int = 8,
    mean_bias_smooth: float = 0.0,
    causal_gain_strength: float = 0.0,
    interval_history_spatial: float = 0.0,
    interval_history_global: float = 0.0,
    causal_anom: float = 0.0,
    tke_head: torch.nn.Module | None = None,
    tke_blend: float | None = None,
    tke_shrink: float | None = None,
    tke_head_uses_tau: bool = False,
    precision: str = "fp32",
    stream_state: str = "",
    state_freeze_n: int = 0,
    causal_v2: bool = False,
    causal_gain_floor: float = 0.0,
    interval_rho_blend: float = 0.0,
    interval_rho_ramp_n0: float = 0.0,
    interval_rho_shape: float = 1.0,
    interval_autocal_n0s: float = 0.0,
    interval_autocal_n0p: float = 50.0,
    interval_autocal_stage1: str = "scalar",
    interval_autocal_shift: float = 0.0,
    interval_autocal_grid: int = 29,
    interval_autocal_lead_sub: int = 1,
    interval_autocal_freeze_n: int = 0,
    interval_recal: str = "",
    interval_recal_overlap: bool = False,
    interval_recal_fused: bool = False,
    interval_recal_merged: bool = False,
    interval_recal_fork: bool = False,
    interval_recal_shift: float = 0.0,
    interval_recal_shift_n0: float = 5.0,
    pinned_io: bool = False,
    state_in_graph: bool = False,
) -> Callable[..., Any]:
    """Track-2 GPU-native ``predict(input_tensor, metadata) -> dict``, for the shipped config.

    ``stream_state`` serves a ``TrainConfig.stream_state`` backbone: this closure accumulates
    the trajectory's running (sum, sum-of-squares, count) over the evaluator-space windows it
    is handed -- the same pattern as the mean-flow bias below, cleared by the same per-
    trajectory ``reset()`` -- and hands the compiled pipeline the two MODEL-SPACE pseudo-frames
    as an ordinary graph input, so CUDA-graph capture cannot freeze them (the 102a rule). The
    second-moment head keeps reading the measured 20 frames only, matching how it is fitted.

    A profiler on the batch-1 step (FINDINGS 76b) puts ``cudaStreamSynchronize`` at 55% of the
    time: the sync-bound tail is the numpy input prep, the host->device copy of the window, the
    eager post-forward kernels each reading back, and the device->host copy of the output. This
    path removes all of it. The input tensor the harness already holds on the GPU stays there:
    the physical bridge, the freestream estimate, the normalisation, the forward, the
    denormalisation, the lead anchor, the width rule and the ordering guard run as ONE compiled
    CUDA graph, and only the final three arrays cross to the host, once.

    It is deliberately narrow -- it reproduces exactly the frozen Track-2 configuration
    (freestream normaliser, ``operating_point`` conditioning from the revealed metadata, a
    uniform-grid width rule, a lead anchor, a bounds head, optionally the second-moment
    head, no geometry/flow planes, the evaluator ``out_transform`` and fp16 wire dtype) and
    nothing else. The head runs on ``pred`` only and after the ordering guard, matching
    ``make_predict``'s ``rescale_reported``: the interval keeps the *unrescaled* centre it was
    fitted against (63j), and its constant feature planes are hoisted out of the graph by
    ``tke_static_planes`` so CUDA-graph capture cannot freeze them. Anything
    outside that must go through :func:`make_predict`, whose numpy tail remains the reference
    this path is checked against. The single graph is what the eager ``compile_mode`` on
    ``make_predict`` cannot reach, because the width rule's ``None``-return branch breaks it.

    ``precision="fp16"`` runs the pipeline under ``torch.autocast`` **inside** the compiled
    graph. This is the opposite call from the host path's fp32 default, and both are measured:
    eager at batch 1, autocast pays per-op cast kernels with no amortisation and fp16 came out
    1.1 ms *slower* -- but under one compiled graph the casts fuse and the batch-1 step is
    compute-bound on fp32 attention/conv/GEMM kernels (profiled 2026-09-15: fmha 0.84, the two
    stem/refine convs 1.28, GEMMs ~1.4 of a 4.4 ms T4 forward), so fp16 cut it to 1.69 ms.
    The tail past the forward stays fp32 either way: the pipeline upcasts the model's outputs
    before the denorm, and autocast does not touch the elementwise tail. Because a single
    non-finite value zeroes all five subscores and this path has no host-side recompute, every
    fp16 step is finiteness-checked on the device; a failure sticky-demotes to the eager fp32
    pipeline, mirroring the compiled-forward contract. bf16 is refused rather than accepted:
    ``torch.autocast`` raises for it on pre-sm_80 hardware, so a bf16 bundle could never be
    validated on the local T4 before shipping.
    """
    if state_in_graph:
        # Withdrawn at the 2026-09-23 merge: it was built on the generic ``cuda_graph_callable``
        # capture, and the board-verified cudagraph path is the manual capture in ``cpipe``
        # below, whose warm-up would fold the window into the state several extra times.
        raise ValueError("state_in_graph is not available on the merged cudagraph path")
    if precision not in ("fp32", "fp16"):
        raise ValueError(
            f"make_gpu_predict supports precision fp32 or fp16, got {precision!r} "
            "(bf16 cannot be validated on the local T4: autocast raises below sm_80)"
        )
    if float(getattr(model, "centre_shrink", 1.0)) != 1.0:
        # This pipeline sizes the interval about ``pred``; a model carrying a shrunk centre
        # (BoundedModel.centre_shrink, a Track-1 ``make_predict`` option) would have its rule
        # applied about the wrong field. Refused rather than served subtly wrong.
        raise ValueError("make_gpu_predict does not implement centre_shrink; use make_predict")
    from .. import N_MEASURED, SIGMA_GLOBAL  # noqa: PLC0415
    from ..features import (  # noqa: PLC0415 - the affine bridge + conditioning builder
        REAL_MEAN_IN,
        REAL_MEAN_TARGET,
        REAL_STD_IN,
        REAL_STD_TARGET,
        conditioning_features_batch,
    )

    dev = torch.device(device)
    c = N_MEASURED
    _mean_in = torch.as_tensor(np.asarray(REAL_MEAN_IN, np.float32), device=dev)
    _std_in = torch.as_tensor(np.asarray(REAL_STD_IN, np.float32), device=dev)
    _std_t = np.asarray(REAL_STD_TARGET, np.float32)
    _mean_t = np.asarray(REAL_MEAN_TARGET, np.float32)
    _out_scale = torch.as_tensor(1.0 / _std_t, device=dev)
    _out_shift = torch.as_tensor(-_mean_t / _std_t, device=dev)

    if width_rule is not None and not width_rule.is_uniform:
        # Same resample make_predict does at construction: a uniform grid turns the rule's bin
        # lookup into an arithmetic index, which is what lets the inlined gather below stay in
        # the compiled graph (a binary search would not).
        width_rule = width_rule.to_uniform()
    if not (getattr(width_rule, "is_uniform", False) and len(width_rule.rho_edges) >= 2):
        raise ValueError("make_gpu_predict needs a uniform-grid width rule")
    _step = float(width_rule.rho_edges[1] - width_rule.rho_edges[0])
    _r0 = float(width_rule.rho_edges[0])
    _nre = len(width_rule.rho_edges)
    _n_ch, _n_lead = width_rule.a_lo.shape[1], width_rule.a_lo.shape[2]
    _a_lo = torch.as_tensor(width_rule.a_lo.reshape(-1), device=dev, dtype=torch.float32)
    _a_up = torch.as_tensor(width_rule.a_up.reshape(-1), device=dev, dtype=torch.float32)
    _lead_edges = torch.as_tensor(np.asarray(width_rule.lead_edges), device=dev)
    _i_lead = torch.bucketize(torch.arange(T_IN, device=dev), _lead_edges).view(1, T_IN, 1, 1, 1)
    _i_ch = (torch.arange(_n_ch, device=dev) * _n_lead).view(1, 1, 1, 1, _n_ch)
    _sigma = float(SIGMA_GLOBAL)

    if tke_head is not None:
        from .tke_head import (  # noqa: PLC0415
            DEFAULT_SHRINK,
            apply_gain_torch,
            inference_log_ratio_torch,
        )

        _blend = _blend_of(tke_blend, tke_head)
        _shrink = DEFAULT_SHRINK if tke_shrink is None else float(tke_shrink)

    from .factory import (  # noqa: PLC0415 - one home for the frame table and phase estimator
        PHASE_MAX_FRAMES,
        STREAM_STATE_FAST_WINDOWS,
        phase_planes_torch,
        stream_state_frames,
    )
    from .factory import STATE_VAR_FLOOR as _VAR_FLOOR  # noqa: PLC0415

    _n_state = stream_state_frames(stream_state)
    _mode = stream_state or ""
    # ``state_freeze_n``: stop UPDATING the state after this many reveals and keep serving the
    # frozen statistics, exactly as ``bias_freeze_n`` does for the mean-flow correction. The
    # running mean/std are converged long before the stream ends (800+ frames by the last
    # window against 320 by reveal 16), so the accuracy cost should be nil -- but training
    # always sees the FULL history, so this is a train/serve mismatch and must be measured,
    # never assumed. 0 = never freeze.
    _freeze_n = int(state_freeze_n)

    def _pipeline(x_eval, cb, sp, acc=None, call=None):
        # The whole GPU tail as one graph: bridge -> freestream -> normalise -> forward ->
        # denormalise -> anchor -> width rule -> guard -> out_transform -> zero-pressure.
        # ``x_eval`` is always the wire's 20-frame window; ``acc`` (stream_state only) is the
        # trajectory's running ``(sum, sum-of-squares, frame count)`` over revealed frames, as
        # three per-step graph INPUTS -- so capture cannot freeze them (the 102a rule) and the
        # state frames are BUILT HERE, inside the graph, where Inductor fuses them into the
        # existing elementwise work and they reuse the freestream ``s`` this pipeline already
        # computes. The first version did this arithmetic eagerly around the graph and cost
        # ~0.5 ms/step in launches and a per-step allocation, against ~0.13 ms for the two
        # extra stem columns it exists to feed.
        x_phys = x_eval * _std_in + _mean_in
        s = x_phys[:, :, :FREESTREAM_ROWS, :, 0].abs().mean(dim=(1, 2, 3)).clamp_min(1e-6)
        mul = s.view(-1, 1, 1, 1, 1)
        xn = x_phys / mul
        if acc is not None:
            # Same construction as ``WindowDataset._state_frames`` and
            # ``FrozenPort._with_state``, built HERE so Inductor fuses it and it reuses the
            # freestream ``s`` this pipeline already computed. ``_mode`` is a python constant
            # at trace time, so each mode specialises into its own graph.
            a_sum, a_sq, a_n = acc[0], acc[1], acc[2]
            n = a_n.clamp_min(1.0)
            mean_e = a_sum / n
            var_e = (a_sq / n - mean_e * mean_e).clamp_min(0.0)
            div = s.view(1, 1, 1)
            std2 = _std_in[..., :2]
            mean_p = (mean_e * std2 + _mean_in[..., :2]) / div
            sd_p = (var_e.sqrt() * std2) / div
            nf = (a_n / 820.0).clamp_max(1.0).view(1, 1, 1).expand(*mean_p.shape[:2], 1)
            zero = torch.zeros_like(nf)
            frames = [torch.cat((mean_p, nf), dim=-1), torch.cat((sd_p, zero), dim=-1)]
            if _mode == "meanvarfast":
                f_sum, f_sq, f_n = acc[3], acc[4], acc[5]
                fn = f_n.clamp_min(1.0)
                fmean_e = f_sum / fn
                fvar_e = (f_sq / fn - fmean_e * fmean_e).clamp_min(0.0)
                frames.append(
                    torch.cat(((fmean_e * std2 + _mean_in[..., :2]) / div, zero), dim=-1)
                )
                frames.append(torch.cat(((fvar_e.sqrt() * std2) / div, zero), dim=-1))
            elif _mode == "meanvarcov":
                # Accumulated in float64 (``_update_acc``): the autocorrelation is a difference
                # of large sums and float32 cancellation would make it noise at the
                # near-constant pixels. Same arithmetic as ``covariance_stats_from_sums``.
                s_uv, s_w2, s_lag = acc[3], acc[4], acc[5]
                std64 = std2.double()
                cov_p = (s_uv / n - mean_e[..., 0] * mean_e[..., 1]) * (std64[0] * std64[1])
                var_p = var_e * std64 * std64
                mean_p64 = mean_e * std64 + _mean_in[..., :2].double()
                has = var_p > _VAR_FLOOR
                rho = (s_lag / (n - 1.0).clamp_min(1.0) - mean_p64 * mean_p64) / torch.where(
                    has, var_p, torch.ones_like(var_p)
                )
                rho = torch.where(has & (a_n > 1.0), rho.clamp(-1.0, 1.0), torch.zeros_like(rho))
                cov_f = (cov_p.sign() * cov_p.abs().sqrt()).unsqueeze(-1)
                vort = (s_w2 / n).clamp_min(0.0).sqrt().unsqueeze(-1)
                frames.append(
                    torch.cat((cov_f / div, vort / div, zero), dim=-1).to(nf.dtype)
                )
                frames.append(torch.cat((rho.to(nf.dtype), zero), dim=-1))
            elif _mode == "meanvarperr":
                p_sum, p_n = acc[3], acc[4]
                # An evaluator-space DIFFERENCE: the additive part of the bridge cancels, so
                # only the scale applies. Getting that wrong would bias the plane by mean_in.
                perr = (p_sum / p_n.clamp_min(1.0)) * std2 / div
                frames.append(torch.cat((perr, zero), dim=-1))
            elif _mode == "meanvarphase":
                # The two planes are already the finished, dimensionless phase/coherence maps
                # (``_update_acc`` recomputed them eagerly); just place them in front.
                frames.append(torch.cat((acc[3], zero), dim=-1))
                frames.append(torch.cat((acc[4], zero), dim=-1))
            stt = torch.stack([f.to(xn.dtype) for f in frames]).unsqueeze(0)
            xn = torch.cat((stt.to(xn.dtype), xn), dim=1)
        out = call_model(model, xn, cb, s) if call is None else call(xn, cb, s)
        pred, lo, hi = split_output(out)
        pred, lo, hi = pred.float() * mul, lo.float() * mul, hi.float() * mul  # freestream denorm
        if lead_anchor is not None:  # unanchored bundles (FINDINGS 119b) skip the shift
            anchored = lead_anchor.apply_torch(pred, x_phys[:, -1:])
            shift = anchored - pred
            pred, lo, hi = anchored, lo + shift, hi + shift
        p = pred[..., :c]
        below, above = p - lo[..., :c], hi[..., :c] - p
        half = 0.5 * (below + above)
        log_rho = torch.log(half.clamp_min(1e-30) / _sigma)
        i_rho = ((log_rho - _r0) / _step + 1.0).to(torch.int32).clamp_(0, _nre).long()
        flat = i_rho * (_n_ch * _n_lead) + _i_ch + _i_lead
        new_lo, new_hi = p - below * _a_lo[flat], p + above * _a_up[flat]
        lo2, hi2 = pred.clone(), pred.clone()
        lo2[..., :c] = torch.minimum(new_lo, new_hi)
        hi2[..., :c] = torch.maximum(new_lo, new_hi)
        if tke_head is not None:
            # Reported field only, and LAST before the transform -- the same order
            # ``rescale_reported`` uses, so the width rule and the guard above read the
            # unrescaled field the interval head was fitted against.
            r = inference_log_ratio_torch(
                pred,
                None,
                None,
                head=tke_head,
                blend=_blend,
                shrink=_shrink,
                uses_tau=tke_head_uses_tau,
                # The measured window only: the head is fitted on 20-frame maps, and the state
                # pseudo-frames (when present) sit in front.
                input_device=(xn[:, _n_state:][..., :c], s),
                static_planes=sp,
            )
            pred = apply_gain_torch(pred, r)
        pred = pred * _out_scale + _out_shift
        lo2 = lo2 * _out_scale + _out_shift
        hi2 = hi2 * _out_scale + _out_shift
        if zero_pressure:
            pred[..., 2] = 0.0
            lo2[..., 2] = 0.0
            hi2[..., 2] = 0.0
        return pred.to(torch.float16), lo2.to(torch.float16), hi2.to(torch.float16)

    # fp16 is CUDA-only, exactly as in ``make_predict``: autocast("cuda") on a CPU build is at
    # best a no-op and the CPU path exists only for tests and smoke runs.
    _amp = torch.float16 if precision == "fp16" and str(dev).startswith("cuda") else None

    def _pipeline_amp(x_eval, cb, sp, acc=None):
        # Autocast around the SAME pipeline, not a second spelling of it (PROTOCOL 5): only the
        # forward's matmul/conv/attention ops change dtype, and ``_pipeline`` already upcasts
        # the model outputs to fp32 before the denorm, so the interval tail is bit-for-bit the
        # fp32 code path either way.
        with torch.autocast("cuda", dtype=_amp):
            return _pipeline(x_eval, cb, sp, acc)

    # Holder for the merged v2 post-hoc pipeline, populated after ``extra_history`` is built
    # (below). ``_src`` dispatches through it so the manual CUDA-graph capture records the v2
    # observe + apply inside the ONE main pipeline graph -- the compiler-free way to make the
    # history-conditioned post-hoc free on time.
    _v2m: dict[str, Any] = {"pipe": None, "state": None}
    _ref_src = _pipeline_amp if _amp is not None else _pipeline

    # The same pipeline over ``fast_serve``'s forward: every function of ``cond`` alone computed
    # once per operating point, and the element-wise chains fused into single kernels that do
    # the same IEEE operations in the same order (FINDINGS 219). Only the CUDA-graph path uses
    # it, and only after the first call has checked it BIT-IDENTICAL to ``_ref_src`` on that
    # call's own inputs (``_fast_prepare``); any failure there serves the reference instead.
    _fast = _fast_full = None
    if compile_mode == "cudagraph" and str(dev).startswith("cuda"):
        from . import fast_serve  # noqa: PLC0415

        if fast_serve.pipeline_supported(
            model,
            lead_anchor=lead_anchor,
            tke_head=tke_head,
            tke_head_uses_tau=tke_head_uses_tau,
            stream_state=_mode,
            zero_pressure=zero_pressure,
            n_ch=_n_ch,
        ):
            # The whole pipeline, for the board bundle's configuration.
            _fast_full = fast_serve.FastPipeline(
                model,
                std_in=_std_in,
                mean_in=_mean_in,
                out_scale=_out_scale,
                out_shift=_out_shift,
                sigma=_sigma,
                r0=_r0,
                step=_step,
                nre=_nre,
                n_lead=_n_lead,
                a_lo=_a_lo,
                a_up=_a_up,
                i_ch=_i_ch,
                i_lead=_i_lead,
                tke_head=tke_head,
                blend=_blend,
                shrink=_shrink,
                fs_rows=FREESTREAM_ROWS,
                half_weights=_amp is not None,
                check_finite=_amp is not None,
            )
        elif fast_serve.supported(model):
            # The model only, inside the reference pipeline.
            _fast = fast_serve.FastBounded(model)
    _fsw: dict[str, Any] = {
        "on": _fast is not None or _fast_full is not None,
        "checked": False,
        "entry": None,
    }

    def _fast_pipeline(x_eval, cb, sp, acc=None):
        if _fast_full is not None:
            if _amp is not None:
                with torch.autocast("cuda", dtype=_amp):
                    return _fast_full(x_eval, cb, sp, acc)
            return _fast_full(x_eval, cb, sp, acc)
        if _amp is not None:
            with torch.autocast("cuda", dtype=_amp):
                return _pipeline(x_eval, cb, sp, acc, call=_fast)
        return _pipeline(x_eval, cb, sp, acc, call=_fast)

    def _fast_prepare(xt, cb, sp, acc, entry) -> None:
        """Refresh the conditioning buffers when the operating point changes; check once."""
        try:
            if _fsw["entry"] is not entry:
                ctx = (
                    torch.autocast("cuda", dtype=_amp)
                    if _amp is not None
                    else contextlib.nullcontext()
                )
                with ctx:
                    if _fast_full is not None:
                        _fast_full.set_cond(cb, sp, xt)
                    else:
                        _fast.set_cond(cb)
                _fsw["entry"] = entry
            if not _fsw["checked"]:
                _fsw["checked"] = True
                ref, got = _ref_src(xt, cb, sp, acc), _fast_pipeline(xt, cb, sp, acc)
                if not all(torch.equal(a, b) for a, b in zip(ref, got, strict=True)):
                    _fsw["on"] = False
        except Exception:  # noqa: BLE001 - the reference pipeline is always correct
            _fsw["on"] = False

    def _base_src(x_eval, cb, sp, acc=None):
        if _fsw["on"]:
            return _fast_pipeline(x_eval, cb, sp, acc)
        return _ref_src(x_eval, cb, sp, acc)

    def _src(x_eval, cb, sp, acc=None):
        pipe = _v2m["pipe"]
        if pipe is not None:
            return pipe(x_eval, cb, sp, acc)
        return _base_src(x_eval, cb, sp, acc)

    # ``torch.compile`` can fail at CONSTRUCTION, not only in the forward: on the eval platform
    # the participant runs in a daemonic subprocess, and inductor's parallel-compile worker pool
    # raises "daemonic processes are not allowed to have children" the moment it is set up. The
    # bundle sets ``TORCHINDUCTOR_COMPILE_THREADS=1`` to avoid the pool, but this construction is
    # still wrapped so ANY compile/inductor failure demotes to the eager pipeline rather than
    # killing the run at import. ``_pipeline`` is always correct -- it is what compile was built
    # from -- so the only cost of a demotion is the fusion speedup (and, under fp16, the reduced
    # precision: the demotion target is the eager FP32 pipeline, the reference).
    # ``compile_mode="cudagraph"`` is the compiler-FREE overhead-removal path (see
    # CUDAGRAPH_WARMUP). The eval image has no C compiler, so ``torch.compile`` silently
    # demotes to eager there (Triton's launcher build raises "Failed to find C compiler");
    # ``reduce-overhead`` thus never actually captured a graph on the board. A manual
    # ``torch.cuda.CUDAGraph`` capture of the *eager* pipeline needs only the CUDA driver
    # (injected by ``--gpus``), so it removes the batch-1 launch overhead the compiled path
    # was supposed to and could not.
    _cudagraph = compile_mode == "cudagraph"
    _compiled = _pipeline
    if compile_mode and not _cudagraph:
        try:
            _compiled = torch.compile(_src, mode=compile_mode)
        except Exception:  # noqa: BLE001 - eager is always correct
            _compiled = _pipeline
    elif _amp is not None and not _cudagraph:
        # No compile requested but fp16 was: run the autocast pipeline eagerly. Reachable only
        # in experiments -- the packaged fp16 path always compiles -- but it must not silently
        # serve fp32 under a recorded fp16 precision (the FINDINGS 102a failure class).
        _compiled = _pipeline_amp
    # Sticky demote on any failure of the compiled forward too, mirroring ``make_predict``'s
    # ``use_compiled`` contract.
    _use = {"compiled": _compiled is not _pipeline}

    def cpipe(xt, cb, sp, acc=None):
        if _use["compiled"]:
            try:
                return _compiled(xt, cb, sp, acc)
            except Exception:  # noqa: BLE001 - eager is always correct
                _use["compiled"] = False
        return _pipeline(xt, cb, sp, acc)

    if _cudagraph and str(dev).startswith("cuda"):
        # Manual capture of the eager ``_src`` (``_pipeline``, or ``_pipeline_amp`` under fp16).
        # Static buffers for the per-step inputs (``xt``/``cb``/``sp``); ``acc`` is already the
        # preallocated static-address tuple ``_update_acc`` mutates in place, so the graph reads
        # this step's freshly-folded statistics on every replay. Any failure demotes, once and
        # for good, to the eager pipeline -- never wrong, only slower.
        _cg: dict[str, Any] = {"g": None, "warm": 0, "on": True,
                               "sx": None, "scb": None, "ssp": None, "out": None}

        def cpipe(xt, cb, sp, acc=None):  # noqa: F811 - the cudagraph specialisation
            if not _cg["on"]:
                # Demoted (by a capture/replay failure or by the fp16 guard below): the eager
                # fp32 reference, which needs no guard. ``_src`` would be unguarded eager fp16.
                return _pipeline(xt, cb, sp, acc)
            if _cg["g"] is None:
                if _cg["warm"] < CUDAGRAPH_WARMUP:
                    _cg["warm"] += 1
                    return _src(xt, cb, sp, acc)
                # The stream-state fold inside the graph (``_acc_graph_intent``): ``predict`` has
                # NOT folded this window, so snapshot the accumulators, let the warm-up and
                # capture calls fold as they like, restore, and let this step's replay fold once.
                acc_g = acc is not None and _acc_graph_intent()
                acc_snap = [a.clone() for a in acc] if acc_g else None

                def _cap(x, c, p):
                    if acc_g:
                        _acc_fold(x, acc)
                    return _src(x, c, p, acc)

                try:
                    with torch.inference_mode(False):
                        _cg["sx"] = xt.detach().clone()
                        _cg["scb"] = cb.detach().clone()
                        _cg["ssp"] = None if sp is None else sp.detach().clone()
                    # The merged v2 path folds ``observe`` (which MUTATES the history buffers)
                    # into the captured graph, so the warm-up + capture calls below would fold
                    # ``sx`` several extra times. Snapshot the history buffers first and restore
                    # them after capture, so only the real replays (one fold per step) advance
                    # the trajectory state.
                    _snap = (
                        [b.clone() for b in _v2m["state"]]
                        if _v2m["state"] is not None
                        else None
                    )
                    cur = torch.cuda.current_stream()
                    side = torch.cuda.Stream()
                    side.wait_stream(cur)
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            _cap(_cg["sx"], _cg["scb"], _cg["ssp"])
                    cur.wait_stream(side)
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        _cg["out"] = _cap(_cg["sx"], _cg["scb"], _cg["ssp"])
                    _cg["g"] = gr
                    _cg["acc"] = acc_g
                    if acc_g:
                        for _a, _s in zip(acc, acc_snap, strict=True):
                            _a.copy_(_s)
                    # The full fast pipeline reads its own conditioning buffers, never the
                    # static cb/sp copies, so a graph captured over it does not need them.
                    _cg["fast"] = bool(_fsw["on"] and _fast_full is not None)
                    if _snap is not None:
                        for _b, _s in zip(_v2m["state"], _snap, strict=True):
                            _b.copy_(_s)
                except Exception:  # noqa: BLE001 - eager is always correct
                    _cg["on"] = False
                    if acc_g:
                        for _a, _s in zip(acc, acc_snap, strict=True):
                            _a.copy_(_s)
                        _acc_fold(xt, acc)
                    return _pipeline(xt, cb, sp, acc)
            try:
                if xt is not _cg["sx"]:
                    _cg["sx"].copy_(xt)
                if not _cg.get("fast"):
                    _cg["scb"].copy_(cb)
                    if _cg["ssp"] is not None:
                        _cg["ssp"].copy_(sp)
                _cg["g"].replay()
                return _cg["out"]
            except Exception:  # noqa: BLE001 - eager is always correct
                _cg["on"] = False
                if _cg.get("acc"):
                    _acc_fold(xt, acc)
                return _pipeline(xt, cb, sp, acc)

    # The conditioning vector is a pure function of (Re, AoA), which the port freezes after its
    # warm-up (FrozenPort.OP_WARMUP), so it is constant for all but the first few steps of a
    # trajectory. Cache it keyed on the operating point rather than rebuild it (numpy +
    # host->device copy) every step.
    _cond_cache: dict[tuple[float, float], Any] = {}

    # Per-trajectory MEAN-FLOW bias (FINDINGS 77), the one adaptation that stacks on drop-path.
    # At stride 20 the current input window IS the previous window's revealed target, so the
    # signed residual ``input - prev_prediction``, averaged over the lead axis, is this
    # trajectory's first-moment error -- a single (H, W, C) field. Added identically to every
    # frame it is a pure mean-flow correction: it moves mvpe/rel_l2 and leaves temporal variance
    # (tke) untouched, which is why it composes with the drop-path tke lever rather than fighting
    # it. All on the device, so the GPU-native step keeps its time. Estimate frozen after
    # ``bias_freeze_n`` windows (it saturates early), shrunk by ``lam = s*n/(n+n0)``.
    st: dict[str, Any] = {
        "prev": None,
        "sum_res": None,
        "n": 0,
        "bias": None,
        # ``stream_state`` accumulators: the running (sum, sum-of-squares, count) tuple, the
        # recent-reveal ring for ``meanvarfast``, the previous window for ``meanvarperr``, and
        # how many reveals have been folded in (which ``state_freeze_n`` reads).
        "acc": None,
        "recent": [],
        "prev_w": None,
        "st_reveals": 0,
        # ``meanvarphase``: the retained revealed frames (evaluator space) and their count. The
        # shedding mode is recomputed from them each step (the frequency is estimated online),
        # eager and off the graph; the two finished planes ride into the graph as ``acc[3:5]``.
        "phase_hist": None,
        "phase_nrev": 0,
    }

    def _smooth_bias(bias):
        """Selected LTTA regulariser: blend raw bias with two 3x3 box passes (triangle)."""
        if not mean_bias_smooth:
            return bias
        b = bias.squeeze(1).permute(0, 3, 1, 2).float()
        smooth = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(b, (1, 1, 1, 1), mode="replicate"), 3, stride=1
        )
        smooth = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(smooth, (1, 1, 1, 1), mode="replicate"), 3, stride=1
        )
        smooth = smooth.permute(0, 2, 3, 1).unsqueeze(1)
        return (1.0 - mean_bias_smooth) * bias + mean_bias_smooth * smooth

    extra_history = None
    _v2_merged = False
    _v2_only = causal_gain_floor or interval_rho_blend
    if _v2_only and not causal_v2:
        raise ValueError(
            "causal_gain_floor / interval_rho_blend exist only in the v2 causal module; "
            "pass causal_v2=True to select it"
        )
    if (
        causal_gain_strength
        or interval_history_spatial
        or interval_history_global
        or causal_anom
        or _v2_only
    ):
        raise ValueError("the causal-history serving paths (models/causal_history*.py) is not included in this package")

    # Self-calibrating intervals (FINDINGS 192, ``models.interval_autocal``): per-trajectory
    # width multipliers picked by the SPS credit the revealed windows would have earned. Off
    # unless ``interval_autocal_n0s`` > 0; it re-sizes the FINAL target-space bounds, after every
    # other correction, and never touches the prediction.
    autocal = None
    if interval_autocal_n0s > 0:
        raise ValueError("interval_autocal (models/interval_autocal.py) is not included in this package")
    if interval_recal:
        # The LEARNED twin (``models.interval_recal``): same wire, same place, same serving
        # hook below. Mutually exclusive with the closed-form calibrator -- both re-size the
        # final bounds and neither was measured on top of the other.
        if autocal is not None:
            raise ValueError("interval_recal and interval_autocal_n0s > 0 are exclusive")
        from .interval_recal import load_calibrator, make_interval_recal  # noqa: PLC0415

        if interval_recal_shift and not (interval_recal_merged and interval_recal_fused):
            raise ValueError("interval_recal_shift needs the fused, merged calibrator step")
        autocal = make_interval_recal(
            load_calibrator(interval_recal, dev),
            dev,
            fused=interval_recal_fused,
            shift_beta=interval_recal_shift,
            shift_n0=interval_recal_shift_n0,
            out_scale=_out_scale,
            out_shift=_out_shift,
            in_mean=_mean_in,
            in_std=_std_in,
            c=c,
        )
    # ``interval_recal_merged``: the calibrator's observe + apply captured INSIDE the one main
    # pipeline graph (the ``_v2m`` mechanism the v2 history uses), instead of as its own graph
    # around it. What that buys against the separate-graph ``_rc`` path: no second graph launch,
    # no stream join, and -- because ``_rc``/``_ac_step`` stay None -- the fast path's direct
    # static-input copy and in-graph finiteness flag stay on. One graph serves every step:
    # ``observe`` is gated by the calibrator's own has-previous scalar (zero after ``reset``), so
    # a trajectory's first call leaves the state untouched with no host branch, and ``apply_``
    # writes into the pipeline's output buffers. ``interval_recal_fork`` runs the observe on a
    # forked stream inside the capture, so the graph holds it as a branch parallel to the
    # backbone rather than serial with it.
    _rcm = None
    if interval_recal and interval_recal_merged:
        if not (_cudagraph and str(dev).startswith("cuda")):
            raise ValueError("interval_recal_merged needs compile_mode='cudagraph' on CUDA")
        if _v2m["pipe"] is not None:
            raise ValueError("interval_recal_merged and the merged v2 history are exclusive")
        if not hasattr(autocal, "step_merged"):
            raise ValueError("interval_recal_merged needs the fast (GRU / no-context) step")
        _rcm = {"side": torch.cuda.Stream(device=dev) if interval_recal_fork else None}

        def _rcm_pipeline(x_eval, cb, sp, acc=None):
            side = _rcm["side"]
            if side is not None:
                cur = torch.cuda.current_stream(dev)
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    autocal.observe_gated(x_eval)
                pred, lo, hi = _base_src(x_eval, cb, sp, acc)
                cur.wait_stream(side)
            else:
                autocal.observe_gated(x_eval)
                pred, lo, hi = _base_src(x_eval, cb, sp, acc)
            autocal.apply_(lo, hi)
            if _fast_full is not None:
                # The pipeline's finiteness flag was reduced before the re-size: redo it over
                # the final buffers so the guard covers the calibrator too.
                flag = _fast_full.finite_flag(pred, lo, hi)
                if flag is not None:
                    torch.all(torch.isfinite(_fast_full.stacked(pred, lo, hi)), out=flag)
            return pred, lo, hi

        _v2m["pipe"] = _rcm_pipeline
        _v2m["state"] = autocal.state_buffers()
    # One serving step (observe the revealed target, re-size the bounds) on fixed buffers, so
    # under CUDA_GRAPH_MODE it replays as its own graph instead of ~30 eager launches. The
    # capture's warm-up runs pollute the state; the port's per-trajectory reset clears it.
    _ac_step = None
    if autocal is not None and _rcm is None:
        _graphed = compile_mode == CUDA_GRAPH_MODE and dev.type == "cuda"
        _ac_fns = {
            "full": (cuda_graph_callable(autocal.step) if _graphed else None, autocal.step),
            "frozen": (
                cuda_graph_callable(autocal.step_frozen) if _graphed else None,
                autocal.step_frozen,
            ),
        }
        if hasattr(autocal, "step_first"):
            # The learned calibrator: a trajectory's first call has nothing revealed, and is
            # served by an apply-only graph chosen below by the host count.
            _ac_fns["first"] = (
                cuda_graph_callable(autocal.step_first) if _graphed else None,
                autocal.step_first,
            )
        _ac_use = {"graph": _graphed, "captured": False}
        _freeze = int(interval_autocal_freeze_n)

        def _ac_step(x, lo, hi):
            # Capture BOTH graphs on the very first call -- the template's untimed warm-up, whose
            # state the first trajectory's reset discards. Capture runs a step several times, so
            # capturing the frozen graph lazily would double-count one trajectory's evidence.
            if _ac_use["graph"] and not _ac_use["captured"]:
                _ac_use["captured"] = True
                try:
                    for graph, _ in _ac_fns.values():
                        graph(x, lo, hi)
                except Exception:  # noqa: BLE001 - eager is always correct
                    _ac_use["graph"] = False
                autocal.reset()
            # Host count of steps since reset: call k reveals window k-1, so the tables take
            # windows 0..freeze-1 (calls up to freeze+1) and every later step runs the cheap
            # frozen graph.
            st["ac_calls"] = st.get("ac_calls", 0) + 1
            which = "frozen" if _freeze and st["ac_calls"] > _freeze + 1 else "full"
            if "first" in _ac_fns and st["ac_calls"] == 1:
                which = "first"
            graph, eager = _ac_fns[which]
            # Sticky demotion, as for the main pipeline: a raise here would zero every subscore.
            # A failed capture may have touched the state, so the trajectory restarts its
            # calibration from the shipped widths -- slower to adapt, never wrong.
            if _ac_use["graph"]:
                try:
                    return graph(x, lo, hi)
                except Exception:  # noqa: BLE001 - eager is always correct
                    _ac_use["graph"] = False
                    autocal.reset()
            return eager(x, lo, hi)

    # The learned calibrator's update needs only the revealed target (this input) and the
    # previous step's bounds, so under CUDA_GRAPH_MODE it runs as its own graph on a SIDE stream
    # while the backbone runs, and only the width multiply waits for it (FINDINGS 208). At batch 1
    # the backbone leaves the GPU partly idle, which is what the overlap spends. Any failure
    # demotes, for good, to the serial ``_ac_step`` path above -- slower, never wrong.
    _rc = None
    if (
        interval_recal
        and interval_recal_overlap
        and _rcm is None
        and compile_mode == CUDA_GRAPH_MODE
        and dev.type == "cuda"
    ):

        def _rc_observe(x):
            autocal.observe(x)
            return ()

        _rc = {
            "side": torch.cuda.Stream(device=dev),
            "obs": cuda_graph_callable(_rc_observe),
            "app": cuda_graph_callable(autocal.apply),
            "live": True,
            "captured": False,
            "launched": False,
        }

    def _rc_launch(x) -> None:
        """Start this step's calibrator update on the side stream (before the backbone)."""
        _rc["launched"] = False
        st["rc_calls"] = st.get("rc_calls", 0) + 1
        if not _rc["live"] or st["rc_calls"] == 1:  # first call: nothing revealed yet
            return
        try:
            cur = torch.cuda.current_stream(dev)
            if not _rc["captured"]:
                # Capture the update on the very first call, as ``_ac_step`` does: capture runs it
                # several times, so the state is reset after. ``apply`` is idempotent (it only
                # re-reads the state and overwrites the remembered bounds), so it captures lazily.
                _rc["captured"] = True
                _rc["obs"](x)
                torch.cuda.synchronize(dev)
                autocal.reset()
            _rc["side"].wait_stream(cur)
            with torch.cuda.stream(_rc["side"]):
                _rc["obs"](x)
            x.record_stream(_rc["side"])
            _rc["launched"] = True
        except Exception:  # noqa: BLE001 - the serial eager path is always correct
            _rc["live"] = False
            torch.cuda.synchronize(dev)
            autocal.reset()

    def _rc_finish(x, lo, hi):
        """Join the side stream and re-size the bounds; the serial path if the launch failed."""
        if _rc["launched"]:
            try:
                torch.cuda.current_stream(dev).wait_stream(_rc["side"])
                return _rc["app"](lo, hi)
            except Exception:  # noqa: BLE001
                _rc["live"] = False
                torch.cuda.synchronize(dev)
                autocal.reset()
        if _rc["live"] and st.get("rc_calls") == 1:
            return _rc["app"](lo, hi)
        return _ac_step(x, lo, hi)

    def _reset() -> None:
        st["ac_calls"] = 0
        st["rc_calls"] = 0
        if autocal is not None:
            autocal.reset()
        if extra_history is not None:
            extra_history.reset()
        st["prev"] = None
        st["sum_res"] = None
        st["n"] = 0
        st["bias"] = None
        st["recent"] = []
        st["prev_w"] = None
        st["st_reveals"] = 0
        st["phase_nrev"] = 0  # keep the buffer allocated across trajectories; just rewind it
        # Merged v2 path: the first served step of a trajectory has no revealed target, so it
        # runs eager (no observe/apply) and just seeds ``_g_prev``; steps 2+ use the graph.
        st["v2_first"] = True
        if st.get("acc") is None:
            st["acc"] = None
        else:
            # Preallocated: the per-trajectory reset is O(1) and allocation-free after the
            # first trajectory, which matters because Track 2 charges reset to the next step.
            for t in st["acc"]:
                t.zero_()

    _reset()

    def _acc_fold(xt, acc):
        """The ``meanvar`` fold of ``_update_acc``, the same ops, for the captured graph."""
        wf = xt[0, ..., :2].float()
        acc[0].add_(wf.sum(0))
        acc[1].add_((wf * wf).sum(0))
        acc[2].add_(float(xt.shape[1]))

    def _acc_graph_intent() -> bool:
        """Fold the stream state INSIDE the CUDA graph instead of as ~6 eager launches before
        it (FINDINGS 219): only on the fast pipeline, the plain ``meanvar`` state, no freeze."""
        return bool(
            _fast_full is not None
            and _fsw["on"]
            and _mode == "meanvar"
            and not _freeze_n
            and st["acc"] is not None
        )

    def _graph_folds_acc() -> bool:
        """Will this step's cpipe fold the state itself (a replay of, or the capture of, a graph
        with the fold)? Then ``predict`` must not fold it eagerly."""
        if not (_cudagraph and str(dev).startswith("cuda") and _cg["on"]):
            return False
        if _cg["g"] is not None:
            return bool(_cg.get("acc"))
        return _cg["warm"] >= CUDAGRAPH_WARMUP and _acc_graph_intent()

    def _update_acc(xt):
        """Fold this window into the trajectory's running statistics; return the graph inputs.

        All the arithmetic runs INSIDE the compiled pipeline; what is left per step is a few
        in-place reductions on PREALLOCATED buffers -- no per-step allocation, no host round
        trip for the count, and no second freestream estimate. Accumulators stay in evaluator
        space, which is what the wire delivers, so no bridge runs here either.
        """
        w = xt[0, ..., :2]
        if st["acc"] is None:
            # Outside inference mode on purpose: an inference tensor cannot be zeroed in place
            # by ``_reset``, which the evaluator calls between trajectories.
            with torch.inference_mode(False):
                shape = tuple(w.shape[1:])
                extra = {
                    "meanvarfast": (shape, shape, ()),
                    "meanvarperr": (shape, ()),
                    # uv sum, vorticity^2 sum, lag-1 product sum, previous window's last frame
                    "meanvarcov": (shape[:-1], shape[:-1], shape, shape),
                    # the two finished (H, W, 2) phase planes, recomputed eagerly each step and
                    # read straight through by the pipeline (static addresses = graph inputs).
                    "meanvarphase": (shape, shape),
                }
                acc_dtype = torch.float64 if _mode == "meanvarcov" else torch.float32
                st["acc"] = tuple(
                    torch.zeros(sh, device=xt.device, dtype=acc_dtype)
                    for sh in ((shape, shape, ()) + extra.get(_mode, ()))
                )
            # The acc buffers are PREALLOCATED and updated in place (``.add_`` below), then
            # passed as inputs to the compiled ``_pipeline``. cudagraph-trees skips capture when
            # a graph input is mutated between replays ("skipping cudagraphs due to mutated
            # inputs"); marking their addresses static tells it these are stable buffers updated
            # in place (the KV-cache pattern), so the batch-1 graph captures and the launch
            # overhead -- the dominant batch-1 cost -- is removed. Zeroed in place by ``_reset``,
            # so the address survives across trajectories and the capture stays valid.
            for _t in st["acc"]:
                torch._dynamo.mark_static_address(_t)
        acc = st["acc"]
        if _freeze_n and st["st_reveals"] >= _freeze_n:
            return acc  # frozen: the eager reductions below are the cost this removes
        st["st_reveals"] += 1
        wf = w.double() if _mode == "meanvarcov" else w.float()
        wsum, wsq = wf.sum(0), (wf * wf).sum(0)
        acc[0].add_(wsum)
        acc[1].add_(wsq)
        acc[2].add_(float(xt.shape[1]))
        if _mode == "meanvarfast":
            st["recent"].append((wsum, wsq))
            del st["recent"][:-STREAM_STATE_FAST_WINDOWS]
            acc[3].copy_(sum(r[0] for r in st["recent"]))
            acc[4].copy_(sum(r[1] for r in st["recent"]))
            acc[5].fill_(float(len(st["recent"]) * xt.shape[1]))
        elif _mode == "meanvarcov":
            std64 = _std_in[..., :2].double()
            # Physical vorticity from evaluator-space frames: the additive bridge cancels in a
            # gradient, so only the per-channel scale applies.
            omega = std64[1] * torch.gradient(wf[..., 1], dim=2)[0] - std64[0] * torch.gradient(
                wf[..., 0], dim=1
            )[0]
            acc[3].add_((wf[..., 0] * wf[..., 1]).sum(0))
            acc[4].add_((omega * omega).sum(0))
            # The lag-1 product is accumulated in PHYSICAL units: it averages n-1 pairs against
            # a mean over n frames, so it is not invariant to the evaluator-space offset (which
            # is large against the fluctuation) and must see the same field the dataset does.
            xp = wf * std64 + _mean_in[..., :2].double()
            if st["st_reveals"] > 1:  # chain the lag-1 product across the window boundary
                acc[5].add_(acc[6] * xp[0])
            acc[5].add_((xp[1:] * xp[:-1]).sum(0))
            acc[6].copy_(xp[-1])
        elif _mode == "meanvarperr":
            if st["prev_w"] is not None:
                acc[3].add_((wf - st["prev_w"]).abs().sum(0))
                acc[4].add_(float(xt.shape[1]))
            st["prev_w"] = wf.clone()
        elif _mode == "meanvarphase":
            # Append this window to the retained history and recompute the shedding mode from the
            # whole revealed span (the frequency is estimated online, so it sharpens as history
            # grows). Eager and off the graph; the two finished planes land in the static-address
            # buffers ``acc[3:5]`` the captured pipeline reads. The history buffer is preallocated
            # once (outside inference mode, so ``_reset`` can rewind it) and frozen past the cap.
            nfr = xt.shape[1]
            if st["phase_hist"] is None:
                with torch.inference_mode(False):
                    st["phase_hist"] = torch.zeros(
                        (PHASE_MAX_FRAMES, *w.shape[1:]), device=xt.device, dtype=torch.float32
                    )
            hb, k = st["phase_hist"], st["phase_nrev"]
            if k + nfr <= PHASE_MAX_FRAMES:
                hb[k : k + nfr].copy_(wf)
                st["phase_nrev"] = k + nfr
            planes = phase_planes_torch(hb[: st["phase_nrev"]])
            acc[3].copy_(planes[0].to(acc[3].dtype))
            acc[4].copy_(planes[1].to(acc[4].dtype))
        return acc

    # ``pinned_io`` (FINDINGS 194): stage the batch-1 window through a persistent PINNED host buffer
    # on the way in and the three outputs through another on the way out. A pageable copy makes
    # the driver stage it itself, synchronously: measured 0.189 -> 0.023 ms in and 0.275 -> 0.176 ms
    # out on the T4. The returned array is still a FRESH copy every step (the caller may keep it),
    # and every step still ends in a stream sync, so the next step's staging never races a copy.
    _pin: dict[str, Any] = {"in": None, "out": None}

    def _to_device(input_array):
        src = input_array if isinstance(input_array, torch.Tensor) else None
        if src is None:
            src = torch.from_numpy(np.ascontiguousarray(input_array))
        if not pinned_io or src.is_cuda or dev.type != "cuda":
            return torch.as_tensor(input_array, device=dev).float()
        src = src.float()
        buf = _pin["in"]
        if buf is None or buf.shape != src.shape:
            buf = _pin["in"] = torch.empty(src.shape, dtype=torch.float32).pin_memory()
        buf.copy_(src)
        return buf.to(dev, non_blocking=True)

    def _stack3(pred, lo, hi):
        """``torch.stack((pred, lo, hi))``, or the fast pipeline's own stacked output buffer when
        the three are its views (the same bytes, one copy fewer)."""
        if _fast_full is not None:
            base = _fast_full.stacked(pred, lo, hi)
            if base is not None:
                return base
        return torch.stack((pred, lo, hi))

    def _to_host(stacked):
        if not pinned_io or dev.type != "cuda":
            return stacked.cpu().numpy()
        buf = _pin["out"]
        if buf is None or buf.shape != stacked.shape or buf.dtype != stacked.dtype:
            buf = _pin["out"] = torch.empty(stacked.shape, dtype=stacked.dtype).pin_memory()
        buf.copy_(stacked, non_blocking=True)
        torch.cuda.current_stream(dev).synchronize()
        return buf.numpy().copy()

    def _to_static(input_array):
        """The window straight into the captured graph's static input buffer, when a fast-path
        graph is live: one host->device copy instead of a copy and a device-side copy."""
        if not (
            _cudagraph
            and str(dev).startswith("cuda")
            and _cg["on"]
            and _cg["g"] is not None
            and _cg.get("fast")
            and extra_history is None
            and _rc is None
            and _ac_step is None
            and not mean_bias_strength
        ):
            return None
        src = input_array if isinstance(input_array, torch.Tensor) else None
        if src is None:
            src = torch.from_numpy(np.ascontiguousarray(input_array))
        sx = _cg["sx"]
        if src.shape != sx.shape or src.dtype != torch.float32:
            return None
        if pinned_io and not src.is_cuda:
            buf = _pin["in"]
            if buf is None or buf.shape != src.shape:
                buf = _pin["in"] = torch.empty(src.shape, dtype=torch.float32).pin_memory()
            buf.copy_(src)
            sx.copy_(buf, non_blocking=True)
        else:
            sx.copy_(src)
        return sx

    def predict(input_array, metadata=None):
        xt = _to_static(input_array)
        if xt is None:
            xt = _to_device(input_array)
        re = float(metadata["re"]) if metadata else 1.0e4
        aoa = float(metadata["aoa"]) if metadata else 10.0
        entry = _cond_cache.get((re, aoa))
        if entry is None:
            cond = conditioning_features_batch(
                np.array([re]),
                np.array([aoa]),
                "real",
                None,  # type: ignore[arg-type]
            )
            cb = torch.as_tensor(np.ascontiguousarray(cond), device=dev, dtype=torch.float32)
            # The head's sdf and conditioning planes are constant for this operating point, and
            # a trajectory is one operating point. Built here rather than inside ``_pipeline``
            # because a tensor materialised during CUDA-graph capture is baked into the graph
            # and replayed unchanged forever -- silently wrong, not an error.
            sp = None
            if tke_head is not None:
                from .tke_head import tke_static_planes  # noqa: PLC0415

                h, w = int(xt.shape[2]), int(xt.shape[3])
                sp = tke_static_planes(cond, h, w, dev, uses_tau=tke_head_uses_tau)
            entry = (cb, sp)
            _cond_cache[(re, aoa)] = entry
        cb, sp = entry
        with torch.inference_mode():
            # Fold the revealed pair (current input == previous target) into the mean-flow
            # residual, before overwriting ``prev`` with this step's prediction.
            if extra_history is not None and not _v2_merged:
                extra_history.observe_input(xt)
            if (
                extra_history is None
                and mean_bias_strength
                and st["prev"] is not None
                and st["bias"] is None
            ):
                res = (xt[..., :c] - st["prev"]).mean(dim=1, keepdim=True)  # over the lead axis
                st["sum_res"] = res if st["sum_res"] is None else st["sum_res"] + res
                st["n"] += 1
                if bias_freeze_n and st["n"] >= bias_freeze_n:
                    lam = mean_bias_strength * st["n"] / (st["n"] + bias_warmup_n0)
                    st["bias"] = _smooth_bias((lam / st["n"]) * st["sum_res"])
            if _n_state and _graph_folds_acc():
                acc = st["acc"]
                st["st_reveals"] += 1
            else:
                acc = _update_acc(xt) if _n_state else None
            if _rc is not None:
                _rc_launch(xt)
            if _v2_merged and st.get("v2_first", False):
                # First step of the trajectory: no revealed target -> no observe/apply. Seed
                # ``_g_prev`` (this prediction is the next step's ``prev``) and return it raw.
                pred, lo, hi = _pipeline(xt, cb, sp, acc)
                extra_history._g_prev.copy_(pred[..., : extra_history._g_prev.shape[-1]].float())
                st["v2_first"] = False
            else:
                if _fsw["on"]:
                    _fast_prepare(xt, cb, sp, acc, entry)
                pred, lo, hi = cpipe(xt, cb, sp, acc)
            # fp16 spans 6e-5 to 65504 against fp32's 1e-38 to 3e38, and a single non-finite
            # value zeroes ALL FIVE subscores. The host path recomputes a bad chunk in fp32
            # (make_predict); this path mirrors that with its sticky demotion: one fused
            # reduction per step (~0.03 ms, and the step ends in a sync anyway), and a failure
            # permanently drops to the eager fp32 pipeline -- slower, never wrong. Checked
            # BEFORE any state is written, so the recomputed step observes nothing twice.
            # The guard must also cover the cudagraph path, whose replays run the autocast
            # pipeline while ``_use["compiled"]`` stays False (``_compiled`` is ``_pipeline``
            # there): a demotion switches the graph off for good.
            _graph_live = _cudagraph and str(dev).startswith("cuda") and _cg["on"]
            flag = None if _fast_full is None else _fast_full.finite_flag(pred, lo, hi)
            if (
                flag is not None
                and _amp is not None
                and _graph_live
                and extra_history is None
                and _rc is None
                and _ac_step is None
                and not mean_bias_strength
            ):
                # The fast pipeline computed the fp16 guard's flag inside the graph: copy the
                # outputs first (the one synchronisation), then read the flag, which is ready.
                # On a non-finite step, demote exactly as below and serve the fp32 recompute.
                host = _to_host(_stack3(pred, lo, hi))
                if bool(flag):
                    return {"prediction": host[0], "lower": host[1], "upper": host[2]}
                _use["compiled"] = False
                _cg["on"] = False
                pred, lo, hi = _pipeline(xt, cb, sp, acc)
                host = _to_host(torch.stack((pred, lo, hi)))
                return {"prediction": host[0], "lower": host[1], "upper": host[2]}
            if (
                _amp is not None
                and (_use["compiled"] or _graph_live)
                and not bool(torch.isfinite(_stack3(pred, lo, hi)).all())
            ):
                _use["compiled"] = False
                if _graph_live:
                    _cg["on"] = False
                pred, lo, hi = _pipeline(xt, cb, sp, acc)
            if extra_history is not None and not _v2_merged:
                pred, lo, hi = extra_history.apply(pred, lo, hi)
            if _rc is not None:
                lo, hi = _rc_finish(xt, lo, hi)
            elif _ac_step is not None:
                lo, hi = _ac_step(xt, lo, hi)
            if extra_history is None and mean_bias_strength:
                st["prev"] = pred[..., :c].detach()  # uncalibrated, for the next residual
                bias = st["bias"]
                if bias is None and st["n"] > 0:
                    lam = mean_bias_strength * st["n"] / (st["n"] + bias_warmup_n0)
                    bias = _smooth_bias((lam / st["n"]) * st["sum_res"])
                if bias is not None:
                    pred = pred.clone()
                    pred[..., :c] = pred[..., :c] + bias.to(pred.dtype)  # bounds stay uncalibrated
            # One stacked D2H transfer instead of three: at batch 1 each .cpu() is its own
            # synchronise, and the three arrays together are ~0.7 MB. Views into the single
            # host array are safe -- a fresh array is made every step.
            host = _to_host(_stack3(pred, lo, hi))
            return {"prediction": host[0], "lower": host[1], "upper": host[2]}

    predict.reset = _reset  # type: ignore[attr-defined]
    return predict


#: Seconds of sustained dummy GPU work to run after the warm-up forwards, to bring the
#: device's clock governor up *before* the timed region. Zero disables it.
#:
#: This exists because of the day-8 anomaly (FINDINGS 30h): the evaluation A800 gave
#: 2.93-3.75x on six artefacts and **0.87x** on the STFormer one, and the property that
#: separates them is how much sustained GPU work the timed region contains -- 1.0-3.8 s on
#: the board for every other artefact, **0.22 s** for that one. An SXM A100-class card idles
#: near 210 MHz and boosts to ~1410, and the ramp is driven by sustained utilisation over
#: 100s of milliseconds, so a short bursty run can spend its whole life un-boosted.
#:
#: 0.5 s is chosen against the *load* budget, not the timed one: model construction runs
#: outside the timed region and a real submission used 1.9 s of the 5 minutes available.
#: It cannot change any output -- it computes on throwaway tensors and keeps nothing.
DEFAULT_SPIN_S = 0.5


def _spin_gpu(seconds: float) -> None:
    """Occupy the GPU with dense fp16 GEMMs for ``seconds``, then discard the result.

    Deliberately a large square matmul rather than more model forwards: the point is to hold
    the device at high utilisation, and the model reaches only 6.75 TFLOP/s against 28.4 for
    a dense fp16 GEMM on the same card, so a GEMM asks the governor for ~4x more per second
    spent. Sized to stay well inside any evaluation GPU's memory.
    """
    if seconds <= 0 or not torch.cuda.is_available():
        return
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(8):
            a = (a @ b).mul_(0.0).add_(a)  # keep values bounded; the result is discarded
        torch.cuda.synchronize()
    del a, b


def time_predict(
    predict_fn: Callable[..., Any],
    batch_size: int,
    reps: int = 3,
    t_in: int = T_IN,
) -> float:
    """Best of ``reps`` synthetic ``predict`` calls, in seconds, or ``inf`` if any raised.

    Exists so the *compiled-or-eager* decision can be made on the evaluation device instead of
    inferred from ours. That inference has failed once at a cost of -8.6 ``time_score``: compile
    measured 1.35x faster on our T4 and came back 5.4x slower on the board, and §30c had already
    recorded this artefact's T4->A800 GPU factor as **0.88x** -- i.e. its kernels do not transfer
    monotonically. Two measurements here cost ~1 s of the untimed 300 s construction budget and
    replace the extrapolation with an observation.

    ``inf`` on failure so a raising path always loses the comparison, never wins it by being
    unmeasurable.
    """
    x = np.zeros((max(1, batch_size), t_in, H, W, C), dtype=np.float32)
    x[..., 0] = WARMUP_U
    best = float("inf")
    for _ in range(max(1, reps)):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            predict_fn(x, None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        except Exception:  # noqa: BLE001 - an unmeasurable path must lose, not crash the run
            return float("inf")
    return best


def compile_backend_is_usable(device: str = "cuda") -> bool:
    """Whether ``torch.compile`` can actually compile anything in THIS environment.

    Probes rather than guesses, because the two cases that matter look identical from the
    outside. Inductor lowers to Triton, and Triton builds a small **C launcher stub for every
    kernel** (`triton/common/build.py:_build`), so it needs a C compiler. The Track-1 evaluation
    image ships none -- no gcc, g++, cc, clang, ld or make anywhere, ``CC`` unset -- and the
    compiled path dies with ``RuntimeError: Failed to find C compiler``. But a submission that
    ships a prebuilt Triton cache has no compiler either and DOES work, because
    ``make_stub`` consults its cache before building. So "is there a compiler" is the wrong
    question and only a real compilation answers the right one.

    Runs at import, in the untimed construction window, and costs a fraction of a second when it
    succeeds. Never raises: the answer on any failure is False, i.e. eager.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return False
    try:
        probe = torch.compile(lambda t: (t * 2.0).relu().sum())
        probe(torch.zeros(8, device=device))
    except Exception:  # noqa: BLE001 - any failure at all means "do not compile"
        return False
    finally:
        # Dynamo caches per code object, and this probe's frames are worthless to the caller.
        try:
            torch._dynamo.reset()
        except Exception:  # noqa: BLE001 - best effort; a failed reset must not decide anything
            pass
    return True


def warm_up_predict(
    predict_fn: Callable[..., Any],
    batch_size: int,
    reps: int = 1,
    spin_s: float = DEFAULT_SPIN_S,
    t_in: int = T_IN,
) -> bool:
    """Run one synthetic ``predict`` so the *timed* calls meet a warm device.

    The platform times ``predict()`` and nothing else -- "model construction and checkpoint
    loading run outside the timed region" -- but the *first* forward is where cuDNN selects
    an algorithm per convolution, kernels are loaded to the device, and the caching
    allocator grows its pool with real ``cudaMalloc``s. Left alone, all of that lands inside
    the first scored call. Measured locally: a first call costs ~10x a steady-state one,
    against a board budget of 21 ms x 210 samples = 4.4 s in total, so this is a large share
    of the axis rather than a rounding error.

    Called with the deployed ``batch_size`` because algorithm choice is per shape, and
    through the whole ``predict`` closure rather than the bare model so that normalisation,
    conditioning and the bound head are warmed too.

    ``reps`` exists for ``torch.compile``. Measured call profiles at batch 8: ``default``
    reaches its steady 0.360 s on call **2**, while ``reduce-overhead`` needs call **3** --
    its second call still costs 0.797 s capturing the CUDA graph. One rep too few leaves that
    capture inside the first *scored* call.

    Returns whether the warm-up ran, and the return value **must be checked** -- see
    ``make_predict``'s ``demote_to_eager``. A False here does not mean "we saved nothing": it
    means the entire first-forward cost is still owed and will be paid inside the first *scored*
    call. Measured on a T4 squeezed to 1.3 GB free, that is 22.5 against 5.7 ms/sample, i.e.
    +3.5 s over the platform's 210 samples.

    Retried once after ``empty_cache()`` because the measured cause is transient: the failure is
    an allocation, in an allocator that has just grown its pool for the build, and releasing the
    cached blocks is enough to make room for the search that needs them. The retry is free when
    the first attempt succeeds.

    **Never raises**: a submission that dies here would zero all five subscores to save a
    fraction of one.
    """
    for _ in range(2):
        try:
            x = np.zeros((max(1, batch_size), t_in, H, W, C), dtype=np.float32)
            x[..., 0] = WARMUP_U
            for _ in range(max(1, reps)):
                predict_fn(x, None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _spin_gpu(spin_s)
            return True
        except Exception:  # noqa: BLE001 - see the docstring: correctness outranks the warm-up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return False
