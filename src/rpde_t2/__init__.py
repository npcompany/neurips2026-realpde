"""Track 2 (LTTTA): the streaming test-time-adaptation track.

Track 2 is Track 1's prediction problem behind a different interface. Same data, same
20-in/20-out windows, same five subscores computed by the same formulas on the same
physical fields -- `vendor/realpde_t2_starting_kit_v6/scoring.py` differs from the Track-1
kit's only by an evaluator-side `sample_id` order check. So everything under
:mod:`rpde_t1` is shared core, and this package holds only what the streaming interface
adds.

Three things are genuinely different, and each has a module here:

* **The interface.** ``get_ttt_model(dir, device)`` returning an object with
  ``reset_ttt_state()`` and ``ttt_step(input_norm, prev_target_norm)``, called once per
  window at batch size 1, instead of one batched ``predict(input_array)``. See :mod:`.ttt`.
* **The space.** Track 1 hands ``predict`` physical fields; Track 2 hands ``ttt_step``
  fields standardized by the frozen official ``train_real`` statistics. The bridge is
  below, and it is two affine maps.
* **The stride.** Evaluation windows are consecutive, stride ``INTERVAL == IN_STEP``, so
  window k's target frames *are* window k+1's input frames. On Track 1 that overlap is the
  leak that reset the phase and :func:`rpde_t1.score.harness.assert_leak_free_stride`
  forbids it; here it is the protocol. See :mod:`.stream`.

A consequence of that stride worth stating once, because it voids a whole class of
adaptation idea: ``prev_target_norm`` is **bit-identical to the current input window**
(measured; the only difference is that the two are standardized with the input and target
statistics respectively, which agree to ~5e-5 relative). It therefore carries no
information about the current target beyond what ``input_norm`` already carries. What the
stream does give you is (a) the whole trajectory history, since the union of the windows
you have seen is every frame up to now, and (b) your own previous prediction paired with
its revealed target, i.e. a *measurement* of your error field on this trajectory.
"""

from __future__ import annotations

import numpy as np

from rpde_t1.features import (
    REAL_MEAN_IN,
    REAL_MEAN_TARGET,
    REAL_STD_IN,
    REAL_STD_TARGET,
    batch_freestream_speed,
)

#: Frames per input window and per prediction horizon. Equal, which is what makes the
#: stride below overlap consecutive windows exactly.
IN_STEP = 20
OUT_STEP = 20

#: Frames advanced between consecutive evaluation windows. Equal to ``IN_STEP`` by
#: protocol -- see :func:`rpde_t2.stream.assert_streaming_stride`.
INTERVAL = 20

#: Spatial subsample applied to the native 64x128 recording before evaluation.
SUB_S = 2

#: The shape of every tensor crossing the interface: (batch, T, H, W, [u, v, p]).
#: Batch is always 1.
EXPECTED_SHAPE = (1, IN_STEP, 32, 64, 3)

#: Extracted-archive size cap, and the wall-clock limit for the whole run. Both identical
#: to Track 1's, but the second one now has to cover per-step adaptation as well as loading.
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024
EXECUTION_LIMIT_S = 600.0

#: Real PIV leaves a substantial region with ``u`` and ``v`` identically zero in every frame
#: -- the opaque airfoil, its laser-sheet shadow, and vectors that failed the correlation
#: check (see :func:`rpde_t1.features.invalid_mask`). Measured over the release, 9-12% of
#: ``u`` pixels are *exactly* 0.0 in physical space. Standardizing maps every one of them to
#: this single distinctive constant, which is what makes the primary check below independent
#: of Reynolds number.
NORMALIZED_ZERO = float((0.0 - REAL_MEAN_IN[0]) / REAL_STD_IN[0])  # -1.600742

#: Smallest invalid fraction the primary check will act on. The release runs 6.6-12%, so 1%
#: is an order of magnitude of headroom; below it the field is treated as having no invalid
#: region at all and the freestream fallback decides.
MIN_INVALID_FRACTION = 0.01

#: Fallback bracket on :func:`rpde_t1.features.batch_freestream_speed`, for fields with no
#: invalid region (synthetic windows, and CFD, where the mask is only ~0.2%).
#:
#: **Not sufficient on its own, which is why it is the fallback.** The physical ladder spans
#: 0.03-0.40, but standardizing maps a trajectory's freestream to ``|u_fs - 0.155| / 0.0968``,
#: which is *small* for mid-ladder cells rather than large: measured, ``10125_0`` normalizes
#: to 0.121, ``12675_0`` to 0.303 and ``13950_5`` to 0.421 -- all inside this bracket. Three
#: of seven real cells sampled slip through it. Only the zero-mask check above is reliable.
PHYSICAL_FREESTREAM = (0.02, 0.6)


def freestream(x: np.ndarray) -> float:
    """The freestream speed a Track-1 predictor would estimate from ``x`` and divide by."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 4:
        x = x[None]
    return float(batch_freestream_speed(x).mean())


def assert_physical(x: np.ndarray) -> None:
    """Raise unless ``x`` is in physical units. Call on anything bound for a predictor.

    This guards the one way a Track-2 port dies **silently**. Every Track-1 model here is
    freestream-normalized: the predictor estimates the freestream from the window's
    undisturbed rows and divides by it. Hand it evaluator-space input instead and that
    divisor is ~1-2 rather than ~0.06, so every field is scaled by a factor of ~20 -- and
    measured, that takes ``rel_l2`` 95.3 -> 29.7 and ``tke`` 79.3 -> 0.2 while still
    returning correct shapes and finite values throughout. Neither the scorer's zero-score
    guards nor a shape check sees it, so the cost is a submission and a day.
    """
    u = np.asarray(x, dtype=np.float32)[..., 0]

    # Primary, and Reynolds-independent: where is the PIV invalid region?
    sentinel = float(np.isclose(u, NORMALIZED_ZERO, rtol=0, atol=1e-3).mean())
    if sentinel > MIN_INVALID_FRACTION:
        raise ValueError(
            f"{sentinel:.1%} of u pixels sit at {NORMALIZED_ZERO:.6f}, the value an exactly-"
            f"zero pixel takes after standardization. This field is still in evaluator space; "
            f"call to_physical() on it first. A freestream-normalized predictor would divide "
            f"by the wrong scale and score around rel_l2 30 while returning correct shapes "
            f"and finite values throughout."
        )
    if float((u == 0.0).mean()) > MIN_INVALID_FRACTION:
        return  # a real PIV window with its invalid region intact: physical

    # No invalid region to read, so fall back to magnitude -- see PHYSICAL_FREESTREAM for
    # why this alone is not enough.
    speed = freestream(x)
    lo, hi = PHYSICAL_FREESTREAM
    if not lo <= speed <= hi:
        raise ValueError(
            f"freestream estimate {speed:.4f} is outside the physical range [{lo}, {hi}] and "
            f"this field has no invalid region to check instead. Either it is still "
            f"normalized, or the official statistics no longer match the evaluator's."
        )


def to_physical(x_norm: np.ndarray, *, check: bool = True) -> np.ndarray:
    """Evaluator space -> physical units, the inverse of the ingestion program's input map.

    ``check`` runs :func:`assert_physical` on the result, so a caller that reaches a
    predictor through this function cannot make the mistake described there.
    """
    raw = np.asarray(x_norm, dtype=np.float32) * REAL_STD_IN + REAL_MEAN_IN
    if check:
        assert_physical(raw)
    return raw


#: Fused forms of the two affine maps, so each is one multiply-add over a preallocated
#: buffer instead of a subtract-then-divide with two temporaries. This is not premature:
#: measured at batch 1, the three ``to_normalized`` calls a bounded prediction needs cost
#: 1.60 ms of a 17.4 ms timed step, and Track 2 charges every one of them.
_TO_PHYS_SCALE = REAL_STD_IN.astype(np.float32)
_TO_PHYS_SHIFT = REAL_MEAN_IN.astype(np.float32)
_TO_NORM_SCALE = (1.0 / REAL_STD_TARGET).astype(np.float32)
_TO_NORM_SHIFT = (-REAL_MEAN_TARGET / REAL_STD_TARGET).astype(np.float32)


def to_normalized_into(raw: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """:func:`to_normalized` as a fused multiply-add, optionally in place.

    ``out=raw`` overwrites the input, which is safe for a buffer the predictor just handed
    back and nothing else holds.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if out is None:
        out = np.empty_like(raw)
    np.multiply(raw, _TO_NORM_SCALE, out=out)
    np.add(out, _TO_NORM_SHIFT, out=out)
    return out


def to_physical_into(x_norm: np.ndarray, *, check: bool = True) -> np.ndarray:
    """:func:`to_physical` as a fused multiply-add into a fresh buffer."""
    x_norm = np.asarray(x_norm, dtype=np.float32)
    out = np.empty_like(x_norm)
    np.multiply(x_norm, _TO_PHYS_SCALE, out=out)
    np.add(out, _TO_PHYS_SHIFT, out=out)
    if check:
        assert_physical(out)
    return out


def to_physical_from_target(y_norm: np.ndarray) -> np.ndarray:
    """Evaluator space -> physical, using the **target** statistics.

    For ``prev_target_norm``, which the evaluator standardized with the target statistics
    rather than the input ones. No :func:`assert_physical` here: a revealed target is not
    bound for a predictor, and the two statistic sets agree to ~5e-5 anyway.
    """
    y_norm = np.asarray(y_norm, dtype=np.float32)
    return y_norm * REAL_STD_TARGET.astype(np.float32) + REAL_MEAN_TARGET.astype(np.float32)


def to_normalized(raw: np.ndarray) -> np.ndarray:
    """Physical units -> evaluator space, using the **target** statistics.

    The ingestion program applies ``x * std_target + mean_target`` to the prediction and to
    both SPS bounds, so this is its exact inverse and applies to all three alike.

    Using the *input* statistics here instead is a 4.9e-5 relative error
    (``std_target / std_in = 1.0000491`` on ``u``), some 2000x below the prediction's own
    error. It is :func:`to_physical` that has to be right, not this.
    """
    return (np.asarray(raw, dtype=np.float32) - REAL_MEAN_TARGET) / REAL_STD_TARGET


__all__ = [
    "EXECUTION_LIMIT_S",
    "REAL_MEAN_TARGET",
    "REAL_STD_TARGET",
    "EXPECTED_SHAPE",
    "IN_STEP",
    "INTERVAL",
    "MAX_EXTRACTED_BYTES",
    "MIN_INVALID_FRACTION",
    "NORMALIZED_ZERO",
    "OUT_STEP",
    "PHYSICAL_FREESTREAM",
    "SUB_S",
    "assert_physical",
    "freestream",
    "to_normalized",
    "to_normalized_into",
    "to_physical",
    "to_physical_from_target",
    "to_physical_into",
]
