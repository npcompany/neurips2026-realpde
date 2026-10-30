"""Per-lead shrinkage of the prediction toward persistence, fitted in closed form.

The premise is FINDINGS 5's measurement, not a new idea: at lead 1 the model is *worse*
than repeating the last observed frame, and the crossover sits at lead 2-3. A model that
cannot reproduce the frame it was just handed is leaving accuracy on the table at exactly
the leads where the flow is most predictable.

What is new is that the fix needs no retraining. Writing ``anchor`` for the last observed
frame, the one-parameter family

    pred'_t = anchor + a_t * (pred_t - anchor)

contains both the model (``a_t = 1``) and persistence (``a_t = 0``), and the ``a_t`` that
minimises squared error against the target is a *least-squares projection* with a closed
form -- one inner-product ratio per lead, no optimiser, no hyperparameter, nothing to
tune and therefore nothing to mis-transfer. Fitted on the training ladder it recovers the
crossover exactly (``0.718`` at lead 1, ``~1.0`` from lead 3 on), which is the check that
this is the effect FINDINGS 5 described rather than a fitting artefact.

Two properties matter for how it is used:

* **It is a correction to the prediction, so it moves the accuracy axes** -- unlike a
  width rule, which the scorer's interval term alone sees. All three improve slightly;
  the lead-1 relative L2 error falls ~11%, diluted to ~+0.002 in the aggregate because
  lead 1 is one of twenty leads inside a whole-window norm.
* **Most of the SPS gain is coverage, not the ceiling**: a smaller error at lead 1 is
  covered by the same interval. So it composes with an interval head rather than
  competing with it, and the two are fitted independently.

Numpy only, three multiplies per element, and applied in
:func:`~rpde_t1.models.predictor.make_predict` on the same displacement path as the
second-moment correction, so the bounds follow the prediction rather than being left
bracketing a field that is no longer what we return.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import N_MEASURED


def _anchor_frame(inputs: np.ndarray, dtype: type = np.float64) -> np.ndarray:
    """The last observed frame, broadcastable over the output horizon.

    The fit accumulates over ~10^8 elements and needs float64; :meth:`LeadAnchor.apply` runs
    *inside the timed region* and stays in the prediction's own float32.

    That dtype is **not** where the cost is, which is worth recording because it is the
    obvious guess and it is wrong. Measured at batch 32: the whole apply is 1.59 ms/window
    against a ~63 ms budget, of which the copy is 0.08 and the dtype choice is nil -- it is
    memory traffic over three full-field temporaries. The in-place rewrite that would remove
    them is *slower* (2.40 ms), because ``out[..., :2]`` of a 3-channel field is a strided
    view. So 1.6 ms is the price of the +0.10 SPS, and it is priced rather than assumed
    (L11: a numpy feature once cost 800x the head it served).
    """
    return inputs[:, -1:, ..., :N_MEASURED].astype(dtype, copy=False)


@dataclass
class LeadAnchor:
    """Per-lead weights in ``anchor + a_t * (pred - anchor)``, one scalar per lead.

    ``weights`` has shape ``(n_lead,)`` and applies to the measured channels only -- the
    pressure channel is unmeasured, never scored, and zeroed downstream.
    """

    weights: np.ndarray

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)

    def apply(self, prediction: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Anchored copy of ``prediction``; ``inputs`` is the raw physical input window."""
        if self.weights.size != prediction.shape[1]:
            raise ValueError(
                f"anchor has {self.weights.size} lead weights but the prediction has "
                f"{prediction.shape[1]} leads"
            )
        dtype = prediction.dtype if prediction.dtype == np.float32 else np.float64
        anchor = _anchor_frame(inputs, dtype)
        a = self.weights.reshape(1, -1, 1, 1, 1).astype(dtype, copy=False)
        out = prediction.copy()
        out[..., :N_MEASURED] = anchor + a * (prediction[..., :N_MEASURED] - anchor)
        return out

    def apply_torch(self, prediction, inputs):
        """:meth:`apply` on whatever device the tensors are already on.

        The docstring of :func:`_anchor_frame` prices the host-side transform at 1.6
        ms/window and correctly identifies the cost as memory traffic over full-field
        temporaries rather than dtype or the copy. Measured paired and interleaved against
        the shipped artefact it is worse than that -- **+2.86 ms/window, +6.2%** -- which at
        the board's exchange rate cancels the +0.087 SPS the anchor buys. Memory traffic
        over three temporaries is precisely what a GPU is for, so the fix is to run it
        before the copy to host rather than to micro-optimise numpy: the same move, and the
        same reason, as :meth:`~.width_rule.WidthRule.apply_torch`.

        ``inputs`` may be the whole input window or just its last frame(s); the slice is
        ``[:, -1:]`` either way, so a caller on the far side of a PCIe bus can upload 0.5 MB
        instead of 10.5. Held to the numpy path elementwise by ``tests/test_lead_anchor.py``,
        because two spellings of one transform is the PROTOCOL 5 trap.
        """
        import torch  # noqa: PLC0415 - the fitting path stays torch-free

        if self.weights.size != prediction.shape[1]:
            raise ValueError(
                f"anchor has {self.weights.size} lead weights but the prediction has "
                f"{prediction.shape[1]} leads"
            )
        c = N_MEASURED
        anchor = inputs[:, -1:, ..., :c].to(prediction.dtype)
        a = torch.as_tensor(self.weights, device=prediction.device, dtype=prediction.dtype)
        out = prediction.clone()
        out[..., :c] = anchor + a.view(1, -1, 1, 1, 1) * (prediction[..., :c] - anchor)
        return out

    def restricted(self, n_lead: int | None = None, max_weight: float | None = None) -> LeadAnchor:
        """A copy with the late leads pinned to 1 and/or the weights capped.

        Both restrictions exist because the fitted weights past the crossover are noise around
        1, and that noise is not harmless. Shrinking toward the anchor shrinks toward a field
        that is *constant in time*, so any ``a_t < 1`` removes temporal variance -- and
        ``tke_score`` scores the temporal-variance field of a model that already retains only
        ~0.41 of the target's fluctuation energy. Whether the noise helps or hurts therefore
        depends on which side of 1 it happens to fall on, which is a property of the backbone
        and not of the method: measured, the U-Net's late weights average 1.0019 and its tke
        *rose* 0.016, while an SWA backbone's average 0.9921 and its tke *fell* 0.012.

        ``n_lead`` keeps the first ``n_lead`` fitted weights and sets the rest to exactly 1,
        which makes the transform the identity past the crossover and removes the leak
        entirely. ``max_weight`` caps the weights, on the prior that the model should never
        want to be pushed *away* from the frame it was just handed.
        """
        w = self.weights.copy()
        if max_weight is not None:
            w = np.minimum(w, max_weight)
        if n_lead is not None:
            w[n_lead:] = 1.0
        return LeadAnchor(weights=w)

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"weights": self.weights.tolist()}, indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: Path) -> LeadAnchor:
        return cls(weights=np.asarray(json.loads(Path(path).read_text())["weights"]))

    @classmethod
    def identity(cls, n_lead: int) -> LeadAnchor:
        """The no-op anchor, i.e. the model's own prediction. The one-parameter family's
        second endpoint, useful as a control that the plumbing changes nothing."""
        return cls(weights=np.ones(n_lead, dtype=np.float32))


def fit_lead_anchor(
    prediction: np.ndarray,
    inputs: np.ndarray,
    target: np.ndarray,
    scored: np.ndarray | None = None,
) -> LeadAnchor:
    """Least-squares ``a_t`` per lead, over the scored measured elements.

    ``a_t = <pred - anchor, target - anchor> / <pred - anchor, pred - anchor>``, which is
    the orthogonal projection of the target's departure from persistence onto the model's.
    Leads where the model predicts persistence exactly carry no information about ``a_t``
    and fall back to 1, i.e. to leaving the prediction alone.

    ``scored`` is the scorer's own mask (elements whose target is exactly zero are PIV
    dropouts and are not scored); fitting on unscored elements would fit the mask.
    """
    anchor = _anchor_frame(inputs)
    if scored is None:
        scored = target[..., :N_MEASURED] != 0.0
    weights = np.ones(prediction.shape[1], dtype=np.float64)
    # Per lead rather than vectorised over the horizon: the pooled fitting set is ~10^8
    # measured elements, and the products below would allocate several float64 copies of
    # it at once -- the same OOM that width_rule.CHUNK exists to avoid.
    for t in range(prediction.shape[1]):
        m = scored[:, t]
        d = (prediction[:, t, ..., :N_MEASURED].astype(np.float64) - anchor[:, 0])[m]
        r = (target[:, t, ..., :N_MEASURED].astype(np.float64) - anchor[:, 0])[m]
        den = float(np.dot(d, d))
        weights[t] = float(np.dot(d, r)) / den if den > 0.0 else 1.0
    return LeadAnchor(weights=weights)
