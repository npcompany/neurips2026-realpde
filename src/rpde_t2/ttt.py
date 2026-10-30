"""The Track-2 streaming interface, and the frozen port that satisfies it.

The evaluator is duck-typed: it only requires ``reset_ttt_state()`` and
``ttt_step(input_norm, prev_target_norm=None)``. :class:`TTTModel` is therefore a
documentation device, not a dependency -- but everything shipped goes through it so the
timing and shape contract sits next to the code that has to honour it.

:class:`FrozenPort` is the control arm and the first thing to submit: a Track-1 predictor,
unchanged, behind the Track-2 interface. It adapts nothing and returns ``adapt_loss=None``
on every step. That is the point -- it isolates the transfer from the modelling, so the
board row measures one thing. Anything adaptive is judged against it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from . import (
    IN_STEP,
    REAL_MEAN_TARGET,
    REAL_STD_TARGET,
    to_normalized_into,
    to_physical_from_target,
    to_physical_into,
)

#: How the absent earlier frames are filled on the FIRST window of a trajectory, where the
#: stream has no history yet (``prev_target_norm`` is ``None`` there by the official contract).
#: Only meaningful when ``t_in > IN_STEP``.
HISTORY_FILLS = ("zeros", "repeat", "hold")


class _OutputMapper:
    """The physical -> evaluator affine map for prediction and both bounds.

    Three full-array maps cost **1.913 ms** in numpy at batch 1, against 9.5 ms for the whole
    GPU forward. Stacking them and doing the arithmetic on the device costs **1.125 ms**
    including both transfers, and is **bit-identical** (max disagreement 0.0 over the field) --
    the same affine on the same float32 values, just not on the host.

    Falls back to the numpy path when there is no CUDA device, so a CPU smoke test still runs.

    ``premapped`` is the third and cheapest option: the predictor was built with
    ``out_transform`` and did the same affine on the device *before* its one copy to host, so
    there is nothing left to do but wrap the buffer. That removes the round trip this class
    exists to make cheap -- three arrays up and three back down -- and it is the mode the
    shipped bundle uses. The two mapping paths stay because the sizing experiments and the
    CPU smoke test both build predictors without ``out_transform``.
    """

    def __init__(self, device: str | None, premapped: bool = False):
        self.premapped = premapped
        self.device = device if device and str(device).startswith("cuda") else None
        if self.device is not None and not premapped:
            self._scale = torch.as_tensor(1.0 / REAL_STD_TARGET, device=self.device)
            self._shift = torch.as_tensor(-REAL_MEAN_TARGET / REAL_STD_TARGET, device=self.device)

    def __call__(self, *arrays: np.ndarray) -> tuple[torch.Tensor, ...]:
        if self.premapped:
            return tuple(_as_tensor(a) for a in arrays)
        if self.device is None:
            return tuple(_as_tensor(to_normalized_into(a, out=a)) for a in arrays)
        stacked = torch.from_numpy(np.stack(arrays)).to(self.device, non_blocking=True)
        stacked = stacked * self._scale + self._shift
        host = stacked.cpu()
        return tuple(host[i] for i in range(len(arrays)))


class TTTModel(ABC):
    """The contract the Track-2 evaluator calls.

    Both methods are **timed**. ``reset_ttt_state`` is charged to the step that follows
    the trajectory boundary, so moving work between the two changes nothing -- and a reset
    that deep-copies a state dict (as the starting kit's reference implementation does)
    pays ~90 MB of copying inside the timed region once per trajectory. Prefer adapting a
    small side module, so the reset is O(1).

    ``ttt_step`` receives ``(1, 20, 32, 64, 3)`` in the evaluator's normalized space and
    returns ``(pred_norm, info)``. The evaluator reads ``info["adapt_loss"]`` -- a python
    float, or ``None`` when nothing adapted -- and optionally ``info["lower"]`` /
    ``info["upper"]``, per-element SPS bounds in the same space and shape. Bounds are
    all-or-nothing **for the whole run**: starting or stopping partway through fails it at
    ingestion, so whether to emit them must be fixed before the stream, never decided
    per step.
    """

    @abstractmethod
    def reset_ttt_state(self) -> None:
        """Restore checkpoint weights and clear adaptation state. Called per trajectory."""

    @abstractmethod
    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Adapt on the previous pair if there is one, then predict ``input_norm``."""


class FrozenPort(TTTModel):
    """A Track-1 ``predict``-style callable behind the Track-2 interface, without adaptation.

    ``predict_fn`` is what :func:`rpde_t1.models.predictor.make_predict` returns: physical
    fields in, physical fields out, either a bare array or a dict with ``prediction`` /
    ``lower`` / ``upper``. Whether bounds come back is decided by that closure at
    construction, which is exactly the right place -- see the all-or-nothing rule on
    :class:`TTTModel`.

    ``prev_target_norm`` is deliberately ignored. At stride 20 it is the current input
    window (see :mod:`rpde_t2`), so for a model whose weights do not move it carries
    nothing at all.

    ``premapped`` says the predictor already returns evaluator-space fields, because it was
    built with ``out_transform``; the wrapper then only wraps the buffers. See
    :class:`_OutputMapper`.
    """

    #: Windows whose operating-point estimates are averaged before the value is frozen for the
    #: rest of the trajectory. 0 disables the cache and re-estimates every step.
    OP_WARMUP = 3

    def __init__(
        self,
        predict_fn: Callable[..., Any],
        cache_operating_point: bool = True,
        device: str | None = None,
        premapped: bool = False,
        gpu_native: bool = False,
        t_in: int = IN_STEP,
        history_fill: str = "zeros",
        state_mode: str = "",
        state_freeze_n: int = 0,
    ):
        self.predict_fn = predict_fn
        self.cache_operating_point = cache_operating_point
        self.map_out = _OutputMapper(device, premapped=premapped)
        # ``state_mode`` serves a ``TrainConfig.stream_state`` backbone (runs/t2_backbone/
        # PLAN.md): two pseudo-frames -- running mean and running temporal std over every
        # frame the stream has revealed through the current input -- prepended to the window.
        # On the HOST path this port builds them (``_with_state``); on the GPU-NATIVE path
        # ``make_gpu_predict(stream_state=...)`` owns the accumulators (they live with the
        # mean-flow bias and share its per-trajectory ``reset``), so the port passes the wire
        # window through untouched -- prepending here too would double the state.
        from rpde_t1.models.factory import STREAM_STATE_FRAMES  # noqa: PLC0415

        if state_mode not in STREAM_STATE_FRAMES:
            raise ValueError(
                f"state_mode must be one of {tuple(STREAM_STATE_FRAMES)}, got {state_mode!r}"
            )
        if state_mode and t_in != IN_STEP:
            raise ValueError("state_mode and t_in > 20 are two spellings of history; pick one")
        self.state_mode = state_mode
        #: Reveals after which the state stops being re-estimated (0 = never). Mirrors
        #: ``make_gpu_predict(state_freeze_n=)`` so the two serving paths stay in parity.
        self.state_freeze_n = int(state_freeze_n)
        # ``t_in`` > 20 serves a longer-input-context backbone. The stream hands out 20 frames
        # per step, so the earlier frames come from windows this object has already seen --
        # which the official interface explicitly sanctions ("Cache the current input for the
        # next step", docs/interface.md). Inputs are CONTIGUOUS and non-overlapping at stride
        # 20 (window k is frames 20k..20k+19), so concatenating the buffer with the current
        # window reconstructs a genuine unbroken block of frames, not a stitched one.
        self.t_in = int(t_in)
        if self.t_in < IN_STEP or self.t_in % IN_STEP:
            raise ValueError(
                f"t_in={t_in} must be a multiple of the stream's {IN_STEP}-frame window"
            )
        if history_fill not in HISTORY_FILLS:
            raise ValueError(f"history_fill={history_fill!r} not in {HISTORY_FILLS}")
        self.history_fill = history_fill
        # ``gpu_native``: ``predict_fn`` is :func:`rpde_t1.models.predictor.make_gpu_predict`,
        # which takes the evaluator-space INPUT TENSOR (kept on the device) and returns
        # evaluator-space fp16 arrays -- the whole GPU tail is one compiled graph. The port then
        # does no host bridge on the input and no output map: it only supplies the cached
        # operating point and wraps the returned buffers. See FINDINGS 76b.
        self.gpu_native = gpu_native
        self._reset_port_state()

    def _reset_port_state(self) -> None:
        self._op: dict[str, float] | None = None
        self._op_acc = np.zeros(2, dtype=np.float64)
        self._op_n = 0
        self._checked = False
        # The earlier frames of a longer context, carried between steps of ONE trajectory and
        # dropped at its boundary -- the next trajectory's first window genuinely has none.
        self._hist: Any = None
        # ``state_mode`` accumulators: per-trajectory running sum / sum-of-squares of the
        # physical (u, v) field and the revealed-frame count. O(1) in the stream length.
        self._st_sum: np.ndarray | None = None
        self._st_sumsq: np.ndarray | None = None
        self._st_n = 0
        # ``meanvarfast``: per-window (sum, sumsq, n) for the last FAST_WINDOWS reveals.
        self._st_recent: list[tuple[np.ndarray, np.ndarray, int]] = []
        # ``meanvarperr``: the previous window, and the running 20-lag persistence error.
        self._st_prev: np.ndarray | None = None
        self._st_perr: np.ndarray | None = None
        self._st_perr_n = 0
        self._st_reveals = 0
        # ``meanvarcov``: running sums of u*v, vorticity^2 and the lag-1 product x_t*x_{t-1}
        # (continued across windows through the previous window's last frame).
        self._st_cov: dict[str, Any] | None = None
        self._st_last: np.ndarray | None = None
        # ``meanvarphase``: the retained revealed frames. The shedding mode is recomputed from
        # the whole revealed span each window (the frequency is estimated online), so the host
        # keeps the physical (u, v) history and hands it to the shared ``phase_planes_np``.
        self._st_phist: list[np.ndarray] = []

    def reset_ttt_state(self) -> None:
        # O(1): a few scalars. Track 2 charges the reset to the step that follows it, so
        # nothing here grows with the trajectory.
        self._reset_port_state()
        # The GPU-native predictor may carry its own per-trajectory state (the mean-flow bias
        # accumulator, FINDINGS 77); clear it on the trajectory boundary.
        reset = getattr(self.predict_fn, "reset", None)
        if callable(reset):
            reset()

    def _metadata(self, raw: np.ndarray) -> dict[str, float] | None:
        """The operating point, estimated once per trajectory instead of once per window.

        ``make_predict`` calls :func:`rpde_t1.features.estimate_batch` on every window to
        recover (Re, AoA) from the freestream and the wake, which their own profiling puts at
        **1.20 ms/window** -- 8% of a 15.6 ms Track-2 step. But a trajectory *is* one operating
        point: the file is a single (Re, AoA) case, and only the *estimate* wobbles (measured
        spread 0.95% on Re, 0.84 deg on AoA). So estimate it over the first few windows,
        average, and pass it through the documented ``metadata`` escape hatch thereafter, which
        short-circuits the estimator entirely.

        Averaging is a small bonus rather than the point: the estimator's error is mostly bias,
        so averaging removes only its spread.
        """
        if not self.cache_operating_point:
            return None
        if self._op is not None:
            return self._op
        from rpde_t1.features import estimate_batch

        est = estimate_batch(raw)[0]
        self._op_acc += est
        self._op_n += 1
        if self._op_n >= self.OP_WARMUP:
            re, aoa = self._op_acc / self._op_n
            self._op = {"re": float(re), "aoa": float(aoa)}
            return self._op
        return {"re": float(est[0]), "aoa": float(est[1])}

    def _with_history(self, cur: Any) -> Any:
        """Build the model's ``t_in``-frame input from the current window and the buffer.

        ``cur`` is the current 20-frame window with a leading batch axis, either a device
        tensor (the GPU-native path, evaluator space) or a physical numpy array (the host
        path). Whichever it is, the buffer holds the same kind, so the two never mix.

        **The first window of a trajectory is the case that only exists here.** The stream
        has no earlier frames then, and nothing can invent them: the official contract passes
        ``prev_target_norm=None`` on that step. A 40-frame backbone fine-tuned on windows that
        always had 40 real frames has never seen whatever we put there, so the fill is a
        train/serve mismatch unless the backbone was trained to expect it -- which is what
        ``--history-dropout`` in the fine-tune is for. The fills differ in what they lie about:
        ``zeros`` matches the warm-start state of a widened stem but is off the data manifold;
        ``repeat`` keeps the input in distribution while asserting a false 20-frame period;
        ``hold`` freezes the oldest frame, which is steady but has no fluctuation.

        Returns the model input and leaves the buffer holding this step's trailing frames.
        """
        if self.state_mode:
            # The gpu-native predictor accumulates its own state (see __init__).
            return cur if self.gpu_native else self._with_state(cur)
        if self.t_in == IN_STEP:
            return cur
        n_old = self.t_in - IN_STEP
        hist = self._hist
        if hist is None:
            hist = self._cold_history(cur, n_old)
        joined = self._concat(hist, cur)
        # Keep the trailing ``n_old`` frames for the next step. Copied because the caller may
        # write into ``cur`` in place (``to_physical_into``) after we return.
        self._hist = self._tail(joined, n_old)
        return joined

    def _with_state(self, cur: np.ndarray) -> np.ndarray:
        """Prepend the ``state_mode`` pseudo-frames to a physical window.

        Bit-compatible with the dataset's construction (``WindowDataset._state_frames``): the
        state covers frames ``[0, 20(k+1))`` -- the accumulators fold the CURRENT window in
        before the statistics are read, matching training's ``arr[: start + 20]`` on the
        stream's aligned starts -- and the predictor divides every frame by the freestream it
        estimates from the LAST 20 frames, which is the same division training applied. The
        n-plane is written into the (unused) pressure channel; training writes it AFTER the
        division, so it is pre-multiplied here by the same freestream estimate and the
        predictor's division restores it exactly.

        ``meanvarfast`` additionally keeps the last ``STREAM_STATE_FAST_WINDOWS`` reveals'
        statistics, and ``meanvarperr`` the running ``|x_t - x_{t-20}|`` persistence error --
        which at stride 20 is exactly ``|this window - the previous window|``, elementwise.
        """
        from rpde_t1.models.factory import (  # noqa: PLC0415
            STREAM_STATE_FAST_WINDOWS,
            STREAM_STATE_FRAMES,
        )
        from rpde_t1.models.predictor import estimate_freestream  # noqa: PLC0415

        frozen = bool(self.state_freeze_n) and self._st_reveals >= self.state_freeze_n
        w = cur[0, ..., :2].astype(np.float64)
        wsum, wsq, wn = w.sum(axis=0), (w * w).sum(axis=0), w.shape[0]
        if self._st_sum is None:
            self._st_sum, self._st_sumsq = wsum.copy(), wsq.copy()
            self._st_reveals = 1
        elif not frozen:
            self._st_sum += wsum
            self._st_sumsq += wsq
            self._st_n += wn
            self._st_reveals += 1
        if self._st_n == 0:
            self._st_n = wn
        n = self._st_n
        m = self._st_sum / n
        sd = np.sqrt(np.maximum(self._st_sumsq / n - m * m, 0.0))

        if self.state_mode == "meanvarfast" and not frozen:
            self._st_recent.append((wsum, wsq, wn))
            del self._st_recent[:-STREAM_STATE_FAST_WINDOWS]
        if self.state_mode == "meanvarfast":
            fn = sum(r[2] for r in self._st_recent)
            fm = sum(r[0] for r in self._st_recent) / fn
            fsd = np.sqrt(np.maximum(sum(r[1] for r in self._st_recent) / fn - fm * fm, 0.0))
        elif self.state_mode == "meanvarperr" and not frozen:
            if self._st_prev is not None:
                diff = np.abs(w - self._st_prev).sum(axis=0)
                self._st_perr = diff if self._st_perr is None else self._st_perr + diff
                self._st_perr_n += wn
            self._st_prev = w

        if self.state_mode == "meanvarphase" and not frozen:
            self._st_phist.append(w)  # w is this window's physical (u, v), float64

        if self.state_mode == "meanvarcov" and not (frozen and self._st_cov is not None):
            omega = np.gradient(w[..., 1], axis=2) - np.gradient(w[..., 0], axis=1)
            chain = w if self._st_last is None else np.concatenate([self._st_last[None], w])
            add = {
                "uv": (w[..., 0] * w[..., 1]).sum(axis=0),
                "w2": (omega * omega).sum(axis=0),
                "lag": (chain[1:] * chain[:-1]).sum(axis=0),
            }
            if self._st_cov is None:
                self._st_cov = add
            else:
                for k, v in add.items():
                    self._st_cov[k] = self._st_cov[k] + v
            self._st_last = w[-1].copy()
        if self.state_mode == "meanvarcov":
            from rpde_t1.models.factory import covariance_stats_from_sums  # noqa: PLC0415

            cs = covariance_stats_from_sums(
                {"n": n, "s": self._st_sum, "sq": self._st_sumsq, **self._st_cov}
            )

        scale = float(estimate_freestream(cur[:, -IN_STEP:])[0])
        n_frames = STREAM_STATE_FRAMES[self.state_mode]
        frames = np.zeros((1, n_frames, *m.shape[:2], cur.shape[-1]), dtype=np.float32)
        frames[0, 0, ..., :2] = m
        frames[0, 0, ..., 2] = min(n / 820.0, 1.0) * scale
        frames[0, 1, ..., :2] = sd
        if self.state_mode == "meanvarfast":
            frames[0, 2, ..., :2] = fm
            frames[0, 3, ..., :2] = fsd
        elif self.state_mode == "meanvarperr" and self._st_perr is not None:
            frames[0, 2, ..., :2] = self._st_perr / self._st_perr_n
        elif self.state_mode == "meanvarcov":
            # Velocity-unit statistics are left physical here, as ``m`` and ``sd`` are: the
            # predictor divides every frame by the freestream. ``rho1`` is dimensionless, so it
            # is pre-multiplied by that same freestream, exactly as the count plane is.
            frames[0, 2, ..., 0] = cs["cov_ssqrt"]
            frames[0, 2, ..., 1] = cs["vort_rms"]
            frames[0, 3, ..., :2] = cs["rho1"] * scale
        elif self.state_mode == "meanvarphase":
            from rpde_t1.models.factory import phase_planes_np  # noqa: PLC0415

            planes = phase_planes_np(np.concatenate(self._st_phist, axis=0))
            # Dimensionless, so -- like ``rho1`` -- pre-multiplied by the freestream the
            # predictor divides every frame by.
            frames[0, 2, ..., :2] = planes[0] * scale
            frames[0, 3, ..., :2] = planes[1] * scale
        return np.concatenate([frames, cur], axis=1)

    def _cold_history(self, cur: Any, n_old: int) -> Any:
        """The stand-in for frames that do not exist, on a trajectory's first window."""
        if self.history_fill == "zeros":
            return cur[:, :1] * 0 if n_old == 1 else self._tile(cur[:, :1] * 0, n_old)
        if self.history_fill == "hold":
            return self._tile(cur[:, :1], n_old)
        return self._tail(self._tile(cur, -(-n_old // cur.shape[1])), n_old)

    @staticmethod
    def _concat(a: Any, b: Any) -> Any:
        if isinstance(a, torch.Tensor):
            return torch.cat((a, b), dim=1)
        return np.concatenate((a, b), axis=1)

    @staticmethod
    def _tile(a: Any, reps: int) -> Any:
        if isinstance(a, torch.Tensor):
            return a.repeat(1, reps, *([1] * (a.dim() - 2)))
        return np.concatenate([a] * reps, axis=1)

    @staticmethod
    def _tail(a: Any, n: int) -> Any:
        out = a[:, -n:]
        return out.clone() if isinstance(out, torch.Tensor) else out.copy()

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Every allocation and every extra pass here is charged to the step. Measured at
        # batch 1, this wrapper cost 3.4 ms of a 17.4 ms step before being fused; the
        # predictor's own pipeline is 3.0 ms and the GPU forward 10.5 ms.
        if self.gpu_native:
            # The input tensor stays on the device. The physical window is materialised only
            # while the operating-point cache is still warming (<= OP_WARMUP steps), because
            # ``estimate_batch`` reads it; once frozen, nothing here touches the host until the
            # output copy inside ``predict_fn``.
            if self._op is None:
                raw = to_physical_into(
                    torch.as_tensor(input_norm).detach().cpu().numpy(), check=not self._checked
                )
                meta = self._metadata(raw)
            else:
                meta = self._op
            self._checked = True
            out = self.predict_fn(self._with_history(input_norm), meta)
            info = {
                "adapt_loss": None,
                "lower": _as_tensor(out["lower"]),
                "upper": _as_tensor(out["upper"]),
            }
            return _as_tensor(out["prediction"]), info

        x = torch.as_tensor(input_norm).detach().cpu().numpy()
        # The space cannot change mid-run, so the guard runs once per trajectory rather than
        # once per step: np.isclose over the field is ~1 ms of a 15.6 ms step, and paying it 42
        # times to re-answer a question settled on the first window is waste.
        raw = to_physical_into(x, check=not self._checked)
        # The operating point is estimated from the CURRENT window only. Every estimate reads
        # the last 20 frames by construction (see `dataset` and `predictor`), so conditioning
        # stays identical to the 20-frame model's whatever the context length is.
        meta = self._metadata(raw)
        out = self.predict_fn(self._with_history(raw), meta)

        if not isinstance(out, dict):
            self._checked = True
            return self.map_out(out)[0], {"adapt_loss": None}

        # A reversed interval fails the whole run at ingestion, but a bound head either orders
        # its output or it does not -- so this is checked on the first window and then trusted.
        if not self._checked and np.any(out["lower"] > out["upper"]):
            out["lower"], out["upper"] = (
                np.minimum(out["lower"], out["upper"]),
                np.maximum(out["lower"], out["upper"]),
            )
        self._checked = True
        pred, lower, upper = self.map_out(out["prediction"], out["lower"], out["upper"])
        return pred, {"adapt_loss": None, "lower": lower, "upper": upper}


def _as_tensor(a: np.ndarray) -> torch.Tensor:
    # Arithmetic output is already C-contiguous float32, so from_numpy is a view, not a copy.
    return torch.from_numpy(a)


__all__ = ["FrozenPort", "TTTModel", "TrajectoryCalibratedPort"]


class TrajectoryCalibratedPort(FrozenPort):
    """Frozen backbone, plus a closed-form per-trajectory calibration from revealed history.

    This is the adaptation the measurements actually support. ``scripts/t2/t2_error_autocorr.py``
    found that the *temporal* anomaly of the error field is unpredictable at lag 1
    (r = +0.013) while its per-trajectory *static* structure is not (r = +0.60). So the thing
    to estimate online is a per-trajectory constant, not a per-step update -- and a constant
    needs no gradients, which is what keeps this affordable on an axis where a full-backbone
    gradient step would cost ~0.7 final in time alone.

    Two statistics accumulate over the steps already revealed, and both are applied to later
    predictions of the same trajectory:

    * ``gain`` -- the ratio of the target's temporal standard deviation to the prediction's,
      per element. `tke_score` is literally a relative error on temporal variance and is our
      weakest axis, so measuring the trajectory's own fluctuation level beats predicting it.
    * ``bias`` -- the mean signed residual. Note this is the *signed* mean, which is a
      different object from the error *magnitude* field the experiment ruled out.

    ``gain_strength`` and ``bias_strength`` shrink the two corrections toward the uncalibrated
    prediction, and they are deliberately separate. They target different moments: the gain
    fixes temporal *variance*, whose 20-frame realisation carries ~0.64 relative sampling
    scatter, while the bias fixes the time *mean*, whose scatter is ~sqrt(20) smaller. A 2D
    sweep on both held-out splits accordingly prefers a larger bias than gain (0.75 against
    0.65) -- though both sit inside the 0.029 paired noise floor of the local eval set, in a
    broad optimum, so that is a mechanistically-motivated setting rather than a measured win.
    Pushed to 1.0 the gain overshoots and *loses* 0.26.

    **The whole correction commutes with the evaluator's affine map**, so it runs in whichever
    space the predictor hands back. ``gain`` is a ratio of temporal standard deviations and is
    therefore scale-free; the centred rescale and the additive bias are both affine images of
    themselves under ``x -> ax + b``. So with a ``premapped`` predictor the statistics accumulate
    directly in evaluator space against ``prev_target_norm`` as delivered -- which drops a full
    denormalisation of the revealed target *and* keeps the device-side output map, where the
    physical-space spelling would force the host round trip back on and cost 2.4 ms/step before
    the calibration bought anything. Identical arithmetic either way, checked elementwise by
    ``tests/test_t2_stream.py``.

    ``warmup_n0`` makes that shrinkage grow with the history instead of being constant:
    ``lam_k = strength * n / (n + warmup_n0)``. The estimate after one revealed window is far
    worse than after ten, and a constant strength over-trusts the first few. Swept over six
    families of estimator in `scripts/t2_gain_lab.py`, this is the only variant that beat the
    constant schedule, and it beat it on **both** held-out splits (+0.1419 mean against
    +0.1222). Set ``warmup_n0=0`` for the constant schedule.
    """

    #: Clamp on the fluctuation gain. A trajectory whose first window happens to be unusually
    #: quiet would otherwise hand a later window an unbounded multiplier.
    GAIN_LIMITS = (0.5, 2.0)

    #: Revealed windows after which the correction stops being re-estimated and the two fields
    #: are cached. The gain is **capacity-limited, not data-limited** -- measured by lead-window
    #: band, it buys +0.70 over history 1-5 and then +1.33 / +1.21 / +1.33 over 6-15 / 16-28 /
    #: 29-41, i.e. it saturates at ~6 windows and the remaining 35 re-derive the same two fields
    #: at full price. Freezing at 8 keeps an honest shrinkage (lam is a function of the sample
    #: count actually used, so a frozen estimate carries a frozen lam by construction) and turns
    #: every later step from two temporal-variance passes plus a residual into one fused
    #: multiply-add. 0 disables the freeze and re-estimates for the whole trajectory.
    FREEZE_N = 8

    def __init__(
        self,
        predict_fn: Callable[..., Any],
        gain_strength: float = 0.5,
        bias_strength: float = 0.5,
        cache_operating_point: bool = True,
        device: str | None = None,
        warmup_n0: float = 2.0,
        premapped: bool = False,
        bias_mode: str = "meanflow",
        t_in: int = IN_STEP,
        history_fill: str = "zeros",
        state_mode: str = "",
    ):
        self.gain_strength = float(gain_strength)
        self.bias_strength = float(bias_strength)
        self.warmup_n0 = float(warmup_n0)
        # ``bias_mode`` decides what the additive correction fixes.
        #   "perframe"  -- the full per-lead residual field. Corrects the mean flow AND the
        #                  per-lead fluctuation profile, so it moves ``tke`` (measured: it
        #                  *hurts* it, because a noisy per-lead residual injected into the
        #                  prediction is uncorrelated with the target's own fluctuations).
        #   "meanflow"  -- the residual averaged over the lead axis, i.e. a single (H, W, C)
        #                  first-moment correction added identically to every frame. A constant
        #                  temporal shift leaves the variance-about-the-mean untouched, so it is
        #                  a pure ``mvpe`` / ``rel_l2`` correction and is ORTHOGONAL to the
        #                  drop-path ``tke`` lever -- which is what lets it stack on the soup.
        self.bias_mode = bias_mode
        super().__init__(
            predict_fn,
            cache_operating_point=cache_operating_point,
            device=device,
            premapped=premapped,
            t_in=t_in,
            history_fill=history_fill,
            state_mode=state_mode,
        )

    def reset_ttt_state(self) -> None:
        # O(1) in the number of steps and allocation-free after the first trajectory, which
        # matters because Track 2 charges the reset to the step that follows it.
        super().reset_ttt_state()
        self._prev_pred = None
        self._n = 0
        self._sum_res: np.ndarray | None = None
        self._sum_var_t: np.ndarray | None = None
        self._sum_var_p: np.ndarray | None = None
        self._frozen: tuple[np.ndarray, np.ndarray | None] | None = None

    def _observe(self, prev_target: np.ndarray) -> float:
        """Fold the revealed previous pair into the running statistics. Returns its MSE."""
        pred = self._prev_pred
        assert pred is not None
        res = prev_target - pred
        var_t = prev_target.std(axis=0) ** 2
        var_p = pred.std(axis=0) ** 2
        if self._sum_res is None:
            self._sum_res = res.astype(np.float64)
            self._sum_var_t = var_t.astype(np.float64)
            self._sum_var_p = var_p.astype(np.float64)
        else:
            self._sum_res += res
            self._sum_var_t += var_t
            self._sum_var_p += var_p
        self._n += 1
        return float(np.mean(res[..., :2] ** 2))

    def _shrinkage(self) -> float:
        """How far to trust the accumulated estimate, given how much history built it."""
        if self.warmup_n0 <= 0.0:
            return self.gain_strength
        return self.gain_strength * self._n / (self._n + self.warmup_n0)

    def _fields(self) -> tuple[np.ndarray, np.ndarray | None]:
        """The multiplicative and additive corrections, cached once the estimate is frozen."""
        if self._frozen is not None:
            return self._frozen
        lam = self._shrinkage()
        lo, hi = self.GAIN_LIMITS
        gain = np.sqrt(
            np.maximum(self._sum_var_t, 1e-12) / np.maximum(self._sum_var_p, 1e-12)
        ).clip(lo, hi)
        scale = (1.0 + lam * (gain - 1.0)).astype(np.float32)
        bias = None
        if self.bias_strength:
            bias_lam = (
                self.bias_strength
                if self.warmup_n0 <= 0.0
                else self.bias_strength * self._n / (self._n + self.warmup_n0)
            )
            bias = (bias_lam / self._n) * self._sum_res
            if self.bias_mode == "meanflow":
                # Average over the lead axis -> a single first-moment field, broadcast back to
                # every frame. Pure mean-flow correction; does not touch temporal variance.
                bias = bias.mean(axis=0, keepdims=True)
            bias = bias.astype(np.float32)
        if self.FREEZE_N and self._n >= self.FREEZE_N:
            self._frozen = (scale, bias)
        return scale, bias

    def _calibrate(self, pred: np.ndarray) -> np.ndarray:
        """Apply the accumulated per-trajectory gain and bias, in place."""
        if self._n == 0:
            return pred
        scale, bias = self._fields()
        mean_t = pred.mean(axis=0, keepdims=True)
        pred -= mean_t
        pred *= scale
        pred += mean_t
        if bias is not None:
            pred += bias
        return pred

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        adapt_loss: float | None = None
        # Once the correction is frozen the accumulators are never read again, so the two
        # temporal-variance passes and the residual that feed them are pure cost. This is the
        # half of the 2.9 ms the freeze actually removes.
        if prev_target_norm is not None and self._prev_pred is not None and self._frozen is None:
            revealed = torch.as_tensor(prev_target_norm).detach().cpu().numpy()[0]
            # Same space as the prediction the statistics are compared against -- evaluator
            # space when the predictor already mapped its output, physical otherwise. Mixing
            # the two would leave `gain` right (it is a ratio) and `bias` silently wrong by a
            # factor of `std_target`, which is the kind of error that scores rather than raises.
            if not self.map_out.premapped:
                revealed = to_physical_from_target(revealed)
            adapt_loss = self._observe(revealed)

        x = torch.as_tensor(input_norm).detach().cpu().numpy()
        raw = to_physical_into(x, check=not self._checked)
        meta = self._metadata(raw)
        out = self.predict_fn(self._with_history(raw), meta)
        bare = not isinstance(out, dict)
        pred = out if bare else out["prediction"]

        # The interval is deliberately NOT recentred on the calibrated prediction: the width
        # rule was fitted against the uncalibrated error field, and moving both the centre and
        # the width at once would make a board row uninterpretable. Bounds keep their own
        # centre, so only the point prediction is adapted.
        self._prev_pred = None if self._frozen is not None else pred[0].copy()
        pred = self._calibrate(pred[0])[None]

        if bare:
            self._checked = True
            return self.map_out(pred)[0], {"adapt_loss": adapt_loss}
        if not self._checked and np.any(out["lower"] > out["upper"]):
            out["lower"], out["upper"] = (
                np.minimum(out["lower"], out["upper"]),
                np.maximum(out["lower"], out["upper"]),
            )
        self._checked = True
        pred_n, lower, upper = self.map_out(pred, out["lower"], out["upper"])
        return pred_n, {"adapt_loss": adapt_loss, "lower": lower, "upper": upper}
