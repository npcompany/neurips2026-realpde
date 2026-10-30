"""The local Track-2 streaming harness: build a stream, drive a TTT model, score it.

This mirrors the official ingestion program's loop closely enough that a local number is
worth something: trajectory order, batch size 1, ``reset_ttt_state`` at boundaries and
timed into the following step, the previous target cached *after* the timed call, and
subscores from the vendored official ``scoring.py`` rather than a re-implementation.

Two invariants are the opposite of Track 1's, and getting either wrong makes every number
here meaningless:

* **Stride must equal 20.** :func:`rpde_t1.score.harness.assert_leak_free_stride` forbids
  exactly this and is correct to, *for Track 1*. Here the overlap is the protocol, and
  :func:`assert_streaming_stride` enforces it in the other direction.
* **Whole trajectories are held out, or nothing is.** With overlapping windows there is no
  frame-level holdout to be had, so contamination is decided entirely by which files the
  backbone trained on. ``splits.assign_splits`` already partitions by whole cell.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from rpde_t1.data.h5io import read_trajectory
from rpde_t1.paths import OFFICIAL_SCORING_T2

from . import IN_STEP, INTERVAL, OUT_STEP, SUB_S

HORIZON = IN_STEP + OUT_STEP


def assert_streaming_stride(stride: int) -> None:
    """Raise unless ``stride`` is the Track-2 protocol stride.

    Deliberately strict, and deliberately the mirror image of Track 1's guard. A Track-2
    stream built at stride 40 is not a safer Track-2 stream -- it is a *different task*,
    because it stops handing the model a previous target that continues into the current
    input, which is the only structure adaptation has to work with.
    """
    if stride != INTERVAL:
        raise ValueError(
            f"stride={stride} is not the Track-2 protocol stride {INTERVAL}. Consecutive "
            f"windows must overlap (window k's target frames are window k+1's input "
            f"frames); rpde_t1.score.harness.assert_leak_free_stride is the Track-1 rule "
            f"and must not be applied here."
        )


@dataclass
class Step:
    """One evaluation step: an input window, its target, and where it came from."""

    inputs: np.ndarray  # (T_in, H, W, 3) physical
    target: np.ndarray  # (T_out, H, W, 3) physical
    sim_id: str
    time_id: int
    is_first: bool

    @property
    def sample_id(self) -> str:
        return f"{self.sim_id}:{self.time_id}"


def build_stream(
    paths: Sequence[Path | str],
    stride: int = INTERVAL,
    sub_s: int = SUB_S,
    max_steps_per_traj: int | None = None,
) -> list[Step]:
    """Consecutive windows over whole trajectories, in trajectory order.

    ``max_steps_per_traj`` truncates each trajectory, for a cheap smoke run. It shortens
    the adaptation history every step sees, so it is not a valid setting to *measure* an
    adaptive method on -- only to check that one runs.
    """
    assert_streaming_stride(stride)
    steps: list[Step] = []
    for path in paths:
        path = Path(path)
        traj = read_trajectory(path, sub_s=sub_s)
        fields = traj.fields
        n_windows = (fields.shape[0] - HORIZON) // stride + 1
        if n_windows < 1:
            continue
        if max_steps_per_traj is not None:
            n_windows = min(n_windows, max_steps_per_traj)
        for k in range(n_windows):
            t0 = k * stride
            steps.append(
                Step(
                    inputs=fields[t0 : t0 + IN_STEP],
                    target=fields[t0 + IN_STEP : t0 + HORIZON],
                    sim_id=path.stem,
                    time_id=t0,
                    is_first=(k == 0),
                )
            )
    return steps


@dataclass
class StreamResult:
    """Stacked predictions, targets and optional bounds, plus the per-step wall times.

    Everything is **physical**, which is the space the official scorer works in: the
    harness denormalizes what ``ttt_step`` returns exactly as the ingestion program does.
    """

    prediction: np.ndarray
    target: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    per_step_s: np.ndarray
    sample_id: list[str] = field(default_factory=list)

    @property
    def mean_t_neural_s(self) -> float:
        return float(np.mean(self.per_step_s))


def run_stream(model: Any, steps: Sequence[Step], device: str = "cpu") -> StreamResult:
    """Drive ``model`` through ``steps`` under the evaluator's timing and caching rules."""
    import torch

    from . import to_normalized

    preds: list[np.ndarray] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    times: list[float] = []
    bounds_active: bool | None = None
    prev_target_norm: np.ndarray | None = None

    for step in steps:
        # reset_ttt_state is timed and charged to the step that follows the boundary, so a
        # local time_score cannot read better than the real one by moving work there.
        reset_s = 0.0
        if step.is_first:
            t = perf_counter()
            model.reset_ttt_state()
            reset_s = perf_counter() - t
            prev_target_norm = None

        input_norm = _to_input_space(step.inputs)[None]
        prev = None if prev_target_norm is None else torch.from_numpy(prev_target_norm)

        t = perf_counter()
        pred_norm, info = model.ttt_step(torch.from_numpy(input_norm), prev)
        times.append(perf_counter() - t + reset_s)

        # Cached AFTER the timed call, so the current target can never reach the current
        # prediction -- the evaluator does the same, and it is the rule the whole track
        # rests on.
        prev_target_norm = to_normalized(step.target)[None]

        if not isinstance(info, dict) or "adapt_loss" not in info:
            raise ValueError('ttt_step must return (pred, info) with info["adapt_loss"]')
        has_bounds = "lower" in info or "upper" in info
        if has_bounds and not ("lower" in info and "upper" in info):
            raise ValueError("SPS bounds must be supplied together, or not at all")
        if bounds_active is None:
            bounds_active = has_bounds
        elif has_bounds != bounds_active:
            raise ValueError(
                "a run must supply SPS bounds on every step or on none; this one "
                f"{'started' if has_bounds else 'stopped'} partway through"
            )

        preds.append(_from_target_space(pred_norm))
        if has_bounds:
            lowers.append(_from_target_space(info["lower"]))
            uppers.append(_from_target_space(info["upper"]))

    return StreamResult(
        prediction=np.stack(preds),
        target=np.stack([s.target for s in steps]),
        lower=np.stack(lowers) if bounds_active else None,
        upper=np.stack(uppers) if bounds_active else None,
        per_step_s=np.asarray(times, dtype=np.float64),
        sample_id=[s.sample_id for s in steps],
    )


def _to_input_space(raw: np.ndarray) -> np.ndarray:
    from rpde_t1.features import REAL_MEAN_IN, REAL_STD_IN

    return ((np.asarray(raw, dtype=np.float32) - REAL_MEAN_IN) / REAL_STD_IN).astype(np.float32)


def _from_target_space(x: Any) -> np.ndarray:
    import torch

    from rpde_t1.features import REAL_MEAN_TARGET, REAL_STD_TARGET

    a = torch.as_tensor(x).detach().cpu().numpy()
    return np.asarray(a[0] * REAL_STD_TARGET + REAL_MEAN_TARGET, dtype=np.float32)


def official_scoring(path: Path | None = None):
    """Import the vendored Track-2 ``scoring.py`` as a module.

    Loaded by path rather than added to ``sys.path`` because the Track-1 kit ships a file
    of the same name, and which one answers ``import scoring`` would then depend on import
    order. The two differ, so that ambiguity is not acceptable in a scoring path.
    """
    path = path or OFFICIAL_SCORING_T2
    spec = importlib.util.spec_from_file_location("rpde_t2_official_scoring", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise ImportError(f"could not load the official Track-2 scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score_stream(result: StreamResult, scorer: Any | None = None) -> dict[str, float]:
    """The five subscores, from the vendored official scorer.

    No ``final_score``: the combination is unpublished. Use
    :func:`rpde_t2.report.board_final` on *board* rows once the rule is confirmed for
    Track 2.
    """
    official = scorer or official_scoring()
    pred, target = result.prediction, result.target
    c = official.measured_channels(target)
    sps, _coverage = official.aggregate_sps(pred, target, c, lower=result.lower, upper=result.upper)
    return {
        "rel_l2_score": official.score_error(
            float(np.mean(official.rel_l2_per_sample(pred, target, c)))
        ),
        "tke_score": official.score_error(
            float(np.mean(official.tke_rel_l2_per_sample(pred, target, c)))
        ),
        "mvpe_score": official.score_error(official.mvpe_rel_l2(pred, target)),
        "time_score": official.score_time(result.mean_t_neural_s),
        "sps_score": official.score_sps(sps),
    }


def time_score_to_ms(time_score: float, t_numerical: float = 0.72896) -> float:
    """Invert ``time_score`` to mean ms/step.

    The board reports the score, not the time. Combined with the platform's wall clock this
    also recovers the hidden stream's step count, which is otherwise unobservable.
    """
    if not 0.0 < time_score < 100.0:
        raise ValueError(f"time_score must lie in (0, 100); got {time_score}")
    return ((100.0 / time_score) - 1.0) ** 2 * t_numerical * 1e3


__all__ = [
    "HORIZON",
    "Step",
    "StreamResult",
    "assert_streaming_stride",
    "build_stream",
    "official_scoring",
    "run_stream",
    "score_stream",
    "time_score_to_ms",
]
