"""Local reproduction of the Codabench Track 1 scoring loop.

The point of this module is that it does **not** reimplement any metric. It
builds the two files the platform exchanges -- ``predictions.npz`` and
``ref/targets.npz`` -- and then shells out to the vendored, byte-identical
``scoring.py``. Anything it reports is what the leaderboard would report on the
same data.

``mean_t_neural_s`` is normalized **per sample (window)**, not per timestep. This
was confirmed against a real submission -- the CNO row recorded in ``FINDINGS.md``
§5: the ingestion log reported "mean per-sample 0.0093s, total timed 1.9s" over 210
samples, and inverting the returned ``time_score`` of 89.8606 gives t = 0.009281 s
-- i.e. ``total_timed / N``, with T_out playing no part. Reading
``T_NUMERICAL_SEC = 0.72896`` as a per-step solver cost, as we first did,
over-rewards a submission by a factor of sqrt(T_out) ~ 4.5.

That row predates the 2026-08 restart, which matters for its ``sps`` and
``final_score`` but **not** here: the ``time_score`` formula and the 210-sample
public set are unchanged across the restart, so the calibration carries over. It
cross-checks independently -- CNO is 174.1 ms/window on a T4 at an 18.8x A800
speedup, i.e. 9.26 ms against the 9.28 ms this score implies.

**Model construction and checkpoint loading are outside the timed region.** The
competition FAQ states it outright: "Model construction and checkpoint loading run
outside the timed region; only inference is timed." So there is nothing to gain by
hoisting a lazy load to import time. Loading *does* count against the separate
platform wall-clock limit (5 minutes since the August 2026 restart, container
execution only), which is not
binding in practice: a real submission used 1.9 s of it.

We therefore time ``predict()`` calls and report the first-call cost separately,
so a lazy load stays visible as warm-up rather than silently inflating the
per-window figure we compare against the leaderboard.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import T_IN, T_OUT
from ..paths import DATA_DIR, LOCAL_EVAL_DIR, OFFICIAL_SCORING, require


@dataclass
class Timing:
    """Wall-clock breakdown of a submission's ``predict()`` calls."""

    total_s: float
    first_call_s: float
    n_windows: int
    t_out: int

    @property
    def per_window_s(self) -> float:
        """Seconds per sample -- the platform's ``mean_t_neural_s`` (confirmed)."""
        return self.total_s / max(1, self.n_windows)

    @property
    def per_step_s(self) -> float:
        """Seconds per predicted timestep. Kept for diagnostics only: this is NOT
        how the platform normalizes, and using it inflates ``time_score``."""
        return self.total_s / max(1, self.n_windows * self.t_out)

    def as_dict(self) -> dict[str, float]:
        return {
            **{k: v for k, v in asdict(self).items()},
            "per_step_s": self.per_step_s,
            "per_window_s": self.per_window_s,
        }


NORMALIZATIONS = {
    "per_step": lambda t: t.per_step_s,
    "per_window": lambda t: t.per_window_s,
    "total": lambda t: t.total_s,
}


# --------------------------------------------------------------------------- #
# Building the exchange files
# --------------------------------------------------------------------------- #


#: Minimum stride for an *evaluation* set: with T_in = T_out = 20, any stride below
#: 40 puts one window's target inside another window's input.
#:
#: This is the flaw the organizers found in the original Track 1 evaluation set and
#: rebuilt the phase to fix ("the windows were cut at a stride shorter than the
#: prediction horizon, which put one window's target inside another window's input").
#: Our own eval sets had it too. A per-window model is unaffected in value, but the
#: set is exploitable by anything that reads across the batch, so it is no longer
#: buildable by accident.
#:
#: Training sets are exempt and *should* use stride 1: overlap there is just more
#: data, and a model trained per-sample cannot leak across samples it never sees
#: together.
MIN_EVAL_STRIDE = T_IN + T_OUT


def assert_leak_free_stride(stride: int, t_in: int = T_IN, t_out: int = T_OUT) -> None:
    """Raise unless ``stride`` prevents cross-window target/input overlap."""
    span = t_in + t_out
    if 0 < stride < span:
        raise ValueError(
            f"stride={stride} would leak: window s has target [s+{t_in}, s+{span}) and "
            f"window s+{stride} has input [s+{stride}, s+{stride + t_in}), which overlap. "
            f"Use stride >= {span} for an evaluation set."
        )


def write_eval_set(
    name: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    metadata: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """Persist a local eval set in the platform's own directory layout.

    Produces ``<root>/<name>/{inputs.npy, ref/targets.npz, meta.json}``. The
    ``ref/`` subdirectory is where Codabench mounts the reference, so the
    official scorer finds ``targets.npz`` without any path coaxing.
    """
    root = root or LOCAL_EVAL_DIR
    out = root / name
    (out / "ref").mkdir(parents=True, exist_ok=True)

    np.save(out / "inputs.npy", np.asarray(inputs, dtype=np.float32))
    np.savez_compressed(out / "ref" / "targets.npz", target=np.asarray(targets, dtype=np.float32))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "n_windows": int(inputs.shape[0]),
                "input_shape": list(inputs.shape),
                "target_shape": list(targets.shape),
                **(metadata or {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def load_eval_set(
    name: str, root: Path | None = None, mmap_inputs: bool = False
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read back an eval set written by :func:`write_eval_set`.

    ``mmap_inputs`` returns the input window as a read-only memory map instead of reading it.
    For a caller that wants the *targets* and touches the inputs only sometimes -- every
    consumer of ``sps_lab.Cached`` -- that turns a guaranteed 368 MB read on the pooled
    ladder into one paid for only if used, which on a 15.8 GB box with no swap is the
    difference between fitting alongside another job and not.
    """
    root = root or LOCAL_EVAL_DIR
    out = root / name
    require(
        out / "inputs.npy",
        # `--split` is not optional for anything but the example set: without it the builder
        # falls back to the example trajectory and writes it under this name, so the old hint
        # (`make eval-set NAME=<name>`) produced a silently WRONG eval set rather than an error.
        f"Build the eval set first: `make eval-set NAME={name} SPLIT={name}` "
        f"(or `make eval-set-example` for the example trajectory).",
    )
    inputs = np.load(out / "inputs.npy", mmap_mode="r" if mmap_inputs else None)
    with np.load(out / "ref" / "targets.npz") as z:
        targets = z["target"]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    return inputs, targets, meta


def _local_source(path: str) -> str:
    """An eval set's recorded source trajectory, on this machine.

    ``meta['sources']`` holds the absolute paths of the box that built the set (e.g.
    ``/home/ubuntu/realpde-t1/data/...``); on a checkout elsewhere the same file lives under
    this repo's ``DATA_DIR``. The path is used as recorded when it exists.
    """
    p = Path(path)
    if p.exists() or "data" not in p.parts:
        return path
    last = len(p.parts) - 1 - p.parts[::-1].index("data")
    return str(DATA_DIR.joinpath(*p.parts[last + 1 :]))


def prepend_eval_state(inputs: np.ndarray, meta: dict, mode: str = "meanvar") -> np.ndarray:
    """Prepend a stream-state backbone's two pseudo-frames to an eval set's windows.

    The counterpart of ``WindowDataset._state_frames`` / ``FrozenPort._with_state`` for the
    batched instruments (``sps_cache``): each window's state is the running mean and temporal
    std of its source trajectory over the frames the Track-2 stream would have revealed
    through the window's input -- ``[0, t0 + 20)``, quantised down to 20-frame reveals -- read
    from the eval set's own recorded provenance (``meta['sources']``/``meta['origins']``).
    Physical units throughout, with the n-plane pre-multiplied by the window's freestream so
    ``make_predict``'s division restores ``n_revealed/820`` exactly (the same trick the port
    uses). Each source's first covered window is asserted equal to the trajectory read back,
    so a stale eval set cannot silently pair a window with the wrong history.
    """
    if mode != "meanvar":
        raise ValueError(f"unknown stream_state mode {mode!r}")
    from ..data.h5io import read_trajectory  # noqa: PLC0415
    from ..models.predictor import estimate_freestream  # noqa: PLC0415

    t_in = int(inputs.shape[1])
    if t_in != 20:
        raise ValueError(f"stream-state eval sets are 20-frame; got t_in={t_in}")
    sources, origins = meta["sources"], meta["origins"]
    sub_s = int(meta.get("sub_s", 2))
    fields: dict[int, np.ndarray] = {}
    checked: set[int] = set()
    out = np.empty((inputs.shape[0], t_in + 2, *inputs.shape[2:]), dtype=np.float32)
    stats: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i, (src, t0) in enumerate(origins):
        arr = fields.get(src)
        if arr is None:
            arr = fields[src] = read_trajectory(_local_source(sources[src]), sub_s=sub_s).fields
        win = np.asarray(inputs[i], dtype=np.float32)
        if src not in checked:
            checked.add(src)
            if not np.allclose(arr[t0 : t0 + t_in, ..., :2], win[..., :2], atol=1e-6):
                raise AssertionError(
                    f"eval-set window {i} does not match {sources[src]} at t0={t0}; the set's "
                    "provenance is stale and no state can be built from it"
                )
        n_rev = max(t_in, (int(t0) + t_in) // 20 * 20)
        key = (src, n_rev)
        got = stats.get(key)
        if got is None:
            hist = np.asarray(arr[:n_rev, ..., :2], dtype=np.float32)
            got = stats[key] = (hist.mean(axis=0), hist.std(axis=0))
        m, sd = got
        scale = float(estimate_freestream(win[None])[0])
        out[i, 2:] = win
        out[i, :2] = 0.0
        out[i, 0, ..., :2] = m
        out[i, 0, ..., 2] = min(n_rev / 820.0, 1.0) * max(scale, 1e-6)
        out[i, 1, ..., :2] = sd
    return out


# --------------------------------------------------------------------------- #
# Running a submission
# --------------------------------------------------------------------------- #


def load_predict(submission_path: str | Path) -> Callable[..., Any]:
    """Import ``predict`` from a submission file the way the platform does.

    The submission's own directory goes on ``sys.path`` first, so sibling
    modules it ships (``load_baseline.py``, ``rpde_baselines/``, ``_vendor/``)
    resolve exactly as they do inside the evaluation container.
    """
    path = Path(submission_path).resolve()
    require(path, "Point --submission at a submission.py-style file.")

    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    spec = importlib.util.spec_from_file_location(f"_submission_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import a module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "predict"):
        raise AttributeError(f"{path} defines no `predict` function.")
    return module.predict


def _split_output(out: Any) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Normalize ``predict()``'s return value into (prediction, lower, upper)."""
    if isinstance(out, dict):
        if "prediction" not in out:
            raise KeyError("A dict return from predict() must contain 'prediction'.")
        pred = np.asarray(out["prediction"], dtype=np.float32)
        lower = out.get("lower")
        upper = out.get("upper")
        return (
            pred,
            None if lower is None else np.asarray(lower, dtype=np.float32),
            None if upper is None else np.asarray(upper, dtype=np.float32),
        )
    return np.asarray(out, dtype=np.float32), None, None


def run_submission(
    predict: Callable[..., Any],
    inputs: np.ndarray,
    batch_size: int | None = None,
    metadata: dict | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Timing]:
    """Run ``predict`` over ``inputs``, timing only the prediction calls.

    ``batch_size=None`` hands the whole array over in one call, which is what the
    organizers' own FNO example expects (it batches internally). Chunking is
    available for models that would otherwise run out of memory.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    n = int(inputs.shape[0])
    chunks = (
        [(0, n)]
        if batch_size is None
        else [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
    )

    preds: list[np.ndarray] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    total = 0.0
    first = 0.0

    for idx, (lo, hi) in enumerate(chunks):
        t0 = time.perf_counter()
        out = predict(inputs[lo:hi], metadata)
        dt = time.perf_counter() - t0
        total += dt
        if idx == 0:
            first = dt

        pred, low, up = _split_output(out)
        preds.append(pred)
        if low is not None:
            lowers.append(low)
        if up is not None:
            uppers.append(up)

    prediction = np.concatenate(preds, axis=0)
    lower = np.concatenate(lowers, axis=0) if len(lowers) == len(chunks) and lowers else None
    upper = np.concatenate(uppers, axis=0) if len(uppers) == len(chunks) and uppers else None

    timing = Timing(total_s=total, first_call_s=first, n_windows=n, t_out=T_OUT)
    return prediction, lower, upper, timing


def write_predictions(
    out_dir: Path,
    prediction: np.ndarray,
    mean_t_neural_s: float,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> Path:
    """Write ``predictions.npz`` in exactly the form the scorer reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "prediction": np.asarray(prediction, dtype=np.float32),
        "mean_t_neural_s": np.asarray([mean_t_neural_s], dtype=np.float64),
    }
    if lower is not None:
        payload["lower"] = np.asarray(lower, dtype=np.float32)
    if upper is not None:
        payload["upper"] = np.asarray(upper, dtype=np.float32)
    path = out_dir / "predictions.npz"
    np.savez_compressed(path, **payload)
    return path


# --------------------------------------------------------------------------- #
# The official scorer
# --------------------------------------------------------------------------- #


def run_official_scorer(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Invoke the vendored ``scoring.py`` as a subprocess, as Codabench does.

    Running it out-of-process rather than importing it keeps the contract
    honest: we cannot accidentally monkeypatch a metric, and the exit path
    (including its catch-all zero-score branch) is the real one.
    """
    require(OFFICIAL_SCORING, "The vendored starting kit is missing from vendor/.")
    output_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [sys.executable, str(OFFICIAL_SCORING), str(input_dir), str(output_dir)],
        capture_output=True,
        text=True,
    )
    scores_path = output_dir / "scores.json"
    if not scores_path.exists():
        raise RuntimeError(
            "The official scorer produced no scores.json.\n"
            f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return json.loads(scores_path.read_text(encoding="utf-8"))
