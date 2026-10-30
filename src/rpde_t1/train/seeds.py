"""Run the same config under several seeds and report the decision threshold.

Why this is a command rather than a shell loop. `scripts/eval/seed_variance.sh` measured the
*frozen bound head* by hand and gave sd 0.03-0.28 SPS. That number is quoted throughout
the project, but it licenses only frozen-head comparisons -- and every remaining item on
`ROADMAP.md` trains a backbone, where the noise floor is unmeasured. Repeating the shell
loop per experiment would reproduce it by hand each time, differently each time. So the
sweep is defined once, here, and is:

*Systematic* -- one entry point takes the training arguments, fans them over seeds, scores
every run with the official scorer, and prints the thresholds. There is no step where a
human transcribes a number.

*Resumable* -- a seed whose run directory already holds a checkpoint is skipped, because a
five-seed backbone sweep is hours on one GPU and losing it to an interruption is the
difference between doing this and not doing it. Use ``--force`` to retrain.

*Inspectable before it costs anything* -- ``--dry-run`` prints the exact plan and exits, so
the wiring is checkable without spending GPU time. That is also how it is tested.

*Two-armed when needed* -- ``--arm NAME=--flag ...`` measures an A/B and its noise in one
sweep, which is the only way an effect and its threshold come from the same conditions.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..paths import REPO_ROOT
from ..score.noise import format_seed_noise, seed_noise
from .config import TrainConfig

#: Where sweeps live. One directory per (arm, seed) so a run is identifiable from its path
#: alone, which matters when a table is assembled from a glob months later.
SEED_ROOT = "runs/seed"

#: cuBLAS needs a fixed workspace before its reductions are run-to-run stable, and
#: ``torch.use_deterministic_algorithms`` raises at the first GEMM if this is unset. It is
#: read when the cuBLAS handle is created, so :func:`seed_everything` must run before the
#: first matmul -- which is why it is the first statement of every entry point rather than
#: something the caller may do later.
CUBLAS_WORKSPACE = ":4096:8"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG this project draws from, and optionally pin the CUDA kernels.

    Seeding alone does **not** make a run reproducible, which is the finding this function
    exists to encode. Measured on a T4, same seed, same data, twelve steps:

    * ``stformer`` -- the shipping architecture -- came back with **130 of 144 tensors
      different** (max 7.3e-7) between two runs. ``cudnn.deterministic`` alone does *not*
      close it: the drift is in the atomics of the bilinear-interpolate backward in the
      decode path, so ``use_deterministic_algorithms`` is the flag that matters. UNet and
      FNO happened to be reproducible already, which is exactly why this went unnoticed.
    * Evaluating one *fixed* checkpoint three times gave three different scores
      (``rel_l2`` spread 2.0e-9, ``tke`` spread 7.9e-8). Cause and fix are the ones
      ``scripts/sps/sps_cache.py`` documents for its own cache: cuDNN picks a convolution
      algorithm against the workspace it can get, and this box shares its GPU, so the same
      forward pass takes a different path depending on what else is resident.

    On A10G with Torch 2.2.2, fused attention also needs to be disabled (2026-09-14).
    A 100-step replay differed in all 121 shipping-model tensors with the flags alone;
    forcing math attention made the replay exact. The backend is pinned below as part
    of the deterministic contract. Older checkpoints trained before this fix cannot
    be expected to regenerate their exact weights.

    The earlier T4 checks for the cuDNN/interpolation fixes were exact and measured
    ~3-5% training overhead. Math attention adds a hardware-dependent cost. This helper
    is absent from packaged inference: ``models.predictor.make_predict`` keeps its
    serving backend, and timings must come from the packaged-artifact benchmark.

    ``deterministic=False`` restores the old behaviour for a run that would rather have the
    throughput; ``TrainConfig.deterministic`` records which was used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device

    if not deterministic:
        return

    # Set rather than required: a caller that has already pinned it to something valid
    # keeps its choice, and one that has not gets a working default instead of a crash at
    # the first GEMM. Only ever widened, never narrowed.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ``warn_only=False`` on purpose. An op with no deterministic kernel must be a loud
    # failure, because the alternative is a run that reports itself reproducible and is
    # not -- the exact defect this function was written to remove. All nine architectures
    # in ``models.factory`` were trained under it before it was made the default.
    torch.use_deterministic_algorithms(True)
    # Torch 2.2.2 on A10G: the fused SDPA training path still diverges with the
    # deterministic-algorithms flag set. Two identical 100-step shipping-model
    # runs differed in all 121 tensors; math SDPA matched every tensor exactly.
    # This helper is training/analysis-only; packaged inference keeps its fast backend.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 -- torch passes it positionally
    """``worker_init_fn`` that gives each DataLoader worker its own numpy/random stream.

    PyTorch seeds ``torch``, ``random`` and ``np.random`` per worker already. What it
    cannot reach is a generator a dataset built in its own ``__init__`` and carried across
    the fork -- ``data.dataset.WindowDataset._rng`` is one, and every augmentation draws
    from it. Forked, all ``num_workers`` copies hold the *same* state, so worker ``w``'s
    k-th draw equals worker 0's k-th draw:

        num_workers=4, four consecutive samples, one draw each:
            [ 0.1257, -0.1321]   <- identical
            [ 0.1257, -0.1321]
            [ 0.1257, -0.1321]
            [ 0.1257, -0.1321]

    Two consequences, both measured on ``train``/``real`` with ``real_noise_scale=0.05``.
    The augmentation is ``num_workers`` times less diverse than the config asks for. And
    the stream is a function of ``num_workers`` -- 0, 2 and 4 workers give different
    samples (max 0.34, i.e. full scale) from one seed -- so a throughput knob silently
    changes the training data.

    Reseeding from the worker's own base seed fixes both, and keeps the result a function
    of ``(seed, num_workers)`` rather than of the fork.
    """
    dataset = torch.utils.data.get_worker_info().dataset
    seed = torch.initial_seed() % 2**32
    for member in _rng_holders(dataset):
        member._rng = np.random.default_rng(seed)


def _rng_holders(dataset) -> list:
    """Every ``_rng``-carrying dataset reachable from ``dataset``, through concatenation.

    ``make_loaders`` wraps the training set in a ``ConcatDataset`` for phase augmentation
    and for the joint sim/real mixture, so the datasets that actually own an ``_rng`` are
    one level down and a check on the top-level object finds nothing.
    """
    inner = getattr(dataset, "datasets", None)
    if inner is not None:
        return [h for d in inner for h in _rng_holders(d)]
    return [dataset] if hasattr(dataset, "_rng") else []


#: Checkpoint a completed run is expected to leave behind. ``last.pt`` rather than
#: ``best.pt`` deliberately: ``best.pt`` is selected on ``quick_eval``, a non-dimensional
#: metric over a subset, which disagrees with the official scorer -- and on the scorer
#: ``last.pt`` measured marginally better.
CHECKPOINT = "last.pt"


def run_dir(arm: str, seed: int, root: str = SEED_ROOT) -> Path:
    """Directory for one (arm, seed). Absolute, so it does not depend on the cwd."""
    path = Path(root) / f"{arm}_s{seed}"
    return path if path.is_absolute() else REPO_ROOT / path


def plan_sweep(
    base: TrainConfig,
    seeds: tuple[int, ...],
    arms: dict[str, dict[str, Any]] | None = None,
    root: str = SEED_ROOT,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate the runs a sweep needs, marking which already exist.

    Pure: touches the filesystem only to ask whether a checkpoint is there. That is what
    makes ``--dry-run`` meaningful and the whole thing testable without a GPU.
    """
    arms = arms or {"base": {}}
    plan = []
    for arm, overrides in arms.items():
        for seed in seeds:
            out = run_dir(arm, seed, root)
            cfg = replace(base, seed=seed, out_dir=str(out), **overrides)
            done = (out / CHECKPOINT).exists()
            plan.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "out_dir": str(out),
                    "checkpoint": str(out / CHECKPOINT),
                    "exists": done,
                    "will_train": force or not done,
                    "overrides": dict(overrides),
                    "config": cfg,
                }
            )
    return plan


def format_plan(plan: list[dict[str, Any]], splits: tuple[str, ...]) -> str:
    """What the sweep will do, and what it will cost, before it does it."""
    to_train = [p for p in plan if p["will_train"]]
    lines = [
        f"sweep: {len(plan)} run(s) over {len({p['arm'] for p in plan})} arm(s) "
        f"x {len({p['seed'] for p in plan})} seed(s)",
        f"  {len(to_train)} to train, {len(plan) - len(to_train)} already present",
        f"  scored on: {', '.join(splits)}",
        "",
    ]
    for p in plan:
        mark = "TRAIN" if p["will_train"] else "skip "
        extra = " ".join(f"{k}={v}" for k, v in p["overrides"].items())
        lines.append(f"  [{mark}] {p['out_dir']}{'  ' + extra if extra else ''}")

    if to_train:
        steps = to_train[0]["config"].steps
        lines += [
            "",
            f"each run is {steps} steps; on one T4 an FNO fine-tune of that length is "
            "roughly 1 GPU-h,",
            "so budget accordingly before launching. Interrupted sweeps resume: rerun the "
            "same command.",
        ]
    else:
        lines += ["", "nothing to train -- this would go straight to scoring."]
    return "\n".join(lines)


def score_sweep(
    plan: list[dict[str, Any]],
    splits: tuple[str, ...],
    bounds: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Score every run in the plan, keyed by ``<arm>_s<seed>``."""
    from .evaluate import evaluate_checkpoint  # noqa: PLC0415  (torch import, keep it lazy)

    table = {}
    for p in plan:
        ckpt = Path(p["checkpoint"])
        if not ckpt.exists():
            continue
        table[Path(p["out_dir"]).name] = evaluate_checkpoint(
            ckpt, splits=tuple(splits), bounds=bounds
        )
    return table


def run_sweep(
    base: TrainConfig,
    seeds: tuple[int, ...],
    splits: tuple[str, ...],
    arms: dict[str, dict[str, Any]] | None = None,
    root: str = SEED_ROOT,
    force: bool = False,
    bounds: bool = False,
    dry_run: bool = False,
    out: Path | None = None,
) -> dict[str, Any] | None:
    """Train the missing seeds, score everything, and report the thresholds."""
    from .loop import train  # noqa: PLC0415

    plan = plan_sweep(base, seeds, arms=arms, root=root, force=force)
    print(format_plan(plan, splits))
    if dry_run:
        print("\n--dry-run: nothing trained, nothing scored.")
        return None

    for p in plan:
        if not p["will_train"]:
            print(f"\n=== skip {p['out_dir']} (checkpoint present) ===")
            continue
        print(f"\n=== train {p['out_dir']} ===")
        train(p["config"])

    table = score_sweep(plan, splits, bounds=bounds)
    if not table:
        raise SystemExit("No checkpoints to score; did every run fail?")

    arm_of = {Path(p["out_dir"]).name: p["arm"] for p in plan}
    noise = seed_noise(table, arm_of=arm_of)
    noise["frozen_backbone"] = bool(base.bounds and base.freeze_backbone)
    print()
    print(format_seed_noise(noise, frozen=noise["frozen_backbone"]))

    if out:
        payload = {"table": table, "noise": noise}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten: {out}")
    return noise
