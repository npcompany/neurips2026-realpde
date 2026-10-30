"""Weights & Biases logging for the training loop.

Optional by construction, in three ways that all matter here:

* **Lazy import.** ``wandb`` is an extra (``uv sync --extra wandb``), never a dependency
  of the scoring path or of a submission bundle -- the evaluation container ships only
  torch + numpy and has no network, so an import at module scope would be one more way
  to fail the smoke test.
* **Never fatal.** A missing package, a refused handshake or a dropped connection
  degrades to a printed note and a run that keeps training. A six-hour fine-tune on a
  shared T4 must not die because a logger could not reach the internet.
* **Never the only record.** Everything logged here is also written to
  ``history.json``/``config.json`` next to the checkpoint. W&B is a second view of a run,
  so a run whose logging was off is not a run with missing data.

Checkpoints are deliberately *not* uploaded as artifacts: ``last.pt`` is ~384 MB and
``scripts/gpu.sh`` already refuses to start below 4 GB free disk.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import TrainConfig

__all__ = ["WandbRun", "score_row", "train_row", "val_row"]

#: W&B turns the text before the first ``/`` of a key into a panel *section*, so the number
#: of sections is the number of distinct prefixes. The layout here follows the one the
#: sibling SimFormer project settled on: keep the handful of curves a person actually watches
#: as **flat keys** -- no ``/`` -- so they all sit together in the default *Charts* section
#: (train and val loss adjacent, the schedule and the health signals beside them), and reach
#: for a ``metric/split`` prefix only where a *family* of splits is worth collecting under one
#: heading. That is three grouped sections (``rel_l2``, ``tke``, ``score``) plus Charts,
#: against the seven a prefix-per-quantity scheme produced -- readable in one scroll.
#:
#: Run-level constants (parameter counts, best step, wall clock) go to ``run.summary``, which
#: is a column in the run table rather than a panel, so they never cost a section at all.
#:
#: These rules only set how the summary column *reduces* each series (``min`` for an error,
#: ``max`` for a score) so a run table sorts on a run's best figure, not its last. Applied
#: best-effort; a W&B build without ``define_metric`` just skips them.
_SUMMARY_RULES: tuple[tuple[str, str], ...] = (
    ("loss_*", "min"),
    ("rel_l2/*", "min"),
    ("tke/*", "min"),
    ("score/*", "max"),
    ("select_score", "min"),
)


def train_row(
    parts: dict[str, float],
    lr: float,
    it_per_s: float,
    grad_norm: float | None = None,
    epoch: float | None = None,
) -> dict[str, float]:
    """The per-``log_every`` row: the loss, the schedule, and training health -- flat.

    The total is ``loss_train``, so it sits in the Charts section right next to the
    ``loss_<split>`` validation losses for the over-fitting read; the individual loss terms
    ride along as ``train_<term>`` in the same section. ``lr``, ``it_per_s``, ``grad_norm``
    and ``epoch`` are the health signals -- ``grad_norm`` because it is the cheapest early
    warning of a diverging run. No key carries a ``/``, so none of this spills into a
    separate section.
    """
    row: dict[str, float] = {}
    for term, value in parts.items():
        # ``fine/vs_upsample`` carries a slash; flatten it so a diagnostic does not open a
        # section of its own.
        row["loss_train" if term == "total" else f"train_{term.replace('/', '_')}"] = value
    row["lr"] = lr
    row["it_per_s"] = it_per_s
    if grad_norm is not None:
        row["grad_norm"] = grad_norm
    if epoch is not None:
        row["epoch"] = epoch
    return row


def val_row(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """The per-``eval_every`` row: the validation loss (flat) and the diagnostics (grouped).

    ``loss_<split>`` is flat so it lands in Charts beside ``loss_train`` -- that adjacency is
    the whole point, train vs val in one place. The non-dimensional diagnostics group under
    ``rel_l2/<split>`` and ``tke/<split>``, one section each collecting all three splits.
    ``loss`` appears only when :func:`~rpde_t1.train.loop.quick_eval` was given a config to
    compute it; ``n_windows`` is bookkeeping and stays in ``history.json``, off the dashboard.

    These are in-loop, non-dimensional diagnostics, not leaderboard scores -- see
    :func:`score_row` for those.
    """
    row: dict[str, float] = {}
    for split, m in metrics.items():
        for metric, value in m.items():
            if metric == "n_windows":
                continue
            if metric == "loss":
                row[f"loss_{split}"] = value
            else:
                row[f"{metric}/{split}"] = value
    return row


def score_row(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """The per-score-point row: the official subscores, ``score/<metric>/<split>``.

    Only the ``*_score`` keys of each split's scorer output become series -- ``rel_l2``,
    ``tke``, ``mvpe``, ``time``, ``sps`` -- so the private diagnostics the scorer also
    returns (per-window seconds, normalised MSE, coverage) stay out of the panels. One
    ``score/`` section, kept apart from the ``rel_l2``/``tke`` diagnostic sections so a
    leaderboard number is never mistaken for the non-dimensional one it tracks.
    """
    return {
        f"score/{k[: -len('_score')]}/{split}": float(v)
        for split, s in scores.items()
        for k, v in s.items()
        if k.endswith("_score")
    }


class WandbRun:
    """A W&B run, or a no-op stand-in for one.

    Always constructed through :meth:`start`, so the training loop never branches on
    whether logging is enabled: a disabled run swallows every call. A logging failure
    mid-run disables the run rather than propagating -- see the module docstring.
    """

    def __init__(self, run: Any | None = None) -> None:
        self._run = run

    @property
    def active(self) -> bool:
        """Whether calls actually reach W&B."""
        return self._run is not None

    @classmethod
    def start(cls, cfg: TrainConfig) -> WandbRun:
        """Open a run for ``cfg``, or return an inactive stand-in.

        The whole ``TrainConfig`` goes up as the run config, which is what makes two W&B
        runs diffable in the same way their ``config.json`` files are.
        """
        if not cfg.wandb:
            return cls()
        try:
            import wandb  # noqa: PLC0415
        except ImportError:
            print("note: wandb requested but not installed (`uv sync --extra wandb`); off")
            return cls()

        try:
            run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
                name=cfg.wandb_name or Path(cfg.out_dir).name,
                group=cfg.wandb_group or None,
                tags=[t for t in cfg.wandb_tags.split(",") if t],
                notes=cfg.notes or None,
                # Keep the run's own files next to its checkpoints instead of in the cwd,
                # so deleting a run directory takes its logs with it.
                dir=cfg.out_dir,
                config=asdict(cfg),
                # No id/resume: two runs may share an output directory (``--force``), and
                # resuming one into the other would splice two step axes together. A
                # short init timeout because the caller is usually holding the GPU lock.
                settings={"init_timeout": 60},
            )
        except Exception as exc:  # noqa: BLE001 -- logging must never end a training run
            print(f"note: wandb.init failed ({type(exc).__name__}: {exc}); continuing without it")
            return cls()

        # Best-effort summary reductions so the run table sorts on a run's best figures.
        # Never fatal and never disabling: a build without define_metric, or an older one
        # that rejects a glob, just leaves the defaults -- the curves are unaffected.
        for pattern, reduction in _SUMMARY_RULES:
            try:
                run.define_metric(pattern, summary=reduction)
            except Exception:  # noqa: BLE001 -- a cosmetic call must never touch the run
                break

        print(f"wandb: logging to {getattr(run, 'url', None) or cfg.wandb_project}")
        return cls(run)

    def log(self, row: dict[str, float], step: int) -> None:
        """Log one row against the training step."""
        self._guard(lambda: self._run.log(row, step=step))

    def summary(self, values: dict[str, Any]) -> None:
        """Set run-level constants (parameter counts, the best step, wall clock).

        Kept out of :meth:`log` on purpose: a constant logged per step would be a flat
        series to scroll past instead of a field to sort a run table by.
        """
        self._guard(lambda: self._run.summary.update(values))

    def finish(self) -> None:
        """Close the run. Idempotent, so the loop can call it on every exit path."""
        self._guard(lambda: self._run.finish())
        self._run = None

    def _guard(self, call) -> None:
        if self._run is None:
            return
        try:
            call()
        except Exception as exc:  # noqa: BLE001 -- see the module docstring
            print(f"note: wandb logging failed ({type(exc).__name__}: {exc}); disabling it")
            self._run = None
