"""Score a trained checkpoint with the official scorer, on the held-out splits.

This is the only place a number may be called a "score". It reuses the existing
harness, which shells out to the vendored ``scoring.py``, and it drives the model
through :func:`rpde_t1.models.predictor.make_predict` -- the same closure a
submission uses. So a validation number here and a leaderboard number differ only
in the data they were computed on.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .. import SUBSCORES
from ..data.normalizers import normalized_mse
from ..models.predictor import make_predict
from ..score.harness import (
    NORMALIZATIONS,
    load_eval_set,
    run_official_scorer,
    run_submission,
    write_predictions,
)
from .config import TrainConfig
from .loop import build_model, flow_names, geom_names
from .seeds import seed_everything

#: Structural ceilings per axis. Since the August 2026 restart sps_score maps its
#: aggregate to 0-100 linearly, so the whole axis is reachable (it was previously
#: confined to [50, 73.1] by a logistic).
AXIS_CEILING = {
    "rel_l2_score": 100.0,
    "tke_score": 100.0,
    "mvpe_score": 100.0,
    "time_score": 100.0,
    "sps_score": 100.0,
}


#: Config fields introduced after checkpoints already existed, mapped to the behaviour
#: those older checkpoints actually have. Anything that changes the *architecture* must
#: be listed here, or loading an old checkpoint builds the wrong shape.
POST_HOC_DEFAULTS = {"bound_cond": False, "head_lr_mult": 1.0}


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, TrainConfig]:
    """Rebuild a model and its config from a checkpoint written by the loop."""
    # Normalised here because every caller in scripts/ passes a plain string: the
    # unknown-field report below reaches for ``path.name`` and only runs for checkpoints that
    # *have* unknown fields, so a str path failed on exactly the newer-config case the report
    # exists to announce.
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    raw = dict(payload["config"])
    raw["val_splits"] = tuple(raw.get("val_splits", ()))
    # Fields added after a checkpoint was written must read back as whatever was true
    # when it was written, not as today's default -- otherwise the rebuilt architecture
    # does not match the saved weights. ``bound_cond`` defaults to True for new runs but
    # did not exist for the earlier bound heads, which have a 10-channel input.
    for key, was in POST_HOC_DEFAULTS.items():
        raw.setdefault(key, was)
    # The mirror case: a checkpoint written by a *newer* config than this working tree's,
    # which happens whenever two branches train into the same ``runs/``. Unknown fields are
    # dropped rather than raising, and named on stderr so the drop is never silent -- if one
    # of them was architectural, ``load_state_dict`` below is strict and will say so.
    unknown = {k: raw.pop(k) for k in set(raw) - {f.name for f in fields(TrainConfig)}}
    if unknown:
        print(f"load_checkpoint({path.name}): dropping unknown config fields {sorted(unknown)}")
    cfg = TrainConfig(**raw)
    model = build_model(cfg)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), cfg


def _load_width_rule(path: str | None):
    if not path:
        return None
    from ..models.width_rule import WidthRule

    return WidthRule.from_json(Path(path))


def evaluate_checkpoint(
    ckpt: Path,
    splits: tuple[str, ...] = ("val_time", "val_interp", "val_extrap"),
    device: str | None = None,
    batch_size: int = 32,
    bounds: bool = False,
    bound_floor: float = 0.0,
    time_normalization: str = "per_window",
    stats_domain: str | None = None,
    tke_blend: float = 0.0,
    tke_head: str | None = None,
    tke_shrink: float | None = None,
    width_rule: str | None = None,
) -> dict[str, dict]:
    """Score one checkpoint on each split. Returns ``{split: scores}``.

    ``tke_blend`` turns on the post-hoc fluctuation rescaling, so the on/off arms of
    that comparison come from one artefact rather than from two training runs.

    ``width_rule`` is the path of a fitted ``width_rule.json``: with ``bounds`` it sizes the
    interval exactly as a packaged bundle serves it. Without it the bounds are the head's raw
    scale, which is ``sps_apply``'s ``head_only`` rung and ~22 sps below the shipped stack.

    Scoring is pinned deterministic here rather than left to the caller, because a score
    is the project's unit of record and this was **not** reproducible: three evaluations
    of one fixed checkpoint returned three different numbers (``rel_l2`` spread 2.0e-9,
    ``tke`` spread 7.9e-8), for the cuDNN algorithm-selection reason
    ``scripts/sps/sps_cache.py`` documents. It is the checkpoint's own recorded choice that is
    honoured, so a run trained with ``deterministic=False`` is not silently re-graded
    under different arithmetic than it was produced with.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(Path(ckpt), device=device)
    seed_everything(cfg.seed, cfg.deterministic)

    head = None
    head_tau = False
    if tke_head:
        from ..models.tke_head import head_uses_tau, load_head

        head = load_head(tke_head, device=device)
        # Read off the checkpoint rather than guessed from the weights: see
        # ``tke_head.head_uses_tau``. Getting it wrong serves the head a channel it never saw.
        head_tau = head_uses_tau(tke_head)

    return score_model(
        model,
        cfg,
        splits=splits,
        device=device,
        batch_size=batch_size,
        bounds=bounds,
        bound_floor=bound_floor,
        time_normalization=time_normalization,
        stats_domain=stats_domain,
        tke_blend=tke_blend,
        tke_head=head,
        tke_shrink=tke_shrink,
        tke_head_uses_tau=head_tau,
        width_rule=_load_width_rule(width_rule),
    )


def score_model(
    model: torch.nn.Module,
    cfg: TrainConfig,
    splits: tuple[str, ...] = ("val_time", "val_interp", "val_extrap"),
    device: str | None = None,
    batch_size: int = 32,
    bounds: bool = False,
    bound_floor: float = 0.0,
    time_normalization: str = "per_window",
    stats_domain: str | None = None,
    tke_blend: float = 0.0,
    tke_head: torch.nn.Module | None = None,
    tke_shrink: float | None = None,
    tke_head_uses_tau: bool = False,
    width_rule=None,
) -> dict[str, dict]:
    """Score an *in-memory* model on each split. Returns ``{split: scores}``.

    The body :func:`evaluate_checkpoint` shares once a checkpoint is loaded, split out so
    the training loop can score the live model at each validation without a save/reload
    round trip (``score_points``). It drives the model through :func:`make_predict` -- the
    same closure a submission uses -- so an in-loop score and a leaderboard score differ
    only in the data they see. ``tke_head`` here is an already-loaded module, not a path.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head = tke_head

    results: dict[str, dict] = {}
    for split in splits:
        inputs, targets, meta = load_eval_set(split)
        # A simulated eval set carries its own operating points, because they cannot be
        # estimated from a non-dimensionalised window. Real sets estimate, which is what
        # the scored submission does and therefore the honest measurement there.
        is_sim = meta.get("kind") == "sim"
        predict = make_predict(
            model,
            device=device,
            batch_size=batch_size,
            return_bounds=bounds or cfg.bounds,
            bound_floor=bound_floor,
            normalizer=cfg.normalizer,
            aux_mask=getattr(cfg, "aux_mask", ""),
            aux_mask_plane=getattr(cfg, "aux_mask_plane", True),
            domain="sim" if is_sim else "real",
            stats_domain=stats_domain,
            tke_blend=tke_blend,
            tke_head=head,
            tke_shrink=tke_shrink,
            tke_head_uses_tau=tke_head_uses_tau,
            width_rule=width_rule,
            # From the checkpoint's own config: a model trained with geometry planes must
            # be served them, and the width mismatch would otherwise surface as a shape
            # error at best and a silent misalignment at worst.
            geom_channels=geom_names(cfg),
            flow_channels=flow_names(cfg),
            cond_mode=cfg.cond_mode,
        )
        metadata = (
            {"re": meta["window_re"], "aoa": meta["window_aoa"]}
            if is_sim and "window_re" in meta
            else None
        )
        pred, lower, upper, timing = run_submission(
            predict, inputs, batch_size=None, metadata=metadata
        )
        # Staged into a private temp directory rather than into data/local_eval/<split>/.
        # That directory is shared: two agents scoring the same split at once would
        # overwrite each other's predictions.npz between the write and the scorer's read,
        # and the loser would silently get a number computed on the other model's
        # predictions. ``train/calibrate.py`` already worked this way; this brings the
        # main scoring path in line with it.
        work = Path(tempfile.mkdtemp(prefix=f"rpde_score_{split}_"))
        try:
            (work / "ref").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(work / "ref" / "targets.npz", target=targets)
            write_predictions(work, pred, NORMALIZATIONS[time_normalization](timing), lower, upper)
            scores = run_official_scorer(work, work / "out")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        scores["_per_window_s"] = timing.per_window_s
        # The organizers' own validation metric, so our runs can be compared against
        # the numbers recorded inside their released checkpoints.
        scores["_normalized_mse"] = normalized_mse(pred, targets, kind="sim" if is_sim else "real")
        results[split] = scores
    return results


def format_table(rows: dict[str, dict[str, dict]]) -> str:
    """Render ``{label: {split: scores}}`` as a comparison table.

    Prints subscores rather than only the composite: the axes have very different
    reachable ranges, so a single total hides where a model actually gained.
    """
    keys = SUBSCORES
    head = (
        f"{'model':<22}{'split':<12}"
        + "".join(f"{k.replace('_score', ''):>9}" for k in keys)
        + f"{'mean*':>9}{'norm_mse':>10}"
    )
    lines = [head, "-" * len(head)]
    for label, per_split in rows.items():
        for split, s in per_split.items():
            vals = [float(s.get(k, 0.0)) for k in keys]
            nm = s.get("_normalized_mse")
            lines.append(
                f"{label:<22}{split:<12}"
                + "".join(f"{v:>9.2f}" for v in vals)
                + f"{sum(vals) / len(vals):>9.2f}"
                + (f"{nm:>10.5f}" if nm is not None else f"{'-':>10}")
            )
    lines += [
        "",
        "* mean is OUR equal-weight average, for ranking runs against each other only.",
        "  The leaderboard's combination rule is not published and v9 emits no",
        "  final_score. norm_mse is the organizers' own validation metric.",
    ]
    return "\n".join(lines)


def save_table(rows: dict[str, dict[str, dict]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return path
