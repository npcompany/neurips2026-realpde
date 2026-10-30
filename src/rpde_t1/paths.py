"""Filesystem layout for the project.

Every path is resolved relative to the repository root, so a fresh clone plus
``make data`` puts everything where the code expects it. Each location can be
overridden by the matching ``RPDE_*`` environment variable, which is what a
shared / read-only dataset mount needs.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/rpde_t1/paths.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


#: Raw competition tarballs and their extracted trajectories.
DATA_DIR = _env_path("RPDE_DATA_DIR", REPO_ROOT / "data")
#: Official baseline checkpoints (``sim_pretrain/`` and ``sim_real_ft/``).
CKPT_DIR = _env_path("RPDE_CKPT_DIR", REPO_ROOT / "data" / "ckpts")
#: The single example PIV trajectory shipped on the competition Drive.
EXAMPLE_DIR = _env_path("RPDE_EXAMPLE_DIR", REPO_ROOT / "data" / "example")
#: Generated local evaluation sets (inputs + reference targets).
LOCAL_EVAL_DIR = _env_path("RPDE_LOCAL_EVAL_DIR", REPO_ROOT / "data" / "local_eval")
#: Training run outputs.
RUNS_DIR = _env_path("RPDE_RUNS_DIR", REPO_ROOT / "runs")
#: Staged submission bundles.
SUBMISSIONS_DIR = _env_path("RPDE_SUBMISSIONS_DIR", REPO_ROOT / "submissions")

#: The organizers' starting kit, vendored and committed. ``scoring.py`` here
#: computes the same five subscores the leaderboard computes, so local subscores are
#: directly comparable. It does **not** produce ``final_score``: since the August
#: 2026 restart the combination rule is not published.
#:
#: v9 replaced v7 at that restart (linear SPS, SPS masking, no composite). The old
#: kit is kept at ``~/realpde_t1_starting_kit_v7.zip`` for reference only -- never
#: score against it, the SPS scale differs.
VENDOR_KIT = REPO_ROOT / "vendor" / "realpde_t1_starting_kit_v9"
OFFICIAL_SCORING = VENDOR_KIT / "scoring.py"

#: The Track-2 kit, vendored on the same terms. Its ``scoring.py`` differs from the Track-1
#: copy only by an evaluator-side ``sample_id`` order check -- every subscore formula and
#: constant is identical -- but the two files must not be confused, so
#: :func:`rpde_t2.stream.official_scoring` loads this one by path rather than by name.
VENDOR_KIT_T2 = REPO_ROOT / "vendor" / "realpde_t2_starting_kit_v6"
OFFICIAL_SCORING_T2 = VENDOR_KIT_T2 / "scoring.py"

EXAMPLE_H5 = EXAMPLE_DIR / "3750_0.h5"


def require(path: Path, hint: str) -> Path:
    """Return ``path``, or fail with an actionable message naming the fix."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. {hint}")
    return path
