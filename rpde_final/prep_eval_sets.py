"""Build the local evaluation sets the chain needs, from the training release only.

Run in the stage environment (RPDE_DATA_DIR / RPDE_LOCAL_EVAL_DIR set):

    python prep_eval_sets.py <t1|t2>

Sets and their spellings (Makefile `val-sets`, scripts/sps/sps_fit_sets.sh, FINDINGS for
`all_real`, scripts/train/ftlr_campaign.sh for `hard5`):

    val_extrap, val_interp         stride 40, whole trajectories of the held-out cells
    val_time                       the tail quarter of the 20 reserved train cells, 3 windows each
    all_real                       all 81 cells at stride 60 (the interval head's fit set)
    train_ends                     train cells at Re {5025, 6300, 24150, 25425}, stride 60, first 651 frames
    train_all (Track 2 only)       the 69 train cells, stride 60, first 651 frames (train-split chains)
    hard5                          16500_{10,15,20}, 26700_{10,15}, stride 40 (diagnostics only)

Track 2 also gets the stream directories `t2_val_interp` and `t2_val_extrap` (Makefile
`t2-stream`): symlinks to the held-out trajectories plus the official mean_std_real.pt.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from rpde_t1.data.splits import assign_splits
from rpde_t1.paths import DATA_DIR, LOCAL_EVAL_DIR
from rpde_t1.score.__main__ import main as score_main

TRAIN_ENDS_RE = (5025, 6300, 24150, 25425)
HARD5 = ("16500_10", "16500_15", "16500_20", "26700_10", "26700_15")


def build(name: str, *args: str) -> None:
    if (LOCAL_EVAL_DIR / name / "meta.json").exists():
        print(f"eval set {name}: exists", flush=True)
        return
    print(f"eval set {name}: building", flush=True)
    rc = score_main(["eval-set", "--name", name, *args])
    if rc not in (0, None):
        raise SystemExit(f"eval-set {name} failed with {rc}")


def main() -> None:
    track = sys.argv[1]
    splits = assign_splits()
    train = sorted(splits["train"])
    build("val_extrap", "--split", "val_extrap")
    build("val_interp", "--split", "val_interp")
    build("val_time", "--split", "val_time", "--max-windows-per-traj", "3")
    build("all_real", "--split", "all", "--stride", "60")
    ends = [p for p in train if int(p.stem.split("_")[0]) in TRAIN_ENDS_RE]
    build("train_ends", "--h5", *map(str, ends), "--stride", "60", "--max-frames", "651")
    hard5 = [DATA_DIR / "train_real" / f"{c}.h5" for c in HARD5]
    build("hard5", "--h5", *map(str, hard5))
    if track == "t2":
        build("train_all", "--h5", *map(str, train), "--stride", "60", "--max-frames", "651")
        for split in ("val_interp", "val_extrap"):
            d = LOCAL_EVAL_DIR / f"t2_{split}" / "test_real"
            d.mkdir(parents=True, exist_ok=True)
            for p in splits[split]:
                dst = d / Path(p).name
                if not dst.exists():
                    dst.symlink_to(Path(p).resolve())
            ms = LOCAL_EVAL_DIR / f"t2_{split}" / "mean_std_real.pt"
            if not ms.exists():
                shutil.copy(DATA_DIR / "mean_std_real.pt", ms)
            print(f"stream dir t2_{split}: {len(list(d.iterdir()))} trajectories", flush=True)


if __name__ == "__main__":
    main()
