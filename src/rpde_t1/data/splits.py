"""Parameter-held-out splits of the real PIV trajectories.

Why not a random split
----------------------
The final ranking is not the leaderboard: the top teams are shortlisted, then the
organizers retrain the solution and score it on a **private test set with unseen
angles-of-attack and Reynolds numbers**. A split that shuffles windows, or even
whole trajectories, at random measures interpolation inside the training
distribution and will overstate our standing.

What the grid tells us
----------------------
The 82 delivered trajectories are named ``{Re}_{AoA}.h5`` and sit on a regular
grid: AoA in {0, 5, 10, 15, 20}, Re on a 1275-wide ladder from 3750 to 26700.
Two gaps are informative:

* **Re = 15225 is absent for every AoA.** It falls exactly on the ladder, so it
  is not a data gap -- it reads as a deliberately withheld *interior* row.
* Our Re spans 3750..26700, while the competition page quotes a dataset range of
  2968..27975. Both ends are missing, so the private set plausibly demands
  **extrapolation** in Re, not only interpolation.

The splits below mirror those two regimes. This is inference from the public
training filenames about how to design a validation set; it neither uses nor
reveals any hidden data.

Cost of holding out an AoA row
------------------------------
A whole AoA row is ~17 of 82 trajectories (20% of an already small real dataset).
Rather than pay that permanently, ``val_interp`` / ``val_extrap`` hold out **Re**
rows (5-7 trajectories each) and AoA generalization is measured by
leave-one-AoA-out cross-validation during development, via
:func:`leave_one_aoa_out`.
"""

from __future__ import annotations

import json
import re as _re
from dataclasses import dataclass
from pathlib import Path

from ..paths import DATA_DIR

#: ``{Re}_{AoA}.h5``
_NAME = _re.compile(r"^(?P<re>\d+)_(?P<aoa>\d+)$")

#: Extreme Re rows -- the closest local stand-in for the private set's
#: out-of-range Reynolds numbers.
VAL_EXTRAP_RE = (3750, 26700)

#: An interior Re row, held out whole. Re=16500 sits directly against the absent
#: 15225 rung, so this reproduces the withheld-interior-row regime.
VAL_INTERP_RE = (16500,)

#: Trajectories excluded from every split.
#:
#: ``7575_0`` is a byte-identical duplicate of ``6300_0`` -- verified by md5 over all
#: 868 frames, and it is the only duplicate group in the release (81 distinct arrays
#: among 82 files). Its measured freestream speed is 0.0920, which a ladder fitted on
#: the other 80 cells maps to Re=6210: 1.4% from the 6300 rung and 17.5% from 7575.
#: So the array's physics belongs to Re=6300 and the ``7575_0`` *label* is wrong.
#:
#: Dropping it costs nothing, because the data still exists under its correct label.
#: Keeping it would double-weight one operating point and, worse, teach a conditioned
#: model that Re=7575 looks like Re=6300 -- a false speed-to-flow mapping.
EXCLUDED_CELLS: frozenset[tuple[int, int]] = frozenset({(7575, 0)})

#: Simulated cells whose ``p`` channel is unusable, so it is zeroed at load time.
#:
#: The second release defect, and a much more expensive one than the duplicate.
#: ``13950_15`` (sim) stores pressure on an **absolute** gauge -- per-frame values
#: around 200 with std 40 -- while every other simulated cell stores a gauge pressure
#: centred near zero with std ~0.15. The raw HDF5 has it too, so it is upstream, not a
#: caching artefact. Its ``u`` and ``v`` are in family and stay in use.
#:
#: Why it mattered: ``field_loss`` supervises ``p`` on simulated samples, and this one
#: trajectory's mean ``p^2`` is 28,000 against every other cell's ~0.3. At 1% of the
#: windows it carried **99.9%** of the pretraining loss mass, which diverged the FNO
#: sim pretrain to a real-validation rel-L2 of 1.06 -- worse than predicting zeros --
#: and damaged CNO's less visibly. See docs/pressure_gauge_defect_report.md.
#:
#: Only the channel is dropped, not the trajectory: the flow itself is fine, and the
#: cell is one of the 20x5 grid's interior rungs.
DEFECTIVE_PRESSURE_CELLS: frozenset[tuple[int, int]] = frozenset({(13950, 15)})

#: Fraction of a *reserved* trajectory's frames held back for ``val_time``.
VAL_TIME_TAIL = 0.25

#: How many training cells reserve that tail. The rest train on full trajectories.
#:
#: Reserving the tail on all 69 training cells costs **26.2%** of training windows
#: (41,311 of 55,978) to buy one diagnostic. Reserving it on 20 costs **7.6%** and
#: still leaves ~89 windows at stride 40 -- a paired-difference CI of about +/-0.47
#: SPS, against the ~1.5 effects we actually act on.
#:
#: What licenses the shrink: the trajectories are **statistically stationary**, so the
#: reserved tail is not a different regime. Across all 81, the last 25% differs from
#: the first 75% by -0.02% +/- 0.41% in freestream speed (worst case 1.1%), with no
#: startup transient. A model that never sees trajectory ends is therefore not
#: handicapped, and ``val_time``'s only real asymmetry is the one that makes it
#: *optimistic*: it reuses cells the model trained on.
N_VAL_TIME_CELLS = 20


@dataclass(frozen=True)
class Params:
    """Operating point of one trajectory."""

    re: int
    aoa: int

    @property
    def key(self) -> str:
        return f"{self.re}_{self.aoa}"


def parse_params(path: str | Path) -> Params:
    """Extract ``(Re, AoA)`` from a trajectory filename."""
    stem = Path(path).stem
    m = _NAME.match(stem)
    if not m:
        raise ValueError(f"Cannot parse Re/AoA from {stem!r}; expected '<Re>_<AoA>.h5'.")
    return Params(int(m.group("re")), int(m.group("aoa")))


def real_trajectories(root: Path | None = None, include_excluded: bool = False) -> list[Path]:
    """Every usable real trajectory, sorted by (AoA, Re) for stable ordering.

    Cells in :data:`EXCLUDED_CELLS` are dropped unless ``include_excluded``.
    """
    root = root or (DATA_DIR / "train_real")
    paths = sorted(root.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files under {root}. Run `make data` then `make extract`.")
    if not include_excluded:
        paths = [
            p for p in paths if (parse_params(p).re, parse_params(p).aoa) not in EXCLUDED_CELLS
        ]
    return sorted(paths, key=lambda p: (parse_params(p).aoa, parse_params(p).re))


def all_cells(paths: list[Path] | None = None) -> list[Path]:
    """Every real trajectory, with nothing held out.

    Exists only to reproduce the organizers' setup: their released ``sim_real_ft``
    checkpoints were fine-tuned on the whole of ``train_real``, which is why they
    cannot be measured cleanly on any of our splits. A model trained on this has no
    honest validation set -- use it to check pipeline fidelity, never to select.
    """
    paths = paths if paths is not None else real_trajectories()
    return [p for p in paths if (parse_params(p).re, parse_params(p).aoa) not in EXCLUDED_CELLS]


def assign_splits(paths: list[Path] | None = None) -> dict[str, list[Path]]:
    """Partition trajectories into ``train`` / ``val_interp`` / ``val_extrap``.

    ``val_time`` is not a separate set of files: it reuses the ``train``
    trajectories and takes windows from their held-out tail frames instead.
    """
    paths = paths if paths is not None else real_trajectories()
    # Enforced here as well as in real_trajectories(): a caller passing an explicit
    # path list would otherwise bypass the exclusion silently.
    paths = [p for p in paths if (parse_params(p).re, parse_params(p).aoa) not in EXCLUDED_CELLS]
    out: dict[str, list[Path]] = {"train": [], "val_interp": [], "val_extrap": []}
    for path in paths:
        params = parse_params(path)
        if params.re in VAL_EXTRAP_RE:
            out["val_extrap"].append(path)
        elif params.re in VAL_INTERP_RE:
            out["val_interp"].append(path)
        else:
            out["train"].append(path)
    return out


def stratify_cells(cells: list[tuple[int, int]], n_cells: int) -> list[tuple[int, int]]:
    """Pick ~``n_cells`` cells spread evenly along the Re ladder within each angle.

    Stratification is not cosmetic. Both SPS and MVPE depend strongly on the operating
    point -- SPS spans 54.5 to 32.5 across freestream quartiles and 47.5 to 36.0 across
    angle of attack -- so an unrepresentative ``val_time`` would carry exactly the kind
    of composition bias that made our SPS disagree with the leaderboard by 12 points.
    Deterministic, so two people's ``val_time`` are the same set.
    """
    by_aoa: dict[int, list[int]] = {}
    for re_, aoa in cells:
        by_aoa.setdefault(aoa, []).append(re_)

    per_aoa = max(1, round(n_cells / max(1, len(by_aoa))))
    picked: list[tuple[int, int]] = []
    for aoa in sorted(by_aoa):
        ladder = sorted(by_aoa[aoa])
        k = min(per_aoa, len(ladder))
        idx = [round(i * (len(ladder) - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
        picked.extend((ladder[i], aoa) for i in dict.fromkeys(idx))
    return sorted(picked)


def val_time_cells(
    paths: list[Path] | None = None, n_cells: int = N_VAL_TIME_CELLS
) -> list[tuple[int, int]]:
    """Training cells that reserve a temporal tail, stratified over the grid."""
    paths = paths if paths is not None else assign_splits()["train"]
    return stratify_cells([(parse_params(p).re, parse_params(p).aoa) for p in paths], n_cells)


#: The two standard angle-generalisation arms of the evaluation protocol.
#:
#: ``extrap`` holds out AoA 20, the ladder end: beyond it the section is in deep stall,
#: a qualitatively different regime. ``interp`` holds out 5 and 15, which the remaining
#: angles bracket. **Which of the two the private test uses is unknown** -- the
#: organizers' Reynolds design withheld both an interior row (15225) and one beyond the
#: real range (27975), so assume they did the same for angle and measure both.
AOA_ARMS: dict[str, tuple[int, ...]] = {"extrap": (20,), "interp": (5, 15)}


def aoa_holdout(
    aoa: int | tuple[int, ...], paths: list[Path] | None = None
) -> dict[str, list[Path]]:
    """A leave-angle-out fold, taken **inside the training split**.

    Deliberately not over all 81 cells. A fold built from everything would train on
    the ``val_interp`` and ``val_extrap`` cells at the other angles, which would make
    the resulting model unscoreable on our two held-out Reynolds sets and break every
    comparison we have. Restricting the fold to ``train`` keeps those two sets clean,
    so an AoA-holdout model can still be scored on them.

    The private test uses unseen angles, and nothing in the standard splits measures
    that: all five angles appear in train *and* in both validation sets. AoA 0 and 20
    are the ladder ends and so the only genuine extrapolation folds; 5, 10 and 15 are
    bracketed by training data and will look deceptively good.
    """
    held = {aoa} if isinstance(aoa, int) else set(aoa)
    paths = paths if paths is not None else assign_splits()["train"]
    # Enforced here as well as in assign_splits(), because a caller passing an explicit
    # path list would otherwise reintroduce the mislabelled duplicate.
    paths = [p for p in paths if (parse_params(p).re, parse_params(p).aoa) not in EXCLUDED_CELLS]
    fold = {
        "train": [p for p in paths if parse_params(p).aoa not in held],
        "val": [p for p in paths if parse_params(p).aoa in held],
    }
    if not fold["val"]:
        raise ValueError(f"No cells at AoA {sorted(held)}; nothing to hold out.")
    if not fold["train"]:
        raise ValueError(f"Holding out AoA {sorted(held)} leaves no training cells.")
    return fold


def aoa_split_paths(split: str) -> list[Path] | None:
    """Trajectories for an ``aoa_holdout_*`` / ``aoa_val_*`` split name, or ``None``.

    ``aoa_holdout_20`` is the *training* side of the leave-out-20 fold; ``aoa_val_20`` is
    its held-out side. Several angles are joined by underscores (``aoa_val_5_15``).

    This exists because the name was being parsed in two places -- ``data.dataset.make_dataset``
    for training and ``score.__main__._resolve_split`` for eval-set building -- and only the
    first one had it, so ``PROTOCOL.md``'s third arm could be *trained* and never *scored*:
    ``score profile`` died on a missing ``data/local_eval/aoa_val_5_15``, and the hint it printed
    named a command that would also have failed. Two FNO candidates were carried through all
    six protocol runs and could not be read. One definition, called from both, per
    ``PROTOCOL.md`` section 5's opening trap.
    """
    if not split.startswith(("aoa_holdout_", "aoa_val_")):
        return None
    side = "val" if split.startswith("aoa_val_") else "train"
    prefix = "aoa_val_" if side == "val" else "aoa_holdout_"
    suffix = split[len(prefix) :]
    try:
        angles = tuple(int(a) for a in suffix.split("_"))
    except ValueError as exc:
        raise ValueError(f"{split!r} is not an angle list: {suffix!r}") from exc
    return aoa_holdout(angles)[side]


def train_frame_range(n_frames: int) -> tuple[int, int]:
    """Frames a training trajectory may use (the leading portion)."""
    return 0, int(round(n_frames * (1.0 - VAL_TIME_TAIL)))


def val_time_frame_range(n_frames: int) -> tuple[int, int]:
    """Frames reserved for ``val_time`` in a trajectory of ``n_frames`` frames.

    Must be called per trajectory: lengths are not uniform, so a single absolute
    offset computed from the modal 868 would exclude the short trajectories.
    """
    return int(round(n_frames * (1.0 - VAL_TIME_TAIL))), n_frames


#: Fraction of a trajectory to skip when reading the ``val_time`` tail.
VAL_TIME_OFFSET_FRAC = 1.0 - VAL_TIME_TAIL


def leave_one_aoa_out(paths: list[Path] | None = None) -> dict[int, dict[str, list[Path]]]:
    """Every leave-one-AoA-out fold, one per angle.

    A thin sweep over :func:`aoa_holdout`, which is deliberate: an earlier version built
    its folds from all 81 cells instead, so each fold trained on the ``val_interp`` and
    ``val_extrap`` cells at the other four angles and the resulting model could not be
    scored on either set. The two helpers disagreed about the same thing, which is the
    failure mode ``PROTOCOL.md`` §5 opens with. Delegating means there is one definition.
    """
    paths = paths if paths is not None else assign_splits()["train"]
    angles = sorted({parse_params(p).aoa for p in paths})
    return {aoa: aoa_holdout(aoa, paths=paths) for aoa in angles}


# --------------------------------------------------------------------------- #
# The simulated grid
# --------------------------------------------------------------------------- #
#
# Sim exists in this project as pretraining, but it is also the only place an
# *architecture* can be compared without spending the real data's scarce operating
# points: 100 complete cells at 1000 noise-free frames each, against real's 81 at
# ~868. The splits below mirror the real protocol one-for-one -- ladder ends held
# out, one interior rung held out, a temporal tail on a stratified subset -- so a
# ranking measured on sim is read against the same three regimes the real arms use.
#
# Two differences from real are deliberate:
#
# * The interior rung held out is **15225**, the one Reynolds row real does not
#   have. It is the natural candidate for the public evaluation set (210 = 5 x 42),
#   so a model is being asked the interpolation question that actually matters.
# * The upper ladder end is **27975**, which is above real's maximum and matches the
#   top of the range quoted on the competition page.

#: Ladder ends of the simulated Reynolds grid. 3750 is shared with real's
#: ``val_extrap``; 27975 exists only in sim.
SIM_VAL_EXTRAP_RE = (3750, 27975)

#: The interior rung real withholds entirely.
SIM_VAL_INTERP_RE = (15225,)

#: How many sim training cells reserve a temporal tail for ``sim_val_time``.
N_SIM_VAL_TIME_CELLS = 20


def sim_trajectories(root: Path | None = None) -> list[Path]:
    """Every simulated trajectory, sorted by (AoA, Re).

    Deliberately parallel to :func:`real_trajectories`, including the sort order, so
    the two grids can be zipped cell-by-cell. ``EXCLUDED_CELLS`` does not apply: the
    duplicate is a defect of the real release only, and sim's ``7575_0`` is a genuine
    distinct trajectory.
    """
    root = root or (DATA_DIR / "train_sim")
    paths = sorted(root.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files under {root}. Run `make data` then `make extract`.")
    return sorted(paths, key=lambda p: (parse_params(p).aoa, parse_params(p).re))


def assign_sim_splits(paths: list[Path] | None = None) -> dict[str, list[Path]]:
    """Partition the simulated trajectories into ``train`` / ``val_interp`` / ``val_extrap``.

    Keys are unprefixed so this reads like :func:`assign_splits`; the dataset and
    eval-set layers address them as ``sim_train`` / ``sim_val_interp`` / ... to keep
    the two grids' split names distinct in run configs and ``data/local_eval/``.
    """
    paths = paths if paths is not None else sim_trajectories()
    out: dict[str, list[Path]] = {"train": [], "val_interp": [], "val_extrap": []}
    for path in paths:
        params = parse_params(path)
        if params.re in SIM_VAL_EXTRAP_RE:
            out["val_extrap"].append(path)
        elif params.re in SIM_VAL_INTERP_RE:
            out["val_interp"].append(path)
        else:
            out["train"].append(path)
    return out


def sim_val_time_cells(
    paths: list[Path] | None = None, n_cells: int = N_SIM_VAL_TIME_CELLS
) -> list[tuple[int, int]]:
    """Sim training cells that reserve a temporal tail, stratified over the grid."""
    paths = paths if paths is not None else assign_sim_splits()["train"]
    return stratify_cells([(parse_params(p).re, parse_params(p).aoa) for p in paths], n_cells)


#: Split names the sim grid understands, and the ``assign_sim_splits`` key each maps
#: to. ``sim_val_time`` reuses the training cells and takes their withheld tail.
SIM_SPLITS: dict[str, str] = {
    "sim_train": "train",
    "sim_val_time": "train",
    "sim_val_interp": "val_interp",
    "sim_val_extrap": "val_extrap",
}


#: The simulated grid minus the Reynolds rows REAL holds out (``VAL_EXTRAP_RE`` 3750 and
#: 26700, ``VAL_INTERP_RE`` 16500): 85 cells, every frame. Nothing to do with sim's own
#: holdouts above. This is the split for one question -- does seeing an operating point
#: in simulation help the real prediction at that same point? -- asked by training every
#: sim stage on it and reading real ``val_interp`` / ``val_extrap``, which the model has
#: then never seen in either domain, against a twin trained on ``sim_all``.
SIM_ALL_BUT_REAL_VAL = "sim_all_but_real_val"


def sim_all_but_real_val_paths(paths: list[Path] | None = None) -> list[Path]:
    """Every simulated trajectory except those on a Reynolds row real validates on."""
    paths = paths if paths is not None else sim_trajectories()
    held = set(VAL_EXTRAP_RE) | set(VAL_INTERP_RE)
    return [p for p in paths if parse_params(p).re not in held]


#: The simulated grid minus the corner where simulation is measurably least faithful:
#: **AoA <= 5 degrees at Re >= 8850**, 32 of 100 cells. Added 2026-09-21 (Bruno's proposal).
#:
#: Chosen by the fluctuation-ratio table, not by hand. ``r = sqrt(TKE_real/TKE_sim)`` measured on
#: the 81 paired cells (:mod:`~rpde_t1.data.fluctuation_scale`) falls with Reynolds at EVERY angle
#: -- 0.43x from Re 3750 to 26700 at AoA 0 against 0.73x at AoA 10 -- so both effects are real and
#: they compound. Every threshold on ``r`` from 0.20 to 0.35 selects only AoA 0 and 5, and ``r <
#: 0.30`` agrees with this box on 31 of 32 cells (the exception, ``8850_0`` at 0.341, sits just
#: outside). In energy terms (``1/r^2``) the removed cells average **19.3x** the real fluctuation
#: energy against **3.8x** for those kept: a distinct bad subpopulation, not a marginal trim.
#:
#: **Stated in (Re, AoA), deliberately.** 19 simulated cells have no real counterpart, so their
#: ``r`` is filled rather than measured -- and five of those fills read exactly 0.187, which is
#: ``16500_0``'s measured value copied by the nearest-neighbour fill. Thresholding on them would be
#: circular. A coordinate rule extrapolates the RULE instead of the value, which is the weaker
#: assumption; it is still an assumption, and it covers 9 of the 32 cut cells.
#:
#: Intended for the JOINT stage only, leaving the pretrain on ``sim_all``: FINDINGS 140 established
#: that simulated *coverage* pays in the pretrain, so cutting cells there would trade a measured
#: gain for an unmeasured one. What this tests is whether the joint stage wants *alignment* with
#: real instead -- an idea the repo has never tried. It has tried CORRECTING the mismatch
#: (``--rescale-fluct``, 158a, negative at n=3) and removing the sim-only rungs (158c, a null);
#: simply not using the worst cells is a third thing.
SIM_HIFI = "sim_hifi"

#: The cut: angles at or below this, at Reynolds at or above ``SIM_HIFI_RE_MIN``.
SIM_HIFI_AOA_MAX = 5
SIM_HIFI_RE_MIN = 8850


def sim_hifi_paths(paths: list[Path] | None = None) -> list[Path]:
    """Every simulated trajectory except the low-incidence, high-Reynolds corner."""
    paths = paths if paths is not None else sim_trajectories()
    return [
        p
        for p in paths
        if not (
            parse_params(p).aoa <= SIM_HIFI_AOA_MAX and parse_params(p).re >= SIM_HIFI_RE_MIN
        )
    ]


SIM_ONLY_ROWS = "sim_all_but_sim_only_rows"


def sim_only_reynolds_rows(paths: list[Path] | None = None) -> tuple[int, ...]:
    """Reynolds rows the simulated grid has and the real release does not.

    Derived from the two grids rather than hardcoded, so it cannot drift from the data:
    sim runs 20 rows, real 18, and the difference is ``(15225, 27975)``.

    They matter out of proportion to their count. 15225 sits between real's 13950 and 16500,
    so real brackets it; **27975 is above real's entire ladder**, which makes the simulated
    cells there the only data of any kind that brackets the top of the grid from above. FINDINGS
    71b places the hidden test set at an unseen, high-Reynolds, mid-to-high-AoA composition.
    """
    real = {parse_params(p).re for p in real_trajectories()}
    sim = {parse_params(p).re for p in (paths if paths is not None else sim_trajectories())}
    return tuple(sorted(sim - real))


def sim_all_but_sim_only_rows_paths(paths: list[Path] | None = None) -> list[Path]:
    """Every simulated trajectory except those on a Reynolds row real does not have.

    The mirror of :func:`sim_all_but_real_val_paths`, and it asks that campaign's unasked
    half. FINDINGS 140 removed simulated coverage at real's *validation* rows and measured
    what it was worth (+0.119 ceiling at Re 3750, where real has no neighbour below;
    +0.013 at Re 26700, where 25425 sits right beneath). But it kept 15225 and 27975 in
    **every** arm, so it measured the value of the exact row while its bracketing rung was
    present throughout, and cannot bound what that rung is worth. This split removes them.
    """
    paths = paths if paths is not None else sim_trajectories()
    held = set(sim_only_reynolds_rows(paths))
    return [p for p in paths if parse_params(p).re not in held]


def is_sim_split(split: str) -> bool:
    """Whether a split name addresses the simulated grid.

    ``sim_all`` is the pre-existing pretraining split (every cell, nothing held out)
    and is included so callers that dispatch on this get the domain right for it too;
    so is ``sim_all_but_real_val``.
    """
    return split in SIM_SPLITS or split in (
        "sim_all",
        SIM_ALL_BUT_REAL_VAL,
        SIM_ONLY_ROWS,
        SIM_HIFI,
    )


def describe_splits(splits: dict[str, list[Path]] | None = None) -> dict:
    """JSON-serializable summary, written to ``rpde_t1/data/tables/splits.json``.

    Committed so that two people's numbers are comparable and a split change is a
    reviewable diff rather than a silent shift in every metric.
    """
    splits = splits or assign_splits()
    summary: dict = {
        "val_extrap_re": list(VAL_EXTRAP_RE),
        "val_interp_re": list(VAL_INTERP_RE),
        "val_time_tail": VAL_TIME_TAIL,
        "splits": {},
    }
    for name, paths in splits.items():
        params = [parse_params(p) for p in paths]
        summary["splits"][name] = {
            "n_trajectories": len(paths),
            "re": sorted({p.re for p in params}),
            "aoa": sorted({p.aoa for p in params}),
            "keys": [p.key for p in params],
        }
    return summary


def write_split_config(path: Path | None = None) -> Path:
    """Persist the split definition for reproducibility."""
    path = path or (Path(__file__).resolve().parent / "tables" / "splits.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(describe_splits(), indent=2) + "\n", encoding="utf-8")
    return path
