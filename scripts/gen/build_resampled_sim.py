"""Resample sim in time so one simulated frame carries one real frame's worth of evolution.

Run: ``.venv/bin/python scripts/gen/build_resampled_sim.py [--figure] [--overwrite]``
Prerequisite: ``docs/figures/clocks.json`` from ``scripts/clock_diagnostic.py``.

**Why.** ``clock_diagnostic.py`` measures the wake's integral timescale in frames and finds
the two domains keep different clocks: real's scales as ``1/U`` (slope -0.88 to -0.97 at
the angles with a clear wake) while sim's is flat (-0.04). Real's frame spacing is a fixed
0.02 s set by the camera; sim's is a fixed fraction of a convective time. So a 20-frame
window holds 1.10-9.04 integral timescales in real and only 1.42-3.34 in sim, and the
mismatch runs the wrong way at both ends of the Reynolds ladder. A model pretrained on
20-frame sim windows is being taught a horizon it will not be asked for.

This writes ``data/cache/sim_resampled/`` in exactly the layout ``rpde_t1.data.cache``
produces, resampling each trajectory by its own ``step = T_int_sim / T_int_real`` so that
a resampled sim frame and a real frame at the same operating point advance the flow by the
same amount.

**Why the step is measured per cell rather than computed.** Converting frames to convective
time needs a length scale, and no length scale collapses this flow (FINDINGS 2: Strouhal
spans 4.2x; ``clock_diagnostic`` reproduces it as ~50% CV on the implied constant). The
*ratio* of two timescales measured the same way is robust where either one alone is not,
so the ratio is what is used. Nothing here needs a Strouhal number or a clock convention.

**Deliberate choice: no anti-alias filter before decimating.** The median step is 2.17, so
most cells are sub-sampled, and sub-sampling without a temporal low-pass aliases. That is
correct here: a camera does not low-pass in time either, so plain sampling is the faithful
emulation of how the real data was acquired. Pre-filtering would make sim *less* like real.

**Three limitations, stated because they bound what this can buy.**

1. *Almost every cell interpolates, but only 12 fabricate resolution.* Steps are measured,
   so only 2 of 100 land on an integer; everywhere else an output frame is a blend of two
   adjacent solver frames. For ``step >= 1`` that is a mild smoothing and nothing more --
   consecutive outputs still come from distinct source frames, and adjacent sim frames
   correlate ~0.9, so the variance lost is a few percent. For the **12 cells with
   ``step < 1``** (min 0.57, all at low Re and AoA >= 10) consecutive outputs are drawn
   from the *same* pair of source frames: the sequence claims finer temporal resolution
   than the solver produced, and those in-between states are linear morphs rather than
   solutions of the equations. That is the group to ablate if a pretrain misbehaves.
2. *19 simulated cells have no real counterpart* (the withheld Reynolds rows, plus
   scattered cells), so their step is **predicted** from a fit over the measured cells
   rather than measured -- and that fit's residual is **sd 0.216 in log space, +/-24%**,
   so those steps are much weaker than the measured ones. ``t_int_real_source`` in the
   index records which is which.
**The units step (``--step-mode units``, 2026-09-10).** The measured ratio above turned out
to be mostly noise: at one Reynolds number the five angles should share one step, and their
spread is 40% (larger than the 28% that Re explains), while above Re ~10000 the real
timescale is 2-3 frames at 50 Hz and the ratio flattens at ~2.5 where the physics says 4-5.5
(``docs/figures/step_formula_vs_measured.png``). The sim is in code units -- its free stream
reads 1.02 at every Re and its ``t`` runs 0.08-20.08 in steps of 0.02002 -- so one sim frame
is 0.02002 chord-over-free-stream units, one real frame is 0.02 s, and the step is simply

    step(Re) = 0.02 s * U_real / (chord * 0.02002),  U_real = Re * nu_water / chord

a straight line in Re (0.77 at Re 3750, 5.5 at Re 26700) with the chord (69.7 mm, from the
Re-vs-speed fit over the real release) as its only fitted quantity. ``--offsets auto`` then
writes ``round(step)`` series per cell, one source frame apart, so every source frame is
used about once and nothing is fabricated: a cell at step 1.03 stays a single series, a
cell at step 5.5 becomes five. ``--kind sim_units`` is the registered name.

3. *This changes what sim is, and the competition forbids augmented data.* Temporal
   resampling of released data is a weaker reading of that rule than injecting noise
   would be, but it is not obviously outside it. Get a ruling before shipping a recipe
   that depends on this.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from rpde_t1.data.cache import CACHE_DIR, load_index
from rpde_t1.features import freestream_speed, invalid_mask
from rpde_t1.lab.clock_diagnostic import cell_clock
from rpde_t1.paths import REPO_ROOT

#: Name of the derived dataset, and therefore its folder and index under ``data/cache``.
#: ``--kind sim_resampled2 --offsets 0,0.5`` writes the two-start-offset variant.
KIND = "sim_resampled"

#: Start offsets, in units of ``step``, at which each cell is decimated. ``(0.0,)`` is the
#: original single series. With ``(0.0, 0.5)`` a cell whose step is ~2 is decimated twice,
#: from source frame 0 and from source frame ~1, so every source frame is kept exactly once
#: across the two series -- the same clock, twice the windows. Extra offsets are only
#: applied where ``step >= 1``: below 1 nothing is discarded by decimation, so a second
#: series would be a second set of linear morphs of the same frames, not new data.
DEFAULT_OFFSETS: tuple[float, ...] = (0.0,)

#: Never fabricate more frames than the source trajectory has: a step below 1 would
#: otherwise inflate the record with interpolated states that carry no new information.
CAP_AT_SOURCE_LENGTH = True

#: The units conversion. Chord from the Re = U c / nu fit over the real release in water
#: (FINDINGS 3, ``rpde_t1.geometry``); the two frame intervals are read off the ``t`` vectors.
CHORD_M = 0.0697
NU_WATER = 1.0e-6
DT_SIM_CONVECTIVE = 0.02002
DT_REAL_S = 0.02


def units_step(re: int) -> float:
    """Sim frames per real frame when sim time is in chord-over-free-stream units."""
    u_real = re * NU_WATER / CHORD_M
    return DT_REAL_S * u_real / CHORD_M / DT_SIM_CONVECTIVE


def auto_offsets(step: float) -> tuple[float, ...]:
    """``round(step)`` start offsets, one source frame apart, in units of ``step``.

    Consecutive series then start ``step / k`` (about one) source frames apart, so no output
    frame in one series blends the same pair of source frames as a frame in another, and
    across the series every source frame is used about once. A step near 1 gives one series.
    """
    k = max(1, int(round(step)))
    return tuple(j / k for j in range(k))


def measured_steps() -> list[dict]:
    """Per-cell timescales from ``docs/figures/clocks.json``."""
    path = REPO_ROOT / "docs" / "figures" / "clocks.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run scripts/clock_diagnostic.py first.")
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def fit_real_timescale(cells: list[dict]):
    """Predict real's ``u_fs`` and integral timescale at an operating point with no real data.

    ``u_fs`` is linear in Reynolds number by definition (``Re = U c / nu``), so that is a
    two-parameter fit. The timescale is fitted as a single power of ``u_fs`` with a free
    intercept per angle: the exponent is the physics (a convective timescale gives -1) and
    the intercept absorbs how the wake width changes with incidence.
    """
    re = np.array([c["re"] for c in cells], dtype=float)
    u_fs = np.array([c["u_fs"] for c in cells])
    aoa = np.array([c["aoa"] for c in cells], dtype=int)
    t_real = np.array([c["t_int_real"] for c in cells])

    u_coef = np.polyfit(re, u_fs, 1)

    angles = sorted(set(aoa.tolist()))
    design = np.zeros((len(cells), len(angles) + 1))
    for j, a in enumerate(angles):
        design[aoa == a, j] = 1.0
    design[:, -1] = np.log(u_fs)
    beta = np.linalg.lstsq(design, np.log(t_real), rcond=None)[0]
    resid = np.log(t_real) - design @ beta
    print(
        f"  u_fs(Re) fit: u_fs = {u_coef[0]:.3e}*Re + {u_coef[1]:+.4f}\n"
        f"  log T_int_real fit: exponent on log u_fs = {beta[-1]:+.3f}, "
        f"residual sd {resid.std():.3f} in log space ({np.exp(resid.std()) - 1:+.1%})"
    )

    def predict(re_: int, aoa_: int) -> tuple[float, float]:
        u = float(np.polyval(u_coef, float(re_)))
        j = angles.index(aoa_) if aoa_ in angles else len(angles) - 1
        return u, float(np.exp(beta[j] + beta[-1] * np.log(u)))

    return predict


def resample(fields: np.ndarray, step: float, offset: float = 0.0) -> np.ndarray:
    """Sample ``fields`` at ``step`` source frames per output frame, linearly interpolated.

    ``offset`` shifts the first sample to source frame ``offset * step`` (so ``0.5`` starts
    half a step in); the output is shortened so it never reads past the record.

    Sim's measurement mask is frame-invariant and stored as exact zeros, so interpolating
    between two frames leaves masked pixels exactly zero -- the mask survives untouched.
    That is not true of real data, where dropout flickers, which is one reason this is
    only ever applied to sim.
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    n_src = fields.shape[0]
    start = offset * step
    n_out = int(np.floor((n_src - 1 - start) / step)) + 1
    if CAP_AT_SOURCE_LENGTH:
        n_out = min(n_out, n_src)
    pos = start + np.arange(n_out) * step
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n_src - 1)
    w = (pos - lo).astype(np.float32)[:, None, None, None]
    return (fields[lo] * (1.0 - w) + fields[hi] * w).astype(np.float32)


def build(
    overwrite: bool,
    kind: str = KIND,
    offsets: tuple[float, ...] | str = DEFAULT_OFFSETS,
    step_mode: str = "measured",
) -> list[dict]:
    """Write the resampled cache and its index; return the per-series record.

    One index record per (cell, offset). Records of the same cell share ``re``/``aoa`` and
    the mask; only ``path``, ``offset`` and ``n_frames`` differ. The data layer keys masks
    by cell and lists trajectories from the index, so duplicated cells are read as extra
    trajectories of that operating point, which is the intent.
    """
    if step_mode not in ("measured", "units"):
        raise ValueError(f"step_mode must be 'measured' or 'units', got {step_mode!r}")
    if step_mode == "measured":
        cells = measured_steps()
        known = {(c["re"], c["aoa"]): c for c in cells}
        print(f"loaded {len(cells)} measured cells from docs/figures/clocks.json")
        predict = fit_real_timescale(cells)

    out_dir = CACHE_DIR / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    records, masks = [], {}

    for c in load_index("sim"):
        if step_mode == "units":
            step = units_step(c.re)
            t_sim = t_real = None
            u_fs_real, source = c.re * NU_WATER / CHORD_M, "units"
        else:
            clock = cell_clock("sim", c.re, c.aoa)
            if clock is None:
                print(f"  SKIP {c.re}_{c.aoa}: no usable wake pixels")
                continue
            t_sim = clock["t_int"]

            pair = next(
                (known[k] for k in known if k[1] == c.aoa and abs(k[0] - c.re) <= 40), None
            )
            if pair is not None:
                t_real, u_fs_real, source = pair["t_int_real"], pair["u_fs"], "measured"
            else:
                u_fs_real, t_real = predict(c.re, c.aoa)
                source = "predicted"
            step = t_sim / t_real

        fields = None
        cell_offsets = auto_offsets(step) if offsets == "auto" else offsets
        series = [o for o in cell_offsets if o == 0.0 or step >= 1.0]
        for o in series:
            suffix = "" if o == 0.0 else f"_o{o:g}".replace(".", "p")
            dst = out_dir / f"{c.re}_{c.aoa}{suffix}.npy"
            if dst.exists() and not overwrite:
                arr = np.load(dst, mmap_mode="r")
                n_out = int(arr.shape[0])
                mask = invalid_mask(np.asarray(arr))
                u_fs = freestream_speed(np.asarray(arr[: min(64, n_out)]))
            else:
                if fields is None:
                    fields = np.asarray(np.load(c.path, mmap_mode="r"), dtype=np.float32)
                out = resample(fields, step, offset=o)
                np.save(dst, out)
                n_out = out.shape[0]
                mask = invalid_mask(out)
                u_fs = freestream_speed(out[: min(64, n_out)])
            masks.setdefault((c.re, c.aoa), mask)

            records.append(
                {
                    "re": c.re,
                    "aoa": c.aoa,
                    "n_frames": n_out,
                    "u_fs": u_fs,
                    "path": dst.name,
                    "invalid_fraction": float(mask.mean()),
                    "step": step,
                    "offset": o,
                    "t_int_sim": t_sim,
                    "t_int_real": t_real,
                    "t_int_real_source": source,
                    "u_fs_real": u_fs_real,
                    "n_frames_source": c.n_frames,
                }
            )
            print(
                f"  {c.re:>5}_{c.aoa:<2} step={step:5.2f} offset={o:g} ({source[:4]})  "
                f"{c.n_frames} -> {n_out} frames",
                flush=True,
            )

    payload = {
        "kind": kind,
        "freestream_rows": json.loads((CACHE_DIR / "index_sim.json").read_text(encoding="utf-8"))[
            "freestream_rows"
        ],
        "derived_from": "sim",
        "step_mode": step_mode,
        "step_definition": (
            "t_int_sim / t_int_real, wake integral timescales in frames"
            if step_mode == "measured"
            else "0.02 s * U_real / (chord * 0.02002), U_real = Re * nu_water / chord; "
            f"chord {CHORD_M} m, nu {NU_WATER} m^2/s"
        ),
        "offsets": "auto (round(step) series, one source frame apart)"
        if offsets == "auto"
        else list(offsets),
        "trajectories": records,
    }
    (CACHE_DIR / f"index_{kind}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        CACHE_DIR / f"masks_{kind}.npz",
        **{f"{r}_{a}": m for (r, a), m in masks.items()},
    )
    return records


def _wake_series(fields: np.ndarray) -> np.ndarray:
    """``(T, P, 2)`` fluctuation on the top-quartile-TKE always-valid pixels."""
    n_t = fields.shape[0]
    uv = fields[..., :2].reshape(n_t, -1, 2)
    always = (~((uv[..., 0] == 0) & (uv[..., 1] == 0))).all(axis=0)
    tke = 0.5 * (uv[..., 0].var(axis=0) + uv[..., 1].var(axis=0))
    cand = np.nonzero(always)[0]
    pix = cand[tke[cand] >= np.quantile(tke[cand], 0.75)]
    x = uv[:, pix, :].astype(np.float64)
    return x - x.mean(axis=0, keepdims=True)


def _acf(x: np.ndarray, n_lag: int) -> np.ndarray:
    """Variance-weighted temporal autocorrelation, ``(T, P)`` in, ``(n_lag,)`` out."""
    var = (x**2).mean(axis=0)
    x = x[:, var > 0]
    var = var[var > 0]
    out = np.ones(n_lag)
    for lag in range(1, n_lag):
        out[lag] = (x[:-lag] * x[lag:]).mean(axis=0).sum() / var.sum()
    return out


def figure(records: list[dict], cell: str | None, out_path) -> None:
    """Filmstrip plus the two quantitative checks that actually confirm the match."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if cell:
        re_, aoa = (int(v) for v in cell.split("_"))
    else:
        pool = [r for r in records if r["aoa"] == 15 and r["t_int_real_source"] == "measured"]
        pick = sorted(pool, key=lambda r: abs(r["step"] - np.median([p["step"] for p in pool])))
        re_, aoa = pick[0]["re"], pick[0]["aoa"]
    rec = next(r for r in records if (r["re"], r["aoa"]) == (re_, aoa))
    real_key = next(
        (c.re, c.aoa) for c in load_index("real") if c.aoa == aoa and abs(c.re - re_) <= 40
    )

    real = np.asarray(
        np.load(CACHE_DIR / "real" / f"{real_key[0]}_{real_key[1]}.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    sim = np.asarray(
        np.load(CACHE_DIR / "sim" / f"{re_}_{aoa}.npy", mmap_mode="r"), dtype=np.float32
    )
    res = np.asarray(
        np.load(CACHE_DIR / KIND / f"{re_}_{aoa}.npy", mmap_mode="r"), dtype=np.float32
    )

    arms = (("real", real), ("sim resampled", res), ("sim original", sim))
    frames = [0, 5, 10, 15, 20]
    fig = plt.figure(figsize=(13.5, 7.2))
    grid = fig.add_gridspec(3, len(frames) + 2, width_ratios=[1] * len(frames) + [1.5, 1.5])

    for row, (name, arr) in enumerate(arms):
        u_fs = freestream_speed(arr[:64])
        for col, k in enumerate(frames):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(
                arr[k, :, :, 0] / u_fs, cmap="RdBu_r", vmin=-0.2, vmax=1.6, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"frame {k}", fontsize=8)
            if col == 0:
                ax.set_ylabel(name, fontsize=8)

    ax_d = fig.add_subplot(grid[:, len(frames)])
    ax_a = fig.add_subplot(grid[:, len(frames) + 1])
    n_lag = 25
    for name, arr in arms:
        x = _wake_series(arr)
        base = x[0]
        growth = np.sqrt(((x[: n_lag + 1] - base) ** 2).mean(axis=(1, 2)))
        growth /= np.sqrt((x**2).mean())
        style = "-" if name != "sim original" else "--"
        ax_d.plot(np.arange(len(growth)), growth, style, label=name, lw=1.6)
        ax_a.plot(
            np.arange(n_lag), _acf(x.reshape(x.shape[0], -1), n_lag), style, label=name, lw=1.6
        )

    ax_d.set_xlabel("frames since frame 0", fontsize=8)
    ax_d.set_ylabel("rms change / rms fluctuation", fontsize=8)
    ax_d.set_title("how fast the flow moves on", fontsize=9)
    ax_a.set_xlabel("lag (frames)", fontsize=8)
    ax_a.set_ylabel("autocorrelation", fontsize=8)
    ax_a.set_title("temporal autocorrelation", fontsize=9)
    for ax in (ax_d, ax_a):
        ax.axvline(20, color="k", lw=0.6, ls=":")
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    ax_a.axhline(0, color="k", lw=0.6)

    fig.suptitle(
        f"Re={re_} AoA={aoa}, step={rec['step']:.2f}  --  snapshots will NOT match "
        f"(independent realisations, fluctuation corr ~ -0.05); the RATE should",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true", help="rebuild existing .npy files")
    p.add_argument("--figure", action="store_true", help="also write the comparison figure")
    p.add_argument("--cell", default=None, help="cell for the figure, e.g. 16500_15")
    p.add_argument("--out", default=str(REPO_ROOT / "docs" / "figures" / "resampled_clock.png"))
    p.add_argument(
        "--kind",
        default=KIND,
        help="cache kind to write (folder and index name); must be registered in "
        "rpde_t1.features.SIM_DOMAINS to be trainable on",
    )
    p.add_argument(
        "--offsets",
        default=",".join(str(o) for o in DEFAULT_OFFSETS),
        help="comma-separated start offsets in units of step, e.g. '0,0.5' keeps every "
        "source frame across two series where step >= 1; 'auto' writes round(step) series "
        "per cell, one source frame apart",
    )
    p.add_argument(
        "--step-mode",
        default="measured",
        choices=["measured", "units"],
        help="'measured': per-cell wake-timescale ratio (needs docs/figures/clocks.json); "
        "'units': the straight line in Re from the frame intervals and the chord",
    )
    a = p.parse_args()
    offsets: tuple[float, ...] | str
    if a.offsets == "auto":
        offsets = "auto"
    else:
        offsets = tuple(float(o) for o in a.offsets.split(","))
        if offsets[0] != 0.0:
            raise SystemExit("--offsets must start with 0 (the unshifted series)")
    if a.kind == KIND and (offsets != DEFAULT_OFFSETS or a.step_mode != "measured"):
        raise SystemExit(f"refusing to change the {KIND} cache's recipe; pass --kind")

    records = build(a.overwrite, kind=a.kind, offsets=offsets, step_mode=a.step_mode)
    step = np.array([r["step"] for r in records if r["offset"] == 0.0])
    n_pred = sum(1 for r in records if r["t_int_real_source"] == "predicted" and r["offset"] == 0.0)
    total = sum(r["n_frames"] for r in records) * 32 * 64 * 3 * 4 / 1e9
    print(
        f"\n{len(step)} cells, {len(records)} series written ({n_pred} cells with a predicted "
        f"step), {sum(r['n_frames'] for r in records)} frames, {total:.2f} GB\n"
        f"step: min {step.min():.2f}  median {np.median(step):.2f}  max {step.max():.2f}; "
        f"{(step < 1).sum()} cells interpolated (step < 1), {(step >= 1).sum()} sub-sampled"
    )
    print(f"index: {CACHE_DIR / f'index_{a.kind}.json'}")
    if a.figure:
        figure(records, a.cell, a.out)


if __name__ == "__main__":
    main()
