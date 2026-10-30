"""Two clocks: is a simulated frame the same physical horizon as a real frame?

Run: ``.venv/bin/python scripts/clock_diagnostic.py [--out docs/figures/clocks.png]``

**The question.** The task is 20 frames in, 20 frames out, in both domains. That is only
one task if 20 frames means the same amount of flow evolution in both. A camera has no
choice about frame spacing -- the PIV system fires every 0.02 s whatever the water is
doing. A simulation does have a choice, and the conventional one is to solve in
non-dimensional variables, where the natural time unit is the convective time ``L/U``:
how long the flow takes to travel one body length. Those two spacings coincide only if
``U`` is fixed, and here ``u_fs`` varies 6.7x across the Reynolds ladder.

**The test, and why it is clean.** Measure the Eulerian integral timescale of the wake
fluctuation, in *frames*, and regress it on freestream speed at fixed angle:

* a **physical** clock (fixed ``dt``, ``U`` varying) gives ``T_int[frames] ~ 1/U``, so a
  log-log slope of **-1**;
* a **convective** clock (fixed ``dt* = dt U/L``) gives ``T_int[frames] ~ const``, slope **0**.

The comparison is between two arms holding geometry, angle and Reynolds ladder fixed, so
a difference in slope cannot be the flow physics -- both arms are the same flow. That is
what makes the slope, rather than any absolute number, the load-bearing measurement.

**What this does NOT establish.** Converting frames to convective time needs a length
scale, and no choice of length scale collapses this flow: FINDINGS 2 records a Strouhal
spread of 4.2x, and the reconciliation printed below reproduces it as a ~50% coefficient
of variation on the implied conversion constant. So the *ratio* between the two clocks is
measurable per cell while neither clock is expressible in absolute units. Downstream work
should use the measured per-cell ratio, not a theoretical conversion.

**Three traps.**

1. *Intermittent PIV dropout is exact zeros*, and a zero in the middle of a time series
   corrupts an autocorrelation badly. Only pixels valid in **every** frame are used.
2. *At AoA 0 the real flow is attached*, so "the wake" is a thin shear layer and the
   timescale is not the wake's. That column is reported but is the weakest, and it is
   where both arms' slopes degrade.
3. *Wake pixels are selected per domain*, by each domain's own top-quartile TKE. Using
   one shared pixel set would compare real's wake against whatever sim has in the same
   place; using each domain's own wake is the intended comparison, but it is a choice.

Also reported: the most negative lobe of the autocorrelation, as a coherence measure (a
periodic wake approaches -1, an incoherent one decays to ~0), which is what refuted the
hypothesis that the simulation is two-dimensional; and the normalised TKE ratio per angle.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from rpde_t1.data.cache import CACHE_DIR, load_index
from rpde_t1.data.splits import EXCLUDED_CELLS
from rpde_t1.geometry import projected_height
from rpde_t1.paths import REPO_ROOT

#: Real frame interval in seconds. Fixed by the camera; the whole point of this script.
DT_REAL = 0.02

#: Analyse the wake, not the freestream: the top quartile of temporal TKE among pixels
#: that are valid in every frame. The freestream carries no timescale worth measuring.
TKE_TOP_FRAC = 0.25

#: Real filenames round the stored Reynolds number, so sim and real keys differ by a few
#: units at the same operating point. FINDINGS 2 records the mismatch on 77 of 82 cells.
RE_MATCH_TOL = 40


def integral_timescale(flat: np.ndarray, pix: np.ndarray) -> tuple[float, float]:
    """``(integral timescale in frames, most negative ACF lobe)`` over ``pix``.

    Autocorrelation by FFT, variance-weighted across the selected pixels and both
    velocity components, then integrated to the first zero crossing. The biased
    estimator (dividing by the full record length at every lag) is used deliberately: it
    tapers the noisy tail instead of amplifying it, which is what makes the integral
    stable on records this short.
    """
    n_t = flat.shape[0]
    n_fft = 1 << (2 * n_t - 1).bit_length()
    acc = np.zeros(n_t)
    weight = 0.0
    for ch in range(2):
        x = flat[:, pix, ch].astype(np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        var = (x**2).mean(axis=0)
        keep = var > 0
        if not keep.any():
            continue
        x, var = x[:, keep], var[keep]
        spec = np.fft.rfft(x, n=n_fft, axis=0)
        corr = np.fft.irfft(spec * np.conj(spec), n=n_fft, axis=0)[:n_t]
        corr /= corr[0]
        acc += (corr * var).sum(axis=1)
        weight += var.sum()
    acf = acc / weight

    crossings = np.nonzero(acf <= 0)[0]
    end = int(crossings[0]) if crossings.size else n_t
    if end < 2:
        return 0.5, float(acf.min())
    t_int = float(np.trapz(acf[:end]))
    lo, hi = acf[end - 1], acf[end] if end < n_t else 0.0
    if lo != hi:
        t_int += 0.5 * lo * (lo / (lo - hi))
    return t_int, float(acf.min())


def cell_clock(kind: str, re_: int, aoa: int) -> dict[str, float] | None:
    """Timescale, coherence and wake TKE for one cached trajectory."""
    path = CACHE_DIR / kind / f"{re_}_{aoa}.npy"
    if not path.exists():
        return None
    fields = np.asarray(np.load(path, mmap_mode="r")[..., :2], dtype=np.float32)
    n_t = fields.shape[0]
    valid_always = (~((fields[..., 0] == 0) & (fields[..., 1] == 0))).all(axis=0).reshape(-1)
    flat = fields.reshape(n_t, -1, 2)
    tke = 0.5 * (flat[..., 0].var(axis=0) + flat[..., 1].var(axis=0))

    candidates = np.nonzero(valid_always)[0]
    if candidates.size < 32:
        return None
    pix = candidates[tke[candidates] >= np.quantile(tke[candidates], 1.0 - TKE_TOP_FRAC)]
    t_int, acf_min = integral_timescale(flat, pix)
    return {
        "n_frames": n_t,
        "t_int": t_int,
        "acf_min": acf_min,
        "tke_wake": float(tke[pix].mean()),
        "n_pix": int(pix.size),
    }


def survey() -> list[dict]:
    """Every paired ``(Re, AoA)`` cell, both domains."""
    real = {(c.re, c.aoa): c for c in load_index("real")}
    sim = {(c.re, c.aoa): c for c in load_index("sim")}
    rows = []
    for (re_, aoa), cr in sorted(real.items()):
        if (re_, aoa) in EXCLUDED_CELLS:
            continue
        key = next((k for k in sim if k[1] == aoa and abs(k[0] - re_) <= RE_MATCH_TOL), None)
        if key is None:
            continue
        a, b = cell_clock("real", re_, aoa), cell_clock("sim", *key)
        if a is None or b is None:
            continue
        height = projected_height(float(aoa))
        rows.append(
            {
                "re": re_,
                "aoa": aoa,
                "u_fs": cr.u_fs,
                "u_fs_sim": sim[key].u_fs,
                "h_m": height,
                "conv_per_frame_real": DT_REAL * cr.u_fs / height,
                "t_int_real": a["t_int"],
                "t_int_sim": b["t_int"],
                "acf_min_real": a["acf_min"],
                "acf_min_sim": b["acf_min"],
                "tke_real": a["tke_wake"],
                "tke_sim": b["tke_wake"],
            }
        )
        print(
            f"  Re={re_:>5} a={aoa:>2}  T_int real={a['t_int']:5.2f}  sim={b['t_int']:5.2f}  "
            f"step={b['t_int'] / a['t_int']:5.2f}",
            flush=True,
        )
    return rows


def loglog_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """``(slope, standard error, correlation)`` of ``log y`` on ``log x``."""
    ly, lx = np.log(y), np.log(x)
    design = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(design, ly, rcond=None)[0]
    resid = ly - (slope * lx + intercept)
    sxx = ((lx - lx.mean()) ** 2).sum()
    se = float(np.sqrt((resid**2).sum() / (len(lx) - 2) / sxx)) if len(lx) > 2 else float("nan")
    return float(slope), se, float(np.corrcoef(lx, ly)[0, 1])


def report(rows: list[dict]) -> dict:
    """Print the slope tests and the reconciliation attempt; return the summary."""
    col = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    angles = sorted({int(a) for a in col["aoa"]})

    print("\n" + "=" * 78)
    print("INTEGRAL TIMESCALE IN FRAMES  --  physical clock => -1, convective clock => 0")
    print("=" * 78)
    summary: dict = {"slopes": {}, "n_cells": len(rows)}
    for name, key in (("REAL", "t_int_real"), ("SIM ", "t_int_sim")):
        slope, se, corr = loglog_slope(col[key], col["u_fs"])
        summary["slopes"][key] = {"all": slope, "se": se, "r": corr}
        print(f"  {name} all cells        slope {slope:+.3f} +/- {se:.3f}   r = {corr:+.3f}")

    print("\n  within each angle (removes the h(AoA) confound):")
    for aoa in angles:
        m = col["aoa"] == aoa
        if m.sum() < 5:
            continue
        sr = loglog_slope(col["t_int_real"][m], col["u_fs"][m])[0]
        ss = loglog_slope(col["t_int_sim"][m], col["u_fs"][m])[0]
        summary["slopes"].setdefault("per_aoa", {})[str(aoa)] = {"real": sr, "sim": ss}
        print(f"    AoA {aoa:2d}  n={int(m.sum()):2d}   real {sr:+.3f}    sim {ss:+.3f}")

    print("\n  T_int summary (frames):")
    for key in ("t_int_real", "t_int_sim"):
        v = col[key]
        print(
            f"    {key:12} mean {v.mean():6.2f}  range {v.min():5.2f}-{v.max():5.2f}  "
            f"CV {v.std() / v.mean():5.1%}"
        )

    print("\n" + "=" * 78)
    print("HOW MUCH EVOLUTION IS IN A 20-FRAME WINDOW?  (assumption-free)")
    print("=" * 78)
    for key, name in (("t_int_real", "real"), ("t_int_sim", "sim ")):
        n = 20.0 / col[key]
        print(
            f"  {name} integral timescales per window: mean {n.mean():5.2f}  "
            f"range {n.min():5.2f}-{n.max():5.2f}  ({n.max() / n.min():.1f}x)"
        )
    step = col["t_int_sim"] / col["t_int_real"]
    summary["step"] = {
        "min": float(step.min()),
        "max": float(step.max()),
        "median": float(np.median(step)),
    }
    print(
        f"  step = T_int_sim/T_int_real: {step.min():.2f} to {step.max():.2f}, "
        f"median {np.median(step):.2f}, corr with log u_fs "
        f"{np.corrcoef(np.log(col['u_fs']), step)[0, 1]:+.3f}"
    )

    print("\n" + "=" * 78)
    print("CAN ONE CONSTANT RECONCILE THE CLOCKS?  (expected: no -- see the docstring)")
    print("=" * 78)
    t_conv = col["t_int_real"] * col["conv_per_frame_real"]
    k = t_conv / col["t_int_sim"]
    summary["reconciliation"] = {
        "real_t_int_conv_cv": float(t_conv.std() / t_conv.mean()),
        "k_median": float(np.median(k)),
        "k_cv": float(k.std() / k.mean()),
    }
    print(
        f"  real T_int in convective units: mean {t_conv.mean():.4f}  "
        f"CV {t_conv.std() / t_conv.mean():.1%}"
    )
    print(
        f"  implied convective time per sim frame: median {np.median(k):.5f}  "
        f"CV {k.std() / k.mean():.1%}   <-- too scattered to use"
    )
    span = 20 * col["conv_per_frame_real"]
    print(
        f"  a 20-frame REAL window spans {span.min():.2f}-{span.max():.2f} convective "
        f"times ({span.max() / span.min():.1f}x, L = projected height)"
    )

    print("\n" + "=" * 78)
    print("COHERENCE AND AMPLITUDE")
    print("=" * 78)
    print(f"{'AoA':>4} {'n':>3} {'ACF min real':>13} {'ACF min sim':>12} {'TKE sim/real':>14}")
    for aoa in angles:
        m = col["aoa"] == aoa
        ratio = (col["tke_sim"][m] / col["u_fs_sim"][m] ** 2) / (
            col["tke_real"][m] / col["u_fs"][m] ** 2
        )
        summary.setdefault("per_aoa_tke_ratio", {})[str(aoa)] = float(np.median(ratio))
        print(
            f"{aoa:4d} {int(m.sum()):3d} {col['acf_min_real'][m].mean():13.3f} "
            f"{col['acf_min_sim'][m].mean():12.3f} {np.median(ratio):14.2f}"
        )
    print("  ACF min near -1 = strongly periodic wake; near 0 = incoherent.")
    print("  Sim is not the more coherent arm, which is what refutes a 2-D simulation.")
    return summary


def plot(rows: list[dict], out_path) -> None:
    """Timescale against freestream speed, one panel per angle, both domains."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    angles = sorted({r["aoa"] for r in rows})
    fig, axes = plt.subplots(1, len(angles), figsize=(3.1 * len(angles), 3.4), sharey=True)
    for ax, aoa in zip(np.atleast_1d(axes), angles, strict=True):
        sel = [r for r in rows if r["aoa"] == aoa]
        u = np.array([r["u_fs"] for r in sel])
        for key, colour, label in (
            ("t_int_real", "tab:blue", "real"),
            ("t_int_sim", "tab:red", "sim"),
        ):
            y = np.array([r[key] for r in sel])
            ax.loglog(u, y, "o", color=colour, ms=4, label=label)
            slope = loglog_slope(y, u)[0]
            grid = np.linspace(u.min(), u.max(), 20)
            ax.loglog(
                grid,
                np.exp(np.polyval(np.polyfit(np.log(u), np.log(y), 1), np.log(grid))),
                "-",
                color=colour,
                lw=1,
            )
            ax.annotate(
                f"{label}: {slope:+.2f}",
                (0.05, 0.16 if label == "real" else 0.05),
                xycoords="axes fraction",
                color=colour,
                fontsize=8,
            )
        ax.set_title(f"AoA {aoa}", fontsize=9)
        ax.set_xlabel("$u_{fs}$ (m/s)", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("integral timescale (frames)", fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.suptitle("A physical clock gives slope -1, a convective clock gives 0", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(REPO_ROOT / "docs" / "figures" / "clocks.png"))
    a = p.parse_args()

    rows = survey()
    print(f"\npaired cells measured: {len(rows)}")
    summary = report(rows)

    json_path = REPO_ROOT / "docs" / "figures" / "clocks.json"
    payload = {"dt_real": DT_REAL, "tke_top_frac": TKE_TOP_FRAC, "summary": summary, "cells": rows}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {json_path}")
    plot(rows, a.out)


if __name__ == "__main__":
    main()
