"""Training-seed noise: the decision threshold every A/B is measured against.

Without this number an experiment has no verdict. The project has already been burned
both ways -- a +0.69 SPS effect was dismissed as "probably seed noise" when the measured
noise was 0.03-0.28, and a set of joint-training results was quoted at n=1 with 50M
parameters unfrozen and no noise estimate at all.

Two noise sources, measured separately because they compose differently:

*Evaluation noise* comes from which windows happen to be in the eval set. It is free to
measure -- a paired bootstrap over the fixed windows, no retraining -- and it is ~8x
tighter for a *paired difference* (+/-0.24 SPS on ``val_extrap``) than for a single
absolute number (+/-2), because both models see identical windows and their errors are
correlated. See ``score.report``.

*Training noise* comes from initialisation and data order, and needs one run per seed.
This module turns those runs into a threshold. It is the expensive one, which is why the
frozen-backbone case was measured first (sd 0.03-0.28 SPS) -- but that figure licenses
only frozen-head comparisons. Anything that trains a backbone needs its own measurement.

**They add in variance, not in quotient.** A comparison of two freshly trained models
carries both, so the threshold below is a floor on what such a comparison can resolve,
not the whole story.
"""

from __future__ import annotations

import math
from typing import Any

# Re-exported: several callers import ``SUBSCORES`` from here. ``_``-prefixed keys in a
# score table are diagnostics (per-window seconds, normalized MSE) rather than
# subscores, and are skipped wherever this is iterated.
from .. import SUBSCORES

#: Two-sided 95% normal quantile. Used rather than a t quantile because the seed counts
#: we can afford (3-5) make the difference smaller than the honesty of the model itself:
#: seeds are not quite exchangeable across data orders. Reported as "~95%", not exact.
Z_95 = 1.959964

#: Axes a frozen backbone cannot move. If a frozen-head sweep shows spread here, the
#: backbone is not actually frozen -- which is a bug worth failing loudly on rather than
#: a noise measurement.
FROZEN_INVARIANT = ("rel_l2_score", "tke_score", "mvpe_score")


def _sd(values: list[float]) -> float:
    """Sample standard deviation, ``ddof=1``. Zero for n<2 rather than undefined."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def min_detectable_effect(sd: float, n_per_arm: int = 1, z: float = Z_95) -> float:
    """Smallest difference between two arms that this much noise lets you resolve.

    Comparing two arms of ``n`` seeds each, the difference of the arm means has standard
    error ``sd * sqrt(2/n)``, so at ~95% confidence an effect must exceed
    ``z * sd * sqrt(2/n)``.

    The ``sqrt(2)`` is the part most often dropped, and dropping it makes the bar look
    ~40% easier than it is: *both* arms are noisy. At ``n=1`` the threshold is ``2.77*sd``,
    not ``1.96*sd``.
    """
    if sd <= 0.0 or n_per_arm < 1:
        return 0.0
    return z * sd * math.sqrt(2.0 / n_per_arm)


def seed_noise(
    table: dict[str, dict[str, dict[str, float]]],
    arm_of: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate a ``{run: {split: subscores}}`` table into per-arm noise statistics.

    ``arm_of`` maps a run name to the arm it belongs to; runs in the same arm differ only
    by seed. Omitted, every run is treated as one arm called ``"all"`` -- correct for a
    plain seed sweep, wrong for a two-arm A/B, so the caller must be explicit there.

    Returns, per arm and split and subscore: ``n``, ``mean``, ``sd``, ``min``, ``max``,
    ``sem``, and ``mde_1`` / ``mde_n`` -- the effect resolvable comparing single runs, and
    comparing arms of this size.
    """
    arms: dict[str, list[str]] = {}
    for run in table:
        arms.setdefault((arm_of or {}).get(run, "all"), []).append(run)

    out: dict[str, Any] = {"arms": {}, "n_runs": len(table)}
    for arm, runs in sorted(arms.items()):
        splits = sorted({s for r in runs for s in table[r]})
        per_split: dict[str, Any] = {}
        for split in splits:
            stats: dict[str, Any] = {}
            for key in SUBSCORES:
                values = [
                    float(table[r][split][key])
                    for r in runs
                    if split in table[r] and key in table[r][split]
                ]
                if not values:
                    continue
                sd = _sd(values)
                n = len(values)
                stats[key] = {
                    "n": n,
                    "mean": sum(values) / n,
                    "sd": sd,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                    "sem": sd / math.sqrt(n) if n else 0.0,
                    "mde_1": min_detectable_effect(sd, 1),
                    "mde_n": min_detectable_effect(sd, n),
                }
            per_split[split] = stats
        out["arms"][arm] = {"runs": sorted(runs), "splits": per_split}
    return out


def frozen_backbone_violations(noise: dict[str, Any], tol: float = 1e-6) -> list[str]:
    """Axes that moved when a frozen backbone says they cannot.

    A frozen-backbone sweep changes only the head, so the prediction is bit-identical and
    rel_l2 / tke / mvpe must be constant across seeds. Spread there means the backbone is
    being trained after all, and every "frozen head" conclusion drawn from those runs is
    void. Returns human-readable descriptions, empty when clean.

    ``time_score`` is deliberately excluded: it is wall-clock and varies run to run for
    reasons that have nothing to do with the weights.
    """
    bad = []
    for arm, body in noise["arms"].items():
        for split, stats in body["splits"].items():
            for key in FROZEN_INVARIANT:
                s = stats.get(key)
                if s and s["n"] > 1 and s["range"] > tol:
                    bad.append(
                        f"{arm}/{split}/{key}: range {s['range']:.4g} across {s['n']} seeds "
                        f"(a frozen backbone must give exactly 0)"
                    )
    return bad


def format_seed_noise(noise: dict[str, Any], frozen: bool = False) -> str:
    """A table plus the thresholds it implies. The thresholds are the point."""
    lines = [f"seed noise over {noise['n_runs']} run(s)", ""]
    for arm, body in noise["arms"].items():
        for split, stats in body["splits"].items():
            lines.append(f"[{arm}] {split}   ({len(body['runs'])} seeds)")
            lines.append(
                f"  {'subscore':<16}{'mean':>9}{'sd':>8}{'range':>8}{'1v1 thr':>9}{'nvn thr':>9}"
            )
            for key in SUBSCORES:
                s = stats.get(key)
                if not s:
                    continue
                lines.append(
                    f"  {key:<16}{s['mean']:>9.3f}{s['sd']:>8.3f}{s['range']:>8.3f}"
                    f"{s['mde_1']:>9.3f}{s['mde_n']:>9.3f}"
                )
            lines.append("")

    lines += [
        "1v1 thr = smallest difference two SINGLE runs can establish (~95%): 2.77*sd.",
        "nvn thr = same, comparing arms of this many seeds each: 1.96*sd*sqrt(2/n).",
        "A measured difference below the relevant threshold is not a result.",
        "",
        "Training noise only. A comparison of two freshly trained models also carries",
        "evaluation noise (paired bootstrap, +/-0.24 SPS on val_extrap); variances add.",
    ]
    if frozen:
        violations = frozen_backbone_violations(noise)
        lines.append("")
        lines += (
            ["FROZEN-BACKBONE CHECK FAILED:", *(f"  {v}" for v in violations)]
            if violations
            else ["Frozen-backbone check passed: rel_l2/tke/mvpe identical across seeds."]
        )
    return "\n".join(lines)
