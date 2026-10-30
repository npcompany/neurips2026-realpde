"""Apply one *fixed* interval configuration to a backbone and read every number it produces.

Why a new script rather than a chain of the existing three. ``sps_scale_head.py`` selects a
head, ``sps_select.py`` selects a rule and ``sps_anchor.py`` adds the anchor -- three entry
points with three fitting conventions, which is right while the configuration is being *chosen*
and wrong once it has been. Comparing backbones needs the opposite discipline: the
configuration is held fixed at the one the selection already picked, refitted on each backbone
because both the head and the rule are fitted against that backbone's own error field, and
every backbone measured the same way. Re-selecting per backbone would let each arm pick its own
best configuration, which is a selection artefact dressed up as an effect.

    uv run python scripts/sps/sps_apply.py --tag st --head rho8_s3 --n-rho 64 --asymmetric \
        --anchor 3 --out runs/sps_apply_st.json

Reports the ladder -- scale head alone, plus the rule, plus the anchor -- so a change that
moves one rung and not the others is visible, and the accuracy axes at each rung, since only
the anchor touches the prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rpde_t1 import N_MEASURED
from rpde_t1.models.lead_anchor import fit_lead_anchor
from rpde_t1.models.width_rule import WidthRule, fit_prescaled
from rpde_t1.score.sps_lab import Cached, cache_path, load_cache, quality_weights, sps_breakdown
from rpde_t1.train.scale_fit import (
    ScaleFitConfig,
    predict_scale,
    predict_tau,
    train_scale_head,
)

#: Head configurations, kept in sync with ``sps_scale_head.ARMS`` for the entries used here.
HEADS: dict[str, dict] = {
    "base": {},
    "l1": {"loss": "l1"},
    "rho8": {"n_rho": 8},
    "rho8_l1": {"n_rho": 8, "loss": "l1"},
    "rho8_s1": {"n_rho": 8, "seed": 1},
    "rho8_s2": {"n_rho": 8, "seed": 2},
    "rho8_s3": {"n_rho": 8, "seed": 3},
}


def _cat(caches: list[Cached], attr: str) -> np.ndarray:
    if len(caches) == 1:
        return getattr(caches[0], attr)
    return np.concatenate([getattr(c, attr) for c in caches])


def fit_rule(
    err: np.ndarray, half: np.ndarray, scored: np.ndarray, n_rho: int, asym: bool, n_lead: int = 1
):
    """The selected sizing procedure. One definition, in ``models.width_rule``.

    ``n_lead`` was pinned at 1 here from the start, so the multiplier is constant over the
    20-frame horizon while the head's predicted scale is not. It is a real axis of
    ``fit_prescaled`` (whose sibling defaults to 4) and it defaults to the shipped 1.
    """
    return fit_prescaled(err, half, scored, n_rho=n_rho, n_lead=n_lead, asymmetric=asym)


def score_split(
    c: Cached,
    scale: np.ndarray,
    rule: WidthRule | None,
    pred: np.ndarray | None = None,
    center: np.ndarray | None = None,
) -> dict:
    """SPS for one split, with the REPORTED prediction and the interval CENTRE separable.

    ``aggregate_sps`` reads the prediction only through its three accuracy factors, and
    computes ``inside`` from ``lower``/``upper`` alone -- nothing requires the interval to
    contain the prediction. ``scripts/sps/sps_ceiling.py`` records that as permitted and unused.
    It matters for any *reported-field* correction that raises the ceiling while making its
    own errors harder to bracket: a second-moment rescaling adds fluctuation uncorrelated
    with the truth, so centring the interval on the corrected field spends more interval
    efficiency than the ceiling gain is worth (measured: +0.45 ceiling against -0.056
    efficiency, a net -3.87 sps). Leaving the interval on the uncorrected field keeps both.

    ``pred`` is what gets reported and graded for accuracy; ``center`` is the field the
    interval is built around, and defaults to ``pred`` so every existing caller is unchanged.
    """
    p = c.pred if pred is None else pred
    ctr = p if center is None else center
    lo, hi = ctr.copy(), ctr.copy()
    lo[..., :N_MEASURED] = ctr[..., :N_MEASURED] - scale
    hi[..., :N_MEASURED] = ctr[..., :N_MEASURED] + scale
    if rule is not None:
        lo, hi = rule.apply(ctr, lo, hi)
    return sps_breakdown(p, lo, hi, c.target, q=quality_weights(p, c.target)["q"]).as_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache-dir", default="runs/sps_cache")
    ap.add_argument("--head", default="rho8_s3", help=f"one of {sorted(HEADS)}")
    ap.add_argument(
        "--head-set",
        dest="head_set",
        default=None,
        help="override any ScaleFitConfig field on top of --head, as `key=value,...` "
        "(e.g. `coords=true,steps=3000`). The named HEADS entries fix only `loss` and "
        "`n_rho`; the rest of the config still carries the FINDINGS 14 defaults, and "
        "`coords`/`steps` measure as worth ~+0.15 sps for no inference cost",
    )
    ap.add_argument("--fit-splits", nargs="+", default=["train_all"])
    ap.add_argument("--select-splits", nargs="+", default=["train_ends", "val_interp", "val_time"])
    ap.add_argument("--report-split", default="val_extrap")
    ap.add_argument(
        "--extra-splits",
        dest="extra_splits",
        nargs="*",
        default=[],
        help="more cached splits to SCORE at every rung (no role in the fit or the selection), "
        "e.g. hard5. Reported after the report split",
    )
    ap.add_argument(
        "--distill-arm",
        dest="distill_arm",
        default=None,
        help="an arm dir whose scales.npz holds a fitted head's error scale on the fit splits: "
        "the head here is then fitted to THAT head's log scale instead of to log|err| "
        "(distillation; scale_fit.train_scale_head). The fit splits must be in the npz",
    )
    ap.add_argument(
        "--no-save-scales",
        dest="save_scales",
        action="store_false",
        help="with --arm-dir: do not write scales.npz (about 80 MB per cached split)",
    )
    ap.add_argument("--n-rho", type=int, default=64)
    ap.add_argument("--asymmetric", action="store_true", default=True)
    ap.add_argument("--symmetric", dest="asymmetric", action="store_false")
    ap.add_argument("--anchor", type=int, default=3, help="leads the anchor fits; 0 disables it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arm-dir", default=None, help="persist the head and rule here")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--report-pred-tag",
        dest="report_pred_tag",
        default=None,
        help="on the report split, take the REPORTED prediction from this cache tag while the "
        "interval stays sized and centred on --tag's field. The decoupling the scorer permits "
        "(see score_split): it is how a reported-field correction keeps its ceiling gain "
        "without paying the interval efficiency that centring on it costs",
    )
    ap.add_argument(
        "--report-pred-all",
        dest="report_pred_all",
        action="store_true",
        help="with --report-pred-tag: take the reported prediction from that tag on the SELECTION "
        "splits too, so a configuration of the interval centre (e.g. a fluctuation-shrunk --tag) "
        "can be chosen on the selection splits against the field that is actually reported",
    )
    args = ap.parse_args()

    root = Path(args.cache_dir)
    names = [
        *dict.fromkeys(
            [*args.fit_splits, *args.select_splits, args.report_split, *args.extra_splits]
        )
    ]
    caches = {s: load_cache(cache_path(root, s, args.tag), s) for s in names}
    fit = [caches[s] for s in args.fit_splits]
    targets = [*dict.fromkeys([*args.select_splits, args.report_split, *args.extra_splits])]
    teacher = None
    if args.distill_arm:
        with np.load(Path(args.distill_arm) / "scales.npz") as z:
            teacher = [z[s] for s in args.fit_splits]

    overrides: dict = {}
    for item in filter(None, (args.head_set or "").split(",")):
        k, _, v = item.partition("=")
        current = getattr(ScaleFitConfig(), k)  # raises on an unknown field, which is the point
        overrides[k] = (
            v.lower() in ("1", "true", "yes") if isinstance(current, bool) else type(current)(v)
        )
    # Merged, not double-splatted: `--head-set loss=nll` on a HEADS entry that already fixes
    # `loss` is the whole point of the flag, and two `**` of overlapping dicts is a TypeError
    # ("got multiple values for keyword argument 'loss'"). The override wins, which is the
    # documented precedence -- "override any ScaleFitConfig field ON TOP OF --head".
    fields = {**HEADS[args.head], **overrides}
    cfg = ScaleFitConfig(
        **fields, notes=f"{args.tag}/{args.head}/{args.head_set or ''}"
    )
    head = train_scale_head(cfg, fit, device=args.device, teacher=teacher)
    scales = {s: predict_scale(head, c, cfg, device=args.device) for s, c in caches.items()}

    err_fit = _cat(fit, "err")
    scored_fit = _cat(fit, "scored")
    half_fit = (
        np.concatenate([scales[s] for s in args.fit_splits])
        if len(fit) > 1
        else scales[args.fit_splits[0]]
    )

    rule = fit_rule(err_fit, half_fit, scored_fit, args.n_rho, args.asymmetric)

    report: dict = {
        "tag": args.tag,
        "head": args.head,
        "head_set": args.head_set,
        "distill_arm": args.distill_arm,
        "extra_splits": args.extra_splits,
        "n_rho": args.n_rho,
        "asymmetric": args.asymmetric,
        "anchor": args.anchor,
        "fit_splits": args.fit_splits,
    }
    # The reported field, per split. Identical to the cache's own unless --report-pred-tag
    # names another one, in which case the interval below still centres on the cache's field.
    rep: dict[str, np.ndarray | None] = dict.fromkeys(targets)
    if args.report_pred_tag:
        # The REPORT split only, as the flag's name says. The selection splits exist to choose a
        # configuration, and the configuration here is fixed (FINDINGS 35's), so overriding them
        # would only require caches the fit side does not keep.
        for s in targets if args.report_pred_all else [args.report_split]:
            rep[s] = load_cache(cache_path(Path(args.cache_dir), s, args.report_pred_tag), s).pred
        report["report_pred_tag"] = args.report_pred_tag
        report["report_pred_all"] = bool(args.report_pred_all)

    rungs: dict[str, dict] = {}
    rungs["head_only"] = {
        s: score_split(caches[s], scales[s], None, pred=rep[s], center=caches[s].pred)
        for s in targets
    }
    rungs["plus_rule"] = {
        s: score_split(caches[s], scales[s], rule, pred=rep[s], center=caches[s].pred)
        for s in targets
    }

    anchor = None
    if args.anchor > 0:
        anchor = fit_lead_anchor(
            _cat(fit, "pred"), _cat(fit, "inputs"), _cat(fit, "target"), scored_fit
        ).restricted(n_lead=args.anchor)
        report["anchor_weights"] = anchor.weights.tolist()
        # The rule is fitted against the realised error field and the anchor moves it, so the
        # pair has to be refitted together -- shipping a rule fitted on the unanchored field is
        # the mismatch ``sps_assemble.py --anchor`` exists to prevent.
        pred_fit = anchor.apply(_cat(fit, "pred"), _cat(fit, "inputs"))
        rule_a = fit_rule(
            _cat(fit, "target")[..., :N_MEASURED] - pred_fit[..., :N_MEASURED],
            half_fit,
            scored_fit,
            args.n_rho,
            args.asymmetric,
        )
        rungs["plus_anchor"] = {
            s: score_split(
                caches[s],
                scales[s],
                rule_a,
                pred=anchor.apply(caches[s].pred if rep[s] is None else rep[s], caches[s].inputs),
                center=anchor.apply(caches[s].pred, caches[s].inputs),
            )
            for s in targets
        }
        rule = rule_a

    hdr = f"{'rung':<14}" + "".join(f"{s[:11]:>12}" for s in args.select_splits)
    hdr += f"{'mean sel':>10}{'REPORT':>9}{'cover':>7}{'eff':>7}{'ceil':>7}"
    hdr += "".join(f"{s[:11]:>12}" for s in args.extra_splits)
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, sc in rungs.items():
        sel = [sc[s]["sps"] for s in args.select_splits]
        r = sc[args.report_split]
        line = f"{name:<14}" + "".join(f"{v:>12.2f}" for v in sel)
        line += f"{np.mean(sel):>10.2f}{r['sps']:>9.2f}{r['coverage']:>7.3f}"
        line += f"{r['efficiency']:>7.3f}{r['ceiling']:>7.2f}"
        line += "".join(f"{sc[s]['sps']:>12.2f}" for s in args.extra_splits)
        print(line)
    report["rungs"] = rungs

    if args.arm_dir:
        d = Path(args.arm_dir)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"head": head.state_dict(), "config": cfg.__dict__}, d / "head.pt")
        if args.save_scales:
            np.savez(d / "scales.npz", **{k: v.astype(np.float32) for k, v in scales.items()})
        if cfg.loss == "nll":
            # A two-parameter head's second output, in the same per-split layout as the scales
            # so `sps_shape.py --tau-npz` can read it as a candidate binning axis. Written here
            # rather than recomputed there: it needs the head, the features and the GPU.
            np.savez(
                d / "tau.npz",
                **{
                    k: predict_tau(head, c, cfg, device=args.device).astype(np.float32)
                    for k, c in caches.items()
                },
            )
        rule.to_json(d / "width_rule.json")
        if anchor is not None:
            anchor.to_json(d / "lead_anchor.json")

    out = Path(args.out or f"runs/sps_apply_{args.tag}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
