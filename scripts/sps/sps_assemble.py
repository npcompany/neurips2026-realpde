"""Join a frozen backbone and an offline-fitted scale head into one shippable checkpoint.

The scale head is fitted on *cached* backbone outputs (``sps_scale_head.py``), so it lands
in ``runs/<arms>/<arm>/head.pt`` while the backbone stays where it was. Nothing downstream
takes two files: ``models.factory.load_for_inference``, the packager and the official
scorer all want a single checkpoint whose config describes the whole model. This writes it.

The width rule is copied next to the checkpoint rather than folded into it. It is a numpy
table fitted *after* the head, sizing what the head ranks, and keeping it separate is what
lets a rule be re-fitted -- or swapped for the one ``sps_select.py`` chose over all arms --
without touching the weights.

    uv run python scripts/sps/sps_assemble.py --arm-dir runs/scale_heads_final/rho8_s2 \
        --rule runs/width_rule_selected.json --out runs/unet_sps_logerr
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from rpde_t1.train.scale_fit import merge_scale_head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="runs/unet_bounds_frozen_cond/last.pt")
    ap.add_argument("--arm-dir", required=True, help="one arm written by sps_scale_head.py")
    ap.add_argument(
        "--rule",
        default=None,
        help="width rule json (default: the arm's own, but sps_select.py's is the "
        "one chosen over all arms and is what should ship)",
    )
    ap.add_argument(
        "--anchor",
        default=None,
        help="per-lead persistence anchor json. It displaces the prediction, so --rule must "
        "be the width rule refitted against the anchored error field; the two are copied in "
        "together so a build cannot pair an anchor with a rule fitted without it",
    )
    ap.add_argument("--out", required=True, help="run directory to write last.pt into")
    args = ap.parse_args()

    arm_dir = Path(args.arm_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = merge_scale_head(Path(args.backbone), arm_dir)
    torch.save(payload, out / "last.pt")

    rule_src = Path(args.rule) if args.rule else arm_dir / "width_rule.json"
    shutil.copy2(rule_src, out / "width_rule.json")
    anchor_src = Path(args.anchor) if args.anchor else None
    if anchor_src is not None:
        shutil.copy2(anchor_src, out / "lead_anchor.json")

    # Rebuild it here rather than trusting the key names: a state dict that does not load
    # is the whole failure this script exists to make impossible, and it is a second of work.
    from rpde_t1.models.factory import build_for_inference

    model = build_for_inference(payload["config"])
    model.load_state_dict(payload["model"])
    n_head = sum(p.numel() for p in model.head.parameters())

    print(f"backbone   : {args.backbone}")
    print(
        f"scale head : {arm_dir}  ({n_head:,} params, {payload['config']['extra']['scale_head']})"
    )
    print(f"width rule : {rule_src}")
    if anchor_src is not None:
        from rpde_t1.models.lead_anchor import LeadAnchor

        w = LeadAnchor.from_json(anchor_src).weights
        print(f"lead anchor: {anchor_src}  a_1..a_3 = {w[:3].round(3)}, {w[3:].size} pinned")
    print(f"written    : {out / 'last.pt'} and {out / 'width_rule.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
