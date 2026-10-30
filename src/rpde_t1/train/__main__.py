"""``python -m rpde_t1.train`` -- train a model, or score a checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from ..data.masking import MASK_MODES
from ..features import COND_MODES
from .calibrate import format_scan, save_scan, scan_factor
from .config import TrainConfig
from .evaluate import evaluate_checkpoint, format_table, save_table
from .loop import BASELINES, train


def _cmd_fit(args: argparse.Namespace) -> int:
    overrides = {
        k: v for k, v in vars(args).items() if k not in ("func", "cmd", "config") and v is not None
    }
    cfg = TrainConfig.load(Path(args.config)) if args.config else TrainConfig()
    for k, v in overrides.items():
        setattr(cfg, k, tuple(v) if k == "val_splits" else v)
    if cfg.out_dir == "runs/default":
        tag = f"{cfg.model}{'_res' if cfg.residual else ''}{'_bounds' if cfg.bounds else ''}"
        cfg.out_dir = f"runs/{tag}_{cfg.train_kind}"
    # Improved operating-point estimators: RPDE_RE_CORNERS enables corners3 Re, RPDE_AOA_NET points
    # at the neural-anchored AoA net. Both no-ops unless set, so the operating-point estimate uses
    # them in both training and inference.
    from ..models.aoa_net import install_from_env  # noqa: PLC0415

    if install_from_env():
        print("[estimators] operating-point overrides active", flush=True)
    train(cfg)
    return 0


def parse_arm(spec: str) -> tuple[str, dict[str, object]]:
    """``"name=field=value,field=value"`` -> ``("name", {field: typed_value})``.

    Values are coerced to the type ``TrainConfig`` declares for that field, so
    ``w_tke=0.5`` becomes a float and ``bounds=true`` a bool. An unknown field is an error
    rather than a silently ignored override -- a typo'd arm that trains the base config
    twice would look like a real null result.
    """
    name, _, rest = spec.partition("=")
    if not name:
        raise SystemExit(f"--arm needs a name: got {spec!r}")
    overrides: dict[str, object] = {}
    types = {f.name: f.type for f in fields(TrainConfig)}
    for item in filter(None, rest.split(",")):
        field, _, raw = item.partition("=")
        if field not in types:
            raise SystemExit(f"--arm {name}: unknown config field {field!r}")
        declared = str(types[field])
        if "bool" in declared:
            overrides[field] = raw.lower() in ("1", "true", "yes", "on")
        elif "int" in declared and "float" not in declared:
            overrides[field] = int(raw)
        elif "float" in declared:
            overrides[field] = float(raw)
        else:
            overrides[field] = raw
    return name, overrides


def _cmd_seeds(args: argparse.Namespace) -> int:
    from .seeds import run_sweep

    cfg = TrainConfig.load(Path(args.config)) if args.config else TrainConfig()
    for dest in TRAIN_ARG_DESTS:
        value = getattr(args, dest, None)
        if value is not None:
            setattr(cfg, dest, value)

    arms = dict(parse_arm(spec) for spec in args.arm) if args.arm else None
    prefix = args.name or cfg.model
    # One W&B group per sweep, so the arms and their seeds read as one experiment rather
    # than as N unrelated runs. Each run's own name still comes from its output directory
    # (``<arm>_s<seed>``), which is the join key back to the checkpoint on disk.
    if cfg.wandb and not cfg.wandb_group:
        cfg.wandb_group = prefix
    if arms:
        arms = {f"{prefix}_{arm}": ov for arm, ov in arms.items()}
    else:
        arms = {prefix: {}}

    run_sweep(
        cfg,
        seeds=tuple(args.seeds),
        splits=tuple(args.splits),
        arms=arms,
        force=args.force,
        bounds=bool(cfg.bounds),
        dry_run=args.dry_run,
        out=Path(args.out) if args.out else None,
    )
    return 0


def _cmd_tke_blend(args: argparse.Namespace) -> int:
    """Tune the post-hoc fluctuation-blend exponent on one split, report it on others."""
    import torch

    from ..models.predictor import make_predict
    from .calibrate import format_blend_scan, scan_tke_blend
    from .evaluate import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(Path(args.checkpoint), device=device)
    # ``cond_mode`` from the CHECKPOINT, as every other serving path does (sps_cache.py, the
    # submission template): it builds the conditioning vector the backbone was trained on. It
    # was omitted until 2026-09-18, so these two commands could only serve the default
    # 3-column vector -- a no-op for every recorded head (all `operating_point`) and a hard
    # shape error for the dt-conditioned recipe (FINDINGS 166c).
    predict = make_predict(model, device=device, normalizer=cfg.normalizer, cond_mode=cfg.cond_mode)

    rows, best = scan_tke_blend(predict, args.fit_split)
    print(format_blend_scan(rows, args.fit_split))
    print(f"\nbest blend on {args.fit_split}: {best:.2f}")

    for split in args.report_splits:
        if split == args.fit_split:
            continue
        held, _ = scan_tke_blend(predict, split, blends=(0.0, best))
        print(f"\nheld out on {split} (the exponent was NOT fitted here):")
        print(format_blend_scan(held, split))

    if args.out:
        payload = {"fit_split": args.fit_split, "best_blend": best, "rows": rows}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.out}")
    return 0


def _cmd_tke_head(args: argparse.Namespace) -> int:
    """Fit the second-moment head on cached TKE maps and report it against the blend."""
    import torch

    from ..models.aoa_net import install_from_env
    from ..models.predictor import make_predict
    from .evaluate import load_checkpoint
    from .loop import geom_names
    from .tke_fit import collect_maps, fit_head, format_report, report, save_report

    # Operating-point overrides (RPDE_RE_CORNERS / RPDE_AOA_NET): the head must be fit on the
    # same conditioning the backbone was trained and will be served with. No-op unless set.
    if install_from_env():
        print("[estimators] operating-point overrides active", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(Path(args.checkpoint), device=device)
    # ``cond_mode`` from the CHECKPOINT, as every other serving path does (sps_cache.py, the
    # submission template): it builds the conditioning vector the backbone was trained on. It
    # was omitted until 2026-09-18, so these two commands could only serve the default
    # 3-column vector -- a no-op for every recorded head (all `operating_point`) and a hard
    # shape error for the dt-conditioned recipe (FINDINGS 166c).
    #
    # ``state_frames`` is the same class of argument for the other axis: a stream-state
    # backbone reads two pseudo-frames the dataset synthesises, and the second-moment head's
    # own input features must stay on the measured 20 frames, which the predictor slices.
    stream_state = str(getattr(cfg, "stream_state", "") or "")
    from ..models.factory import stream_state_frames  # noqa: PLC0415

    predict = make_predict(
        model,
        device=device,
        normalizer=cfg.normalizer,
        cond_mode=cfg.cond_mode,
        state_frames=stream_state_frames(stream_state),
        # A geometry-plane backbone (the `unet2d` SDF plane) needs its planes here exactly as
        # sps_cache.py / arm_report.py build them; without this the fit is a channel-count error.
        geom_channels=geom_names(cfg),
    )

    splits = [args.fit_split, *[s for s in args.report_splits if s != args.fit_split]]
    maps = {}
    for split in splits:
        # Only the fit split may come from a different domain. Sim supplies 10x the
        # supervision and, crucially, covers the two Reynolds rows real withholds
        # (15225 and 27975) -- the regime where the real-fitted head is weakest.
        kind = args.fit_kind if split == args.fit_split else "real"
        print(f"collecting maps for {split} ({kind}) ...")
        maps[split] = collect_maps(
            predict,
            split,
            kind=kind,
            stride=args.stride,
            max_windows=args.max_windows,
            cond_mode=args.cond_mode,
            stream_state=stream_state,
        )
        print(f"  {maps[split]['ke_pred'].shape[0]} windows")

    # Every arm reuses the one map collection, so an ablation costs one backbone pass
    # rather than one per arm. ``--arm name=feat,feat`` names the channels to zero.
    arms: dict[str, tuple[str, ...]] = {"full": ()}
    for spec in args.arm or []:
        name, _, dropped = spec.partition("=")
        arms[name] = tuple(f for f in dropped.split(",") if f)

    select = maps.get(args.select_split) if args.select_split else None
    if args.select_split and select is None:
        raise SystemExit(f"--select-split {args.select_split!r} is not among {splits}")

    heads: dict[str, tuple[object, tuple[str, ...]]] = {}
    histories: dict[str, list] = {}
    for name, drop in arms.items():
        print(f"\nfitting arm {name!r}" + (f" (dropping {list(drop)})" if drop else ""))
        head, hist = fit_head(
            maps[args.fit_split],
            steps=args.steps,
            lr=args.lr,
            width=args.width,
            depth=args.depth,
            seed=args.seed,
            device=device,
            drop=drop,
            select_maps=select,
            weight_decay=args.weight_decay,
            objective_mode=args.objective_mode,
            per_component=args.per_component,
        )
        heads[name] = (head, drop)
        histories[name] = hist

    rep = report(heads, maps, args.fit_split)
    rep["select_split"] = args.select_split
    rep["arms"] = {k: list(v) for k, v in arms.items()}
    rep["histories"] = histories
    print()
    print(format_report(rep))

    if args.out:
        print(f"\nwritten: {save_report(rep, Path(args.out))}")
        ckpt = Path(args.out).with_suffix(".pt")
        torch.save(
            {
                "heads": {k: h.state_dict() for k, (h, _) in heads.items()},
                "drops": {k: list(d) for k, (_, d) in heads.items()},
                "width": args.width,
                "depth": args.depth,
                # Recorded because the deployed path cannot infer it: ``tke_features`` always
                # emits every plane, zeroing ``tau`` when the conditioning vector does not carry
                # it, so a head fitted WITH tau and one fitted with the plane zeroed have the
                # SAME input width. Without this, ``inference_log_ratio`` guessed from the width
                # and served live tau to a head that was trained on zeros -- measured as a
                # log-ratio shift of up to 0.31 (FINDINGS 63m).
                "cond_mode": args.cond_mode,
                # Informational: ``load_head`` reads the output width from the weights.
                "per_component": bool(args.per_component),
            },
            ckpt,
        )
        print(f"written: {ckpt}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from ..models.aoa_net import install_from_env

    # Score a checkpoint under the same operating-point overrides it was trained with, so an
    # estimator-conditioned model is not silently scored on §122. No-op unless the env is set.
    if install_from_env():
        print("[estimators] operating-point overrides active", flush=True)
    rows = {}
    for ckpt in args.checkpoints:
        # Qualify with the filename when a run contributes more than one checkpoint:
        # the run directory alone collides between last.pt and best.pt, and the second
        # silently overwrote the first, so half the requested rows vanished from the
        # table with no warning.
        label = Path(ckpt).parent.name
        if label in rows or sum(Path(c).parent.name == label for c in args.checkpoints) > 1:
            label = f"{label}/{Path(ckpt).stem}"
        rows[label] = evaluate_checkpoint(
            Path(ckpt),
            splits=tuple(args.splits),
            bounds=args.bounds,
            bound_floor=args.bound_floor,
            stats_domain=args.stats_domain,
            tke_blend=args.tke_blend,
            tke_head=args.tke_head,
            tke_shrink=args.tke_shrink,
            width_rule=args.width_rule,
        )
    print()
    print(format_table(rows))
    if args.out:
        print(f"\nwritten: {save_table(rows, Path(args.out))}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit the interval width multiplier on one split, report it on another."""
    import torch

    from ..models.predictor import make_predict
    from .evaluate import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(Path(args.checkpoint), device=device)
    predict = make_predict(model, device=device, return_bounds=True)

    rows, best = scan_factor(predict, args.fit_split)
    print(format_scan(rows, args.fit_split))
    print(f"\nbest width multiplier on {args.fit_split}: {best:.2f}")

    if args.report_split and args.report_split != args.fit_split:
        held, _ = scan_factor(predict, args.report_split, factors=(1.0, best))
        base = next(r for r in held if r["factor"] == 1.0)
        tuned = next(r for r in held if r["factor"] == best)
        print(f"\nheld out on {args.report_split} (the multiplier was NOT fitted here):")
        print(
            f"  uncalibrated (factor 1.00): sps={base['sps_score']:.2f} "
            f"coverage={base['coverage_scored']:.3f}"
        )
        print(
            f"  calibrated   (factor {best:.2f}): sps={tuned['sps_score']:.2f} "
            f"coverage={tuned['coverage_scored']:.3f}"
        )
        print(f"  delta: {tuned['sps_score'] - base['sps_score']:+.2f} sps points")

    if args.out:
        print(f"\nwritten: {save_scan(rows, best, Path(args.out))}")
    return 0


#: Arguments that describe *what to train*, as opposed to what to do with it. Shared by
#: ``fit`` and ``seeds`` so a sweep cannot drift from a single run: duplicating thirty
#: flags is how the two would end up accepting different options.
TRAIN_ARG_DESTS: tuple[str, ...] = (
    "model",
    "cond_planes",
    "train_split",
    "train_kind",
    "steps",
    "batch_size",
    "grad_accum",
    "grad_clip",
    "normalizer",
    "mask_prob",
    "mask_swap_prob",
    "real_noise_scale",
    "joint_sim_kind",
    "joint_rescale_fluct",
    "joint_sim_split",
    "distill_from",
    "distill_alpha",
    "distill_tke_truth",
    "distill_part",
    "re_tilt",
    "noise_scale",
    "zero_p",
    "p_target",
    "w_p",
    "w_pvar",
    "rescale_fluct",
    "pe_content",
    "arch",
    "phase_aug",
    "lr",
    "warmup_steps",
    "stride",
    "t_in",
    "history_dropout",
    "recent_dropout",
    "recent_dropout_max",
    "stream_state",
    "state_dropout",
    "state_dropout_max",
    "stream_corr_lam",
    "stream_corr_ema",
    "recur_k",
    "recur_burn",
    "train_compile",
    "state_lr_mult",
    "state_norm",
    "state_anchor",
    "state_planes",
    "state_norm_floor",
    "state_norm_mode",
    "dt_aug",
    "dt_ratio_aug",
    "dt_ratio_p1",
    "cond_mode",
    "cond_source",
    # Declared with a `--re-tilt` flag and a TrainConfig field, but absent here until
    # 2026-08-17: `train seeds` accepted the flag and never applied it, so a sweep arm that
    # asked for a tilted sampler trained on plain shuffling while its help text said
    # otherwise. `train fit` was unaffected (it routes every arg generically), which is why
    # this survived. Caught by `test_every_fit_flag_reaches_the_config`, not by a run.
    "re_tilt",
    "residual",
    "residual_mode",
    "residual_tau",
    "bounds",
    "freeze_backbone",
    "dropout",
    "drop_path",
    "drop_path_hi",
    "dropout_sites",
    "dropout_kind",
    "w_jepa",
    "jepa_ema",
    "amplitude_free",
    "field_norm",
    "fluct_aug",
    "aoa_jitter_deg",
    "fluct_aug_hi",
    "tgt_amp_aug",
    "tgt_amp_aug_lo",
    "tgt_fluct_aug",
    "tgt_fluct_aug_hi",
    "mean_aug",
    "cond_drop",
    "w_tke_amp",
    "tke_amp_delta",
    "w_errhomog",
    "errhomog_k",
    "moment_joint",
    "w_moment",
    "moment_of",
    "llrd",
    "freeze_patterns",
    "bound_cond",
    "bound_width",
    "bound_depth",
    "bound_noise",
    "bound_mode",
    "bound_logs",
    "bound_coords",
    "head_lr_mult",
    "w_tke",
    "w_spectral",
    "w_tke_shape",
    "w_tke_huber",
    "tke_huber_delta",
    "tke_huber_normalize",
    "tke_gate",
    "tke_head_e2e",
    "kmap_channel",
    "kmap_e2e",
    "w_kmap",
    "kmap_base",
    "w_field",
    "lead_decay",
    "w_ceiling",
    "w_absorb",
    "distill_head",
    "distill_head_shrink",
    "absorb_head",
    "absorb_shrink",
    "tke_loop_head",
    "tke_loop_head_shrink",
    "tke_loop_head_field",
    "ceiling_huber",
    "w_relfield",
    "w_meanfield",
    "w_vorticity",
    "w_enstrophy",
    "w_mvpe",
    "w_rstress",
    "w_espec",
    "helm_lambda",
    "w_sps",
    "w_field",
    "w_credit",
    "credit_k",
    "w_headcredit",
    "headcredit_arm",
    "headcredit_tau",
    "headcredit_lo",
    "headcredit_track_lr",
    "sam_rho",
    "rdrop_alpha",
    "rollout_w",
    "rollout_mode",
    "rollout_aoa_min",
    "rollout_re_min",
    "nbr_prior",
    "nbr_prior_mode",
    "headcredit_notch_lo",
    "headcredit_notch_re",
    "headcredit_notch_width",
    "huber_mix",
    "mean_fluct_lambda",
    "mean_fluct_robust",
    "piv_noise",
    "piv_noise_target",
    "aux_mask",
    "aux_mask_ratio",
    "aux_mask_plane",
    "w_recon",
    "w_recon_tke",
    "recon_tke_scope",
    "recon_tke_target",
    "recon_tke_spatial",
    "recon_stop_frac",
    "w_recon_acf",
    # Same omission as ``re_tilt`` below and found the same way: declared on the ``fit``
    # parser, backed by a ``TrainConfig`` field, and absent here -- so ``fit`` picked it up
    # (it copies ``vars(args)`` wholesale) while every ``seeds`` arm silently trained with
    # the default. A reconstruction sweep would have run every arm on the noised field.
    "recon_clean_field",
    "w_fine",
    "w_subgrid",
    "w_paux",
    "w_opreg",
    "huber_delta",
    "weight_decay",
    "ema_decay",
    "joint_sim_frac",
    "joint_sim_frac_end",
    "synth_path",
    "synth_frac",
    # Pre-existing omission, caught by ``test_every_fit_flag_reaches_the_config`` while the two
    # above were added: ``--re-tilt`` has been accepted by the ``fit`` parser and dropped on the
    # floor by ``seeds`` since it landed. A tilt sweep would have run every arm untilted.
    "re_tilt",
    "align_weight",
    # And its sibling: ``--coral-weight`` is the second domain-alignment term and was
    # dropped by ``seeds`` for the same reason ``align_weight`` was not. Anything comparing
    # the two across seeds was comparing align-vs-nothing.
    "coral_weight",
    "l2sp",
    "trunk_lr_scale",
    "init_from",
    "unfreeze",
    "geom_channels",
    "flow_channels",
    "num_workers",
    "deterministic",
    "eval_every",
    "score_points",
    "snapshot_every",
    "val_splits",
    "wandb",
    "wandb_project",
    "wandb_entity",
    "wandb_name",
    "wandb_group",
    "wandb_tags",
    "tke_lead_dropout",
    "w_tke_tprofile",
    "log_domain_split",
)


def _on_off(raw: str) -> bool:
    """``on``/``off`` -> bool, for a tri-state flag whose third state is *absent*.

    ``store_true`` cannot express this: its default would collide with "inherit". Rejecting
    anything else keeps a typo from landing as ``bool("of") == True``.
    """
    if raw.lower() in ("on", "true", "1", "yes"):
        return True
    if raw.lower() in ("off", "false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected on|off, got {raw!r}")


def _add_train_args(f: argparse._ActionsContainer) -> None:
    """Attach the training arguments to a subparser."""
    f.add_argument("--config", default=None, help="JSON config to start from")
    f.add_argument("--model", default=None, help=f"one of {BASELINES} or our own")
    f.add_argument("--train-split", dest="train_split", default=None)
    f.add_argument(
        "--train-kind",
        dest="train_kind",
        default=None,
        choices=["real", "sim", "sim_resampled", "sim_resampled2", "sim_units"],
    )
    f.add_argument("--steps", type=int, default=None)
    f.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    f.add_argument("--grad-accum", dest="grad_accum", type=int, default=None)
    f.add_argument("--grad-clip", dest="grad_clip", type=float, default=None)
    f.add_argument("--normalizer", default=None, choices=["freestream", "gaussian", "none"])
    f.add_argument("--mask-prob", dest="mask_prob", type=float, default=None)
    f.add_argument("--noise-scale", dest="noise_scale", type=float, default=None)
    f.add_argument(
        "--real-noise-scale",
        dest="real_noise_scale",
        type=float,
        default=None,
        help="multiplicative noise on the real input window only (not the target, not sim)",
    )
    f.add_argument(
        "--joint-sim-kind",
        dest="joint_sim_kind",
        default=None,
        choices=["sim", "sim_resampled", "sim_resampled2", "sim_units"],
        help="which simulated grid the joint half draws from; empty inherits the pretrain's",
    )
    f.add_argument(
        "--joint-sim-split",
        dest="joint_sim_split",
        default=None,
        choices=[
            "sim_train",
            "sim_all",
            "sim_all_but_real_val",
            "sim_all_but_sim_only_rows",
            "sim_hifi",
        ],
        help="which split the joint half draws from (default sim_train). `sim_all` adds sim's "
        "own held-out rungs 15225/3750/27975, 18%% more operating points, and makes "
        "val_interp/val_extrap in-sample as conditioning -- see TrainConfig.joint_sim_split "
        "and FINDINGS 66 for how much of that is new coverage (8 of 12 cells are sim-covered "
        "either way)",
    )
    f.add_argument(
        "--mask-swap-prob",
        dest="mask_swap_prob",
        type=float,
        default=None,
        help="probability of unioning another training cell's PIV measurement mask into a "
        "sample, so hole geometry is augmented rather than memorised (train loader only)",
    )
    f.add_argument(
        "--rescale-fluct",
        dest="rescale_fluct",
        action="store_true",
        default=None,
        help="rescale simulated fluctuations per cell toward real's turbulence intensity "
        "(see TrainConfig.rescale_fluct)",
    )
    f.add_argument(
        "--arch",
        default=None,
        help="architecture hyper-parameters as key=value pairs, e.g. "
        "'dim=512,depth=10,patch=20x2x2'. Tuples are written AxBxC. Unknown keys raise "
        "from the constructor rather than being dropped (see TrainConfig.arch)",
    )
    f.add_argument(
        "--no-rescale-fluct",
        dest="rescale_fluct",
        action="store_false",
        default=None,
        help="turn the per-cell fluctuation rescaling off. Needed when a real fine-tune "
        "starts from a `--config` written by a sim pretrain that used it: the correction "
        "is sim-only and make_dataset raises on real data",
    )
    f.add_argument(
        "--pe-content",
        dest="pe_content",
        action="store_true",
        default=None,
        help="ModPCT only: add the centroid Fourier encoding to the tokens as content as "
        "well as feeding the adaLN conditioning (see TrainConfig.pe_content)",
    )
    f.add_argument(
        "--zero-p",
        dest="zero_p",
        action="store_true",
        default=None,
        help="zero the simulated pressure plane in input and target, so a simulated "
        "window looks exactly like a real one (see TrainConfig.zero_p)",
    )
    f.add_argument(
        "--joint-rescale-fluct",
        dest="joint_rescale_fluct",
        action="store_true",
        default=None,
        help="rescale the JOINT stage's simulated half toward real turbulence intensity, "
        "overriding what the pretrain recorded (see TrainConfig.joint_rescale_fluct). "
        "Unset inherits, which is what --rescale-fluct alone silently does",
    )
    f.add_argument(
        "--no-joint-rescale-fluct",
        dest="joint_rescale_fluct",
        action="store_false",
        default=None,
        help="force the joint stage's simulated half UNrescaled even if the pretrain was",
    )
    f.add_argument(
        "--p-target",
        dest="p_target",
        default=None,
        choices=["", "gauge"],
        help="supervise simulated pressure on the backbone's own third output channel, "
        "gauge-removed and scaled by u_fs**2 (see TrainConfig.p_target)",
    )
    f.add_argument(
        "--w-p",
        dest="w_p",
        type=float,
        default=None,
        help="weight on the field loss's pressure term relative to u and v (default 1.0)",
    )
    f.add_argument(
        "--w-pvar",
        dest="w_pvar",
        type=float,
        default=None,
        help="weight on the pressure channel's temporal variance (needs --p-target gauge)",
    )
    f.add_argument("--lr", type=float, default=None)
    f.add_argument("--stride", type=int, default=None)
    f.add_argument(
        "--t-in",
        dest="t_in",
        type=int,
        default=None,
        help="input frames the model reads (default 20; 40 folds the previous 20 into the stem)",
    )
    f.add_argument(
        "--history-dropout",
        dest="history_dropout",
        type=float,
        default=None,
        help="probability a window's earlier frames are zeroed (t_in>20; Track-2 cold start)",
    )
    f.add_argument(
        "--recent-dropout",
        dest="recent_dropout",
        type=float,
        default=None,
        help="probability a window's MOST RECENT frames are replaced by its time mean "
        "(tests whether the 2-frame effective memory is a limit or a shortcut)",
    )
    f.add_argument(
        "--recent-dropout-max",
        dest="recent_dropout_max",
        type=int,
        default=None,
        help="deepest recent-frame dropout in frames, uniform in 1..this (default 3)",
    )
    f.add_argument(
        "--stream-state",
        dest="stream_state",
        default=None,
        help="causal trajectory-state pseudo-frames: 'meanvar' prepends running mean/std "
        "frames over everything the Track-2 stream has revealed (runs/t2_backbone/PLAN.md)",
    )
    f.add_argument(
        "--state-dropout",
        dest="state_dropout",
        type=float,
        default=None,
        help="cold-state dropout (needs --stream-state): probability a training window's state "
        "is rebuilt from only its first k reveals (k uniform 1..state-dropout-max), presenting "
        "the noisy early-stream state and its small count plane; 0 off",
    )
    f.add_argument(
        "--state-dropout-max",
        dest="state_dropout_max",
        type=int,
        default=None,
        help="deepest cold state in 20-frame reveals, uniform 1..this when --state-dropout fires "
        "(default 4 => <=80 frames of history)",
    )
    f.add_argument(
        "--stream-corr-lam",
        dest="stream_corr_lam",
        type=float,
        default=None,
        help="correction-in-the-loop: loss on pred + stopgrad(lam * per-trajectory mean-flow "
        "residual EMA), the shipped Track-2 correction's frozen regime (0.667); 0 off",
    )
    f.add_argument(
        "--stream-corr-ema",
        dest="stream_corr_ema",
        type=float,
        default=None,
        help="EMA decay of the per-trajectory residual estimate stream_corr_lam applies",
    )
    f.add_argument(
        "--recur-k",
        dest="recur_k",
        type=int,
        default=None,
        help="TBPTT chain length over consecutive stream windows (train/recurrent.py); 0 off",
    )
    f.add_argument(
        "--recur-burn",
        dest="recur_burn",
        type=int,
        default=None,
        help="gradient-free burn-in windows before each chain's graded part (default 8)",
    )
    f.add_argument(
        "--train-compile",
        dest="train_compile",
        choices=["default", "reduce-overhead"],
        default=None,
        help="torch.compile the chain step's forwards (needs --recur-k); eager when omitted",
    )
    f.add_argument(
        "--state-planes",
        dest="state_planes",
        action="store_true",
        default=None,
        help="feed the state as extra channels on every frame (reaching the full-resolution "
        "refinement) instead of as two extra frames consumed by the stem",
    )
    f.add_argument(
        "--state-anchor",
        dest="state_anchor",
        action="store_true",
        default=None,
        help="blend the residual anchor per lead toward the trajectory mean, with a LEARNED "
        "weight initialised at today's behaviour (models.wrappers.StateAnchorWrapper)",
    )
    f.add_argument(
        "--state-norm",
        dest="state_norm",
        action="store_true",
        default=None,
        help="standardise the window by the stream_state statistics and restore the affine on "
        "the output, instead of feeding them to the stem (models/wrappers.StateNormWrapper)",
    )
    f.add_argument(
        "--state-norm-mode",
        dest="state_norm_mode",
        default=None,
        choices=["meanstd", "mean"],
        help="'meanstd' standardises by both moments (measured negative: -0.63 tke); 'mean' "
        "removes only the first moment, which is the mean-flow correction done exactly",
    )
    f.add_argument(
        "--state-norm-floor",
        dest="state_norm_floor",
        type=float,
        default=None,
        help="floor on the state std before dividing (freestream-normalised units)",
    )
    f.add_argument(
        "--state-lr-mult",
        dest="state_lr_mult",
        type=float,
        default=None,
        help="gradient multiplier on the stem columns that read the stream_state frames; they "
        "start at zero while the rest of the network is pretrained, so they are the "
        "undertrained part at any fine-tune length (1.0 = off)",
    )
    f.add_argument(
        "--dt-aug",
        dest="dt_aug",
        default=None,
        help="all-to-all temporal strides within a window, e.g. '1,1,2,3' (repeats weight)",
    )
    f.add_argument(
        "--dt-ratio-aug",
        dest="dt_ratio_aug",
        default=None,
        help="sim-only time-step augmentation 'lo,hi': frame interval as a multiple of the real "
        "step at the cell's Re, log-uniform, never below one source frame "
        "(TrainConfig.dt_ratio_aug)",
    )
    f.add_argument(
        "--dt-ratio-p1",
        dest="dt_ratio_p1",
        type=float,
        default=None,
        help="share of --dt-ratio-aug draws pinned at ratio 1 (default 0.5)",
    )
    f.add_argument("--cond-mode", dest="cond_mode", default=None, choices=list(COND_MODES))
    f.add_argument(
        "--cond-source",
        dest="cond_source",
        default=None,
        choices=["label", "estimate"],
        help="real windows' (Re, AoA) in the conditioning vector: the cell label (default) or "
        "the estimate from the raw window, as inference computes it; sim always uses labels",
    )
    f.add_argument(
        "--cond-planes",
        dest="cond_planes",
        action="store_true",
        default=None,
        help="feed the operating point to a vendored baseline as constant input planes, "
        "the only way CNO/FNO can be conditioned without editing vendor/",
    )
    f.add_argument("--re-tilt", dest="re_tilt", type=float, default=None)
    f.add_argument("--residual", action="store_true", default=None)
    f.add_argument(
        "--residual-mode",
        dest="residual_mode",
        choices=("last", "damped"),
        default=None,
        help="what --residual anchors on: the last frame (default) or the damped-persistence "
        "reference (TrainConfig.residual_mode)",
    )
    f.add_argument("--residual-tau", dest="residual_tau", type=float, default=None)
    f.add_argument("--bounds", action="store_true", default=None)
    f.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        default=None,
        help="train the backbone jointly with the bound head instead of freezing it. "
        "Frozen is the safe reference -- it leaves the other four subscores bit-identical, "
        "so any change there is attributable to joint training alone",
    )
    f.add_argument(
        "--w-jepa",
        dest="w_jepa",
        type=float,
        default=None,
        help="JEPA-style latent prediction as a CO-OBJECTIVE during supervised training: the live "
        "model sees the masked input, an EMA copy sees the clean one, and the term is the MSE "
        "between their deepest representations. Requires --aux-mask (see TrainConfig.w_jepa)",
    )
    f.add_argument("--jepa-ema", dest="jepa_ema", type=float, default=None)
    f.add_argument(
        "--dropout-sites",
        dest="dropout_sites",
        default=None,
        help="comma-separated module components dropout is injected after, e.g. 'mlp,proj'. "
        "Empty uses the per-architecture default, which on transformers is 'mlp' only -- half "
        "of the standard ViT placement (see TrainConfig.dropout_sites)",
    )
    f.add_argument(
        "--dropout-kind",
        dest="dropout_kind",
        choices=["element", "channel"],
        default=None,
        help="'element' (F.dropout) or 'channel' (F.dropout2d, whole feature maps of a conv "
        "output; the conv-net form). See TrainConfig.dropout_kind",
    )
    f.add_argument(
        "--drop-path",
        dest="drop_path",
        type=float,
        default=None,
        help="stochastic depth: drop whole residual branches per sample, rescaling survivors. "
        "A different prior from --dropout, not a stronger dose (see TrainConfig.drop_path)",
    )
    f.add_argument(
        "--log-domain-split",
        dest="log_domain_split",
        action="store_true",
        default=None,
        help="diagnostic: also log train-batch rel_l2 split by domain (rel_real/rel_sim) via the "
        "is_sim flag; detached, does not affect training (see TrainConfig.log_domain_split)",
    )
    f.add_argument(
        "--drop-path-hi",
        dest="drop_path_hi",
        type=float,
        default=None,
        help="operating-point-dosed drop-path: the per-sample rate runs linearly in log Re from "
        "--drop-path at Re 3750 to this value at Re 26700 (see TrainConfig.drop_path_hi). "
        "Unset = flat --drop-path dose",
    )
    f.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="inject dropout into the backbone via forward hooks, so MC dropout is a "
        "valid posterior approximation rather than test-time noise on a net that never "
        "saw dropout (see models.uq)",
    )
    f.add_argument(
        "--amplitude-free",
        dest="amplitude_free",
        action="store_true",
        default=None,
        help="SIM STAGE ONLY: grade the mean flow and the fluctuation shape, not its amplitude, "
        "so a longer pretrain cannot install sim's 1.8-20x over-fluctuation "
        "(see TrainConfig.amplitude_free)",
    )
    f.add_argument(
        "--bound-cond",
        dest="bound_cond",
        action="store_true",
        default=None,
        help="give the interval head the conditioning vector, so it can make the width "
        "Reynolds-dependent (default: on)",
    )
    f.add_argument(
        "--no-bound-cond",
        dest="bound_cond",
        action="store_false",
        default=None,
        help="reproduce the original scale-blind head",
    )
    f.add_argument(
        "--bound-width",
        dest="bound_width",
        type=int,
        default=None,
        help="channels in the interval head. 16 was tuned on FNO and has never been "
        "swept for another backbone",
    )
    f.add_argument("--bound-depth", dest="bound_depth", type=int, default=None)
    f.add_argument(
        "--bound-noise",
        dest="bound_noise",
        action="store_true",
        default=None,
        help="give the interval head PIV measurement-noise estimates (models.bounds)",
    )
    f.add_argument(
        "--bound-mode",
        dest="bound_mode",
        default=None,
        choices=["offsets", "scale", "logerr"],
        help="'scale': the head predicts an error scale and the width follows the Gaussian "
        "closed-form optimum. 'logerr': the head predicts a scale and the width is left to a "
        "fitted WidthRule -- the FINDINGS 14 configuration, and only worth anything with it",
    )
    f.add_argument(
        "--bound-logs",
        dest="bound_logs",
        action="store_true",
        default=None,
        help="log-magnitude feature channels for the interval head; the one feature change "
        "that measured (-0.36 SPS without it)",
    )
    f.add_argument(
        "--bound-coords",
        dest="bound_coords",
        action="store_true",
        default=None,
        help="normalised row/column planes for the interval head; measured no effect",
    )
    f.add_argument(
        "--head-lr-mult",
        dest="head_lr_mult",
        type=float,
        default=None,
        help="bound-head LR as a multiple of --lr; only meaningful with --no-freeze-backbone",
    )
    f.add_argument(
        "--piv-noise",
        dest="piv_noise",
        type=float,
        default=None,
        help="synthetic PIV noise on the input, as a multiple of the measured level (1.0 = real)",
    )
    f.add_argument(
        "--piv-noise-target",
        dest="piv_noise_target",
        type=float,
        default=None,
        help="the same on the target; equal to --piv-noise is the test-time task, 0 is denoising",
    )
    f.add_argument(
        "--aux-mask",
        dest="aux_mask",
        default=None,
        choices=[*MASK_MODES, ""],
        help="input mask geometry for masked spatio-temporal modelling",
    )
    f.add_argument("--aux-mask-ratio", dest="aux_mask_ratio", type=float, default=None)
    f.add_argument(
        "--no-aux-mask-plane",
        dest="aux_mask_plane",
        action="store_false",
        default=None,
        help="zero-mask only the auxiliary view; keep the production model at its original width",
    )
    f.add_argument("--w-recon", dest="w_recon", type=float, default=None)
    f.add_argument("--w-recon-tke", dest="w_recon_tke", type=float, default=None)
    f.add_argument(
        "--recon-tke-scope",
        dest="recon_tke_scope",
        choices=["hidden", "full"],
        default=None,
        help="form the amplitude target from hidden frames or the full clean input window",
    )
    f.add_argument(
        "--recon-tke-target",
        dest="recon_tke_target",
        choices=["amplitude", "energy"],
        default=None,
        help="directly decode RMS amplitude or the scored temporal energy statistic",
    )
    f.add_argument(
        "--recon-tke-spatial",
        dest="recon_tke_spatial",
        choices=["map", "global"],
        default=None,
        help="supervise the dense moment map or only its valid spatial mean",
    )
    f.add_argument("--w-recon-acf", dest="w_recon_acf", type=float, default=None)
    f.add_argument(
        "--recon-clean-field",
        dest="recon_clean_field",
        action="store_true",
        default=None,
        help="forecast from the clean view and compute reconstruction on a second masked view",
    )
    f.add_argument(
        "--recon-stop-frac",
        dest="recon_stop_frac",
        type=float,
        default=None,
        help="fraction of training that retains the masked auxiliary forward",
    )
    f.add_argument("--w-fine", dest="w_fine", type=float, default=None)
    f.add_argument("--w-subgrid", dest="w_subgrid", type=float, default=None)
    f.add_argument("--w-paux", dest="w_paux", type=float, default=None)
    f.add_argument(
        "--w-opreg",
        dest="w_opreg",
        type=float,
        default=None,
        help="regress (log-Re, AoA) off the pooled trunk embedding; see TrainConfig.w_opreg",
    )
    f.add_argument(
        "--w-field",
        dest="w_field",
        type=float,
        default=None,
        help="weight on the field term (default 1.0). Two uses. It turns `--huber-delta` from "
        "smooth-L1 into true Huber: huber(d) = d * smooth_l1(beta=d), so `--w-field d/0.02` "
        "fixes the body gradient across a delta sweep and leaves only the tail cap moving. "
        "And `--w-field 0` is what lets `--w-ceiling` / `--w-relfield` REPLACE this term "
        "rather than add to it -- all three grade the same residual",
    )
    f.add_argument("--w-tke", dest="w_tke", type=float, default=None)
    f.add_argument("--w-spectral", dest="w_spectral", type=float, default=None)
    f.add_argument(
        "--w-tke-shape",
        dest="w_tke_shape",
        type=float,
        default=None,
        help="scale-invariant TKE term; MEASURED TO FAIL, see FINDINGS 27",
    )
    f.add_argument(
        "--w-tke-tprofile",
        dest="w_tke_tprofile",
        type=float,
        default=None,
        help="per-lead temporal energy-profile term (FINDINGS 143/140 lead-16 dump)",
    )
    f.add_argument(
        "--w-tke-huber",
        dest="w_tke_huber",
        type=float,
        default=None,
        help="robust smooth-L1 TKE term; an alternative to --w-tke, better on val_extrap",
    )
    f.add_argument(
        "--tke-head-e2e",
        dest="tke_head_e2e",
        action="store_true",
        default=None,
        help="train the w8/d2 second-moment head end to end with the backbone: the stage "
        "objective is evaluated on the head-corrected output (serve with --tke-shrink 1.0)",
    )
    f.add_argument(
        "--kmap-channel",
        dest="kmap_channel",
        action="store_true",
        default=None,
        help="emit a fourth output channel carrying the temporal-variance map (models/kmap.py)",
    )
    f.add_argument(
        "--kmap-e2e",
        dest="kmap_e2e",
        action="store_true",
        default=None,
        help="apply the map channel's mean-preserving gain to the frames inside the step, so "
        "the objective grades the corrected output (needs --kmap-channel; see kmap_e2e)",
    )
    f.add_argument(
        "--kmap-base",
        dest="kmap_base",
        choices=("input", "frames"),
        default=None,
        help="anchor of the map channel's log ratio: the input window's map or the predicted "
        "frames' own map (detached)",
    )
    f.add_argument(
        "--w-kmap",
        dest="w_kmap",
        type=float,
        default=None,
        help="weight on the map channel's relative-L2 term (raw ~0.76 at step 0; matched ~0.003)",
    )
    f.add_argument(
        "--tke-huber-delta",
        dest="tke_huber_delta",
        type=float,
        default=None,
        help="smooth-L1 crossover for --w-tke-huber, in units of the k-map RMS (default 1.0)",
    )
    f.add_argument(
        "--tke-huber-raw",
        dest="tke_huber_normalize",
        action="store_false",
        default=None,
        help="drop --w-tke-huber's per-window RMS normalisation; --tke-huber-delta becomes an "
        "absolute crossover in freestream units (median target-map RMS ~0.005)",
    )
    f.add_argument(
        "--tke-gate",
        dest="tke_gate",
        choices=("none", "huber", "both", "pred"),
        default=None,
        help="predictability gate on the dense second-moment terms "
        "(losses.unpredictability_gate): 'huber' gates --w-tke-huber, 'both' also --w-tke-shape; "
        "'pred' is the inverted gate (press where tracked) on the huber term (FINDINGS 162)",
    )
    f.add_argument(
        "--tke-lead-dropout",
        dest="tke_lead_dropout",
        type=float,
        default=None,
        help="training-only fraction of output leads randomly omitted from TKE-family maps",
    )
    f.add_argument(
        "--lead-decay",
        dest="lead_decay",
        type=float,
        default=None,
        help="tilt the accuracy term toward short leads, weights exp(-d*t/T) at mean 1 "
        "(FINDINGS 42: fluctuation correlation falls 0.85 -> 0.32 over the horizon)",
    )
    f.add_argument(
        "--llrd",
        dest="llrd",
        type=float,
        default=None,
        help="layer-wise lr decay factor per depth from the head backwards (1.0 = off); needs a "
        "backbone with lr_depth(), see TrainConfig.llrd",
    )
    f.add_argument(
        "--freeze-patterns",
        dest="freeze_patterns",
        default=None,
        help="comma-separated parameter-name substrings to freeze for the run, e.g. "
        "'stem,encoder,down' (see TrainConfig.freeze_patterns)",
    )
    f.add_argument(
        "--moment-of",
        dest="moment_of",
        default=None,
        help="train a moment network: frozen mean-optimal checkpoint whose squared residuals "
        "are the target; the model outputs log sigma^2 (2 channels) under a Gaussian NLL "
        "(see TrainConfig.moment_of)",
    )
    f.add_argument(
        "--field-norm",
        dest="field_norm",
        choices=["none", "frame"],
        default=None,
        help="'frame': normalise each output frame's field error by its own detached RMS and "
        "restrict it to PIV-valid cells (see TrainConfig.field_norm); 'none' is the shipped loss",
    )
    f.add_argument(
        "--w-tke-amp",
        dest="w_tke_amp",
        type=float,
        default=None,
        help="weight on the fluctuation energy's TOTAL, graded alone through a log ratio "
        "(see TrainConfig.w_tke_amp and losses.tke_amplitude_loss); 0 = off",
    )
    f.add_argument(
        "--tke-amp-delta",
        dest="tke_amp_delta",
        type=float,
        default=None,
        help="tail cap of --w-tke-amp's smooth-L1, in log units (default 0.5 = a factor 1.65)",
    )
    f.add_argument(
        "--w-errhomog",
        dest="w_errhomog",
        type=float,
        default=None,
        help="weight on the local-error-homogeneity term: makes the field error locally "
        "homogeneous so it is predictable from its neighbourhood, raising SPS interval "
        "efficiency (see TrainConfig.w_errhomog and losses.error_homogeneity_loss); 0 = off",
    )
    f.add_argument(
        "--errhomog-k",
        dest="errhomog_k",
        type=int,
        default=None,
        help="spatial box size (odd) of --w-errhomog's neighbourhood (default 3)",
    )
    f.add_argument(
        "--fluct-aug",
        dest="fluct_aug",
        type=float,
        default=None,
        help="sigma of the log-normal fluctuation-amplitude augmentation of real training "
        "windows (input and target together); 0 = off (see TrainConfig.fluct_aug)",
    )
    f.add_argument(
        "--aoa-jitter",
        dest="aoa_jitter_deg",
        type=float,
        default=None,
        help="degrees of uniform AoA-conditioning jitter on real training windows for angle "
        "generalisation; ~2.5 (half the 5-deg spacing) is the coverage sweet spot, 0 = off "
        "(see TrainConfig.aoa_jitter_deg)",
    )
    f.add_argument(
        "--fluct-aug-hi",
        dest="fluct_aug_hi",
        type=float,
        default=None,
        help="--fluct-aug's sigma at Re 26700; the dose is then linear in log Re from --fluct-aug "
        "at Re 3750 (see TrainConfig.fluct_aug_hi). Unset = flat dose",
    )
    f.add_argument(
        "--tgt-amp-aug",
        dest="tgt_amp_aug",
        type=float,
        default=None,
        help="one-sided TARGET-only fluctuation boost at Re 26700: target fluctuation x exp(U(0, "
        "dose)), input untouched, linear in log Re from --tgt-amp-aug-lo (TrainConfig.tgt_amp_aug)",
    )
    f.add_argument(
        "--tgt-amp-aug-lo",
        dest="tgt_amp_aug_lo",
        type=float,
        default=None,
        help="--tgt-amp-aug's dose at Re 3750 (default 0)",
    )
    f.add_argument(
        "--tgt-fluct-aug",
        dest="tgt_fluct_aug",
        type=float,
        default=None,
        help="sigma of a symmetric log-normal on the TARGET frames' fluctuation only (input "
        "untouched): the target half of --fluct-aug, a mechanism probe (TrainConfig.tgt_fluct_aug)",
    )
    f.add_argument(
        "--tgt-fluct-aug-hi",
        dest="tgt_fluct_aug_hi",
        type=float,
        default=None,
        help="--tgt-fluct-aug's sigma at Re 26700, linear in log Re from --tgt-fluct-aug at 3750",
    )
    f.add_argument(
        "--mean-aug",
        dest="mean_aug",
        type=float,
        default=None,
        help="mean-flow perturbation of real windows: alpha ~ U(0, this) of the climatology "
        "difference to an adjacent-Reynolds training cell, same angle (TrainConfig.mean_aug)",
    )
    f.add_argument(
        "--cond-drop",
        dest="cond_drop",
        type=float,
        default=None,
        help="probability per sample of replacing the operating-point vector by another "
        "sample's during training; 0 = off (see TrainConfig.cond_drop)",
    )
    f.add_argument(
        "--moment-joint",
        dest="moment_joint",
        action="store_true",
        default=None,
        help="emit log sigma^2 for u, v as two extra output channels, trained with the Gaussian "
        "NLL against the detached residual (see TrainConfig.moment_joint)",
    )
    f.add_argument(
        "--w-moment",
        dest="w_moment",
        type=float,
        default=None,
        help="weight of the joint model's NLL term (TrainConfig.w_moment)",
    )
    f.add_argument(
        "--w-absorb",
        dest="w_absorb",
        type=float,
        default=None,
        help="weight on pulling the prediction toward a frozen second-moment head's proposal "
        "(train.absorb); needs --absorb-head",
    )
    f.add_argument("--absorb-head", dest="absorb_head", default=None)
    f.add_argument(
        "--distill-head",
        dest="distill_head",
        default=None,
        help="frozen second-moment head applied to the teachers' prediction before it becomes "
        "the distillation target (student learns teacher+head, serves without a head)",
    )
    f.add_argument("--distill-head-shrink", dest="distill_head_shrink", type=float, default=None)
    f.add_argument("--absorb-shrink", dest="absorb_shrink", type=float, default=None)
    f.add_argument(
        "--tke-loop-head",
        dest="tke_loop_head",
        default=None,
        help="frozen second-moment head applied (gain detached) to the prediction the dense tke "
        "terms grade; the field term still grades the bare prediction (FINDINGS 196k)",
    )
    f.add_argument("--tke-loop-head-shrink", dest="tke_loop_head_shrink", type=float, default=None)
    f.add_argument(
        "--tke-loop-head-field",
        dest="tke_loop_head_field",
        action="store_true",
        default=None,
        help="with --tke-loop-head: the field term grades the head-corrected prediction too; "
        "the band credit keeps the bare prediction",
    )
    f.add_argument(
        "--w-ceiling",
        dest="w_ceiling",
        type=float,
        default=None,
        help="differentiable SPS ceiling -- the scored composite itself, .5/.3/.2 over "
        "rel_l2/tke/mvpe with self-balancing marginals. Pair with --w-field 0",
    )
    f.add_argument(
        "--ceiling-huber",
        dest="ceiling_huber",
        type=float,
        default=None,
        help="tail-cap on --w-ceiling's rel_l2 branch. --w-field 0 drops --huber-delta with the "
        "field term (FINDINGS 55a); set this to the same value to keep the swap one-factor",
    )
    f.add_argument(
        "--w-meanfield",
        dest="w_meanfield",
        type=float,
        default=None,
        help="relative L2 on the window's 20-frame TIME MEAN. MEASURED AND CUT (FINDINGS 63, and "
        "the motivation is withdrawn in 63b-bis): it does not move rel_l2 at either dose and "
        "costs tke monotonically. A redistribution of pressure, so dose it at gradient parity "
        "(measured 0.507 on this architecture, where the obvious 0.05 is 10x under)",
    )
    f.add_argument(
        "--w-relfield",
        dest="w_relfield",
        type=float,
        default=None,
        help="per-sample RELATIVE field loss (TR.1), matching rel_l2_score exactly; an "
        "alternative to --w-field, not an addition",
    )
    f.add_argument(
        "--w-vorticity",
        dest="w_vorticity",
        type=float,
        default=None,
        help="relative L2 on the instantaneous vorticity field: a high-pass reweighting of "
        "the field term. Calibrate per backbone (FINDINGS 24a)",
    )
    f.add_argument(
        "--w-enstrophy",
        dest="w_enstrophy",
        type=float,
        default=None,
        help="relative L2 on var_t(vorticity): the TKE term one derivative up, aimed at the "
        "shear layers rather than the whole wake",
    )
    f.add_argument("--w-mvpe", dest="w_mvpe", type=float, default=None)
    f.add_argument(
        "--w-rstress",
        dest="w_rstress",
        type=float,
        default=None,
        help="Reynolds-stress SHAPE term: the anisotropy and shear the trace-only tke_score "
        "cannot see, so orthogonal to --w-tke rather than a heavier version of it",
    )
    f.add_argument(
        "--w-espec",
        dest="w_espec",
        type=float,
        default=None,
        help="SPATIAL radial energy-spectrum term E(k) -- not --w-spectral, which is temporal "
        "and is cut (FINDINGS 24)",
    )
    f.add_argument(
        "--helm-lambda",
        dest="helm_lambda",
        type=float,
        default=None,
        help="weight on the DIVERGENT part of the field error; 1.0 is exactly MSE. <1 trains on "
        "the incompressible part, >1 on the divergent one. Excludes --huber-delta",
    )
    f.add_argument("--w-sps", dest="w_sps", type=float, default=None)
    f.add_argument(
        "--w-credit",
        dest="w_credit",
        type=float,
        default=None,
        help="oracle-interval credit term: makes the field cheap to bracket (SPS's own factor)",
    )
    f.add_argument("--credit-k", dest="credit_k", type=float, default=None)
    f.add_argument(
        "--w-headcredit",
        dest="w_headcredit",
        type=float,
        default=None,
        help="band credit through a frozen interval head (width detached; FINDINGS 214)",
    )
    f.add_argument("--headcredit-arm", dest="headcredit_arm", default=None)
    f.add_argument("--headcredit-tau", dest="headcredit_tau", type=float, default=None)
    f.add_argument(
        "--headcredit-notch-lo",
        dest="headcredit_notch_lo",
        type=float,
        default=None,
        help="w_headcredit multiplier at --headcredit-notch-re, back to 1 at "
        "|log(Re/centre)| = --headcredit-notch-width (1 = off; FINDINGS 223)",
    )
    f.add_argument("--headcredit-notch-re", dest="headcredit_notch_re", type=float, default=None)
    f.add_argument(
        "--headcredit-notch-width", dest="headcredit_notch_width", type=float, default=None
    )
    f.add_argument(
        "--nbr-prior",
        dest="nbr_prior",
        action="store_true",
        default=None,
        help="append the nearest OTHER training cells' climatology (mean, std of u, v) as input "
        "planes, looked up from the estimated operating point (FINDINGS 224)",
    )
    f.add_argument(
        "--nbr-prior-mode",
        dest="nbr_prior_mode",
        choices=("all", "std"),
        default=None,
        help="nbr_prior planes: all = mean + std of u, v + distance; std = std + distance",
    )
    f.add_argument(
        "--rollout-w",
        dest="rollout_w",
        type=float,
        default=None,
        help="> 0: weight of the rollout-consistency loss -- the forecast fed back as the next "
        "input window, scored against the following real frames (TrainConfig.rollout_w)",
    )
    f.add_argument(
        "--rollout-mode",
        dest="rollout_mode",
        choices=("full", "tke"),
        default=None,
        help="rollout second-window loss: full, or tke terms only (TrainConfig.rollout_mode)",
    )
    f.add_argument(
        "--rollout-aoa-min",
        dest="rollout_aoa_min",
        type=float,
        default=None,
        help="> 0: the rollout term only on windows with label angle >= this (degrees); "
        "the rest get weight zero (TrainConfig.rollout_aoa_min)",
    )
    f.add_argument(
        "--rollout-re-min",
        dest="rollout_re_min",
        type=float,
        default=None,
        help="> 0: the rollout term only on windows with label Reynolds number >= this "
        "(TrainConfig.rollout_re_min)",
    )
    f.add_argument(
        "--sam-rho",
        dest="sam_rho",
        type=float,
        default=None,
        help="> 0: sharpness-aware minimisation radius (one extra forward/backward per step on "
        "the same batch; TrainConfig.sam_rho, FINDINGS 224). 0 = off",
    )
    f.add_argument(
        "--rdrop-alpha",
        dest="rdrop_alpha",
        type=float,
        default=None,
        help="> 0: R-drop consistency weight (two forward passes/step, MSE between the two u,v "
        "predictions; TrainConfig.rdrop_alpha). 0 = off",
    )
    f.add_argument(
        "--headcredit-track-lr",
        dest="headcredit_track_lr",
        type=float,
        default=None,
        help="> 0: the w_headcredit head keeps fitting its own objective on the current "
        "backbone's errors during the fine-tune, at this lr (0 = frozen; FINDINGS 223)",
    )
    f.add_argument(
        "--headcredit-lo",
        dest="headcredit_lo",
        type=float,
        default=None,
        help="w_headcredit multiplier at Re 3750, linear in log Re to 1 at 26700 (1 = flat)",
    )
    f.add_argument("--huber-delta", dest="huber_delta", type=float, default=None)
    f.add_argument(
        "--huber-mix",
        dest="huber_mix",
        type=float,
        default=None,
        help="blend MSE and smooth-L1: 1 = pure smooth-L1 (default), 0 = pure MSE. A mechanism "
        "probe -- a blend keeps MSE's unbounded tail, which no huber_delta does",
    )
    f.add_argument(
        "--mean-fluct-lambda",
        type=float,
        default=None,
        help="replace field loss with robust mean error plus weighted fluctuation MSE; "
        "0 disables, 100 matches small-error curvature at huber-delta .005",
    )
    f.add_argument(
        "--mean-fluct-robust",
        dest="mean_fluct_robust",
        action="store_true",
        default=None,
        help="with --mean-fluct-lambda: grade the fluctuation half with the same smooth-L1 as the "
        "mean half instead of MSE, so lambda is a pure price (1 = shipped term applied per half)",
    )
    f.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    f.add_argument(
        "--warmup-steps",
        dest="warmup_steps",
        type=int,
        default=None,
        help="linear-warmup length before the cosine decay (default 200). At 200 steps a "
        "4,000-step stage spends 5%% of its budget there, so it is part of the schedule "
        "rather than a constant when short stages are being compared",
    )
    f.add_argument("--init-from", dest="init_from", default=None)
    f.add_argument(
        "--distill-from",
        dest="distill_from",
        default=None,
        help="comma-separated teacher checkpoints for ensemble distillation",
    )
    f.add_argument(
        "--distill-alpha",
        dest="distill_alpha",
        type=float,
        default=None,
        help="blend: target = a*teacher + (1-a)*truth (default 1.0)",
    )
    f.add_argument(
        "--distill-part",
        dest="distill_part",
        choices=("mean", "fluct"),
        default=None,
        help="distil only one half of the teacher's Reynolds decomposition: 'mean' (teacher's "
        "temporal mean + truth's fluctuation) or 'fluct' (truth's mean + teacher's fluctuation)",
    )
    f.add_argument(
        "--distill-tke-truth",
        dest="distill_tke_truth",
        action="store_true",
        default=None,
        help="anchor tke-family terms to ground truth (not the teacher) during distillation",
    )
    f.add_argument(
        "--unfreeze",
        default=None,
        help="FT.2 ladder: cond | io | blocks:N | all (default all). Needs --init-from",
    )
    f.add_argument(
        "--ema-decay",
        dest="ema_decay",
        type=float,
        default=None,
        help="keep an exponential moving average of the weights and write it as ema.pt "
        "alongside last.pt, so one run yields both arms of the comparison (see "
        "TrainConfig.ema_decay)",
    )
    f.add_argument(
        "--joint-sim-frac",
        dest="joint_sim_frac",
        type=float,
        default=None,
        help="fraction of each fine-tuning batch drawn from sim_train instead of real, "
        "keeping the pretraining distribution in the gradient as a regulariser (FT.1)",
    )
    f.add_argument(
        "--synth-path",
        dest="synth_path",
        default=None,
        help="synthetic block set (from `train gen-sample`) mixed into the training batches",
    )
    f.add_argument(
        "--synth-frac",
        dest="synth_frac",
        type=float,
        default=None,
        help="fraction of each batch drawn from --synth-path instead of real",
    )
    f.add_argument(
        "--joint-sim-frac-end",
        dest="joint_sim_frac_end",
        type=float,
        default=None,
        help="anneal the sim fraction linearly to this value over training; negative "
        "disables (FINDINGS 32-bis: keep the tke gain, drop the imported mean-flow offset)",
    )
    f.add_argument(
        "--align-weight",
        dest="align_weight",
        type=float,
        default=None,
        help="weight on the sim/real latent-alignment term: match the per-channel mean and "
        "spread of the encoder's deepest features across domains. Needs --joint-sim-frac",
    )
    f.add_argument(
        "--coral-weight",
        dest="coral_weight",
        type=float,
        default=None,
        help="weight on cross-channel sim/real correlation alignment. Needs --joint-sim-frac",
    )
    f.add_argument(
        "--l2sp",
        type=float,
        default=None,
        help="weight on ||w - w_init||^2, pulling the weights back toward the pretrained "
        "point instead of re-showing the pretraining data. Needs --init-from",
    )
    f.add_argument(
        "--trunk-lr-scale",
        dest="trunk_lr_scale",
        type=float,
        default=None,
        help="LR multiplier on the trunk -- everything --unfreeze io would freeze. The graded "
        "form of that rung: 0 trains the same parameters, values in between are the continuum "
        "the ladder cannot express. Distinct from backbone_lr_scale, which scopes all but the "
        "0.092M head and is a near-uniform cut",
    )
    f.add_argument(
        "--geom-channels",
        dest="geom_channels",
        default=None,
        help="comma-separated geometry planes appended to the input, e.g. "
        "'sdf,inlet,aoa_fourier'. An unseen angle costs ~-4 final and AoA currently "
        "reaches a baseline model not at all (BaselineAdapter discards cond). A "
        "3-channel pretrain still loads: fc0 is widened with zero columns",
    )
    f.add_argument(
        "--flow-channels",
        dest="flow_channels",
        default=None,
        help="comma-separated planes derived from the observed window: 'vort' (linear, so "
        "the stem can already build it), 'enstrophy' and 'tke' (quadratic, so it cannot). "
        "An STFormer pretrain still loads: the stem is widened with zero columns",
    )
    f.add_argument(
        "--phase-aug",
        dest="phase_aug",
        action="store_true",
        default=None,
        help="also train on the three unscored [::2,::2] phases of the recorded 64x128 "
        "grid: 4x the windows, same task, free at inference (needs `make phase-cache`)",
    )
    f.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    f.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        default=None,
        help="let the CUDA kernels pick nondeterministic algorithms, for ~3-5%% training "
        "wall clock. The run is then NOT reproducible from its seed -- two stformer runs "
        "at one seed differ on 130 of 144 tensors (see TrainConfig.deterministic)",
    )
    f.add_argument(
        "--snapshot-every",
        dest="snapshot_every",
        type=int,
        default=None,
        help="write snap_<step>.pt every N steps as ingredients for a trajectory soup "
        "(`train soup`); ~24 MB each",
    )
    f.add_argument("--eval-every", dest="eval_every", type=int, default=None)
    f.add_argument(
        "--score-points",
        dest="score_points",
        type=int,
        default=None,
        help="run the official scorer at N evenly-spaced checkpoints over the run (0 off); "
        "e.g. 20 scores every 5%% of the budget. Logs rel_l2/tke/mvpe/time/sps subscores "
        "and writes scores.json. Expensive; needs `make eval-set`",
    )
    f.add_argument(
        "--val-splits",
        dest="val_splits",
        nargs="+",
        default=None,
        help="splits for the in-loop diagnostic (default: the three real ones). A "
        "sim-trained run must name the sim splits, or it validates on the wrong grid.",
    )
    f.add_argument(
        "--wandb",
        action="store_true",
        default=None,
        help="mirror the loop's logging to Weights & Biases. Needs the extra "
        "(`uv sync --extra wandb`) and WANDB_API_KEY; a missing package or a failed "
        "handshake prints a note and trains anyway. history.json is written either way",
    )
    f.add_argument("--wandb-project", dest="wandb_project", default=None)
    f.add_argument(
        "--wandb-entity", dest="wandb_entity", default=None, help="W&B team (default: account)"
    )
    f.add_argument(
        "--wandb-name",
        dest="wandb_name",
        default=None,
        help="W&B run name (default: the --out-dir basename, so the run and its "
        "directory share a name)",
    )
    f.add_argument(
        "--wandb-group",
        dest="wandb_group",
        default=None,
        help="W&B group; `seeds` fills it with the sweep prefix so an arms x seeds "
        "sweep lands under one heading",
    )
    f.add_argument("--wandb-tags", dest="wandb_tags", default=None, help="comma-separated tags")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rpde_t1.train", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="train a model")
    _add_train_args(f)
    f.add_argument("--out-dir", dest="out_dir", default=None)
    f.add_argument("--seed", type=int, default=None)
    f.set_defaults(func=_cmd_fit)

    q = sub.add_parser(
        "seeds",
        help="run one config under several seeds and report the decision threshold",
        description=(
            "Measure training-seed noise, which every A/B is judged against. The only "
            "figure we have (sd 0.03-0.28 SPS) is for a FROZEN bound head and licenses "
            "nothing that trains a backbone. Resumable: rerun to continue an interrupted "
            "sweep. Use --dry-run to inspect the plan before spending GPU time."
        ),
    )
    _add_train_args(q)
    q.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds to run (default 0 1 2)"
    )
    q.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME=FIELD=VALUE[,FIELD=VALUE]",
        help="add an A/B arm, e.g. --arm base= --arm tke=w_tke=0.5 . Both arms get every "
        "seed, so an effect and its threshold come from the same sweep. Omitted, a single "
        "arm named 'base' is used.",
    )
    q.add_argument("--name", default=None, help="prefix for run directories (default: model)")
    q.add_argument("--splits", nargs="+", default=["val_extrap", "val_time"])
    q.add_argument("--force", action="store_true", help="retrain seeds that already exist")
    q.add_argument("--dry-run", dest="dry_run", action="store_true", help="print the plan, stop")
    q.add_argument("--out", default=None, help="write table + noise as JSON")
    q.set_defaults(func=_cmd_seeds)

    s = sub.add_parser("score", help="score checkpoints with the OFFICIAL scorer")
    s.add_argument("checkpoints", nargs="+")
    s.add_argument("--splits", nargs="+", default=["val_time", "val_interp", "val_extrap"])
    s.add_argument("--bounds", action="store_true")
    s.add_argument(
        "--stats-domain",
        dest="stats_domain",
        default=None,
        choices=["sim", "real"],
        help="gaussian only: whose normalisation constants to apply, independently of "
        "which domain the conditioning is built for. 'sim' is the strict zero-shot "
        "measurement (nothing from the target dataset); 'real' is deployment-realistic.",
    )
    s.add_argument("--bound-floor", dest="bound_floor", type=float, default=0.0)
    s.add_argument(
        "--tke-blend",
        dest="tke_blend",
        type=float,
        default=0.0,
        help="post-hoc fluctuation-rescaling exponent (0 = off; see `tke-blend`)",
    )
    s.add_argument(
        "--tke-head",
        dest="tke_head",
        default=None,
        help="second-moment head checkpoint from `tke-head`. Scored through the same "
        "make_predict closure a submission uses, so this measures the artefact rather "
        "than the analysis path that fitted it",
    )
    s.add_argument(
        "--tke-shrink",
        dest="tke_shrink",
        type=float,
        default=None,
        help="shrinkage of the head toward the blend (default: the measured 0.75). "
        "1.0 is the unshrunk head, which loses to the free blend on Reynolds extrapolation",
    )
    s.add_argument(
        "--width-rule",
        dest="width_rule",
        default=None,
        help="fitted width_rule.json (with --bounds): size the interval as a packaged bundle "
        "serves it. Without it --bounds scores the head's raw scale (sps_apply's head_only rung)",
    )
    s.add_argument("--out", default=None, help="write the table as JSON")
    s.set_defaults(func=_cmd_score)

    c = sub.add_parser("calibrate", help="fit the interval width multiplier")
    c.add_argument("checkpoint")
    c.add_argument("--fit-split", dest="fit_split", default="val_interp")
    c.add_argument(
        "--report-split",
        dest="report_split",
        default="val_extrap",
        help="a split the multiplier was NOT fitted on",
    )
    c.add_argument("--out", default=None)
    c.set_defaults(func=_cmd_calibrate)

    b = sub.add_parser(
        "tke-blend",
        help="tune the post-hoc fluctuation-rescaling exponent",
        description=(
            "An L2-trained model is unbiased in the mean and retains only 0.36-0.44 of "
            "the target's fluctuation energy, while the input window's total is "
            "unbiased. Rescaling each pixel's fluctuations toward a blend of the two "
            "variance maps recovers part of the tke axis with no training and leaves "
            "mvpe invariant by construction. Fitted on one split, reported on others."
        ),
    )
    b.add_argument("checkpoint")
    b.add_argument("--fit-split", dest="fit_split", default="val_time")
    b.add_argument(
        "--report-splits",
        dest="report_splits",
        nargs="+",
        default=["val_interp", "val_extrap"],
        help="splits the exponent was NOT fitted on",
    )
    b.add_argument("--out", default=None)
    b.set_defaults(func=_cmd_tke_blend)

    t = sub.add_parser(
        "tke-head",
        help="fit the second-moment (variance-map) head on a frozen backbone",
        description=(
            "tke_score grades the temporal variance field, which an L2-trained backbone "
            "under-predicts by ~3x and mis-localises (spatial corr 0.758). This fits a "
            "small CNN to the per-pixel log-ratio between the target and predicted "
            "energy maps. Cheap because the objective touches the backbone only through "
            "ke(pred): the backbone runs once, the head trains on cached 32x64 maps. "
            "Reported against the closed-form blend it generalises, and against the "
            "oracle map, on splits it was not fitted on."
        ),
    )
    t.add_argument("checkpoint")
    t.add_argument("--fit-split", dest="fit_split", default="train")
    t.add_argument(
        "--cond-mode",
        dest="cond_mode",
        default="operating_point",
        choices=list(COND_MODES),
        help="conditioning vector the cached maps carry. operating_point_tau adds the "
        "window decorrelation time, which is the horizon feature ROADMAP N40 asks for.",
    )
    t.add_argument(
        "--fit-kind",
        dest="fit_kind",
        default="real",
        choices=["real", "sim", "sim_resampled", "sim_resampled2", "sim_units"],
        help="domain of the fit split only; report splits are always real. Sim can be a "
        "target for *statistics* even though it provably cannot be one for pointwise "
        "fields (fluctuation correlation -0.05), and this head predicts a statistic",
    )
    t.add_argument(
        "--report-splits",
        dest="report_splits",
        nargs="+",
        default=["val_time", "val_interp", "val_extrap"],
    )
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--width", type=int, default=32)
    t.add_argument("--depth", type=int, default=3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument(
        "--stride",
        type=int,
        default=40,
        help="window stride; >= T_in+T_out or one window's target sits in another's input",
    )
    t.add_argument(
        "--max-windows",
        dest="max_windows",
        type=int,
        default=3000,
        help="cap on windows per split, since the maps are held in memory",
    )
    t.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    t.add_argument(
        "--select-split",
        dest="select_split",
        default="val_time",
        help="split used only to choose the stopping step; must be in --report-splits. "
        "The first version of this head overfitted 1322 windows over 73 epochs and lost "
        "to the closed-form blend on Reynolds extrapolation, so the stopping point is "
        "chosen rather than assumed",
    )
    t.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME=FEATURE[,FEATURE]",
        help="add an ablation arm that zeroes the named feature channels, e.g. "
        "--arm no_cond=log_re,aoa . All arms share one map collection. An arm named "
        "'full' (no ablation) is always included",
    )
    t.add_argument("--out", default=None, help="write the report JSON and the head .pt")
    from .tke_fit import OBJECTIVE_MODES  # noqa: PLC0415  (parser build time only)

    t.add_argument(
        "--objective",
        dest="objective_mode",
        default="tke",
        choices=list(OBJECTIVE_MODES),
        help="what the head optimises. 'tke' is the scorer's tke error alone, which lets "
        "the head trade rel_l2 away for free (measured -0.27 rel_l2 for +2.67 tke, on axes "
        "weighted 0.5 and 0.3 in the ceiling, so ~1/5 of the apparent gain survived). "
        "'ceiling' weights rel_l2/tke by the scorer's published SPS branch weights "
        "(0.5/0.3, ratio 1.67). 'board' weights them by the FITTED final-score marginals "
        "from score.report, direct plus the part arriving through the SPS ceiling "
        "(ratio ~3.42), i.e. how the leaderboard has actually paid.",
    )
    t.add_argument(
        "--per-component",
        dest="per_component",
        action="store_true",
        help="fit one gain per velocity component (two outputs, four extra input planes) "
        "instead of one shared gain. The scorer's tke sees only the two components' summed "
        "energy, so this head may put the energy into the component that agrees with the "
        "truth; pair it with --objective ceiling or board, since on tke alone it can only "
        "match each component's own variance, which is worth nothing over a shared gain.",
    )
    t.set_defaults(func=_cmd_tke_head)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
