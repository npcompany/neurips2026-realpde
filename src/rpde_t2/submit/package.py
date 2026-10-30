"""Build and verify a Track-2 submission zip.

The archive layout and the 256 MB extracted cap are Track 1's, so staging and zipping are
:mod:`rpde_t1.submit.package`'s job and are reused verbatim. What differs is three facts
about the Track-2 platform, and each is a constant here rather than a fork of that module:

* the entry point is ``get_ttt_model`` returning a streaming object, not ``predict``, so
  the verifier drives two steps through it instead of one batched call;
* the Track-2 image ships every library the Track-1 image lacks, so nothing is blocked and
  nothing has to be vendored;
* ``_vendor/einops`` is therefore droppable, which is the only size saving on offer.

It is kept in the payload anyway. 86 MB of a 256 MB budget is not worth spending a
submission to find out about, and the Track-1 bundle has already proven this exact payload
imports cleanly on this exact torch build.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rpde_t1.paths import RUNS_DIR, SUBMISSIONS_DIR, VENDOR_KIT, require
from rpde_t1.submit.package import (
    PACKAGE_PAYLOAD,
    Bundle,
    run_in_extracted,
    stage_bundle,
    zip_bundle,
)

from .. import EXPECTED_SHAPE, IN_STEP

#: Nothing. The Track-2 image ships torch/numpy plus scipy, h5py, einops, scikit-learn,
#: pandas, matplotlib, safetensors, tensorboard, openai, pillow, sympy, networkx, pyyaml,
#: requests and tqdm. Blocking the Track-1 list here would reject a valid bundle, which is
#: the whole reason this is a separate constant and not a shared one.
MISSING_IN_EVAL_IMAGE_T2: tuple[str, ...] = ()

#: Copied out of the Track-1 kit, not the Track-2 one: these are the *baseline model* files,
#: identical between the kits apart from the Track-2 copy importing scipy eagerly. The
#: Track-1 copy's lazy import is strictly more portable, so it is the one to ship.
KIT_PAYLOAD_T2 = ("load_baseline.py", "rpde_baselines", "_vendor")

#: The subset of *this* package a bundle needs. ``stream.py`` is excluded because it reads
#: h5 trajectories, and ``submit/`` because a bundle does not build bundles -- same rule
#: that keeps ``rpde_t1.data`` out of :data:`rpde_t1.submit.package.PACKAGE_PAYLOAD`.
T2_PACKAGE_PAYLOAD = ("__init__.py", "ttt.py")

TEMPLATE = Path(__file__).with_name("ttt_submission.py.tmpl")


def render_ttt_submission(
    run_name: str,
    ckpt_name: str = "last.pt",
    return_bounds: bool = True,
    bound_floor: float = 0.0,
    bound_scale: float = 1.0,
    width_rule_name: str = "",
    lead_anchor_name: str = "",
    tke_head_name: str = "",
    tke_blend: float = 0.0,
    tke_shrink: float = 0.0,
    precision: str = "fp32",
    geom_channels: str = "",
    gain_strength: float = 0.0,
    bias_strength: float = 0.0,
    warmup_n0: float = 2.0,
    compile_mode: str = "",
    gpu_native: bool = False,
    tf32: bool = False,
    bias_mode: str = "meanflow",
    mean_bias_strength: float = 0.0,
    bias_freeze_n: int = 8,
    mean_bias_smooth: float = 0.0,
    history_fill: str = "zeros",
    state_freeze_n: int = 0,
    causal_v2: bool = False,
    causal_gain_floor: float = 0.0,
    interval_rho_blend: float = 0.0,
    interval_rho_ramp_n0: float = 0.0,
    interval_rho_shape: float = 1.0,
    interval_autocal_n0s: float = 0.0,
    interval_autocal_n0p: float = 50.0,
    interval_autocal_stage1: str = "scalar",
    interval_autocal_shift: float = 0.0,
    interval_autocal_grid: int = 29,
    interval_autocal_lead_sub: int = 1,
    interval_autocal_freeze_n: int = 0,
    pinned_io: bool = False,
    interval_recal_name: str = "",
    interval_recal_fused: bool = False,
    interval_recal_merged: bool = False,
    interval_recal_fork: bool = False,
    causal_gain_strength: float = 0.0,
    interval_history_spatial: float = 0.0,
    interval_history_global: float = 0.0,
    causal_anom: float = 0.0,
    port_warmup: int = 0,
    step_log: bool = False,
) -> str:
    """Fill the Track-2 submission template.

    Deliberately no ``batch`` or ``pad_batches`` knob. Track 2 calls ``ttt_step`` once per
    window at batch size 1, so the batch is 1 and padding is off -- and inheriting the
    Track-1 bundle's 210 / True would multiply every timed step's compute by 210.

    ``precision`` defaults to fp32, not fp16. At batch 1 autocast adds cast kernels with no
    tensor-core payoff at this size, so fp16 measured 1.1 ms *slower* -- the opposite of the
    batch-210 regime the Track-1 bundles were tuned in -- and fp32 is the reference anyway.

    ``compile_mode`` is a Track-2 win, unlike on Track 1. Track 1's batch-210 bundles regressed
    under compile (variable tail shape -> recompile in the timed region); Track 2's batch-1 single
    fixed shape captures a CUDA graph once in the untimed warm-up. Measured ``reduce-overhead``:
    step 11.2 -> 7.3 ms, time_score 88.97 -> 90.91, all subscores bit-identical. Default "" is off.
    """
    return TEMPLATE.read_text(encoding="utf-8").format(
        RUN_NAME=run_name,
        CKPT_NAME=ckpt_name,
        RETURN_BOUNDS="True" if return_bounds else "False",
        BOUND_FLOOR=repr(float(bound_floor)),
        BOUND_SCALE=repr(float(bound_scale)),
        WIDTH_RULE_NAME=width_rule_name,
        LEAD_ANCHOR_NAME=lead_anchor_name,
        TKE_HEAD_NAME=tke_head_name,
        TKE_BLEND=repr(float(tke_blend)),
        TKE_SHRINK=repr(float(tke_shrink)),
        PRECISION=precision,
        GEOM_CHANNELS=geom_channels,
        GAIN_STRENGTH=repr(float(gain_strength)),
        BIAS_STRENGTH=repr(float(bias_strength)),
        WARMUP_N0=repr(float(warmup_n0)),
        COMPILE_MODE=compile_mode,
        GPU_NATIVE=repr(bool(gpu_native)),
        TF32=repr(bool(tf32)),
        BIAS_MODE=bias_mode,
        MEAN_BIAS_STRENGTH=repr(float(mean_bias_strength)),
        BIAS_FREEZE_N=repr(int(bias_freeze_n)),
        MEAN_BIAS_SMOOTH=repr(float(mean_bias_smooth)),
        HISTORY_FILL=history_fill,
        STATE_FREEZE_N=state_freeze_n,
        CAUSAL_V2=causal_v2,
        CAUSAL_GAIN_FLOOR=causal_gain_floor,
        INTERVAL_RHO_BLEND=interval_rho_blend,
        INTERVAL_RHO_RAMP_N0=interval_rho_ramp_n0,
        INTERVAL_RHO_SHAPE=interval_rho_shape,
        INTERVAL_AUTOCAL_N0S=repr(float(interval_autocal_n0s)),
        INTERVAL_AUTOCAL_N0P=repr(float(interval_autocal_n0p)),
        INTERVAL_AUTOCAL_STAGE1=interval_autocal_stage1,
        INTERVAL_AUTOCAL_SHIFT=repr(float(interval_autocal_shift)),
        INTERVAL_AUTOCAL_GRID=int(interval_autocal_grid),
        INTERVAL_AUTOCAL_LEAD_SUB=int(interval_autocal_lead_sub),
        INTERVAL_AUTOCAL_FREEZE_N=int(interval_autocal_freeze_n),
        PINNED_IO=bool(pinned_io),
        INTERVAL_RECAL_NAME=interval_recal_name,
        INTERVAL_RECAL_FUSED=bool(interval_recal_fused),
        INTERVAL_RECAL_MERGED=bool(interval_recal_merged),
        INTERVAL_RECAL_FORK=bool(interval_recal_fork),
        CAUSAL_GAIN_STRENGTH=repr(float(causal_gain_strength)),
        INTERVAL_HISTORY_SPATIAL=repr(float(interval_history_spatial)),
        INTERVAL_HISTORY_GLOBAL=repr(float(interval_history_global)),
        CAUSAL_ANOM=repr(float(causal_anom)),
        PORT_WARMUP=int(port_warmup),
        STEP_LOG=repr(bool(step_log)),
    )


def build_ttt_bundle(
    run: str | Path,
    name: str | None = None,
    out_root: Path | None = None,
    tke_head: str | Path | None = None,
    interval_recal: str | Path | None = None,
    **render_kwargs: object,
) -> Bundle:
    """Stage and zip a Track-2 bundle around one of our own training runs.

    ``run`` is a run directory (``runs/<name>/``) or an explicit ``last.pt``. Artefacts the
    run wrote beside the weights -- ``width_rule.json``, ``lead_anchor.json``, a TKE head --
    are picked up automatically and wired into the template, because a bundle that ships a
    width rule the submission does not reference is the same bug as one that references a
    rule it does not ship.
    """
    run_dir = Path(run)
    ckpt = run_dir if run_dir.is_file() else run_dir / "last.pt"
    require(ckpt, f"No checkpoint at {ckpt}. Train it, or point --run at the right place.")
    run_dir = ckpt.parent
    run_name = name or f"t2_{run_dir.name}"

    extras: list[Path] = []
    keys = [
        ("width_rule_name", "width_rule.json"),
        ("lead_anchor_name", "lead_anchor.json"),
        # `.pt`, not `.json`: `train tke-head --out x.json` writes the report as JSON and the
        # WEIGHTS as the .pt sibling, and `load_head` reads the latter. Auto-discovering the
        # JSON would have handed a report to torch.load.
        ("tke_head_name", "tke_head.pt"),
    ]
    if tke_head is not None:
        # An explicit head from anywhere on disk, staged under its own name -- the same
        # contract as Track 1's `--tke-head`. Without this the only way in was a file already
        # sitting beside the checkpoint, which meant a head fitted by `train tke-head` had to
        # be copied by hand.
        head_path = Path(tke_head)
        require(head_path, f"{head_path} is not a second-moment head checkpoint.")
        render_kwargs["tke_head_name"] = head_path.name
        extras.append(head_path)
        # Already staged from an absolute path; the run-directory lookup below must not also
        # try to resolve the same name against run_dir and fail.
        keys = [k for k in keys if k[0] != "tke_head_name"]
    if interval_recal is not None:
        # The learned interval calibrator's weights (models/interval_recal.py), staged under
        # their own name like an explicit --tke-head. Only the GPU-native path serves it: the
        # eager host closure has no calibrator hook, so a bundle that took it there would ship
        # the file and silently ignore it (found packaging the 2026-09-24 fp32 reference).
        if not render_kwargs.get("gpu_native") or not render_kwargs.get("compile_mode"):
            raise ValueError(
                "--interval-recal is served only by the GPU-native captured path; add "
                "--gpu-native --compile-mode cudagraph (or drop the calibrator)."
            )
        recal_path = Path(interval_recal)
        require(recal_path, f"{recal_path} is not a calibrator checkpoint.")
        render_kwargs["interval_recal_name"] = recal_path.name
        extras.append(recal_path)
    elif render_kwargs.get("interval_recal_fused") or render_kwargs.get("interval_recal_merged"):
        raise ValueError("--interval-recal-fused / --interval-recal-merged need --interval-recal.")
    for key, filename in keys:
        candidate = run_dir / filename
        if candidate.is_file() and not render_kwargs.get(key):
            render_kwargs[key] = filename
            extras.append(candidate)
        elif render_kwargs.get(key):
            named = run_dir / str(render_kwargs[key])
            extras.append(require(named, f"Template references {named.name}, which is absent."))

    text = render_ttt_submission(run_name=run_name, ckpt_name=ckpt.name, **render_kwargs)  # type: ignore[arg-type]
    out_root = out_root or SUBMISSIONS_DIR
    staged = stage_bundle(
        None,
        ckpt,
        out_root / run_name,
        submission_text=text,
        include_package=True,
        extra_files=tuple(extras),
        kit_payload=KIT_PAYLOAD_T2,
        kit_dir=VENDOR_KIT,
    )
    _stage_t2_package(staged)
    return zip_bundle(staged, out_root / f"{run_name}.zip")


def _stage_t2_package(staged: Path) -> None:
    """Copy the ``rpde_t2`` inference subset in beside the ``rpde_t1`` one."""
    pkg_root = Path(__file__).resolve().parents[1]
    dst_root = staged / pkg_root.name
    dst_root.mkdir(exist_ok=True)
    for item in T2_PACKAGE_PAYLOAD:
        src = require(pkg_root / item, f"Missing {item} from the Track-2 package payload.")
        shutil.copy2(src, dst_root / src.name)


def verify_ttt_bundle(bundle: Bundle, n_steps: int = 3) -> dict:
    """Extract the zip elsewhere and drive ``n_steps`` of the streaming protocol from it.

    Checks what the platform checks and this repo otherwise cannot: that ``get_ttt_model``
    resolves from inside the archive, that ``reset_ttt_state`` and ``ttt_step`` exist, that
    the prediction shape and finiteness hold, that ``adapt_loss`` is present, that bounds
    are all-or-nothing across steps and correctly ordered -- and that the input handed in is
    treated as *normalized*, which is the one mistake that produces a plausible-looking run.

    The synthetic input is built in evaluator space from a physical field at 0.15 m/s, so a
    submission that forgets to denormalize sees the same near-zero freestream the platform
    would hand it.
    """
    body = (
        "import json, time\n"
        "import numpy as np, torch\n"
        "import submission\n"
        "MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)\n"
        "STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)\n"
        "MEAN_T = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)\n"
        "STD_T = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)\n"
        f"shape = {EXPECTED_SHAPE!r}\n"
        "raw = np.zeros(shape, dtype=np.float32); raw[..., 0] = 0.15\n"
        "x = torch.from_numpy(((raw - MEAN_IN) / STD_IN).astype(np.float32))\n"
        "m = submission.get_ttt_model('.', 'cuda' if torch.cuda.is_available() else 'cpu')\n"
        "assert hasattr(m, 'reset_ttt_state') and hasattr(m, 'ttt_step')\n"
        "m.reset_ttt_state()\n"
        "keys, shapes, finite, ordered, losses, times = [], [], [], [], [], []\n"
        "prev = None\n"
        f"for _ in range({n_steps}):\n"
        "    t0 = time.perf_counter()\n"
        "    pred, info = m.ttt_step(x, prev)\n"
        "    times.append(time.perf_counter() - t0)\n"
        "    p = torch.as_tensor(pred).detach().cpu().numpy()\n"
        "    shapes.append(list(p.shape)); finite.append(bool(np.isfinite(p).all()))\n"
        "    keys.append(sorted(k for k in info if k in ('lower', 'upper')))\n"
        "    losses.append(info['adapt_loss'] is None or isinstance(info['adapt_loss'], float))\n"
        "    if 'lower' in info and 'upper' in info:\n"
        "        lo = torch.as_tensor(info['lower']).detach().cpu().numpy()\n"
        "        hi = torch.as_tensor(info['upper']).detach().cpu().numpy()\n"
        "        ordered.append(bool(np.all(lo <= hi) and np.isfinite(lo).all()\n"
        "                            and np.isfinite(hi).all() and lo.shape == p.shape))\n"
        "    prev = torch.from_numpy(((raw - MEAN_T) / STD_T).astype(np.float32))\n"
        "res = {'shape': shapes[0], 'shapes_stable': len(set(map(tuple, shapes))) == 1,\n"
        "       'finite': all(finite), 'adapt_loss_ok': all(losses),\n"
        "       'bounds_all_or_none': len(set(map(tuple, keys))) == 1,\n"
        "       'bounds': keys[0], 'bounds_ordered': all(ordered) if ordered else None,\n"
        "       'mean_step_s': float(np.mean(times)), 'blocked_ok': True}\n"
        "print('VERIFY_JSON' + json.dumps(res))\n"
    )
    result = run_in_extracted(
        bundle,
        body,
        blocked=MISSING_IN_EVAL_IMAGE_T2,
        failure_hint="failed to run the streaming protocol",
    )
    result["expected_shape"] = list(EXPECTED_SHAPE)
    result["shape_ok"] = result["shape"] == list(EXPECTED_SHAPE)
    # A bounds-emitting run must emit them on the first step too, so a `None` here means the
    # run is point-only, not that the check was skipped.
    result["ok"] = bool(
        result["shape_ok"]
        and result["shapes_stable"]
        and result["finite"]
        and result["adapt_loss_ok"]
        and result["bounds_all_or_none"]
        and result["bounds_ordered"] is not False
    )
    return result


__all__ = [
    "IN_STEP",
    "KIT_PAYLOAD_T2",
    "T2_PACKAGE_PAYLOAD",
    "MISSING_IN_EVAL_IMAGE_T2",
    "PACKAGE_PAYLOAD",
    "RUNS_DIR",
    "build_ttt_bundle",
    "render_ttt_submission",
    "verify_ttt_bundle",
]
