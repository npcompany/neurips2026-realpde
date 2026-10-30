"""Build and verify a Track 1 submission zip.

The platform expects a flat archive: ``submission.py`` plus everything it imports
sitting at the **root** of the zip, not inside a directory. For a wrapped
baseline that means::

    submission.py          the predict() entry point
    <checkpoint>.pth       the weights
    load_baseline.py       from the starting kit
    rpde_baselines/        vendored model subtree
    _vendor/               vendored einops (Transolver needs rearrange)

Two rules are enforced here rather than discovered on the leaderboard:

* **256 MB extracted.** Checked against the sum of uncompressed member sizes,
  which is what the limit applies to -- a zip that compresses under the cap but
  extracts over it would still be rejected.
* **It has to actually run.** ``verify_bundle`` extracts the archive to a
  scratch directory, imports ``submission.py`` from there with the repository
  *off* ``sys.path``, and runs one ``predict()`` call. That catches a forgotten
  file or a stray absolute import before it costs a submission.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch

from .. import INPUT_SHAPE, T_OUT, H, W
from ..paths import CKPT_DIR, RUNS_DIR, SUBMISSIONS_DIR, VENDOR_KIT, require

#: Codabench rejects a submission whose extracted size exceeds this.
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024

#: Files/directories copied verbatim out of the starting kit into every bundle.
KIT_PAYLOAD = ("load_baseline.py", "rpde_baselines", "_vendor")

#: The subset of our own package shipped when the submission wraps one of *our*
#: checkpoints. Deliberately not the whole of ``rpde_t1``: ``data/``, ``train/`` and
#: ``score/`` reach ``h5py``, which the evaluation container does not ship, so leaving
#: the files out makes the bad import impossible rather than merely unexercised.
PACKAGE_PAYLOAD = ("__init__.py", "features.py", "geometry.py", "models")

TEMPLATE = Path(__file__).with_name("baseline_submission.py.tmpl")
MODEL_TEMPLATE = Path(__file__).with_name("model_submission.py.tmpl")

#: SPS-optimal constant bound half-width measured on a local eval set
#: (~1.5x the per-element error std). See report.py for the sweep.
DEFAULT_BOUND_HALF_WIDTH = 0.005


@dataclass
class Bundle:
    """A built submission archive."""

    zip_path: Path
    staged_dir: Path
    extracted_bytes: int
    members: int

    @property
    def extracted_mb(self) -> float:
        return self.extracted_bytes / 1024 / 1024

    @property
    def zip_mb(self) -> float:
        return self.zip_path.stat().st_size / 1024 / 1024


def render_submission(
    ckpt_name: str,
    model_name: str,
    batch: int = 16,
    return_bounds: bool = False,
    bound_half_width: float = DEFAULT_BOUND_HALF_WIDTH,
) -> str:
    """Fill the baseline submission template for one checkpoint."""
    text = TEMPLATE.read_text(encoding="utf-8")
    # The template uses doubled braces for the literal dicts it emits, so a plain
    # str.format resolves both the placeholders and those braces in one pass.
    return text.format(
        CKPT_NAME=ckpt_name,
        MODEL_NAME=model_name,
        BATCH=batch,
        RETURN_BOUNDS="True" if return_bounds else "False",
        BOUND_HALF_WIDTH=repr(float(bound_half_width)),
    )


def stage_bundle(
    submission_source: str | Path | None,
    checkpoint: Path | None,
    out_dir: Path,
    submission_text: str | None = None,
    include_package: bool = False,
    extra_files: tuple[Path, ...] = (),
    kit_payload: tuple[str, ...] = KIT_PAYLOAD,
    kit_dir: Path | None = None,
) -> Path:
    """Assemble the flat bundle layout in ``out_dir`` and return it.

    With ``include_package`` the inference subset of ``rpde_t1`` is copied in too,
    which is what a submission wrapping one of our own checkpoints needs.

    ``extra_files`` are additional artefacts placed at the bundle root under their own
    names -- currently the second-moment head, which is a separate ~40 kB checkpoint
    rather than part of the backbone's state dict.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if submission_text is not None:
        (out_dir / "submission.py").write_text(submission_text, encoding="utf-8")
    else:
        src = Path(require(Path(submission_source), "Need a submission source file."))
        shutil.copy2(src, out_dir / "submission.py")

    if checkpoint is not None:
        ckpt = Path(require(checkpoint, "Run `make data` to fetch the baseline checkpoints."))
        shutil.copy2(ckpt, out_dir / ckpt.name)

    for extra in extra_files:
        src = Path(require(extra, f"Missing bundled artefact {extra}."))
        shutil.copy2(src, out_dir / src.name)

    for item in kit_payload:
        src = (kit_dir or VENDOR_KIT) / item
        require(src, "The vendored starting kit is incomplete.")
        dst = out_dir / item
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)

    if include_package:
        pkg_root = Path(__file__).resolve().parents[1]
        (out_dir / pkg_root.name).mkdir()
        for item in PACKAGE_PAYLOAD:
            src = pkg_root / item
            require(src, f"Missing {item} from the package payload.")
            dst = out_dir / pkg_root.name / item
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dst)
    return out_dir


def zip_bundle(staged_dir: Path, zip_path: Path) -> Bundle:
    """Zip a staged bundle with every member at the archive root."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    total = 0
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(staged_dir.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            zf.write(path, path.relative_to(staged_dir))
            total += path.stat().st_size
            count += 1

    bundle = Bundle(zip_path, staged_dir, total, count)
    if total > MAX_EXTRACTED_BYTES:
        raise ValueError(
            f"Bundle extracts to {bundle.extracted_mb:.1f} MB, over the "
            f"{MAX_EXTRACTED_BYTES / 1024 / 1024:.0f} MB cap. "
            "For FNO use the fp16-packed checkpoint (see pack_ckpt_fp16.py)."
        )
    return bundle


#: Absent from the **Track-1** evaluation image (verified by the kit's smoke test by
#: importing them inside it). A submission that reaches for any of these passes locally and
#: fails on the platform, so the verifier blocks them.
#:
#: Track-1 specific, and not a fact about the competition. The Track-2 image
#: (``w3nhao/realpde-track2``) ships all four plus einops, scikit-learn, safetensors and
#: openai, so blocking them there would reject a valid bundle -- see
#: :data:`rpde_t2.submit.package.MISSING_IN_EVAL_IMAGE_T2`.
MISSING_IN_EVAL_IMAGE = ("scipy", "h5py", "matplotlib", "pandas")

#: The blocker installed into the verification subprocess. ``_VENDORED`` names modules the
#: image does not ship but the bundle carries its own copy of, and which are therefore
#: allowed *only* when they resolve from inside the extracted archive -- which is what
#: proves the vendored copy is self-sufficient. On Track 1 that is ``einops``; on Track 2
#: the image ships it, so the list is empty.
_BLOCKER_SRC = """
import importlib.machinery, os, sys

_BLOCKED = {blocked!r}
_VENDORED = {vendored!r}
_ROOT = {root!r}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in _BLOCKED:
            raise ModuleNotFoundError(
                "evaluation image does not ship %s (blocked by the verifier)" % top
            )
        if top in _VENDORED:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec and spec.origin and spec.origin not in ("built-in", "frozen"):
                if not os.path.abspath(spec.origin).startswith(_ROOT):
                    raise ModuleNotFoundError(
                        "%s must come from the bundle's _vendor/, not %s" % (top, spec.origin)
                    )
            return spec
        return None


sys.meta_path.insert(0, _Blocker())
"""


def verify_bundle(bundle: Bundle, n_windows: int = 2) -> dict:
    """Extract the zip elsewhere and run ``predict()`` from it in a subprocess.

    Two things make this a meaningful pre-flight check rather than a formality:

    * ``-I`` (isolated) keeps this repository and the user site-packages out, so
      only the archive's own contents resolve -- a file left out of the zip fails
      here instead of on the leaderboard;
    * the packages the evaluation image *lacks* are actively blocked, so a stray
      ``import h5py`` cannot pass locally and then fail on the platform.

    What it still cannot check: the container's exact torch build and GPU. Use
    ``make smoke`` (which runs the kit's own parity test) and, for full fidelity,
    the Docker one-liner in the kit README.
    """
    body = (
        "import json, time\n"
        "import numpy as np\n"
        "import submission\n"
        f"x = np.zeros(({n_windows},) + {INPUT_SHAPE!r}, dtype=np.float32)\n"
        "x[..., 0] = 0.15\n"
        "t0 = time.perf_counter()\n"
        "out = submission.predict(x)\n"
        "dt = time.perf_counter() - t0\n"
        "pred = out['prediction'] if isinstance(out, dict) else out\n"
        "res = {'shape': list(pred.shape), 'dtype': str(pred.dtype),\n"
        "       'finite': bool(np.isfinite(pred).all()), 'seconds': dt,\n"
        "       'is_dict': isinstance(out, dict),\n"
        "       'keys': sorted(out) if isinstance(out, dict) else [],\n"
        "       'blocked_ok': True,\n"
        "       'p_channel_zero': bool(np.all(pred[..., 2] == 0))}\n"
        "print('VERIFY_JSON' + json.dumps(res))\n"
    )
    result = run_in_extracted(
        bundle, body, vendored=("einops",), failure_hint="failed to produce a prediction"
    )
    expected = [n_windows, T_OUT, H, W, 3]
    result["shape_ok"] = result["shape"] == expected
    result["expected_shape"] = expected
    return result


def run_in_extracted(
    bundle: Bundle,
    body: str,
    blocked: tuple[str, ...] = MISSING_IN_EVAL_IMAGE,
    vendored: tuple[str, ...] = (),
    failure_hint: str = "failed to run",
) -> dict:
    """Extract ``bundle`` to a scratch dir and run ``body`` against it, isolated.

    The mechanics both tracks share: ``-I`` keeps this repository and the user
    site-packages off ``sys.path`` so only the archive's own contents resolve, the modules
    the evaluation image lacks are actively blocked, and the child reports back by printing
    one ``VERIFY_JSON<json>`` line. ``body`` may assume ``_ROOT`` is the extraction root and
    is responsible for putting it on ``sys.path``.

    The two tracks differ only in ``blocked`` / ``vendored`` and in the body -- Track 1
    calls ``submission.predict``, Track 2 drives ``get_ttt_model`` through two steps.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "extracted"
        root.mkdir()
        with zipfile.ZipFile(bundle.zip_path) as zf:
            zf.extractall(root)

        script = root / "_verify.py"
        script.write_text(
            _BLOCKER_SRC.format(blocked=list(blocked), vendored=list(vendored), root=str(root))
            + f"sys.path.insert(0, {str(root)!r})\n"
            + body,
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-I", str(script)],
            capture_output=True,
            text=True,
            cwd=tmp,
        )

    marker = "VERIFY_JSON"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])

    raise RuntimeError(
        f"The extracted bundle {failure_hint}.\n"
        f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def build_baseline_bundle(
    checkpoint: str,
    name: str | None = None,
    return_bounds: bool = False,
    bound_half_width: float = DEFAULT_BOUND_HALF_WIDTH,
    batch: int = 16,
    out_root: Path | None = None,
) -> Bundle:
    """Build a submission wrapping one of the official baseline checkpoints.

    ``checkpoint`` is a filename (searched under ``data/ckpts/``) or an explicit path.
    The model type is inferred from the filename by ``load_baseline``.
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        matches = sorted(CKPT_DIR.rglob(ckpt_path.name))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint named {ckpt_path.name} under {CKPT_DIR}. Run `make data`."
            )
        ckpt_path = matches[0]

    model_name = next(
        (m for m in ("transolver", "cno", "fno") if m in ckpt_path.name.lower()),
        "unknown",
    )
    name = name or ckpt_path.stem + ("_bounds" if return_bounds else "")
    out_root = out_root or SUBMISSIONS_DIR

    text = render_submission(
        ckpt_path.name,
        model_name,
        batch=batch,
        return_bounds=return_bounds,
        bound_half_width=bound_half_width,
    )
    staged = stage_bundle(None, ckpt_path, out_root / name, submission_text=text)
    return zip_bundle(staged, out_root / f"{name}.zip")


#: Pack a checkpoint to fp16 once its tensors exceed this. Set below the 256 MB cap
#: with room for the kit payload (~2.5 MB) and our own package (~0.2 MB).
FP16_PACK_ABOVE_BYTES = 200 * 1024 * 1024


def pack_fp16(payload: dict, dst: Path) -> Path:
    """Write the fp16-packed form of a training checkpoint.

    Same scheme as the kit's ``pack_ckpt_fp16.py`` — real tensors halved, complex
    tensors stored as real/imag pairs — but keeps our ``config`` alongside so
    ``load_for_inference`` can still rebuild the architecture.
    """
    packed, complex_keys = {}, []
    for key, value in payload["model"].items():
        if torch.is_tensor(value) and value.is_complex():
            packed[key] = torch.view_as_real(value).half()
            complex_keys.append(key)
        elif torch.is_tensor(value) and torch.is_floating_point(value):
            packed[key] = value.half()
        else:
            packed[key] = value
    torch.save(
        {
            "state_fp16": packed,
            "complex_keys": complex_keys,
            "config": payload.get("config", {}),
            "fp16_packed": True,
        },
        dst,
    )
    return dst


#: Call sizes the emitted ``submission.py`` warms at import when none are requested.
DEFAULT_WARMUP_SIZES = "1,2,4,8,16,32,64,105"


def warmup_literal(spec: str, batch: int) -> str:
    """Render ``--warmup-sizes`` (``"1-32,64,105"``; ranges inclusive) as a tuple literal.

    The bundle's batch size is always warmed last, so the clock-boost spin that follows the
    final size lands on the shape the scored calls are most likely to use.
    """
    sizes: list[int] = []
    for part in (spec or DEFAULT_WARMUP_SIZES).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            sizes.extend(range(lo, hi + 1))
        else:
            sizes.append(int(part))
    sizes = sorted({n for n in sizes if 0 < n < batch})
    return "(" + "".join(f"{n}, " for n in sizes) + "_BATCH)"


def build_run_bundle(
    run: str | Path,
    name: str | None = None,
    checkpoint_file: str = "last.pt",
    return_bounds: bool | None = None,
    bound_floor: float = 0.0,
    bound_scale: float = 1.0,
    precision: str = "fp32",
    pad_batches: bool = False,
    batch: int = 32,
    out_root: Path | None = None,
    tke_head: str | Path | None = None,
    tke_blend: float | None = None,
    tke_shrink: float | None = None,
    width_rule: str | Path | None = None,
    lead_anchor: str | Path | None = None,
    compile_mode: str = "",
    compile_dynamic: bool = False,
    cond_device: bool = False,
    zero_p_device: bool = False,
    cudagraph: str = "",
    re_corners: bool = False,
    warmup_sizes: str = "",
    centre_shrink: float = 1.0,
) -> Bundle:
    """Build a submission around one of **our** training runs.

    ``run`` is a run directory (or a name under ``runs/``). ``return_bounds``
    defaults to whatever the checkpoint's config says about its bound head, which is
    the only setting that cannot be wrong.

    ``width_rule`` bundles a fitted :class:`~rpde_t1.models.width_rule.WidthRule` beside
    the weights. It supersedes ``bound_scale`` -- refused here rather than silently
    composed, matching ``make_predict``, because two calibrations of the same width
    multiplying each other is a bug no subscore would flag.

    ``compile_mode`` compiles the forward with :func:`torch.compile`; empty (the default)
    ships the historical no-compile bundle. It was long refused on Track 1 -- the earlier
    losses (local 85.1 -> board 80.6, 92.4 -> board 83.6, 2026-08-19's 4.8 -> board 27.5,
    FINDINGS 49) were really ``pad_batches`` and the daemon-import crash (FINDINGS 76c), not
    compile itself. ``compile_dynamic`` reopens it safely: a single ``dynamic=True``
    symbolic-shape graph serves every N with no per-shape recompile, so it needs neither
    padding nor a warmed shape ladder, measured +25-29% ms/sample on the batch ladder held
    out at the fp16 quantum (FINDINGS 148). ``reduce-overhead`` is refused unless
    ``pad_batches`` -- its CUDA graphs raise on a second input shape -- and the emitted
    ``submission.py`` forces ``TORCHINDUCTOR_COMPILE_THREADS=1``, guards N=1 to eager, and
    sticky-demotes to eager on any compile failure. ``cudnn.benchmark`` stays off (it bought
    0.02 ms/sample of steady state for 3.6 s of one-off per-shape algorithm search).

    ``centre_shrink`` bakes ``make_predict(centre_shrink=...)`` into the bundle (FINDINGS
    196i): the interval is centred on the fluctuation-shrunk prediction while the reported
    field is untouched. It needs an interval head and width rule fitted on the shrunk field
    and a lead-anchor-free build; 1.0 renders the previous template exactly.
    """
    run_dir = Path(run)
    if not run_dir.is_dir():
        run_dir = RUNS_DIR / str(run)
    ckpt_path = Path(require(run_dir / checkpoint_file, f"No {checkpoint_file} in {run_dir}."))

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = dict(payload.get("config", {}))
    has_bounds = bool(config.get("bounds", False))
    if return_bounds is None:
        return_bounds = has_bounds
    if return_bounds and not has_bounds:
        raise ValueError(f"{ckpt_path} has no bound head, so it cannot return bounds.")

    name = name or run_dir.name
    out_root = out_root or SUBMISSIONS_DIR

    # Oversized checkpoints (FNO is 384 MB in fp32) are halved on the way in. Doing it
    # here rather than asking the caller to pre-pack keeps one command per submission.
    ckpt_bytes = sum(
        v.numel() * v.element_size() for v in payload["model"].values() if torch.is_tensor(v)
    )
    packed_path = None
    if ckpt_bytes > FP16_PACK_ABOVE_BYTES:
        out_root.mkdir(parents=True, exist_ok=True)
        packed_path = pack_fp16(payload, out_root / f"{ckpt_path.stem}_fp16.pt")
        ckpt_path = packed_path

    from ..models.tke_head import DEFAULT_BLEND, DEFAULT_SHRINK

    head_path = Path(tke_head) if tke_head else None
    if head_path is not None and not head_path.is_file():
        raise ValueError(f"{head_path} is not a second-moment head checkpoint.")

    # Default to the rule sitting in the run directory. `sps_assemble.py` puts the
    # selected one there next to the weights precisely so a build cannot ship a head
    # with no sizing, or with a rule fitted for a different one.
    rule_path = Path(width_rule) if width_rule else run_dir / "width_rule.json"
    rule_path = rule_path if rule_path.is_file() else None
    if rule_path is not None and bound_scale != 1.0:
        raise ValueError(
            f"{rule_path} and --bound-scale {bound_scale} are two calibrations of the same "
            "width; they would multiply. Ship one."
        )
    anchor_path = Path(lead_anchor) if lead_anchor else run_dir / "lead_anchor.json"
    anchor_path = anchor_path if anchor_path.is_file() else None
    if anchor_path is not None and rule_path is None:
        # The anchor displaces the prediction, so the interval is sized against a different
        # error field. A rule is not optional alongside it -- and the rule that ships has to
        # be the *anchored* refit, which no file inspection can confirm, hence the note in
        # the template and the end-to-end score check in the build log.
        raise ValueError(
            f"{anchor_path} shifts the prediction, so it needs the width rule refitted "
            "against the anchored error field. Pass --width-rule."
        )
    if config.get("bound_mode") == "logerr" and rule_path is None:
        # The head returns a raw error scale, not a width. Shipping it unsized is not a
        # milder version of the same submission -- it is an uncalibrated interval.
        raise ValueError(
            f"{ckpt_path} carries a log-error scale head, which is sized by a width rule. "
            "Pass --width-rule, or put width_rule.json in the run directory."
        )
    if cudagraph and compile_mode:
        # Both target the forward; graphing a torch.compile'd module is neither tested nor
        # useful here. The image cannot run compile at all, so cudagraph is the shippable one.
        raise ValueError("--cudagraph and --compile-mode are mutually exclusive.")
    if compile_dynamic and not compile_mode:
        raise ValueError("--compile-dynamic needs --compile-mode (e.g. default).")
    if compile_mode == "reduce-overhead" and not pad_batches:
        # Its CUDA graphs raise on a second input shape, and the platform calls predict()
        # with a small unknown N repeatedly. Only a single shape (padding) makes it safe.
        raise ValueError(
            "--compile-mode reduce-overhead raises on a second input shape; it needs "
            "--pad-batches (discouraged), or use --compile-mode default --compile-dynamic."
        )
    centre_shrink = float(centre_shrink)
    if centre_shrink != 1.0:
        # The same refusals make_predict applies at construction, here so a build fails at
        # packaging rather than in the clean-room import.
        if not (0.0 < centre_shrink <= 1.0):
            raise ValueError(f"--centre-shrink must be in (0, 1], got {centre_shrink}")
        if not return_bounds or rule_path is None:
            raise ValueError(
                "--centre-shrink centres the INTERVAL; it needs --bounds and a width rule "
                "fitted on the shrunk field (scripts/sps/shrink_cache.py, sps_apply.py)."
            )
        if anchor_path is not None:
            raise ValueError("--centre-shrink and a lead anchor are not composed; ship one.")

    text = MODEL_TEMPLATE.read_text(encoding="utf-8").format(
        CKPT_NAME=ckpt_path.name,
        RUN_NAME=run_dir.name,
        COMPILE_MODE=compile_mode,
        COMPILE_DYNAMIC=repr(bool(compile_dynamic)),
        CUDAGRAPH_BUCKETS=cudagraph,
        MODEL_NAME=config.get("model", "unknown"),
        BOUNDS_SUFFIX=" + bounds" if return_bounds else "",
        BATCH=batch,
        RETURN_BOUNDS="True" if return_bounds else "False",
        BOUND_FLOOR=repr(float(bound_floor)),
        BOUND_SCALE=repr(float(bound_scale)),
        PRECISION=precision,
        PAD_BATCHES=repr(bool(pad_batches)),
        COND_DEVICE=repr(bool(cond_device)),
        ZERO_P_DEVICE=repr(bool(zero_p_device)),
        RE_CORNERS=repr(bool(re_corners)),
        WARMUP_SIZES=warmup_literal(warmup_sizes, batch),
        CENTRE_SHRINK=repr(centre_shrink),
        # From the checkpoint's own config, never a build-time flag: the backbone's
        # first layer was sized for these planes, so serving a different set is a shape
        # error at best. Reading it off the config is what makes the two agree.
        GEOM_CHANNELS=str(config.get("geom_channels", "")),
        FLOW_CHANNELS=str(config.get("flow_channels", "")),
        WIDTH_RULE_NAME=rule_path.name if rule_path else "",
        LEAD_ANCHOR_NAME=anchor_path.name if anchor_path else "",
        TKE_HEAD_NAME=head_path.name if head_path else "",
        # ``None`` -- the default -- takes the measured blend when a head is present, since
        # a head shrunk toward *nothing* was the configuration 63f measured as worse than
        # doing nothing at the ends of the Reynolds ladder. An explicit ``0.0`` is honoured
        # as no blend, which the previous ``or`` form silently turned back into 0.5.
        TKE_BLEND=repr(
            float((DEFAULT_BLEND if head_path else 0.0) if tke_blend is None else tke_blend)
        ),
        TKE_SHRINK=repr(float(DEFAULT_SHRINK if tke_shrink is None else tke_shrink)),
    )
    staged = stage_bundle(
        None,
        ckpt_path,
        out_root / name,
        submission_text=text,
        include_package=True,
        extra_files=tuple(p for p in (head_path, rule_path, anchor_path) if p is not None),
    )
    bundle = zip_bundle(staged, out_root / f"{name}.zip")
    if packed_path is not None:
        packed_path.unlink()  # it lives in the bundle now
    return bundle
