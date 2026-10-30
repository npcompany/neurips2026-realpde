"""RealPDE Decision phase, Track 2 (long-term test-time adaptation): team np-user.

    train(data_dir, output_dir, device)   trains the whole chain from scratch (see METHOD.md)
    get_ttt_model(output_dir, device)     returns an object with reset_ttt_state() / ttt_step()

`train` writes everything into `output_dir`: the packaged bundle served by `get_ttt_model`
(`t2_bundle/`), the per-stage logs (`logs/`), `PROVENANCE.txt` (sha256 of every produced
checkpoint, stage durations, peak GPU memory) and `work/` (caches, eval sets, intermediate runs).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# The participant runs in a daemonic process on the board; Inductor must not spawn a pool.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

HERE = Path(__file__).resolve().parent
TRACK = "t2"
BUNDLE_NAME = "t2_bundle"


def train(data_dir: str, output_dir: str, device: str = "cuda") -> None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from rpde_final import recipes
    from rpde_final.runtime import Runtime, log

    rt = Runtime(data_dir, output_dir, TRACK)
    rt.setup()
    log("Track 2 training chain starts")
    bundle, recal = recipes.train_t2(rt)
    final = recipes.finalize_t2(rt, bundle, recal)
    recipes.check_t2(rt, final)
    summary = rt.summary()
    (rt.out / "TRAIN_SUMMARY.txt").write_text(summary + "\n")
    rt.prov("summary", summary.replace("\n", " | "))
    log("done\n" + summary)


def get_ttt_model(output_dir: str, device: str = "cuda"):
    """The packaged bundle's own `get_ttt_model`: frozen port, one CUDA graph of the fp16 fast
    pipeline with the interval calibrator fused in, normalized in / normalized out, bounds in
    `info`, `adapt_loss` None (no parameter adaptation)."""
    bundle = Path(output_dir).resolve() / BUNDLE_NAME
    if not (bundle / "submission.py").exists():
        raise FileNotFoundError(f"{bundle}/submission.py: run train() first")
    sys.path.insert(0, str(bundle))
    for k in [k for k in sys.modules if k.split(".")[0] in ("rpde_t1", "rpde_t2", "submission")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("submission", bundle / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = mod
    spec.loader.exec_module(mod)
    return mod.get_ttt_model(str(bundle), device)
