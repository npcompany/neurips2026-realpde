"""RealPDE Decision phase, Track 1 (sim-to-real transfer): team np-user.

    train(data_dir, output_dir, device)   trains the whole chain from scratch (see METHOD.md)
    load_predictor(output_dir, device)    returns an object with predict(input_array)

`train` writes everything into `output_dir`: the trained bundle served by `load_predictor`
(`t1_bundle/`), the per-stage logs (`logs/`), `PROVENANCE.txt` (sha256 of every produced
checkpoint, stage durations, peak GPU memory) and `work/` (caches, eval sets, intermediate runs).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

HERE = Path(__file__).resolve().parent
TRACK = "t1"
BUNDLE_NAME = "t1_bundle"


def train(data_dir: str, output_dir: str, device: str = "cuda") -> None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from rpde_final import recipes
    from rpde_final.runtime import Runtime, log

    rt = Runtime(data_dir, output_dir, TRACK)
    rt.setup()
    log("Track 1 training chain starts")
    bundle = recipes.train_t1(rt)
    final = recipes.package_t1(rt, bundle)
    recipes.check_t1(rt, final)
    summary = rt.summary()
    (rt.out / "TRAIN_SUMMARY.txt").write_text(summary + "\n")
    rt.prov("summary", summary.replace("\n", " | "))
    log("done\n" + summary)


class Predictor:
    """The packaged bundle's `predict`, unchanged: fp16 whole-call CUDA graph, float32 outputs,
    prediction + lower/upper bounds in physical units, pressure zeroed."""

    def __init__(self, output_dir: str, device: str = "cuda"):
        bundle = Path(output_dir).resolve() / BUNDLE_NAME
        if not (bundle / "submission.py").exists():
            raise FileNotFoundError(f"{bundle}/submission.py: run train() first")
        # The bundle carries its own (pruned, constant-folded) copy of the model code; it must
        # shadow any other rpde_t1 on the path.
        sys.path.insert(0, str(bundle))
        for k in [k for k in sys.modules if k.split(".")[0] in ("rpde_t1", "rpde_t2", "submission")]:
            del sys.modules[k]
        spec = importlib.util.spec_from_file_location("submission", bundle / "submission.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = mod
        spec.loader.exec_module(mod)
        self._predict = mod.predict
        self.device = device

    def predict(self, input_array):
        import numpy as np

        x = np.ascontiguousarray(np.asarray(input_array, dtype=np.float32))
        out = self._predict(x, {})
        if isinstance(out, dict):
            return {k: np.asarray(v) for k, v in out.items() if v is not None}
        return np.asarray(out)


def load_predictor(output_dir: str, device: str = "cuda") -> Predictor:
    return Predictor(output_dir, device)
