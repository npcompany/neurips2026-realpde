"""Run a packaged Track-1 bundle over a local eval set and score it with the kit's scorer.

    python check_t1.py <bundle_dir> <split> <out.json>

Calls `predict` on parts of 5 windows (the Codabench harness's call size) after one untimed
same-shape call, writes the scores and the per-call milliseconds. The scores are in-sample
diagnostics for an all-cells backbone, not held-out evidence; the run-through is the check.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from rpde_t1.paths import LOCAL_EVAL_DIR, VENDOR_KIT


def load_submission(bundle: Path):
    sys.path.insert(0, str(bundle))
    for k in [k for k in sys.modules if k == "rpde_t1" or k.startswith("rpde_t1.")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("submission", bundle / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    bundle, split, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    x = np.load(LOCAL_EVAL_DIR / split / "inputs.npy")
    mod = load_submission(bundle)
    part = 5
    mod.predict(np.random.default_rng(0).standard_normal((part, *x.shape[1:]), dtype=np.float32), {})
    preds, los, his, ms = [], [], [], []
    for i in range(0, len(x), part):
        xb = np.ascontiguousarray(x[i:i + part])
        t0 = time.perf_counter()
        o = mod.predict(xb, {})
        ms.append(1e3 * (time.perf_counter() - t0) / len(xb))
        if isinstance(o, dict):
            preds.append(np.asarray(o["prediction"], dtype=np.float32))
            los.append(np.asarray(o["lower"], dtype=np.float32))
            his.append(np.asarray(o["upper"], dtype=np.float32))
        else:
            preds.append(np.asarray(o, dtype=np.float32))
    pred = np.concatenate(preds)
    rec = {"bundle": str(bundle), "split": split, "n_windows": int(len(x)),
           "ms_per_sample_median": float(np.median(ms)), "ms_per_sample_mean": float(np.mean(ms)),
           "finite": bool(np.isfinite(pred).all()), "pressure_zero": bool((pred[..., 2] == 0).all())}
    sys.path.insert(0, str(VENDOR_KIT))
    import scoring  # noqa: E402

    tmp = Path(tempfile.mkdtemp())
    (tmp / "pred").mkdir()
    (tmp / "ref").mkdir()
    arrays = {"prediction": pred}
    if los:
        arrays["lower"], arrays["upper"] = np.concatenate(los), np.concatenate(his)
        rec["bounds_ordered"] = bool((arrays["lower"] <= arrays["upper"]).all())
    np.savez(tmp / "pred" / "predictions.npz", **arrays)
    np.savez(tmp / "ref" / "targets.npz", target=scoring.load_targets(LOCAL_EVAL_DIR / split / "ref"))
    scores = scoring._score_submission(tmp / "pred", tmp / "ref")
    rec["scores"] = {k: float(v) for k, v in scores.items()}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"check {split}: {rec}", flush=True)


if __name__ == "__main__":
    main()
