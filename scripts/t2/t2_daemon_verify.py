#!/usr/bin/env python
"""Import a Track-2 bundle in a daemonic worker, matching the board process constraint."""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path


def _load(bundle: str) -> None:
    root = Path(bundle).resolve()
    os.chdir(root)
    spec = importlib.util.spec_from_file_location("submission", root / "submission.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {root / 'submission.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.get_ttt_model(str(root), "cuda")
    model.reset_ttt_state()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    args = ap.parse_args()
    if not (args.bundle / "submission.py").is_file():
        ap.error(f"missing {args.bundle / 'submission.py'}")
    proc = mp.Process(target=_load, args=(str(args.bundle),), daemon=True)
    proc.start()
    proc.join(300)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise RuntimeError("daemon import exceeded 300 seconds")
    if proc.exitcode != 0:
        raise RuntimeError(f"daemon import failed with exit code {proc.exitcode}")
    print(f"OK: daemon import + get_ttt_model + reset: {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
