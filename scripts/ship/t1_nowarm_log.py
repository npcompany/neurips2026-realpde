#!/usr/bin/env python
"""Turn a CLEAN Track-1 bundle into the "no warm-up + log lines" variant the board runs.

    uv run python scripts/ship/t1_nowarm_log.py <clean_bundle_dir> <out_dir> \
        [--expect <submission.py>]

The 2026-09-24 board zip (`opt_ft24_fatap_hc06_alldata_fast_b210_clean_nowarm_log`, 81.999) was
made from the clean bundle by editing submission.py by hand. This script makes the same edits,
so nobody has to remember them:

1. no import-time warm-up loop: the harness makes its own untimed same-shape call in every
   fresh process (Codabench Track-1 FAQ), so the bundle's own warm-up bought nothing and cost
   import time. Construction still runs at import (`_get_predict()`).
2. log lines on stdout (the FAQ allows shapes and timings): at import the pid and load seconds,
   and per call the pid, call index, input shape, dtype, our own predict milliseconds and the
   seconds since load. Every print sits in try/except so logging can never fail a run.

Nothing else changes: the model, the heads, `predict`'s arithmetic and the package are byte for
byte the clean bundle's. Every edit is an exact-text replacement that must match exactly once,
so a change in the clean generator fails here loudly instead of producing a different bundle.
`--expect` compares the result with a reference submission.py (the board zip's) and fails on
any difference.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

EDITS = [
    (
        "Model construction, checkpoint loading and the warm-up run at import time; "
        "``predict`` runs\n",
        "Model construction and checkpoint loading run at import time; ``predict`` runs\n",
    ),
    (
        "import os\nimport sys\nfrom pathlib import Path\n",
        "import os\nimport sys\nimport time\n\n_T_START = time.perf_counter()\n"
        "from pathlib import Path\n",
    ),
    (
        "from rpde_t1.models.predictor import DEFAULT_SPIN_S, make_predict, warm_up_predict"
        "  # noqa: E402\n",
        "from rpde_t1.models.predictor import make_predict  # noqa: E402\n",
    ),
    ('_state = {"fn": None}\n', '_state = {"fn": None, "calls": 0}\n'),
    (
        "for _size in WARMUP_SIZES:\n"
        "    _spin = DEFAULT_SPIN_S if _size == WARMUP_SIZES[-1] else 0.0\n"
        "    warm_up_predict(_get_predict(), _size, spin_s=_spin)\n",
        "_get_predict()\n"
        "\n"
        "try:\n"
        '    print(f"[t1] pid={os.getpid()} load_s={time.perf_counter() - _T_START:.2f}", '
        "flush=True)\n"
        "except Exception:  # noqa: BLE001 - logging must never fail a run\n"
        "    pass\n",
    ),
    (
        '    """Track 1 API: (N, T_in, H, W, C) raw fields -> dict(prediction, lower, upper)."""\n'
        "    x = np.asarray(input_array, dtype=np.float32)\n",
        '    """Track 1 API: (N, T_in, H, W, C) raw fields -> dict(prediction, lower, upper)."""\n'
        "    t0 = time.perf_counter()\n"
        "    x = np.asarray(input_array, dtype=np.float32)\n",
    ),
    (
        '        raise RuntimeError("bounds requested but predict returned a bare array")\n'
        "    return out\n",
        '        raise RuntimeError("bounds requested but predict returned a bare array")\n'
        "    try:\n"
        '        _state["calls"] += 1\n'
        "        ms = (time.perf_counter() - t0) * 1e3\n"
        "        print(\n"
        "            f\"[t1] pid={os.getpid()} call={_state['calls']} "
        "shape={tuple(np.shape(input_array))} \"\n"
        "            f\"dtype={getattr(input_array, 'dtype', '?')} predict_ms={ms:.1f} \"\n"
        '            f"since_load_s={time.perf_counter() - _T_START:.2f}",\n'
        "            flush=True,\n"
        "        )\n"
        "    except Exception:  # noqa: BLE001 - logging must never fail a run\n"
        "        pass\n"
        "    return out\n",
    ),
]

WARMUP_LINE = re.compile(r"^WARMUP_SIZES = .*\n", re.MULTILINE)


def transform(text: str) -> str:
    """The clean submission.py -> the no-warm-up + log-lines submission.py."""
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"submission.py does not match the clean generator's text: {old!r}")
        text = text.replace(old, new)
    if len(WARMUP_LINE.findall(text)) != 1:
        raise SystemExit("submission.py has no single WARMUP_SIZES line")
    text = WARMUP_LINE.sub("", text)
    for word in ("warm_up_predict", "WARMUP_SIZES", "DEFAULT_SPIN_S"):
        if word in text:
            raise SystemExit(f"{word} survived the edit")
    return text


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    expect = None
    if "--expect" in sys.argv:
        expect = Path(sys.argv[sys.argv.index("--expect") + 1])
        args = [a for a in args if a != str(expect)]
    if len(args) != 2:
        raise SystemExit(__doc__)
    src_dir, out_dir = Path(args[0]), Path(args[1])
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists")
    shutil.copytree(src_dir, out_dir, ignore=shutil.ignore_patterns("__pycache__"))
    sub = out_dir / "submission.py"
    new = transform(sub.read_text())
    sub.write_text(new)
    if expect is not None:
        ref = expect.read_text()
        if ref != new:
            raise SystemExit(f"result differs from {expect}")
        print(f"identical to {expect}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
