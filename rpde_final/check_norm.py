"""Compare the official mean_std_real.pt with the constants the code carries (features.py).

The repository hard-codes the organizers' normalization statistics (extracted from the kit's
own mean_std_real.pt). This never fails: a mismatch is printed and recorded so it is visible
in the training log, because every development run and the Codabench rows used these values.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from rpde_t1 import features


def main() -> None:
    mi, mt, si, st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    si = torch.where(si == 0, torch.ones_like(si), si)
    st = torch.where(st == 0, torch.ones_like(st), st)
    ok = True
    for name, got, want in (
        ("mean_in", mi, features.REAL_MEAN_IN), ("std_in", si, features.REAL_STD_IN),
        ("mean_target", mt, features.REAL_MEAN_TARGET), ("std_target", st, features.REAL_STD_TARGET),
    ):
        g = np.asarray(got, dtype=np.float64).reshape(-1)
        w = np.asarray(want, dtype=np.float64).reshape(-1)
        same = g.shape == w.shape and np.allclose(g, w, rtol=0, atol=1e-7)
        ok &= same
        print(f"{name}: file {g.tolist()} code {w.tolist()} {'OK' if same else 'MISMATCH'}", flush=True)
    print("normalization constants: " + ("match the official file" if ok else "MISMATCH (see above)"))


if __name__ == "__main__":
    main()
