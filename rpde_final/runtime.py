"""Shared runtime for the Decision-phase training packages (both tracks).

`train(data_dir, output_dir, device)` runs inside the organizers' container: `/data` is read-only
and holds `train_sim/`, `train_real/` and `mean_std_real.pt`; everything else must be produced
here. This module

* sets up a writable data root under `output_dir` that symlinks the read-only release, so the
  repository code (which caches under `DATA_DIR/cache` and reads eval sets from
  `LOCAL_EVAL_DIR`) runs unchanged,
* runs every training stage as a fresh `python` subprocess with the exact argv the shipping
  scripts used (one process per stage, as `scripts/ship/*.sh` did, so the RNG/CUDA state of a
  stage is the same as in the development runs),
* makes every stage resumable (a stage is skipped when its `done` marker exists), tees logs into
  `output_dir/logs/`, samples peak GPU memory, and writes `output_dir/PROVENANCE.txt`.

Nothing here imports torch: the environment variables must be in place before `rpde_t1` is
imported anywhere, and the stage subprocesses are where the GPU work happens.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]  # /workspace in the image
SRC = WORKSPACE / "src"
SCRIPTS = WORKSPACE / "scripts"
#: holds `sitecustomize.py`: makes shutil's chmod/utime failures non-fatal on mounted output
#: volumes that refuse metadata changes (see that file). On PYTHONPATH for every stage process,
#: and applied in this process too.
PYINIT = Path(__file__).resolve().parent / "pyinit"


def _tolerate_metadata_errors() -> None:
    """Same patch as pyinit/sitecustomize.py, for the orchestrating process itself."""
    import shutil as _sh

    def tolerant(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except PermissionError:
                return None

        wrapper.__name__ = fn.__name__
        return wrapper

    if not getattr(_sh.copystat, "__wrapped_tolerant__", False):
        _sh.copystat = tolerant(_sh.copystat)
        _sh.copymode = tolerant(_sh.copymode)
        _sh.copystat.__wrapped_tolerant__ = True  # type: ignore[attr-defined]


_tolerate_metadata_errors()


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{stamp} UTC] final: {msg}", flush=True)


class Runtime:
    """Paths, environment and the stage runner for one `train()` call."""

    def __init__(self, data_dir: str | os.PathLike, output_dir: str | os.PathLike, track: str):
        self.track = track
        self.data_src = Path(data_dir).resolve()
        self.out = Path(output_dir).resolve()
        self.work = self.out / "work"
        self.data = self.work / "data"
        self.runs = self.work / "runs"
        self.local_eval = self.data / "local_eval"
        self.logs = self.out / "logs"
        self.sps_cache = self.work / "sps_cache"
        self.t0 = time.time()
        self.stage_times: dict[str, float] = {}
        self.peak_mib: dict[str, int] = {}
        # A resumed run keeps the earlier runs' provenance lines (the RunPod rehearsal of
        # 2026-10-05 resumed after a crash and the first run's sha256s were overwritten).
        prov = self.out / "PROVENANCE.txt"
        self._prov_lines: list[str] = prov.read_text().splitlines() if prov.exists() else []
        if self._prov_lines:
            self._prov_lines.append(
                f"resumed {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
            )

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        for d in (self.work, self.data, self.runs, self.local_eval, self.logs, self.sps_cache):
            d.mkdir(parents=True, exist_ok=True)
        for name in ("train_real", "train_sim", "mean_std_real.pt"):
            src = self.data_src / name
            dst = self.data / name
            if not src.exists():
                raise FileNotFoundError(f"{src} is missing from the data directory")
            if dst.is_symlink() or dst.exists():
                continue
            dst.symlink_to(src)
        # A few scripts resolve `REPO_ROOT/data` and `REPO_ROOT/runs` directly (e.g. the Track-2
        # stream cache's `data/local_eval`); point those at the work dir too.
        for name, target in (("data", self.data), ("runs", self.runs), ("submissions", self.work / "submissions")):
            link = WORKSPACE / name
            if not link.exists() and not link.is_symlink():
                try:
                    link.symlink_to(target)
                except OSError as exc:  # read-only workspace: the env vars still cover the code paths
                    log(f"note: could not link {link} -> {target} ({exc})")
        (self.work / "submissions").mkdir(exist_ok=True)
        # Consumed by rpde_t1.paths at import time (src/rpde_t1/paths.py).
        self.env = dict(os.environ)
        self.env.update(
            RPDE_DATA_DIR=str(self.data),
            RPDE_LOCAL_EVAL_DIR=str(self.local_eval),
            RPDE_RUNS_DIR=str(self.runs),
            RPDE_SUBMISSIONS_DIR=str(self.work / "submissions"),
            PYTHONPATH=os.pathsep.join(
                [str(PYINIT), str(SRC)]
                + ([self.env["PYTHONPATH"]] if self.env.get("PYTHONPATH") else [])
            ),
            PYTHONUNBUFFERED="1",
            WANDB_MODE="disabled",
            TOKENIZERS_PARALLELISM="false",
        )
        # The development drivers of both tracks set these for the Track-2 chains; Track 1's
        # driver did not. Keep each track's own setting (it is only the CPU-side thread pools).
        if self.track == "t2":
            self.env.setdefault("OMP_NUM_THREADS", "1")
            self.env.setdefault("MKL_NUM_THREADS", "1")
        real = sorted(p.name for p in (self.data_src / "train_real").glob("*.h5"))
        sim = sorted(p.name for p in (self.data_src / "train_sim").glob("*.h5"))
        log(f"data: {len(real)} real cells, {len(sim)} sim cells; work dir {self.work}")
        self.prov("track", self.track)
        self.prov("data_dir", str(self.data_src))
        self.prov("n_real_files", str(len(real)))
        self.prov("n_sim_files", str(len(sim)))
        self.prov("has_7575_0", str("7575_0.h5" in real))
        self.prov("python", sys.version.split()[0])
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            gpu = "nvidia-smi unavailable"
        self.prov("gpu", gpu)
        log(f"gpu: {gpu}")

    # ------------------------------------------------------------- provenance
    def prov(self, key: str, value: str) -> None:
        self._prov_lines.append(f"{key} {value}")
        (self.out / "PROVENANCE.txt").write_text("\n".join(self._prov_lines) + "\n")

    def sha(self, path: str | os.PathLike, key: str | None = None) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        self.prov(key or f"sha256 {Path(path).relative_to(self.out)}", digest)
        return digest

    # ------------------------------------------------------------ stage runner
    def done_marker(self, name: str) -> Path:
        return self.logs / f"{name}.done"

    def is_done(self, name: str) -> bool:
        return self.done_marker(name).exists()

    def mark_done(self, name: str) -> None:
        self.done_marker(name).write_text(json.dumps({
            "seconds": round(self.stage_times.get(name, 0.0), 1),
            "peak_gpu_mib": self.peak_mib.get(name),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))

    def run(self, name: str, argv: list[str], *, cwd: Path | None = None, gpu: bool = True) -> None:
        """Run one stage as a subprocess; skip when its marker exists."""
        if self.is_done(name):
            log(f"stage {name}: already done, skipped")
            try:  # carry the earlier run's time and peak memory into this run's summary
                rec = json.loads(self.done_marker(name).read_text())
                self.stage_times.setdefault(name, float(rec.get("seconds") or 0.0))
                if rec.get("peak_gpu_mib"):
                    self.peak_mib.setdefault(name, int(rec["peak_gpu_mib"]))
            except (OSError, ValueError):
                pass
            return
        argv = [str(a) for a in argv]
        log(f"stage {name}: {' '.join(argv)}")
        logfile = self.logs / f"{name}.log"
        t0 = time.time()
        monitor = _GpuMonitor() if gpu else None
        if monitor:
            monitor.start()
        with open(logfile, "a") as lf:
            lf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(argv)}\n")
            lf.flush()
            proc = subprocess.Popen(
                argv, cwd=str(cwd or WORKSPACE), env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                sys.stdout.write(line)
            sys.stdout.flush()
            rc = proc.wait()
        if monitor:
            monitor.stop()
            self.peak_mib[name] = monitor.peak_mib
        dt = time.time() - t0
        self.stage_times[name] = dt
        if rc != 0:
            raise RuntimeError(f"stage {name} failed with exit code {rc}; see {logfile}")
        peak = f", peak GPU {monitor.peak_mib} MiB" if monitor and monitor.peak_mib else ""
        log(f"stage {name}: done in {dt / 60:.1f} min{peak}")
        self.prov(f"stage {name} minutes", f"{dt / 60:.1f}")
        if monitor and monitor.peak_mib:
            self.prov(f"stage {name} peak_gpu_mib", str(monitor.peak_mib))
        self.mark_done(name)

    def py(self, *args: str | os.PathLike) -> list[str]:
        return [sys.executable, *map(str, args)]

    def module(self, mod: str, *args: str | os.PathLike) -> list[str]:
        return [sys.executable, "-m", mod, *map(str, args)]

    def script(self, rel: str, *args: str | os.PathLike) -> list[str]:
        return [sys.executable, str(SCRIPTS / rel), *map(str, args)]

    # ----------------------------------------------------------------- helpers
    def rm(self, *paths: str | os.PathLike) -> None:
        for p in paths:
            p = Path(p)
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists() or p.is_symlink():
                p.unlink()

    def summary(self) -> str:
        total = (time.time() - self.t0) / 60
        stages = sum(self.stage_times.values()) / 60
        lines = [
            f"total wall time {total:.1f} min (this run)",
            f"sum of stage times {stages:.1f} min (all runs, resumed stages included)",
        ]
        for k, v in self.stage_times.items():
            lines.append(f"  {k}: {v / 60:.1f} min, peak GPU {self.peak_mib.get(k)} MiB")
        peak = max((v for v in self.peak_mib.values() if v), default=None)
        lines.append(f"peak GPU memory over all stages: {peak} MiB")
        return "\n".join(lines)


class _GpuMonitor:
    """Samples `nvidia-smi` memory.used every few seconds while a stage runs."""

    def __init__(self, period: float = 5.0):
        self.period = period
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            for tok in out.split():
                self.peak_mib = max(self.peak_mib, int(float(tok)))
        except Exception:  # noqa: BLE001
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.period)
        self._sample()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)
