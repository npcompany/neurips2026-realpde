# RealPDE Decision phase, Track 1 training package (team np-user).
# Base: the official Track-1 evaluation image (torch 2.2.2, CUDA 12.1, Python 3.10).
# Added: h5py (reads the released .h5 trajectories) and ruff (the packaging step
# scripts/ship/clean_bundle.py formats the pruned inference bundle with `ruff format`; it is
# not imported by any training or inference code). Nothing else is installed.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
RUN pip install --no-cache-dir h5py==3.11.0 ruff==0.16.1
WORKDIR /workspace
COPY . /workspace
ENV PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=1
