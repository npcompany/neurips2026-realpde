# RealPDE Decision phase, Track 2 training package (team np-user).
# Base: the official Track-2 evaluation image, by digest (torch 2.2.2, CUDA 12.1, Python 3.10,
# h5py 3.16.0 already present). Added: ruff, which the packaging step at the end of train()
# (scripts/ship/clean_bundle.py) runs as a subprocess to drop the unused import lines of the
# pruned inference bundle; it is not imported by any training or inference code. Nothing else.
FROM w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972
RUN pip install --no-cache-dir ruff==0.16.1
WORKDIR /workspace
COPY . /workspace
ENV PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
