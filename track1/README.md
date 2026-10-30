# np-user: Track 1 training package

## Codabench
- Username: np-user
- Submission id this package reproduces: 948701 (`t1_0927_mix`, submitted 2026-09-27)
- final_score of that submission: 82.115288 (rel_l2 94.759456, tke 79.785085, mvpe 94.258036, time 92.982946, sps 44.576571)

## How to reproduce
- Build command: `docker build -t <name> .` (no difference)
- Train command: the default,
  `docker run --rm --network none --gpus '"device=0"' --shm-size 32g -v <train_data>:/data:ro -v <out>:/output IMAGE python -c "import solution; solution.train('/data', '/output', 'cuda')"`.
  `train()` is resumable: a second run on the same `/output` skips every finished stage
  (`/output/logs/<stage>.done`). It writes `/output/t1_bundle/` (what `load_predictor` serves),
  `/output/logs/` (one log per stage), `/output/PROVENANCE.txt` (sha256 of every checkpoint,
  stage durations, peak GPU memory) and `/output/work/` (data caches, eval sets, intermediate runs;
  about 12 GB).
- `load_predictor('/output', 'cuda').predict(x)` takes a float32 `(N, 20, 32, 64, 3)` array in
  physical units, any N, and returns `{"prediction", "lower", "upper"}` as float32 arrays of the same
  shape, pressure zeroed.

## Hardware your submitted model was trained on
The Codabench submission was built across our development machines, stage by stage (the stage
list is in METHOD.md section 4):
- GPU model and count: sim pretrain on one NVIDIA Tesla T4 (16 GB); joint stage, both fine-tunes,
  the heads and the packaging on one NVIDIA A10G (24 GB) each, one GPU per stage.
- CPU: 4 vCPU (T4 box, AWS g4dn.xlarge) and 8 vCPU (A10G boxes, AWS g5.2xlarge)
- Memory: 16 GB (T4 box), 32 GB (A10G boxes)
- Training time on this hardware: about 3 h of GPU time end to end: pretrain 12 min (T4), joint
  6 min, plain fine-tune 38 min, teacher head 8 min, credited fine-tune 65 min, interval head 8 min,
  second-moment head 2 min, packaging and checks about 30 min (A10G).
- Peak GPU memory on this hardware: under 8 GB in every stage (batch 16, fp16 autocast).
- This package, end to end from scratch on one A100 80 GB (our rehearsal of the organizers' run,
  2026-10-05): 2.7 h of wall time, of which 31 min of data preparation
  on a slow network volume. Peak GPU memory 4.3 GB.
  The produced model scored, through the kit's scorer on our two local validation sets, within
  0.09 points per subscore of the Codabench artifact: inside the seed
  spread.

## Environment
- Base image: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` (torch 2.2.2, CUDA 12.1, Python 3.10)
- Added packages and versions (pinned the same way in the Dockerfile): `h5py==3.11.0`, and
  `ruff==0.16.1`, a code formatter the packaging step runs on the pruned inference bundle (a
  subprocess, never imported). Nothing else; the training and inference code imports torch, numpy
  and h5py only.

## Data, weights and code
- Training data used: the public release only, `train_sim/` (100 cells) and `train_real/`
  (81 cells; the duplicate `7575_0.h5` is excluded by the code, `src/rpde_t1/data/splits.py`,
  so its absence from `/data` changes nothing). The official `mean_std_real.pt` statistics are
  hard-coded in `src/rpde_t1/features.py` and checked against the file at the start of `train()`.
- Augmentation: in the real fine-tunes, a log-normal rescaling of the fluctuating part of input
  and target windows (sigma 0.2 at Re 3750 tapering to 0 at Re 26700), a uniform +-2.5 degree
  jitter of the angle-of-attack conditioning, R-drop consistency (two stochastic-depth forward
  passes per step, MSE between their predictions). The sim pretrain uses the simulation resampled
  to the real frame clock (`sim_units`). No other augmentation.
- Official baseline checkpoints used, if any: none. Every checkpoint is produced by `train()`.
- Third-party code and its license: PyTorch (BSD-3), NumPy (BSD-3), h5py (BSD-3), ruff (MIT,
  packaging step only). The files under `vendor/` are the organizers' starting kits, unchanged
  (their scorer, which the interval-head fit and the final check read, and the kit payload the
  packager copies into a bundle). All other code is ours (team np-user).

## Randomness
- Seed(s): 0 at every stage (`--seed 0`, deterministic cuDNN, math-kernel attention). Runs are
  bit-reproducible on the GPU type that produced them; a different GPU type (the A800 against our
  T4/A10G) is a fresh draw of the same recipe. Our measured run-to-run spread of the full chain,
  fresh pretrain seed included, is about 0.02 final-score points (standard deviation over 5 seeds
  on our held-out splits), so the reproduced score is expected within a few hundredths of 82.115.
- How runs or checkpoints are selected, if at all: no selection between runs. Inside each stage the
  last checkpoint is used (the EMA weights of the fine-tunes). The interval head's fit uses a
  stopping/selection rule on training-data splits only (`train_ends`, `val_interp`, `val_time`
  are all cells of the training release), and the second-moment head stops on the `val_time`
  split, the last quarter of 20 training trajectories. All of it runs inside `train()`.

## Track 2 only
- Does the method call the LLM gateway: not applicable (Track 1).

## Contact
- Team np-user, competition@npco.ai (reachable during the Decision Phase, replies within 48 hours)

## Notes
- `src/` is our library pruned to what the two training chains run: the modules an import
  trace of a smoke run of each chain (same stages and flags, a few steps each) loaded, with the
  branches that reached the dropped modules (other backbones, other training modes, development
  reports) removed and replaced by an explicit error. The pruned package reproduces the unpruned
  one's checkpoints bit for bit in the same smoke run (same hashes in `PROVENANCE.txt`).
  `vendor/` holds the organizer-kit files the chain imports or copies.
- `train()` runs every stage as a fresh `python` subprocess with the command line our development
  scripts used (`scripts/ship/t1_ship_final.sh` in our repository); the orchestration is
  `rpde_final/recipes.py`.
