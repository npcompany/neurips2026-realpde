# np-user: Track 2 training package

## Codabench
- Username: np-user
- Submission id this package reproduces: 948730 (`t2_0927_final`, submitted 2026-09-27)
- final_score of that submission: 82.450962 (rel_l2 94.904762, tke 80.627427, mvpe 94.415925, time 92.648529, sps 45.396755)

## How to reproduce
- Build command: `docker build -t <name> .` (no difference)
- Train command: the default,
  `docker run --rm --network none --gpus '"device=0"' --shm-size 32g -v <train_data>:/data:ro -v <out>:/output IMAGE python -c "import solution; solution.train('/data', '/output', 'cuda')"`.
  `train()` is resumable: a second run on the same `/output` skips every finished stage
  (`/output/logs/<stage>.done`). It writes `/output/t2_bundle/` (what `get_ttt_model` serves),
  `/output/logs/`, `/output/PROVENANCE.txt` (sha256 of every checkpoint, stage durations, peak GPU
  memory) and `/output/work/` (data caches, eval sets, intermediate runs; about 15 GB).
  `t2_bundle/` is the packaged bundle, pruned to the modules it executes as the Codabench zip was
  (`PROVENANCE.txt` says if the unpruned bundle, same weights and outputs, was kept instead).
- `get_ttt_model('/output', 'cuda')` returns the Codabench object: `reset_ttt_state()` and
  `ttt_step(input_norm, prev_target_norm=None) -> (pred_norm, {"adapt_loss": None, "lower": ..., "upper": ...})`,
  normalized in and out with the official statistics, bounds on every step.

## Hardware your submitted model was trained on
The Codabench submission was built across our development machines, one GPU per stage (the
stage list is in METHOD.md section 4):
- GPU model and count: one NVIDIA A10G (24 GB) per stage (two A10G machines were used for
  different stages, and the three train-split chains behind the interval calibrator ran one per
  machine on two A10G and one Tesla T4).
- CPU: 8 vCPU (A10G boxes, AWS g5.2xlarge); 4 vCPU (T4 box)
- Memory: 32 GB (A10G boxes); 16 GB (T4 box)
- Training time on this hardware: main chain about 2.3 h of A10G time (pretrain 16 min, joint 8 min,
  fine-tune 55 min, interval head 8 min, second-moment head 2 min, packaging and checks about 40 min),
  plus the three train-split calibration chains at about 1.5 h each and the calibrator fit (minutes).
- Peak GPU memory on this hardware: under 8 GB in every stage (batch 16, fp16 autocast).
- This package, end to end from scratch on one A100 80 GB (our rehearsal of the organizers' run,
  2026-10-05/06): 6.7 h of wall time, of which 33 min of data
  preparation on a slow network volume, 1.8 h for the served chain, 3.9 h for the three
  calibration chains, 8 min for the calibrator fit.
  Peak GPU memory 6.5 GB. The produced package scored, through the kit's `local_eval.py` on our
  two local streams, within 0.04 points per subscore of the Codabench artifact (rel_l2 96.184 vs
  96.183, tke 83.064 vs 83.095, mvpe 96.775 vs 96.757, sps 58.451 vs 58.47 on the unseen-Reynolds
  stream), at the same speed (2.16 vs 2.14 ms/step on an A100): inside the seed spread.

## Environment
- Base image: `w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972`
  (torch 2.2.2, CUDA 12.1, Python 3.10.14, h5py 3.16.0, numpy 1.26.4)
- Added packages and versions (pinned the same way in the Dockerfile): `ruff==0.16.1`, which the
  packaging step at the end of `train()` runs as a subprocess on the pruned inference bundle
  (unused-import removal and formatting); it is never imported. The training and inference code
  imports torch, numpy and h5py only.

## Data, weights and code
- Training data used: the public release only, `train_sim/` (100 cells) and `train_real/`
  (81 cells; the duplicate `7575_0.h5` is excluded by the code, `src/rpde_t1/data/splits.py`,
  so its absence from `/data` changes nothing). The official `mean_std_real.pt` is used for the
  normalized/physical bridge; its values are hard-coded in `src/rpde_t1/features.py` and checked
  against the file at the start of `train()`.
- Augmentation: in the real fine-tunes, a log-normal rescaling of the fluctuating part of input
  and target windows (sigma 0.2 at Re 3750 tapering to 0 at Re 26700) and a uniform +-2.5 degree
  jitter of the angle-of-attack conditioning. The sim pretrain uses the simulation resampled to
  the real frame clock (`sim_units`). No other augmentation.
- Official baseline checkpoints used, if any: none. Every checkpoint is produced by `train()`.
- Third-party code and its license: PyTorch (BSD-3), NumPy (BSD-3), h5py (BSD-3), ruff (MIT,
  packaging step only). The files under `vendor/` are the organizers' starting kits, unchanged
  (the scorer, the kit payload the packager copies into a bundle, and the Track-2 kit's
  `local_eval.py` the final check runs). All other code is ours (team np-user).

## Randomness
- Seed(s): 0 for the served chain; the three calibration chains use seeds 0, 1 and 2 (`--seed`,
  deterministic cuDNN, math-kernel attention). Runs are bit-reproducible on the GPU type that
  produced them; the A800 is a fresh draw of the same recipe. Our measured run-to-run spread of the
  full Track-2 chain, fresh pretrain seed included, is about 0.011 to 0.014 final-score points
  (standard deviation over 5 seeds on our held-out splits), so the reproduced score is expected
  within a few hundredths of 82.451.
- How runs or checkpoints are selected, if at all: no selection between served runs; the fine-tunes'
  EMA weights are used. The interval head's fit uses a selection rule on splits of the training
  release only (`train_ends`, `val_interp`, `val_time`), the second-moment head stops on `val_time`
  (the last quarter of 20 training trajectories). In each calibration chain the interval head is
  fitted at three seeds and the one with the best mean interval score on those same training-data
  splits is kept. All of it runs inside `train()`.

## Track 2 only
- Does the method call the LLM gateway: no. The method is a frozen port: no test-time parameter
  updates, no external calls. The only state that evolves along a trajectory is the running
  mean/std of the revealed frames (fed to the model as two extra input frames) and the interval
  calibrator's small recurrent state; both reset in `reset_ttt_state()`.

## Contact
- Team np-user, competition@npco.ai (reachable during the Decision Phase, replies within 48 hours)

## Notes
- If the calibration chains do not finish inside the time budget (a cutoff at 17 h of elapsed
  training), `train()` ships the same backbone without the interval calibrator and says so in
  `PROVENANCE.txt`. The whole chain took 6.7 h on an A100, so this is a safeguard, not the
  expected path.
- `train()` runs every stage as a fresh `python` subprocess with the command line our development
  scripts used (`scripts/ship/t2_ship_final.sh`, `scripts/t2/t2full_20260927.sh`,
  `scripts/t2/t2full_recal.sh` in our repository); the orchestration is `rpde_final/recipes.py`.
- `src/` is our library pruned to what the two training chains run: the modules an import
  trace of a smoke run of each chain (same stages and flags, a few steps each) loaded, with the
  branches that reached the dropped modules (other backbones, other training modes, development
  reports) removed and replaced by an explicit error. The pruned package reproduces the unpruned
  one's checkpoints bit for bit in the same smoke run (same hashes in `PROVENANCE.txt`).
  `vendor/` holds the organizer-kit files the chain imports or copies.
