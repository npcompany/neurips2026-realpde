# Light ViT and Careful Transfer for Sim-to-Real Flow Forecasting

This is the solution released by **np-user**, a team representing [NPco](https://www.npco.ai/) and
composed of Benjamin Adjadj (benjamin@npco.ai) and Bruno Régaldo-Saint Blancard (bruno@npco.ai),
for the [NeurIPS 2026 RealPDE Competition](https://realpdecompetition.github.io/):
**1st place on Track 1 (simulation-to-real transfer)** and **3rd place ex aequo on Track 2
(long-term test-time adaptation)** ([final results](https://realpdecompetition.github.io/results.html)).

| Track | Place | Final score (mean +- sd over 15 evaluations) |
|---|---|---|
| 1: Sim2Real | 1st | 82.9555 +- 0.0281 |
| 2: LTTTA | 3rd (tied) | 82.9560 +- 0.0226 |

The organizers retrained each package from scratch three times and evaluated each trained model
five times. This repository is the code of those two packages; only their training logs are left
out.

## The task

Forecast 20 future frames of real PIV measurements of the flow around an airfoil (32 x 64 grid,
velocity `u, v`) from 20 observed frames, with pointwise `lower`/`upper` bounds. The training data
is 100 simulated (CFD) operating points and 81 real ones, each a (Reynolds number, angle of attack)
pair. The score combines relative L2 error, turbulent kinetic energy, mean velocity profile error,
inference time, and an interval score (SPS). Track 1 predicts a whole batch in one call. Track 2
serves the same problem as a batch-1 stream that reveals each target after the prediction.

## Method in one paragraph

A small ViT (4.6 M parameters) predicts the 20 future frames as a residual on the last observed
frame. Each token is a 2 x 2 pixel patch spanning all 20 input frames, so attention is spatial
only, and the time axis lives in the channels. A convolutional stem and a full-resolution
convolutional refinement decoder sit on either side of the 6 transformer blocks. Every block is
conditioned (adaLN, FiLM in the stem) on the operating point, which is **estimated from the input
window itself** by a fixed linear estimator rather than taken from a label, so training and
inference see the same quantity. The transfer recipe has four stages:

1. Pretrain on the simulation resampled in time to the real frame clock.
2. Train jointly on real and simulated data, 50/50 per batch.
3. Fine-tune on real data with drop-path 0.3 and EMA.
4. Fine-tune again with fluctuation-amplitude augmentation, angle-of-attack jitter, R-drop, a
   Huber/MSE loss mix, and a credit term from a teacher interval head.

Two small post-hoc heads finish the model:

- An **interval head** with a calibrated asymmetric width rule produces the bounds.
- A **second-moment head** corrects the energy of the predicted fluctuations.

Inference runs as a single CUDA graph per call. Track 2 uses the same backbone on a finer token
grid (1 x 2 patches). Two extra input frames carry the running mean and standard deviation of the
trajectory so far, and a small recurrent calibrator rescales the interval from the stream's recent
errors. Track 2 makes no test-time weight updates: a frozen model with streaming state beat every
adaptation scheme we measured.

Full details, effect sizes for each component and references:
**[`track1/METHOD.md`](track1/METHOD.md)** and **[`track2/METHOD.md`](track2/METHOD.md)**.

## Repository layout

```
src/rpde_t1/      shared library: data, models (stformer.py is the ViT), training, scoring, packaging
src/rpde_t2/      streaming additions for Track 2
rpde_final/       the training chains train() runs, stage by stage (recipes.py)
scripts/          the helper scripts the chains invoke (sim resampling, interval heads, packaging)
track1/ track2/   solution.py (train / load_predictor / get_ttt_model), Dockerfile, README.md,
                  METHOD.md
data_prep/        downloads the public release into the layout train() expects
build.sh          assembles dist/<track>/, the self-contained package, and builds its image
```

The model class is called `STFormer` for historical reasons: it began as a space-time transformer
before each token was widened to span the whole 20-frame window.

## Reproduce

Requirements: one GPU with at least 16 GB of memory (we trained on a T4, an A10G and an A100),
Docker with the NVIDIA runtime, and about 50 GB of disk for Track 1 or 80 GB for Track 2.

1. **Organizers' kit files.** The packages import the official scorer and copy parts of the
   official starting kit, which we do not redistribute. Download the Track 1 starting kit (v9)
   and the Track 2 starting kit (v6) from the competition's Codabench pages
   ([Track 1](https://www.codabench.org/competitions/17363/),
   [Track 2](https://www.codabench.org/competitions/17385/)) and unpack them as
   `vendor/realpde_t1_starting_kit_v9/` and `vendor/realpde_t2_starting_kit_v6/`.
   `vendor_manifest.txt` lists the files used.
2. **Data** (about 16 GB):
   ```bash
   data_prep/prepare_data.sh /path/to/data   # train_sim/, train_real/ (81 files), mean_std_real.pt
   ```
3. **Build and train** (on one A100 80 GB, Track 1 took 2.7 h and Track 2 6.7 h):
   ```bash
   ./build.sh track1 realpde-t1
   docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
       -v /path/to/data:/data:ro -v /path/to/out:/output realpde-t1 \
       python -c "import solution; solution.train('/data', '/output', 'cuda')"
   ```
   `train()` is resumable: rerunning the same command skips finished stages. It writes the
   served bundle, per-stage logs, `PROVENANCE.txt` (sha256 of every checkpoint) and
   `TRAIN_SUMMARY.txt`. Track 2 is the same with `track2`.
4. **Predict.** Track 1: `solution.load_predictor('/output', 'cuda').predict(x)` on a float32
   `(N, 20, 32, 64, 3)` array returns `prediction`, `lower` and `upper`. Track 2:
   `solution.get_ttt_model('/output', 'cuda')` follows the official streaming interface. See each track's
   `README.md`.

Seeds are fixed and cuDNN is deterministic, so a run is bit-reproducible on the same GPU type. A
different GPU type is a fresh draw of the same recipe; our measured run-to-run spread is about
0.02 final-score points.

## Notes

- Code comments cite numbered sections ("FINDINGS 124", "§219") of our internal experiment
  log. It is not part of this repository; the METHOD files summarise the evidence it holds.
- The code under `src/` is pruned to the modules these two chains execute. Some options and
  branches in it belong to experiments that did not make the final recipe.

## License

MIT for our code (see `LICENSE`). The organizers' starting-kit files are not included and remain
under their own terms.
