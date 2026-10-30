# np-user: Track 2 method

## 1. Summary
The same simulation-to-real forecasting model as our Track-1 entry (a 4.6 M-parameter space-time
transformer conditioned on the operating point), on a finer token grid and fine-tuned with two
extra input frames that carry the running mean and standard deviation of the frames revealed so
far on the trajectory, served as a frozen port: no test-time parameter updates. The only other
use of the stream's history is a small learned recurrent calibrator that rescales and shifts the
interval from the previous windows' realised errors. What matters most for the score: the
backbone and its transfer recipe (shared with Track 1), the interval head, width rule and
calibrator (SPS), the second-moment head (TKE), and a batch-1 serving path that runs one CUDA
graph of an fp16 pipeline per step (time).

## 2. Model
- Architecture and number of parameters: four parts, 4.70 M parameters in all, each fitted in
  turn by `train()` (section 4): the backbone below (4.65 M); a post-hoc interval head (47 234
  parameters plus a fitted width-rule table) and a learned interval calibrator (3 448 parameters,
  one GRU cell) that produce the bounds (section 5); a post-hoc second-moment head (1 105
  parameters) that corrects the energy of the fluctuating part of the prediction (section 6).
- Backbone: `stformer`, `dim=192, depth=6, heads=6, cond_conv=true,
  patch=20x1x2`, 4.65 M parameters (the Track-1 architecture of our METHOD there, with 1024 spatial
  tokens instead of 512, and a stem widened by two input frames for the trajectory state).
- Inputs and outputs, including preprocessing and normalization: the evaluator hands
  `input_norm = (x - mi) / si` (official statistics). Inside the pipeline the window is mapped back
  to physical units (`x * si + mi`), the operating point is estimated from the window by the same
  fixed linear estimator as Track 1 (averaged over a trajectory's first 3 windows, then frozen for
  the trajectory), the window is normalized by its free-stream speed, and two pseudo-frames (the
  running mean and running std of all frames revealed so far on this trajectory, causal) are
  prepended. The prediction is a residual on the last frame; prediction and both bounds are mapped
  to normalized target space (`(y - mt) / st`) before being returned. Pressure is zero.
- Starting point: random initialization. No official baseline checkpoint is used anywhere.

## 3. Data
- Training data used (`train_sim`, `train_real`) and how the two are combined: as in Track 1: sim
  resampled to the real frame clock (`sim_units`) for the pretrain, a 50/50 real + sim joint stage,
  then real-only fine-tunes on all 81 cells. The fine-tunes add the trajectory-state pseudo-frames,
  computed from each training window's own trajectory up to the window's position (causal), so
  training sees the state the stream will provide.
- Validation split carved out of the training data, if any: the served chain is trained on all 81
  real cells (as the Codabench row was). Three training-data splits are used for post-hoc
  fitting/stopping only (`train_ends`, `val_interp`, `val_time`; see the Track-1 METHOD). The
  **interval calibrator** is the exception: it needs genuinely held-out streams, so `train()` also
  runs three auxiliary chains (seeds 0, 1, 2) on the 69-cell `train` split (Re 3750, 16500, 26700
  held out), streams their held-out cells (`val_interp`: Re 16500; `val_extrap`: Re 3750 and
  26700) through the full serving path, and fits one calibrator on those six streams. The served
  backbone itself never sees the calibrator's fit data; the calibrator never sees the served
  backbone (we measured that it transfers across backbones).
- Augmentation: fluctuation-amplitude rescaling tapered in log Re, +-2.5 degree conditioning
  jitter (README). No R-drop and no Huber/MSE mix in this track's recipe.
- Data left out, and why: `7575_0.h5` (duplicate of `6300_0.h5`), excluded by the code.

## 4. Training
Common settings as in Track 1: batch 16, Adam (no weight decay), 200 warm-up steps then cosine
decay to 0.1x the peak learning rate, gradient clipping 1.0, fp16 autocast with a GradScaler,
deterministic cuDNN. "Huber" is smooth-L1 with delta 0.005 on the normalized residual over u and
v. The loss terms and augmentation marked (a), (b), (d) are the ones defined in our Track-1
METHOD section 4: (a) the two TKE terms (0.02 and 0.0038), (b) the band credit through a frozen
"teacher" interval head fitted on a plain fine-tune, (d) the fluctuation-amplitude rescaling
tapered in log Re and the +-2.5 degree conditioning jitter. Stages 1 to 6 build the served
model; stages 7 and 8 build the interval calibrator on held-out streams.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1. sim pretrain | sim_units, 100 sim cells, stride-1 windows | Huber on the residual | Adam, peak 1e-3, warm-up 200, cosine to 1e-4 | 8 000 | 16 |
| 2. joint | 50 % real (81 cells) + 50 % sim_units; init from stage 1 | Huber | Adam, peak 3e-4, cosine | 4 000 | 16 |
| 3. plain fine-tune (teacher's backbone, not served) | real, 81 cells, with the trajectory-state frames; init from stage 2 | Huber + TKE terms (a); augmentation (d); drop-path 0.3, EMA 0.999 | Adam, peak 3e-4, cosine | 24 000 | 16 |
| 3b. teacher interval head | stage-3 predictions on all 81 cells | L1 on the log error scale | Adam 5e-4 | 24 000 | 4 |
| 4. credited fine-tune (served) | as stage 3; init from stage 2 | as stage 3 + band credit 0.08 (b) | Adam, peak 4.5e-4, cosine | 24 000 | 16 |
| 5. interval head + width rule | stage-4 EMA predictions on all 81 cells | L1 on the log error scale; width rule by the scorer's credit (section 5) | Adam 5e-4 | 24 000 | 4 |
| 6. second-moment head | stage-4 bundle, 81 cells (the val_time cells without their last quarter) | the scorer's TKE error of the corrected prediction | Adam 3e-3 | 3 000 (stop on val_time) | - |
| 7. three calibration chains | the 69-cell `train` split (Re 3750, 16500, 26700 held out) | stages 1 to 6 again, see below | as above | as above | as above |
| 8. interval calibrator | the six held-out streams of stage 7 | smoothed SPS credit of the calibrated interval, see below | Adam 3e-3 | 1 000 | one backward per stream per step |

- Stage 7, one chain per seed k = 0, 1, 2: pretrain with seed k (k = 0 reuses stage 1), joint,
  then on the 69 train cells a plain fine-tune and its teacher head (seed 0, once, fitted on the
  69 cells), the credited fine-tune with seed k, the interval head (fitted at three seeds, the
  one with the best mean interval score on `train_ends`, `val_interp`, `val_time` kept, as our
  development scripts did) and the second-moment head. Each chain's held-out cells
  (`val_interp`: Re 16500; `val_extrap`: Re 3750 and 26700) are then streamed through the full
  serving path at shrink 0.625, which gives six held-out streams with their realised errors.
- Stage 8 fits the calibrator of section 5 on those streams: the loss is the scorer's per-element
  interval credit with the inside-the-band indicator relaxed to a sigmoid (temperature annealed
  from 0.2 to 0.02), windows weighted by `1 - accuracy factor`.
- How runs or checkpoints are selected, if at all: none between served runs (EMA weights at the
  last step). Post-hoc fits use training-data splits only (section 3), except the calibrator,
  which by design uses the held-out streams of the auxiliary chains. The teachers are interval
  heads of plain fine-tunes; their fit data is what matters, not their weights.
- Seeds, and the run-to-run spread you have seen: seed 0 for the served chain, 0, 1 and 2 for the
  calibration chains. Full Track-2 chain with a fresh pretrain seed: standard deviation 0.011
  final-score points on our unseen-Reynolds validation split and 0.014 on the hard cells (5
  seeds); pretrain fixed: 0.008 / 0.011.

## 5. Uncertainty bounds
Returned on every step through `info["lower"]` / `info["upper"]`, normalized target space.
Three parts, all post-hoc:
- the **interval head** (per-pixel error scale from the prediction's own features, FiLM on the
  estimated operating point) and the calibrated **asymmetric width rule** (64 bins of
  `log(scale/sigma)` x channel x lead, multipliers maximising the scorer's per-element credit), as
  in our Track-1 METHOD section 5, fitted on the served backbone's predictions over all 81 cells;
- the **learned interval calibrator**: a GRU cell of hidden size 16 shared across every pixel and
  channel, which after each revealed window folds in that window's realised interval statistics
  per lead bucket (mean `log|err|/h`, coverage at 1 h, mean `log(h/sigma)`, mean signed `err/h`)
  and outputs, for the next window, a log multiplier in [1/2, 2] on the half-width and a centre
  shift of up to 0.5 half-widths (both zero-initialised, so an untrained calibrator is the
  identity). Window k is calibrated from windows before k only; the state resets with the
  trajectory. Fitted once on held-out streams (section 4, stage 8), fused into the CUDA graph.
- `lower <= upper` is enforced on the device; the prediction is never shifted.

## 6. Inference
- Track 2: the adaptation rule in `ttt_step`: **no parameters are updated** (`adapt_loss` is
  `None`). Per step: the normalized window is uploaded to a pinned buffer, mapped to physical units,
  the operating point is estimated (first 3 windows, then frozen), the running mean/std of the
  revealed frames is updated and prepended as two pseudo-frames, and one CUDA graph runs the fp16
  pipeline: backbone, interval head, width rule, second-moment correction (shrink 0.625), the
  calibrator's gated observe-and-apply, the map back to normalized space and the finiteness check.
  Outputs are fp16 tensors (the evaluator upcasts). A non-finite fp16 step falls back to eager
  fp32 for the rest of the run. `reset_ttt_state` zeroes the trajectory state, the operating-point
  cache and the calibrator state (O(1), no reload). `get_ttt_model` builds the graph and runs two
  synthetic trajectories (52 steps) untimed so the first scored step pays no one-time cost.
  `prev_target_norm` is not used directly: at stride 20 it equals the current input window, from
  which the calibrator recovers the previous target.

## 7. Speed
Batch-1 serving on the leaderboard went from 15.4 ms/step (a plain port of the Track-1 bundle) to
about 4.6 ms/step: GPU-native pipeline (no host round trips), TF32 matmuls, a single CUDA graph of the
eager pipeline (no compiler; the evaluation image has none), the fp16 "fast pipeline" with
half-precision weights (bit-identity-checked against the reference on its first call), pinned host
I/O, the normalized<->physical map folded into the copies, the port warm-up at load. The finer
token grid and the calibrator cost about +0.3 ms/step and were priced against their SPS/accuracy
gains before shipping. `TORCHINDUCTOR_COMPILE_THREADS=1` is forced because the participant runs in
a daemonic process.

## 8. What mattered
Rough effect sizes (final-score points, held out, unless stated; leaderboard noise is about
+-0.07 points per submission):
- the shared backbone and transfer recipe (see the Track-1 METHOD): the bulk of the score.
- batch-1 time: about +5 time subscore points (about +0.5 final) from 15.4 to about 4.6 ms/step,
  mostly the GPU-native graph and the fp16 fast pipeline.
- trajectory-state pseudo-frames (running mean/std of the revealed frames) in the fine-tune: the
  mechanism that absorbs the mean-flow drift; every explicit streaming correction (mean-bias
  blending, test-time updates) then measured as a wash on this backbone and is off.
- interval head + width rule, refit on the stream-state backbone: several SPS points; the finer
  token grid + lr 4.5e-4 + credit 0.08 + calibrator: +0.049 / +0.026 (unseen-Reynolds split /
  hard cells) net of their time cost, at three paired seeds, of which the calibrator +0.08 to
  +0.11 sps points.
- second-moment head at shrink 0.625: +2 to +4 tke subscore points; the shrink trade (0.5 ->
  0.625) is worth a few thousandths of final here.
- angle jitter 2.5, fluctuation augmentation taper, band credit: +0.01 to +0.03 each.

## 9. Compute
Codabench run: main chain about 2.3 h of A10G time (one GPU per stage), three calibration chains at
about 1.5 h each (two A10G, one T4) and the calibrator fit; peak GPU memory under 8 GB. This
package from scratch on one A100 80 GB (rehearsal, 2026-10-05/06): 6.7 h wall time in one run,
including 33 min of data preparation on a network volume; main chain 1.8 h, the three calibration
chains 3.9 h, calibrator fit 8 min; peak GPU memory 6.5 GB; the produced package reproduces the
Codabench artifact's scores on our two local streams within 0.04 points per subscore. The A800 has the same compute as the A100, so the 24 h budget holds
with a wide margin and the 17 h calibration cutoff is never reached.

## 10. References
- As in the Track-1 METHOD (transformer, ViT, adaLN, early convolutions, stochastic depth, EMA,
  FiLM).
- Cho et al., Learning phrase representations using RNN encoder-decoder (GRU, 2014) for the
  calibrator cell.
- PyTorch 2.2.2 (CUDA graphs: `torch.cuda.CUDAGraph`), NumPy 1.26, h5py; the organizers' Track-2
  starting kit (`vendor/`) for the scorer and the kit payload the packager copies.
