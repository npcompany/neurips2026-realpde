# np-user: Track 1 method

## 1. Summary
A small space-time transformer (4.6 M parameters), conditioned on the operating point, predicts
the 20 future frames as a residual on the last observed frame. It is pretrained on the simulation
resampled to the real frame clock, trained jointly on real and simulation data, then fine-tuned on
all 81 real cells; two small post-hoc heads build the interval (`lower`/`upper`) and correct the
predicted fluctuation energy. What matters most for the score: the simulation-to-real transfer
recipe (accuracy), the fitted interval head with its calibrated width rule (SPS), the
second-moment head (TKE), and serving each call as a single CUDA graph (time).

## 2. Model
- Architecture and number of parameters: three parts, 4.68 M parameters in all, each fitted in
  turn by `train()` (section 4): the backbone below (4.63 M); a post-hoc interval head (47 234
  parameters plus a fitted width-rule table) that produces the bounds (section 5); a post-hoc
  second-moment head (1 105 parameters) that corrects the energy of the fluctuating part of the
  prediction (section 6).
- Backbone: `stformer`, `dim=192, depth=6, heads=6, cond_conv=true`, 4.63 M parameters. A 3x3 convolutional stem on the 20x32x64x3 window; patch embedding with
  patches of 20 frames x 2 x 2 pixels (512 tokens, attention is spatial only, time lives in the
  channel dimension); 6 transformer blocks (pre-LayerNorm, multi-head self-attention with 6 heads,
  MLP) each modulated by the operating-point conditioning through an identity-initialised
  scale/shift/gate ("adaLN" with the gate initialised at identity rather than zero); a linear
  patch decode followed by a convolutional refinement at full resolution with a skip from the
  stem, whose last convolution is zero-initialised. Stochastic depth (drop-path 0.3) in the
  fine-tunes. Inference uses PyTorch's math-kernel attention (no fused attention), in fp16 autocast.
- Inputs and outputs, including preprocessing and normalization: input `(N, 20, 32, 64, 3)` in
  physical units. The window is normalized per sample by its own free-stream speed (estimated from
  the top rows of the field) and by the official per-channel mean/std; the conditioning vector is
  the operating point (Reynolds number, angle of attack) *estimated from the window itself* by a
  fixed linear estimator (free-stream speed, wake geometry; coefficients fitted once on the public
  real release and hard-coded in `src/rpde_t1/features.py`), never a label, so inference and
  training see the same quantity. Output: `(N, 20, 32, 64, 3)` physical, `pred = x[:, -1] + f(x)`
  (residual on the last frame), pressure zeroed, plus `lower`/`upper` of the same shape.
- Starting point: random initialization. No official baseline checkpoint is used anywhere.

## 3. Data
- Training data used (`train_sim`, `train_real`) and how the two are combined: all 100 sim cells
  and all 81 real cells. Sim is first resampled in time to the real frame clock (`sim_units`: one
  sim frame is 0.02002 chord/free-stream time units, one real frame 0.02 s, so the resampling step
  is Re x 2.06e-4 frames, with round(step) start offsets so every source frame is used once;
  linear interpolation; pressure zeroed so a sim window looks like a real one). Stage 1 trains on
  sim only; stage 2 mixes real and sim 50/50 per batch; stages 3 and 4 are real only.
- Validation split carved out of the training data, if any: the model is trained on all 81 real
  cells (the Codabench row was too), so there is no held-out real validation inside `train()`.
  Three training-data splits are used for post-hoc fitting/stopping only: `train_ends` (the train
  cells at the ends of the Reynolds ladder), `val_interp` (Re 16500) and `val_time` (the last
  quarter of 20 training trajectories). During development the real stages were run on a 69-cell
  `train` split with Re 3750/16500/26700 held out; that is where every recipe decision was made.
- Augmentation: see README (fluctuation-amplitude rescaling tapered in log Re, +-2.5 degree
  conditioning jitter, R-drop). In the sim pretrain, stride-1 windows over the resampled series.
- Data left out, and why: the real file `7575_0.h5`, a byte-duplicate of `6300_0.h5`, is excluded
  by the code (and is absent from the Decision-phase mount anyway). One sim cell (`13950_15`) has a
  defective pressure channel, which is zeroed like every other sim pressure plane.

## 4. Training
One row per stage, in the order `train()` runs them. Common settings: batch 16, Adam (no weight
decay), 200 warm-up steps then cosine decay to 0.1x the peak learning rate, gradient clipping 1.0,
fp16 autocast with a GradScaler, deterministic cuDNN, seed 0 everywhere. "Huber" is smooth-L1 with
delta 0.005 on the normalized residual, over u and v (p is unscored and zeroed). The loss terms
and the augmentation marked (a) to (d) are defined under the table. Stages 3 and 3b exist only to
produce the "teacher" interval head that stage 4 trains against; the served model is stage 4.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1. sim pretrain | sim_units, all 100 sim cells, stride-1 windows | Huber on the residual | Adam, peak 1e-3, warm-up 200, cosine to 1e-4 | 8 000 | 16 |
| 2. joint | 50 % real (81 cells, all frames) + 50 % sim_units; init from stage 1 | Huber | Adam, peak 3e-4, cosine | 4 000 | 16 |
| 3. plain fine-tune (teacher's backbone, not served) | real, 81 cells; init from stage 2 | Huber + TKE terms (a); drop-path 0.3, EMA 0.999 | Adam, peak 3e-4, cosine | 24 000 | 16 |
| 3b. teacher interval head | stage-3 predictions on all 81 cells | L1 on the log error scale (section 5) | Adam 5e-4 | 24 000 | 4 |
| 4. credited fine-tune (served) | real, 81 cells; init from stage 2 | 0.5 Huber + 0.5 MSE on the residual + TKE terms (a) + band credit (b) + R-drop (c); augmentation (d); drop-path 0.3, EMA 0.999 | Adam, peak 3e-4, cosine | 24 000 | 16 |
| 5. interval head + width rule | stage-4 EMA predictions on all 81 cells | L1 on the log error scale; width rule by the scorer's credit (section 5) | Adam 5e-4 | 24 000 | 4 |
| 6. second-moment head | stage-4 bundle, all 81 cells (the val_time cells without their last quarter) | the scorer's TKE error of the corrected prediction | Adam 3e-3 | 3 000 (stop on val_time) | - |

- (a) TKE terms: 0.02 x a Huber penalty between the predicted and the true per-pixel temporal
  fluctuation energy of the window (each window's map normalized by its RMS), + 0.0038 x the same
  comparison after dividing both maps by their spatial mean, which matches where the fluctuations
  are rather than how much energy there is.
- (b) Band credit, weight 0.06: the teacher interval head (3b, frozen) draws its interval around
  the stage-4 prediction on every training window, and the term is minus the scorer's own smoothed
  per-element credit of that interval (truth inside the band, times `exp(-2 h / sigma)`), with the
  half-width `h` detached. The only gradient is on the prediction, and it is largest for errors
  just outside the band, so the prediction learns an error size consistent with the level an
  interval head will predict. Held out, the teacher's exact weights do not matter, its fit data
  does.
- (c) R-drop (alpha 1): two forward passes with independent stochastic-depth draws, plus the MSE
  between their two predictions.
- (d) Augmentation of the served fine-tune: a log-normal rescaling of the fluctuating part of the
  input and target windows (sigma 0.2 at Re 3750, tapering to 0 at Re 26700), and a uniform
  +-2.5 degree jitter of the angle-of-attack conditioning.
- How runs or checkpoints are selected, if at all: none between runs. The fine-tunes serve their
  EMA weights at the last step. The interval head's configuration is frozen (`coords, grad,
  steps 24000, width 32, depth 3, lr 5e-4, FiLM`) and its width rule is fitted on all 81 cells;
  the second-moment head stops on `val_time`.
- Seeds, and the run-to-run spread you have seen: seed 0 at every stage. Full-chain spread with a
  fresh pretrain seed: standard deviation about 0.022 final-score points on our unseen-Reynolds
  validation split and 0.021 on the hard cells (5 seeds); with the pretrain fixed and only the
  joint and fine-tune seeds varied, 0.007 and 0.010. Per subscore (every seed varied): rel_l2
  0.004 to 0.011, tke 0.03 to 0.075, mvpe 0.014 to 0.036, sps 0.03 to 0.05 points.

## 5. Uncertainty bounds
`lower`/`upper` come from a post-hoc **interval head** plus a calibrated **width rule**, fitted on
the served backbone's own predictions (never on another backbone's):

- The head is a small convolutional network (width 32, depth 3, kernels over the lead axis) on
  inference-available features of the prediction: the predicted field, its spatial gradients,
  pixel coordinates, the prediction's temporal spread, with the estimated operating point
  injected through FiLM. It predicts, per pixel, lead and channel, a scale `s` of the error, and
  is fitted by L1 on `log|err| - log s` (the median counterpart of a lognormal fit; robust to the
  heavy error tails).
- The width rule turns `s` into an asymmetric interval: per bin of `log(s/sigma)` x channel x
  lead (64 bins over `rho`), multipliers `(a_lo, a_up)` are chosen to maximise the scorer's own
  per-element credit `1[-a_lo s <= err <= a_up s] * exp(-(a_lo + a_up) s / sigma)` on the fit
  sample. The prediction itself is untouched; the interval may be shifted toward the conditional
  median. No lead anchor (anchor 0), no bound floor, no global scale.
- At inference the head and the rule run on the device inside the same graph as the backbone;
  `lower <= upper` is enforced, pressure bounds are zero.

## 6. Inference
- Track 1: `predict` takes the whole batch, uploads it once, and runs the device prefix
  (operating-point estimate, free-stream scale, normalization), the backbone, the interval head,
  the width rule, the second-moment correction (per-pixel log-ratio gain on the fluctuating part,
  shrunk by 0.5 toward no correction; the closed-form geometric blend is its special case) and
  pressure zeroing in fp16 autocast, as **one CUDA graph per call size**, captured on the first
  call at that size and used only if its replay reproduces the eager reference element for element
  (otherwise the eager path serves). Outputs are float32 holding the fp16-rounded values. No
  `torch.compile`, no import-time warm-up (the evaluator's own untimed warm-up call captures the
  graph), a one-time host-memory warm-up (`mallopt`) so later host buffers find resident pages.
  Batch size 210 internally, no padding of small batches.

## 7. Speed
Measured on an A100 and on the leaderboard, over three successive submissions (5.91 -> 4.30 ->
4.0 ms/sample, worth about +0.3 final-score points in total): eager fp16 serving at batch 210
with the operating-point estimate and the pressure zeroing on the device (3 kernels for the
prefix); a re-implementation of the inference pipeline in fewer kernels, checked bit-identical
to the reference; then the whole call captured as one CUDA graph; float32 outputs produced inside
the graph (a host fp16 -> fp32 cast cost about 1 ms/sample); no import-time warm-up; the
host-memory warm-up. The model itself is small by design (4.6 M parameters, 512 tokens).

## 8. What mattered
Rough effect sizes from our held-out campaigns (final-score points on our unseen-Reynolds
validation split unless stated; leaderboard noise is about +-0.07 points per submission):
- stformer over the FNO/CNO baselines and a U-Net: the largest single step (several points).
- sim pretrain on the time-resampled simulation + estimate-conditioned joint stage: the transfer
  recipe behind the backbone; conditioning on the window's own estimated operating point
  (instead of the label) made training match inference.
- fine-tune 24k with drop-path 0.3 and EMA: about +0.05 over 12k without.
- interval head + calibrated asymmetric width rule: several SPS points together; the later
  head configuration (FiLM conditioning, longer fit) +0.04 on the leaderboard with the same
  backbone weights.
- second-moment head at shrink 0.5: +2 to +4 tke subscore points over no correction.
- fluctuation augmentation tapered in log Re, band credit 0.06, angle jitter 2.5: each
  +0.01 to +0.03 held out at three seeds.
- R-drop (+0.008 / +0.019 on the unseen-Reynolds split / hard cells) and the Huber/MSE mix
  (+0.011 / +0.023), the two last additions of this submission.
- serving (section 7): about +0.3 final through the time subscore across three submissions.

## 9. Compute
Codabench run: about 3 h of GPU time across one T4 (pretrain) and A10G machines (all other
stages), peak GPU memory under 8 GB. This package from scratch on one A100 80 GB (rehearsal,
2026-10-05): 2.7 h wall time including 31 min of data preparation on a network volume, 2.0 h of
training stages (the two 24k fine-tunes 25 and 45 min), peak GPU memory 4.3 GB; the produced
model reproduces the Codabench artifact's local scores within 0.09 points per subscore. The A800 has the same compute as the A100.

## 10. References
- Vaswani et al., Attention is all you need (2017); Dosovitskiy et al., An image is worth 16x16
  words (2021); Peebles & Xie, Scalable diffusion models with transformers (adaLN-Zero, 2023);
  Xiao et al., Early convolutions help transformers see better (2021).
- Huang et al., Deep networks with stochastic depth (2016); Polyak averaging / EMA of weights.
- Liang et al., R-Drop: regularized dropout for neural networks (2021).
- Perez et al., FiLM: visual reasoning with a general conditioning layer (2018).
- PyTorch 2.2.2, NumPy 1.26, h5py; the organizers' starting kit (`vendor/`) for the scorer.
