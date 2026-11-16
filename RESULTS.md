# Random split vs leave-one-subject-out (fengxhou)

Same settings (`ifl_aligned` cache, poses at −200 ms, pair Δt 0.1–3 s, 6000 world-model steps, diffusion decoder up to 25,000 steps); only the split differs.

- **Random**: episodes split at random 70/15/15; validation/test are new episodes of subjects seen in training.
- **LOSO fengxhou**: validation = subject fengxhou, test = xuesong, both never seen in training. fengxhou was the best fold of the previous LOSO run (most favourable case).

## World model (validation, 500 pairs, 20 per episode)

Recall@k: the predicted latent must retrieve the true target frame among the 500 pairs.

| Run | | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|---|---|---|---|---|
| Random (`mocap_random`) | model | 0.280 | 0.654 | 0.794 |
|  | zero motion | 0.074 | 0.338 | 0.488 |
|  | copy context | 0.032 | 0.366 | 0.496 |
| LOSO fengxhou (`loso_fengxhou_dt01`) | model | 0.068 | 0.204 | 0.290 |
|  | zero motion | 0.026 | 0.150 | 0.230 |
|  | copy context | 0.030 | 0.238 | 0.330 |

| Run | val loss ↓ | spatial ↓ | motion ↓ | checkpoint |
|---|---|---|---|---|
| Random (`mocap_random`) | 0.594 | 0.127 | 4.675 | best.pt, re-evaluated on 500 pairs |
| LOSO fengxhou (`loso_fengxhou_dt01`) | 0.776 | 0.162 | 6.144 | step 5500 (best) |

## Decoder (generated image vs real frame)

MAE fg: ultrasound sector only (target > 0.05). Motion sensitivity: MAE between the predictions with the true and with a null motion. At most 8 pairs per episode, preferred horizon 1 s.

### Validation, forward motion

| Run | | MAE ↓ | MAE fg ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|
| Random (`diffusion_random`) | model | 0.0358 | 0.0827 | 0.0048 | 23.64 | 0.592 |
|  | copy context | 0.0396 | 0.0939 | 0.0070 | 22.89 | 0.695 |
|  | zero motion | 0.0447 | 0.0991 | 0.0074 | 21.95 | 0.554 |
| LOSO fengxhou (`diffusion_loso_fengxhou_dt01`) | model | 0.0488 | 0.1042 | 0.0087 | 20.88 | 0.591 |
|  | copy context | 0.0424 | 0.0956 | 0.0078 | 22.29 | 0.694 |
|  | zero motion | 0.0536 | 0.1109 | 0.0101 | 20.21 | 0.568 |

- Random: 200 pairs, mean Δt 0.88 s, mean translation 14.4 mm, mean rotation 3.5°, motion sensitivity 0.0308
- LOSO fengxhou: 200 pairs, mean Δt 0.89 s, mean translation 12.5 mm, mean rotation 2.4°, motion sensitivity 0.0357

### Validation, reverse motion

| Run | | MAE ↓ | MAE fg ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|
| Random (`diffusion_random`) | model | 0.0354 | 0.0826 | 0.0048 | 23.68 | 0.601 |
|  | copy context | 0.0396 | 0.0942 | 0.0070 | 22.89 | 0.695 |
|  | zero motion | 0.0451 | 0.0992 | 0.0075 | 21.88 | 0.551 |
| LOSO fengxhou (`diffusion_loso_fengxhou_dt01`) | model | 0.0475 | 0.1032 | 0.0083 | 21.03 | 0.597 |
|  | copy context | 0.0424 | 0.0957 | 0.0078 | 22.29 | 0.694 |
|  | zero motion | 0.0525 | 0.1103 | 0.0101 | 20.26 | 0.578 |

- Random: 200 pairs, mean Δt 0.88 s, mean translation 14.4 mm, mean rotation 3.5°, motion sensitivity 0.0304
- LOSO fengxhou: 200 pairs, mean Δt 0.89 s, mean translation 12.5 mm, mean rotation 2.4°, motion sensitivity 0.0389

### Test, forward motion

| Run | | MAE ↓ | MAE fg ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|
| Random (`diffusion_random`) | model | 0.0370 | 0.0854 | 0.0052 | 23.33 | 0.583 |
|  | copy context | 0.0400 | 0.0976 | 0.0072 | 22.61 | 0.698 |
|  | zero motion | 0.0460 | 0.1034 | 0.0078 | 21.68 | 0.550 |
| LOSO fengxhou (`diffusion_loso_fengxhou_dt01`) | model | 0.0454 | 0.0989 | 0.0074 | 21.69 | 0.581 |
|  | copy context | 0.0397 | 0.0963 | 0.0069 | 22.80 | 0.686 |
|  | zero motion | 0.0507 | 0.1070 | 0.0090 | 20.86 | 0.561 |

- Random: 192 pairs, mean Δt 0.87 s, mean translation 14.3 mm, mean rotation 3.8°, motion sensitivity 0.0310
- LOSO fengxhou: 120 pairs, mean Δt 0.89 s, mean translation 13.3 mm, mean rotation 3.1°, motion sensitivity 0.0347

## Summary

- **World model**: with the random split the model almost doubles the baselines (R@5 0.654 vs 0.366); on the unseen subject fengxhou it is only slightly above zero motion and below copy context (0.204 vs 0.238). The minimum Δt does not change the picture (previous fold with Δt 0.4 s: 0.200).
- **Decoder**: with the random split it beats copy context in MAE, MAE fg, MSE and PSNR; on unseen subjects (fengxhou, xuesong) it is worse than copy context on every metric, but better than zero motion.
- Copy context has the highest SSIM everywhere (~0.69 vs ~0.59): the decoder does not reconstruct the fine texture (speckle) of the real frame.
- Motion sensitivity is similar in both models (~0.031 random, ~0.035 LOSO): the LOSO model uses the motion as much, but turns it into a correct image worse on a new subject.
- The random split overestimates generalization: the model exploits the anatomy of subjects already seen. This is a single fold, the most favourable one.
- The LOSO decoder reached the 25,000-step limit without early stopping (best at the last step); the random one stopped at 16,250 (best 11,250).
- The `diffusion_random` test set contains episodes of subjects seen in training (xuesong has 10 episodes in train, 5 in val, 1 in test): the two test results are not a like-for-like comparison.

## Files

- Videos: `runs/mocap_random/diffusion_random/eval_best/` and `runs/loso_fengxhou_dt01/diffusion_loso_fengxhou_dt01/eval_best/` (`videos_all.mp4`, `videos_all_reverse.mp4`, `videos_all_test.mp4`, `trajectoryN` / `simulationN`).
- Metrics: `eval_best/metrics.json` of both decoders; world model: `runs/mocap_random/mocap_random/eval_val_20pairs/metrics.json`, `runs/loso_fengxhou_dt01/loso_fengxhou_dt01/val_005500/metrics.json`.


# Canonical-frame poses vs raw tracker poses (2026-11-12)

Canonical: SKEL refit at the RGB-D fit marker of each recording, probe projected on the fitted skin and transferred to the canonical SMPL body (Canonical_US); trimmed episodes (data/ifl_canonical); otherwise the same settings and splits as the raw runs. Raw: tracker poses on the untrimmed cache (ifl_aligned). The canonical runs also use the trimmed windows, so trimming and canonical poses change together; the validation galleries differ slightly (episodes without a canonical pose are dropped).

## World model (validation, 500 pairs)

| Run | | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|---|---|---|---|---|
| Random raw (`mocap_random`) | model | 0.280 | 0.654 | 0.794 |
|  | zero motion | 0.074 | 0.338 | 0.488 |
|  | copy context | 0.032 | 0.366 | 0.496 |
| Random canonical (`mocap_canonical_random`) | model | 0.230 | 0.652 | 0.804 |
|  | zero motion | 0.118 | 0.392 | 0.552 |
|  | copy context | 0.086 | 0.442 | 0.576 |
| LOSO fengxhou raw (`loso_fengxhou_dt01`) | model | 0.068 | 0.204 | 0.290 |
|  | zero motion | 0.026 | 0.150 | 0.230 |
|  | copy context | 0.030 | 0.238 | 0.330 |
| LOSO fengxhou canonical (`loso_fengxhou_canonical`) | model | 0.068 | 0.202 | 0.290 |
|  | zero motion | 0.034 | 0.196 | 0.298 |
|  | copy context | 0.044 | 0.342 | 0.440 |

| Run | val loss ↓ | spatial ↓ | motion ↓ | pairs |
|---|---|---|---|---|
| Random raw (`mocap_random`) | 0.594 | 0.127 | 4.675 | 500 |
| Random canonical (`mocap_canonical_random`) | 0.607 | 0.127 | 4.802 | 500 |
| LOSO fengxhou raw (`loso_fengxhou_dt01`) | 0.776 | 0.162 | 6.144 | 500 |
| LOSO fengxhou canonical (`loso_fengxhou_canonical`) | 0.788 | 0.162 | 6.262 | 500 |

## Decoder (generated image vs real frame)

### Validation, forward motion

| Run | | MAE ↓ | MAE fg ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|
| Random raw (`diffusion_random`) | model | 0.0358 | 0.0827 | 0.0048 | 23.64 | 0.592 |
|  | copy context | 0.0396 | 0.0939 | 0.0070 | 22.89 | 0.695 |
|  | zero motion | 0.0447 | 0.0991 | 0.0074 | 21.95 | 0.554 |
| Random canonical (`diffusion_canonical_random`) | model | 0.0362 | 0.0827 | 0.0051 | 23.42 | 0.638 |
|  | copy context | 0.0380 | 0.0907 | 0.0069 | 24.24 | 0.715 |
|  | zero motion | 0.0461 | 0.1007 | 0.0080 | 21.75 | 0.592 |
| LOSO fengxhou raw (`diffusion_loso_fengxhou_dt01`) | model | 0.0488 | 0.1042 | 0.0087 | 20.88 | 0.591 |
|  | copy context | 0.0424 | 0.0956 | 0.0078 | 22.29 | 0.694 |
|  | zero motion | 0.0536 | 0.1109 | 0.0101 | 20.21 | 0.568 |
| LOSO fengxhou canonical (`diffusion_loso_fengxhou_canonical`) | model | 0.0492 | 0.1055 | 0.0087 | 20.81 | 0.589 |
|  | copy context | 0.0382 | 0.0865 | 0.0070 | 24.47 | 0.727 |
|  | zero motion | 0.0524 | 0.1099 | 0.0100 | 20.31 | 0.583 |

- Random raw (`diffusion_random`): 200 pairs, mean Δt 0.88 s, mean translation 14.4 mm, mean rotation 3.5°, motion sensitivity 0.0308
- Random canonical (`diffusion_canonical_random`): 200 pairs, mean Δt 0.84 s, mean translation 15.8 mm, mean rotation 3.7°, motion sensitivity 0.0327
- LOSO fengxhou raw (`diffusion_loso_fengxhou_dt01`): 200 pairs, mean Δt 0.89 s, mean translation 12.5 mm, mean rotation 2.4°, motion sensitivity 0.0357
- LOSO fengxhou canonical (`diffusion_loso_fengxhou_canonical`): 200 pairs, mean Δt 0.87 s, mean translation 14.0 mm, mean rotation 2.6°, motion sensitivity 0.0395

### Validation, reverse motion

| Run | | MAE ↓ | MAE fg ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|
| Random raw (`diffusion_random`) | model | 0.0354 | 0.0826 | 0.0048 | 23.68 | 0.601 |
|  | copy context | 0.0396 | 0.0942 | 0.0070 | 22.89 | 0.695 |
|  | zero motion | 0.0451 | 0.0992 | 0.0075 | 21.88 | 0.551 |
| Random canonical (`diffusion_canonical_random`) | model | 0.0361 | 0.0821 | 0.0051 | 23.45 | 0.643 |
|  | copy context | 0.0380 | 0.0870 | 0.0069 | 24.24 | 0.715 |
|  | zero motion | 0.0460 | 0.0968 | 0.0080 | 21.75 | 0.593 |
| LOSO fengxhou raw (`diffusion_loso_fengxhou_dt01`) | model | 0.0475 | 0.1032 | 0.0083 | 21.03 | 0.597 |
|  | copy context | 0.0424 | 0.0957 | 0.0078 | 22.29 | 0.694 |
|  | zero motion | 0.0525 | 0.1103 | 0.0101 | 20.26 | 0.578 |
| LOSO fengxhou canonical (`diffusion_loso_fengxhou_canonical`) | model | 0.0492 | 0.1072 | 0.0091 | 20.78 | 0.596 |
|  | copy context | 0.0382 | 0.0855 | 0.0070 | 24.47 | 0.727 |
|  | zero motion | 0.0526 | 0.1106 | 0.0103 | 20.19 | 0.587 |

- Random raw (`diffusion_random`): 200 pairs, mean Δt 0.88 s, mean translation 14.4 mm, mean rotation 3.5°, motion sensitivity 0.0304
- Random canonical (`diffusion_canonical_random`): 200 pairs, mean Δt 0.84 s, mean translation 15.8 mm, mean rotation 3.7°, motion sensitivity 0.0327
- LOSO fengxhou raw (`diffusion_loso_fengxhou_dt01`): 200 pairs, mean Δt 0.89 s, mean translation 12.5 mm, mean rotation 2.4°, motion sensitivity 0.0389
- LOSO fengxhou canonical (`diffusion_loso_fengxhou_canonical`): 200 pairs, mean Δt 0.87 s, mean translation 14.0 mm, mean rotation 2.6°, motion sensitivity 0.0390

### Test, forward motion

| Run | | MAE ↓ | MAE fg ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|
| Random raw (`diffusion_random`) | model | 0.0370 | 0.0854 | 0.0052 | 23.33 | 0.583 |
|  | copy context | 0.0400 | 0.0976 | 0.0072 | 22.61 | 0.698 |
|  | zero motion | 0.0460 | 0.1034 | 0.0078 | 21.68 | 0.550 |
| Random canonical (`diffusion_canonical_random`) | model | 0.0371 | 0.0855 | 0.0054 | 23.17 | 0.634 |
|  | copy context | 0.0373 | 0.0919 | 0.0069 | 24.33 | 0.720 |
|  | zero motion | 0.0460 | 0.1026 | 0.0080 | 21.65 | 0.593 |
| LOSO fengxhou raw (`diffusion_loso_fengxhou_dt01`) | model | 0.0454 | 0.0989 | 0.0074 | 21.69 | 0.581 |
|  | copy context | 0.0397 | 0.0963 | 0.0069 | 22.80 | 0.686 |
|  | zero motion | 0.0507 | 0.1070 | 0.0090 | 20.86 | 0.561 |
| LOSO fengxhou canonical (`diffusion_loso_fengxhou_canonical`) | model | 0.0483 | 0.1063 | 0.0084 | 21.12 | 0.572 |
|  | copy context | 0.0385 | 0.0918 | 0.0068 | 24.25 | 0.702 |
|  | zero motion | 0.0513 | 0.1072 | 0.0094 | 20.78 | 0.561 |

- Random raw (`diffusion_random`): 192 pairs, mean Δt 0.87 s, mean translation 14.3 mm, mean rotation 3.8°, motion sensitivity 0.0310
- Random canonical (`diffusion_canonical_random`): 184 pairs, mean Δt 0.84 s, mean translation 14.2 mm, mean rotation 3.9°, motion sensitivity 0.0302
- LOSO fengxhou raw (`diffusion_loso_fengxhou_dt01`): 120 pairs, mean Δt 0.89 s, mean translation 13.3 mm, mean rotation 3.1°, motion sensitivity 0.0347
- LOSO fengxhou canonical (`diffusion_loso_fengxhou_canonical`): 120 pairs, mean Δt 0.87 s, mean translation 12.9 mm, mean rotation 3.1°, motion sensitivity 0.0319

## Files

- Random raw (`mocap_random`): `runs/mocap_random/mocap_random/eval_val_20pairs/metrics.json`
- Random canonical (`mocap_canonical_random`): `runs/mocap_canonical_random/mocap_canonical_random/eval_val_20pairs/metrics.json`
- LOSO fengxhou raw (`loso_fengxhou_dt01`): `runs/loso_fengxhou_dt01/loso_fengxhou_dt01/val_005500/metrics.json`
- LOSO fengxhou canonical (`loso_fengxhou_canonical`): `runs/loso_fengxhou_canonical/loso_fengxhou_canonical/eval_val_20pairs/metrics.json`
- Random raw (`diffusion_random`): `runs/mocap_random/diffusion_random/eval_best/metrics.json`
- Random canonical (`diffusion_canonical_random`): `runs/mocap_canonical_random/diffusion_canonical_random/eval_best/metrics.json`
- LOSO fengxhou raw (`diffusion_loso_fengxhou_dt01`): `runs/loso_fengxhou_dt01/diffusion_loso_fengxhou_dt01/eval_best/metrics.json`
- LOSO fengxhou canonical (`diffusion_loso_fengxhou_canonical`): `runs/loso_fengxhou_canonical/diffusion_loso_fengxhou_canonical/eval_best/metrics.json`


# Anatomical information in the world model (2026-11-13)

Can knowing *where* the probe is on the body improve the world model? All runs use the corrected canonical
caches: 157 trimmed episodes (current trims, expert36 and nardi1 excluded, segments broken only at real
tracking gaps), the same frozen random split and the same settings as above (6000 steps, Δt 0.1–3 s).
The canonical runs of the previous section used an older cache (old trims, spurious segment breaks) and
are superseded by the 6-D baselines below. 3 seeds (42, 1, 2) per variant on the random split; the seed
changes initialization and sampling only. Validation: 500 pairs (20 per episode).

- **Fit**: body model fitted to the RGB-D frame of the fit marker, probe poses transferred to the canonical
  SMPL body. *SKEL*: Canonical_US SKEL fit (Chamfer + ICP, beta L2 1). *SMPL*: SMPL with 300 betas, Chamfer
  only with normal-aware matching (nearest neighbours on [x, 0.05·n], so back-facing skin is not matched
  to the visible surface), beta L2 1 (`data/ifl_canonical_smpl_chamfer_normals`). SMPL follows the cloud
  better (cloud→skin median 4.5 vs 6.4 mm, p90 13.7 vs 22.8 mm, better in 157/157 recordings) but the
  probe→skin distance is slightly worse (median 15.6 vs 12.8 mm).
- **Anatomy as input** (`model.anatomy`): the attachment of the current probe position on the canonical
  body (triangle + barycentric coordinates) becomes an extra token of the predictor, between the context
  tokens and the action token; the 6-D action is unchanged and only the position of the frame the
  prediction starts from is given (never the predicted frame's). Encodings: learnable per-vertex
  embeddings interpolated with the barycentric weights (6890 × 384), optionally with token dropout 0.5 in
  training; or a fixed Fourier encoding of the canonical position (8 wavelengths, 1 m to 1 cm) followed by
  a small MLP.
- **Anatomy as auxiliary target** (`model.anatomy_target`): no anatomical input. For both images of each
  pair, the online encoder tokens (the same context tokens used by the predictor) are averaged and an MLP
  (384 → 384 → 3) regresses the canonical probe position of that image (body-centred, in units of 10 cm);
  loss L = L_spatial + 0.1 L_motion + 0.1 SmoothL1(position). The predictor and inference are those of the
  baseline; the head is created last, so with the same seed all other weights start identical to the
  baseline (paired comparison).

## World model, random split (validation, mean ± std over seeds)

| Fit | Variant | Seeds | R@1 ↑ | R@5 ↑ | R@10 ↑ | R@5 per seed (42, 1, 2) | zero motion R@5 | copy context R@5 | val motion ↓ |
|---|---|---|---|---|---|---|---|---|---|
| SKEL | 6-D actions (baseline) (`mocap_canonical_6d`) | 3 | 0.209 ± 0.009 | 0.595 ± 0.031 | 0.753 ± 0.041 | 0.608, 0.618, 0.560 | 0.367 ± 0.010 | 0.417 ± 0.013 | 4.846 ± 0.064 |
| SKEL | + anatomy input, per-vertex embedding (`mocap_canonical_anatomy`) | 1 | 0.152 | 0.446 | 0.602 | 0.446 | 0.268 | 0.366 | 5.093 |
| SKEL | + anatomy input, per-vertex embedding, token dropout 0.5 (`mocap_canonical_anatomy_drop50`) | 3 | 0.213 ± 0.017 | 0.579 ± 0.029 | 0.763 ± 0.034 | 0.588, 0.546, 0.602 | 0.327 ± 0.022 | 0.387 ± 0.001 | 4.827 ± 0.114 |
| SKEL | + anatomy input, Fourier encoding (`mocap_canonical_anatomy_fourier`) | 3 | 0.144 ± 0.028 | 0.432 ± 0.040 | 0.590 ± 0.047 | 0.414, 0.404, 0.478 | 0.263 ± 0.023 | 0.372 ± 0.021 | 5.106 ± 0.037 |
| SMPL | 6-D actions (baseline) (`mocap_smpl_6d`) | 3 | 0.207 ± 0.034 | 0.581 ± 0.048 | 0.745 ± 0.048 | 0.614, 0.602, 0.526 | 0.359 ± 0.025 | 0.385 ± 0.010 | 4.851 ± 0.065 |
| SMPL | + anatomy input, Fourier encoding (`mocap_smpl_anatomy_fourier`) | 3 | 0.148 ± 0.016 | 0.436 ± 0.060 | 0.595 ± 0.046 | 0.436, 0.376, 0.496 | 0.251 ± 0.009 | 0.383 ± 0.015 | 5.083 ± 0.096 |
| SMPL | **+ auxiliary anatomical target** (`mocap_smpl_anatomy_aux`) | 3 | 0.232 ± 0.040 | 0.647 ± 0.024 | 0.812 ± 0.014 | 0.656, 0.666, 0.620 | 0.387 ± 0.009 | 0.419 ± 0.022 | 4.729 ± 0.055 |

Paired comparison on the SMPL fit (same seed = same initial weights):

| Seed | SMPL 6-D R@5 | SMPL + auxiliary target R@5 | Δ |
|---|---|---|---|
| 42 | 0.614 | 0.656 | +0.042 |
| 1 | 0.602 | 0.666 | +0.064 |
| 2 | 0.526 | 0.620 | +0.094 |

Auxiliary head position error on validation: 31.0 mm (mean over seeds) (56 mm at initialization).

## World model, LOSO fengxhou (validation = unseen subject, 1 seed, SKEL fit)

| Variant (SKEL fit) | R@1 ↑ | R@5 ↑ | R@10 ↑ | zero motion R@5 | copy context R@5 | val motion ↓ |
|---|---|---|---|---|---|---|
| 6-D actions (baseline) (`loso_fengxhou_canonical_6d`) | 0.058 | 0.198 | 0.302 | 0.158 | 0.306 | 6.359 |
| + anatomy input, per-vertex, dropout 0.5 (`loso_fengxhou_canonical_anatomy_drop50`) | 0.044 | 0.156 | 0.254 | 0.154 | 0.336 | 6.729 |
| + anatomy input, Fourier (`loso_fengxhou_canonical_anatomy_fourier`) | 0.014 | 0.104 | 0.164 | 0.112 | 0.308 | 6.265 |


## Summary

- **Anatomy as input always hurts.** The predictor uses the position as a shortcut to recognize the episode:
  training motion loss goes down, validation recall collapses (per-vertex 0.446, Fourier 0.43 vs 0.595). Token
  dropout removes most of the damage (0.579) but does not beat the baseline. On the unseen subject the loss is
  larger and grows with the precision of the position (0.198 → 0.156 with dropout → 0.104 with Fourier).
- **Anatomy as an auxiliary target helps**: R@5 0.647 ± 0.024 vs 0.581 ± 0.048 on the same data, better in all
  three paired seeds (+0.042, +0.064, +0.094), also in R@1 and R@10, with a lower validation motion loss and a
  higher copy-context recall (better encoder features). Nothing of the position is needed at inference.
- **SKEL vs SMPL fit**: no measurable difference for the 6-D baseline (0.595 vs 0.581, within seed variation).
  The relative actions are little affected by the fit; its quality matters through the anatomical labels.
- Still to do: LOSO of the auxiliary target (and of the SMPL 6-D baseline) to check the cross-subject case;
  the decoder on the best world model.

## Files

- Runs: `runs/<variant>/<variant>{,_s1,_s2}/` (`eval_val_20pairs/metrics.json`), LOSO `runs/loso_fengxhou_canonical_*/`.
- Configs: `configs/mocap_canonical_6d.json`, `mocap_canonical_anatomy*.json`, `mocap_smpl_6d.json`,
  `mocap_smpl_anatomy_fourier.json`, `mocap_smpl_anatomy_aux.json`; seeds via `runs/seeds_pipeline.sh <config>`.
- Code: `mocap/pretraining/model.py` (`AnatomicalEmbedding`, `FourierAnatomicalEmbedding`, `AnatomicalPositionTarget`),
  attachments in `mocap/processing/canonical.py anatomy`, SMPL fit: Canonical_US `--body-model smpl
  --config-dir canonical_us/config/smpl_ifl_normals`.
