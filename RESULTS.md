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

- Videos: `runs/diffusion_random/eval_best/` and `runs/diffusion_loso_fengxhou_dt01/eval_best/` (`videos_all.mp4`, `videos_all_reverse.mp4`, `videos_all_test.mp4`, `trajectoryN` / `simulationN`).
- Metrics: `eval_best/metrics.json` of both decoders; world model: `runs/mocap_random/eval_val_20pairs/metrics.json`, `runs/loso_fengxhou_dt01/val_005500/metrics.json`.
