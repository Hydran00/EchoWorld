# EchoWorld on the IFL mocap recordings

## Goal and implementation plan

Train on the original episodes in
`/media/davide.nardi-1/CrucialX9/Nardi/ifl_recording/<name>/<bag>` using **only
ultrasound images and measured 6-DoF probe poses**. RGB, depth, point clouds,
force, anatomical canonicalization, and `probe_reconstruction` are not inputs.
The original recordings remain read-only; derived data and runs live here.

1. Inspect the MCAP topics, timestamps, tracked rigid bodies and recorded probe
   calibration. Decode ultrasound following `../Canonical_US/rgbd_bag.py`,
   including its sequential FFmpeg transport handling. Synchronize images to
   measured poses with an explicit tolerance and reject invalid tracking.
2. Export a compact, reproducible cache with images, timestamps, transforms,
   validity statistics, and the exact preprocessing settings. Inspect the image
   field of view before selecting a crop. Never interpolate across tracking gaps
   or form training pairs across episode boundaries.
3. Split by recording group (`<name>`) before making pairs, so adjacent frames of
   one episode cannot leak into validation. Persist the split in a manifest.
4. Reuse the released EchoWorld ViT encoder and predictor. Implement the missing
   motion-conditioning path and joint spatial/motion learning for these pairs.
   Record the transform convention, translation units and rotation encoding.
5. Keep pre-training at the paper objective (no pixel losses). To view images,
   train a separate decoder on the frozen world model, as in paper C.3, and
   evaluate it against a copy-context baseline and a zero-motion ablation. Generation requires a context ultrasound
   image and a relative probe trajectory; a trajectory alone does not specify
   patient anatomy.
6. Test decoding, synchronization, SE(3) math, splits, batching, gradient flow,
   checkpoint/resume and inference. Run real GPU training, stop it gracefully,
   resume it, and export held-out image comparisons and numerical metrics.

## What the released repository provides

The attached EchoWorld paper (arXiv:2504.13065v1, Sections C.1 and C.3) learns
representations with spatial and motion objectives. Its displayed ultrasound
predictions require an additional RCDM diffusion decoder trained for 300,000
iterations. That image generator is not included in this checkout.

The released `pretrain/main.py` contains the joint spatial/motion loop. It
initializes `policy_net` and `mask_token_extra` there, outside the model class,
and the loop is tied to the authors' dataset and distributed setup. This integration
ports that loop to these pairs. A short successful run demonstrates the
engineering pipeline, not reproduction of the paper's results or useful
generalization. Images come only from a separately trained decoder,
**not a reproduction of RCDM**.

## Planned local artifacts

- `mocap/`: extraction, dataset, model, training and inference commands, split by role:
  `processing/` (bags, geometry, prepare, trims, dataset), `pretraining/` (model, train, loso),
  `decoder/` (diffusion, generate), `finetuning/` (guidance), `evaluation/` (evaluate,
  compare_rollout, compare_motion, temporal_eval, embedding_pca, guidance_report) and
  `visualization/` (visualize, tensorboard).
- `configs/mocap*.json`: reproducible single-GPU configurations; `configs/archive/`: configs of past experiments (fixed 200 ms dt, `mocap_full_v2`).
- `tests/`: focused correctness and integration tests.
- `data/ifl_full/`: ignored, regenerable local cache and split manifest (all frames, `--fps 100`).
- `runs/`: ignored checkpoints, metrics and generated comparisons.

## Measured data and coordinate conventions

The source contains 168 bag directories in nine recording groups, approximately
340 GB of MCAP data. The observed ultrasound transport is
`/ultrasound_image/ffmpeg` (1080x1920 images). Raw `/ultrasound/image_raw` is also
supported and takes precedence if both exist. FFmpeg decoding starts at a
keyframe, preserves packet/frame timestamp associations and flushes delayed
frames. RGB-D and force messages are never deserialized, although their bytes
can share compressed MCAP chunks that must be read from disk.
Packets before the first keyframe and packets that never yield a decoded frame
are counted in `video_stats`; only frames with an unambiguous timestamp are
retained. Some recordings have incomplete video tails, so such packets are not
silently assigned to other frames or used as image samples.

`/rigid_poses` supplies `polaris_frame_T_marker` in metres and ROS xyzw
quaternions, with `inbound` indicating valid tracking. `/tf_static` supplies the
recorded `marker_T_probe` calibration. The stored probe transform is
`polaris_frame_T_probe = polaris_frame_T_marker @ marker_T_probe`. Missing
calibration is an error; there is no silent fallback to the marker position.
The static calibration is bag-specific and stored in each cache's metadata.

Ultrasound and pose **header timestamps** are used. During extraction, the
ultrasound header time is corrected by `--pose-time-offset-ms` (default **-200 ms**
for the current IFL setup), and the probe pose is sampled at that acquisition time.
Translation is interpolated linearly and orientation by quaternion SLERP. Both bracketing measurements must
be valid and within 50 ms; brackets over 100 ms and extrapolation are rejected.
Invalid poses remain barriers, including when no sampled ultrasound frame lands
inside that gap. No clock-offset correction is inferred automatically.

The default crop is `(left=520, top=110, right=1400, bottom=800)`, inspected on
these recordings. It retains the displayed ultrasound field and removes almost
all scanner UI (a small static icon remains). The crop is converted to grayscale
and letterboxed to 224x224; it is not stretched. This is configurable and must be
rechecked if scanner layout changes. `--fps` caps the cache frame rate (default 5 Hz); the current cache keeps all frames (`--fps 100`). Training
uses frames with mean intensity >= 8/255 to exclude mostly black/no-contact
images; this simple ultrasound-only heuristic is not contact annotation.

For source pose `T_a` and target pose `T_b`, the model action is
`inverse(T_a) @ T_b`, encoded as `[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]`.
The last three values form an axis-angle rotation vector, not Euler angles.
Actions are divided by `[30,30,30,0.3,0.3,0.3]` before entering the motion MLP.
This convention is invariant to the global tracker coordinate frame and is an
intentional adaptation from the paper's pose convention. It does not perform
anatomical canonicalization. An arbitrary robot-frame trajectory must first
refer to the same calibrated physical probe frame.

Pairs are sampled 0.1--3 seconds (default `min_dt`/`max_dt`) apart, within one episode and one valid tracking
segment, and may be reversed during training. Validation uses deterministic
pairs near one second apart. The train/val/test split is chosen by `data.split_mode`:
`random` (default for new runs) shuffles episodes with `split_seed` (42) and divides them by
`split_fractions` (0.7/0.15/0.15), so the same subject appears in every split; `leave-out`
uses the subject split of the manifest, where `yuan` is validation and `xuesong` is test and all
other groups are training. Checkpoints without `split_mode` were trained with `leave-out`, and
the decoder inherits the split of its world model unless its config sets `split_mode`.
These are folder-based groups, not independently
verified patient identities. No image from validation/test is used to update
the model. Test data must not be used to choose checkpoints.

## Model and differences from the paper

`mocap/pretraining/model.py` follows the released `pretrain/main.py` and
`configs/cardiac_vits16_p384x6_simclr.yaml`. It reuses the released
`VisionTransformer`, `VisionTransformerPredictor` and multiblock `MaskCollator`.

- Context encoder: ViT-S/16, 384 channels, 12 blocks, 6 heads (21.7M parameters).
  The target encoder is its EMA copy (0.996 -> 1.0) and receives no gradients.
- Predictor: 6 blocks, width 384, 6 heads (11.2M parameters, including the
  motion MLP). One predictor serves both objectives.
- Motion MLP `policy_net`: `Linear(6,384)-SiLU-Linear(384,384)`, added to
  `mask_token_extra` to form the single action token used by `forward_extra`.
- Projector: `Linear(384,1024)-BatchNorm-ReLU-Linear(1024,1024)`, with an EMA target copy.
- Masks: one context block (scale 0.85--1.0) and four target blocks
  (scale 0.15--0.2, aspect ratio 0.75--1.5), without overlap.

The objectives are symmetric over the pair `(I_a, I_b)`, as in the released code.
`L_spatial` is smooth-L1 between the predicted and layer-normalized target
features of masked blocks. `L_motion` is SimCLR-style InfoNCE (temperature 0.1)
between the projected motion prediction and the projected pooled target feature
of the other image. The loss is `L_spatial + 0.1 L_motion`. No pixel, latent
alignment or reconstruction loss is used.

Remaining differences from the paper/release are:

- the action convention described above;
- training from scratch on this data;
- the smaller batch and shorter schedule.

For images smaller than 224 px, lower `mask.min_keep` in the model configuration (e.g. 4 at
128 px). The upstream collator cannot form valid blocks on very small patch grids.

The image decoder in `mocap/decoder/diffusion.py` freezes the trained world model and learns a
conditional U-Net from its predicted feature to the target ultrasound, following the
representation-conditioning idea of paper C.3. The U-Net (`diffusers.UNet2DModel`) and the
cosine noise schedule and DDIM sampler (`DDPMScheduler`, `DDIMScheduler`) come from Hugging Face
diffusers. The EchoWorld feature is projected by an MLP and added to the timestep embedding
(`class_embed_type="identity"`). Each residual block applies that embedding as a per-channel scale and shift
(`resnet_time_scale_shift="scale_shift"`). Training uses classifier-free conditioning dropout
and an EMA denoiser. The selected configuration predicts `v` (velocity), which was more stable
in this experiment than epsilon prediction. This decoder is **not the authors' RCDM
implementation**. It can produce ultrasound texture, but plausible texture is not evidence that
the predicted anatomy is correct.

## Environment

Run all commands below from the EchoWorld repository root. The local `.venv` is
ready on this workstation. To avoid modifying the existing robotics environment
or duplicating its CUDA packages, `.venv/lib/python3.12/site-packages/workstation_torch.pth`
references `/home/davide.nardi-1/i4h_ws/us_dp/.venv/lib/python3.12/site-packages`.
This is a workstation-specific optimization; that existing environment must
remain available. New bag/video dependencies were installed only into this
fork's `.venv`. The detected Torch is `2.14.0+cu130`, with Blackwell `sm_120`
support. GPU 0 is used; GPU 1 has an existing vLLM process.

On another machine, create an independent Python 3.12 environment, install a
CUDA-enabled PyTorch build compatible with that GPU, then install
`requirements-mocap.txt`. No ROS runtime or Canonical_US Python installation is
required by the new code. Tests run on CPU except the separately exercised real
GPU training/inference commands.

## Extraction and training

To reproduce the original zero-offset baseline, export all frames and pre-train the ViT-S world model (the corrected IFL export command is documented below):

```bash
.venv/bin/python -m mocap.processing.prepare \
  --root /media/davide.nardi-1/CrucialX9/Nardi/ifl_recording \
  --output data/ifl_full --fps 100 --workers 2 --skip-errors --pose-time-offset-ms 0

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m mocap.pretraining.train \
  --config configs/archive/mocap_full_v2.json --output runs/mocap_full_v2 > runs/mocap_full_v2.log 2>&1 &
```

`--fps 100` exceeds the ~33 Hz ultrasound rate, so every decoded frame is kept. Four recordings
(`nardi/bag_narditest`, `xuesong/bag_xuesong3-5`) have truncated MCAP files and are listed in
`export_errors.json`. `--limit-per-group N` exports a small subset for quick experiments.

Extraction is restartable: matching complete caches are reused. Source file
sizes/mtimes and export settings are fingerprinted. A changed crop/rate/source
requires a different cache directory; mixing cached preprocessing is rejected.
Each episode contains `images.npy`, `poses.npz`, and `metadata.json`. The global
`manifest.json` records all splits/settings and errors. The command fails if an
episode cannot be exported; `--skip-errors` explicitly permits a partial export
while preserving every failure in `export_errors.json` and the manifest.

For the current IFL recording setup, the temporal investigation supports an ultrasound
latency of approximately 200 ms relative to tracking. The independent force/contact
check also finds delayed image transitions, but contact mechanics do not identify an
exact hardware latency. The fixed **-200 ms** correction is the default for new
exports from the original tracker samples:

```bash
.venv/bin/python -m mocap.processing.prepare \
  --root /media/davide.nardi-1/CrucialX9/Nardi/ifl_recording \
  --output data/ifl_raw_aligned_200ms --fps 100 --workers 2 --skip-errors
```

An image with header time `t` is paired with the pose at `t - 0.2 s`. The cache keeps
`image_header_timestamps_ns` unchanged, and uses the corrected acquisition time as
`timestamps_ns`. Tracking gaps and unavailable poses are rejected, not extrapolated.
The offset is recorded in the cache settings/signature and manifest. Extraction
always starts from original recordings; a matching cache is reused, and changed
settings require a separate output directory. Use `--pose-time-offset-ms 0` to
explicitly disable correction or specify a different calibrated offset for another setup.

The data flow is **original recordings -> extraction with offset -> prepared cache ->
training**. The dataset loader, world-model training and diffusion training only
read prepared timestamps and poses: they apply no offset and perform no temporal
realignment. There is no separate cache-realignment command.

The existing `ifl_aligned` (formerly `ifl_aligned_200ms`) and control caches are historical inputs of the
completed comparison and remain available for reproducing those checkpoints and
metrics. New datasets should be exported directly from the original recordings;
set the new training configuration's `data.manifest` to that exported manifest.
Do not add the bag recording/transport delay to this correction.
This corrects training synchronization, not the physical latency of the live preview.

Training uses BF16 on CUDA, finite-loss/gradient checks, gradient clipping and
an EMA target. Checkpoints are written atomically as `last.pt` and `best.pt`.
`best.pt` is selected by the `selection` metric (default `val_loss`, with fixed masks); `patience` stops a run after that many validations without improvement. Training metrics are in `train.jsonl`,
hardware/dataset details in `run_info.json`, and validation metrics in
`val_<step>/metrics.json`. Run folders and datasets are git-ignored.

`configs/archive/mocap_full_v2.json` includes the regularization of the released configuration. An
earlier run without it reached motion retrieval recall@5 0.99 on training pairs, but stayed below
the zero-motion and copy-context baselines on validation. The regularization settings are:

- `crop_scale: [0.3, 1.0]`: RandomResizedCrop on training pairs. One box per pair is applied to
  both frames, so both views keep the same image-to-probe geometry.
- `weight_decay: 0.04 -> final_weight_decay: 0.4`: the cosine schedule of the release, with
  biases and 1-D parameters excluded from decay.
- `selection: model_recall@5` and `patience: 4`: checkpoint selection and early stopping on held-out
  motion retrieval.

The diffusion decoder does not inherit `crop_scale`; it is trained on uncropped frames.

To monitor all current and future runs in TensorBoard, discover their JSON logs under `runs/` and mirror them into `<run>/tensorboard`:

```bash
setsid nohup .venv/bin/python -m mocap.visualization.tensorboard --watch-root runs --follow --interval 5 > runs/tensorboard_sync.log 2>&1 < /dev/null &
.venv/bin/python -m tensorboard.main --logdir runs --port 6006 --reload_interval 5 --load_fast false
```

The mirror can be started at any time, including during or after training. It rebuilds
the events from the logs on every start and never writes to the training files. The *Custom
Scalars* tab overlays model and baseline metrics. With `--watch-root`, the mirror discovers new run folders at every poll and stays active after existing runs finish. Start only one mirror per run/root. Runs that already write TensorBoard events directly are read by the server without conversion.

`--init-from <run>/best.pt` of a matching architecture can initialize matching weights while resetting the optimizer and step. This is
different from exact resume; the initialization source is recorded separately.

## Stop and resume

Press Ctrl-C, or send SIGTERM to the PID recorded by this run:

```bash
kill -TERM "$(cat runs/mocap_full_v2/pid.txt)"
```

The handler completes the current optimizer step and saves `last.pt`. SIGKILL
cannot be handled. Resume with the identical configuration and manifest:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m mocap.pretraining.train \
  --config configs/archive/mocap_full_v2.json --output runs/mocap_full_v2 \
  --resume runs/mocap_full_v2/last.pt
```

`--max-steps` is a **global** stopping step, independent of the configured total
LR schedule. Model, EMA, optimizer, RNG and current step are restored. Per-step
pair sampling is reproducible across worker prefetch and resume. A test compares
CPU uninterrupted/resumed weights exactly. GPU kernels can still have numeric
nondeterminism. Configuration/manifest changes are rejected on resume rather
than silently starting over; only `patience` may be added or changed. Only load checkpoints from trusted sources.

## Evaluate and generate

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m mocap.evaluation.evaluate \
  --checkpoint runs/mocap_full_v2/best.pt --split val --output runs/mocap_full_v2/evaluation
```

For a world-model checkpoint, evaluation writes representation metrics only. These are
validation spatial/motion losses and motion retrieval recall@1/5/10 over all
held-out pairs. Retrieval compares the prediction against the zero-motion
prediction and against the copy-context baseline, which uses the context image's
own target feature as the query. Motion conditioning is useful only if the model
beats both.

Pixel evaluation and generation require a diffusion checkpoint (see below).
For those, evaluation writes `metrics.json`, `per_pair_metrics.csv`, standalone predicted
PNGs, and the following headless figures (`mocap/visualization/visualize.py`: Pillow and PyAV, with no
window or OpenGL context):

- `comparison1.png`, `comparison2.png`, `comparison3.png`: 36 pairs, 12 per file, spread
  round-robin over the validation episodes. The columns are zero-motion prediction, context,
  prediction, target, absolute error (x3), and probe motion. The probe-motion panel shows the
  recorded trajectory between the two frames in the frame of the context probe, with the start
  (faded) and end RGB axis triads (x red, y green, z blue), Δt, |translation| and rotation angle.
- `trajectory1.mp4` … `trajectory10.mp4` (ten validation episodes): anchored generation from one fixed context frame
  to the recorded poses at +0.1 s, +0.2 s, … up to `max_dt`, played at 5 fps (half
  speed). All video frames share one initial noise image, so the changes are due to the
  motion only. Horizons below `min_dt` are labelled as outside the training range.

TensorBoard shows each grid as `val/comparison1`, `val/comparison2`, ... (videos are not mirrored).
Metrics include MAE, PSNR, local SSIM and foreground MAE
(target intensity > 0.05), so black display background cannot hide all failures.
Pairs shown in the grid are identified in `metrics.json`.

Generate a measured trajectory from any cached episode, choosing an initial
frame and a range inside a single valid tracking segment:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m mocap.decoder.generate \
  --checkpoint runs/diffusion_v2/best.pt \
  --episode data/ifl_full/episodes/<group>/<bag> \
  --start-index 10 --stride 10 --frames 8 --output runs/diffusion_v2/trajectory
```

The first cached image is the only ultrasound input. Future ground-truth images
are used solely for the comparison strip. Outputs include individual PNGs,
`generated.npy`, `generated.gif` and `sequence.png` (generated row, then real
images). Default generation stays anchored to the initial image; `--rollout`
feeds generated frames back and will generally accumulate error.

For a new planned trajectory, provide a raw context image and an NPZ containing
`world_T_probe` with shape `(N+1,4,4)`, in metres. The first pose corresponds to
the context image. This needs no target images or reference dataset at inference:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m mocap.decoder.generate \
  --checkpoint runs/diffusion_v2/best.pt --context context.png \
  --trajectory trajectory.npz --output runs/diffusion_v2/planned
```

Use `--context-is-preprocessed` for an already cropped/grayscale context image.
The checkpoint embeds preprocessing settings for raw image inputs. Long motions
outside the training interval/distribution remain extrapolation experiments.

## Tests and recorded experiment

```bash
.venv/bin/python -m pytest -q tests/test_mocap.py
```

The suite checks raw image stride/endian/channel handling, real H.264
encode/decode including delayed frames, pose interpolation and tracking gaps,
calibration chains, global-frame invariance, split leakage, episode-local pair
sampling, InfoNCE pairing, motion/projector gradients, checkpoint reload,
representation evaluation, diffusion freezing,
metrics, and actual training CLI resume equivalence.

## Diffusion image decoder

After world-model training has completed and `best.pt` is fixed
(`world_checkpoint` in the config):

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m mocap.decoder.diffusion \
  --config configs/archive/mocap_diffusion_full_v2.json --output runs/diffusion_v2 > runs/diffusion_v2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m mocap.evaluation.evaluate \
  --checkpoint runs/diffusion_v2/best.pt --split val --output runs/diffusion_v2/evaluation

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m mocap.decoder.generate \
  --checkpoint runs/diffusion_v2/best.pt \
  --episode data/ifl_full/episodes/yuan/bag_yuan0_2026-10-24_16-20-02 \
  --start-index 100 --stride 10 --frames 8 --output runs/diffusion_v2/trajectory
```

The same `--resume`, `--max-steps`, `patience`, SIGINT/SIGTERM conventions apply. Diffusion
checkpoints include the frozen world-model weights and preprocessing, so image
generation needs only the checkpoint, initial ultrasound and trajectory. Resume
also checks the original world checkpoint and dataset hashes. The default sampler
uses 40 DDIM steps, guidance 1.5 and seed 42. Motion/zero-motion comparisons use
identical initial noise, so their difference is due to conditioning. The fixed
seed is also reused across generated trajectory frames. No target frame is used
as an inference starting image or denoising input.

## Paired test: direct variable-step prediction versus fixed-step rollout

Compare the two **diffusion checkpoints** on identical random starts and targets.
A receives the initial real image and the complete relative probe motion once.
B receives the same initial image, then feeds back its own predictions along the
measured trajectory at 200 ms intervals; future ultrasound images are never inputs.

Run from `EchoWorld/`:

```bash
.venv/bin/python -m mocap.evaluation.compare_rollout \
  --checkpoint-a runs/diffusion_alignment_aligned/best.pt \
  --checkpoint-b runs/diffusion_dt200ms/best.pt \
  --manifest data/ifl_aligned/manifest.json \
  --output runs/compare_direct_vs_rollout_test_new \
  --split test --pairs-per-episode 32 --seed 42 --noise-seed 42 \
  --step 0.2 --target-tolerance 0.01 --max-horizon 3.0 \
  --batch-size 32 --device cuda:0 --amp
```

Sampling is independent of checkpoint predictions and reproducible from the seed.
Within each episode, choose eligible starts uniformly **without replacement**,
then choose uniformly among available nominal target horizons
**0.2, 0.4, …, 2.8 seconds**. Use the nearest valid real target frame within
±10 ms. Both nominal and actual horizons must be strictly below 3 seconds.
Short segments remain eligible, so the overall horizon distribution is conditional
on how much valid future is available after each random start, not globally uniform.
The usual brightness threshold is applied to source and target frames.

Intermediate poses use linear translation and SLERP on the cached recorded trajectory.
B's intermediate steps are exactly 200 ms; the last step absorbs timestamp jitter
and remains in **190–210 ms**. There is no extra short residual step, including when
the actual target is just after a multiple of 200 ms. Neither sampling nor
interpolation crosses a tracking segment or a cached pose gap above 100 ms.
The evaluator rejects a step/tolerance outside B's training interval and flags
A endpoints outside A's training interval.

As in the existing rollout videos, generated feedback is quantised to uint8.
Both final predictions are scored before quantisation. All calls use the same
seeded diffusion noise (`shared_noise=True`), including the zero-action controls.
B's zero-action control repeats the same number of autoregressive steps as B.
The endpoint metrics compare the complete systems, including their decoders.

Outputs:

- `protocol.json`: checkpoint and manifest hashes, configuration, sampling rules,
  evaluator hash, software versions, and per-episode sample coverage.
- `samples.json`: exact source/target frame IDs and timestamps, nominal/actual
  durations, step durations, all actions, and training-range flags.
- `per_pair_metrics.csv`: MAE, MSE, PSNR, local SSIM and foreground MAE for
  A, B, context copying, A with zero action, and B with zero action.
- `metrics.json`: equal-episode means, B−A differences with paired episode
  bootstrap 95% confidence intervals, win fractions, gains over baselines,
  and breakdowns by nominal horizon and endpoint translation.
- `report.md`, `error_by_horizon.png` / `.pdf`: readable results and horizon curves.
- `comparison_*.png`: the first sampled pair in each episode, selected without
  inspecting results.

Existing output directories are refused. Checkpoints are selected on validation;
this command does not train or select models on the test set. Bootstrap intervals
describe variation across the tested episodes, not across independent training runs
or unseen subjects (the current test manifest contains only the `xuesong` group).

Focused protocol tests:

```bash
.venv/bin/python -m pytest -q tests/test_compare_rollout.py tests/test_temporal_eval.py
```

## Motion dependence: correct, zero and wrong actions

Run both diffusion models in **one direct step** on the same held-out image pairs:

```bash
.venv/bin/python -u -m mocap.evaluation.compare_motion \
  --checkpoint-a runs/diffusion_alignment_aligned/best.pt \
  --checkpoint-b runs/diffusion_dt200ms/best.pt \
  --manifest data/ifl_aligned/manifest.json \
  --output runs/motion_ablation_test_new \
  --split test --pairs-per-episode 32 --seed 42 --noise-seed 42 \
  --step 0.2 --target-tolerance 0.01 --max-horizon 3.0 \
  --batch-size 32 --device cuda:0 --amp
```

The sampler and its defaults match `compare_rollout`, so using the same manifest,
seed and sampling parameters selects the same pairs. Both A and B always start
from the real source image; neither performs a rollout in this experiment.

The three conditions are the recorded relative action, a zero action, and a
synthetic wrong action with negated translation and rotation vectors. The wrong
action preserves translation magnitude and rotation angle exactly, while reversing
their directions; it is not the SE(3) inverse, which would also change the
translation's coordinate frame. All conditions and both models use the same noise
seed with shared noise. Exactly zero recorded actions have no distinct wrong
control and are excluded from action-effect statistics, while their image metrics
remain in reconstruction averages.

Outputs include `samples.json` with correct/wrong vectors, `protocol.json` with
checkpoint/manifest/code hashes, `per_pair_metrics.csv`, `metrics.json`,
`report.md`, and example comparison PNGs. MAE, MSE, foreground MAE, PSNR and SSIM
are computed on the final decoded images. The report includes paired episode
bootstrap intervals for the correct-action advantage over zero and wrong motion,
and for B's advantage minus A's advantage. Positive advantage always means that
the correct action improves prediction (control minus correct for errors,
correct minus control for PSNR/SSIM). Episodes are equally weighted.

Interpret the per-horizon results and training-range flags: A was trained at
0.4–3 s and B at 0.19–0.21 s, so they have no common training horizon. This diagnostic
measures action dependence under a common inference protocol; pooled scores
alone cannot rank their intrinsic ability to learn motion. It tests the full
encoder/predictor/decoder system, not just latent representations.

```bash
.venv/bin/python -m pytest -q tests/test_compare_motion.py tests/test_compare_rollout.py
```
