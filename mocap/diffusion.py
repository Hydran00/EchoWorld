"""Representation-conditioned diffusion decoder, following paper C.3, built on Hugging Face diffusers.

The U-Net (diffusers.UNet2DModel) and the noise schedule / DDIM sampler (diffusers schedulers)
come from the library; this module only wires EchoWorld conditioning into them. It is not the
unreleased RCDM code or weights. It sees predicted features, never target pixels at inference.
Frozen EchoWorld weights are embedded in its checkpoints.

Data flow, following the representation-conditioned decoding of the paper (C.3):

    I_a, motion a -> b --frozen world model--> h_hat [B, 384]   predictor output (predict_latent)
    h_hat --MLP--> c [B, 4*base], added to the U-Net timestep embedding
    noisy I_b x_t [B, 1, S, S] --UNet2DModel, scale/shift (FiLM) by the embedding in every block--> v (or epsilon)

Training minimizes the standard denoising error only; it changes the denoiser, never the world
model. Sampling starts from Gaussian noise and runs DDIM, so the target image is never an input
at inference. Because the condition is a single pooled vector, fine texture (speckle) is
sampled from the learned distribution rather than reconstructed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from .dataset import MocapPairs
from .model import MotionWorldModel
from .train import StepBatches, lr_at, same_run

class Denoiser(nn.Module):
    """diffusers UNet2DModel conditioned on an EchoWorld representation."""
    def __init__(self, latent_dim, image_size, base=32):
        super().__init__()
        # The "identity" class embedding is added to the timestep embedding (dimension 4 * base),
        # which every residual block applies as a per-channel scale and shift
        # (resnet_time_scale_shift="scale_shift"), the analogue of RCDM's conditional normalization.
        self.condition = nn.Sequential(nn.Linear(latent_dim, 512), nn.SiLU(), nn.Linear(512, 4 * base))
        self.unet = UNet2DModel(
            sample_size=image_size, in_channels=1, out_channels=1,
            block_out_channels=(base, 2 * base, 4 * base, 6 * base), layers_per_block=1,
            down_block_types=('DownBlock2D',) * 4, up_block_types=('UpBlock2D',) * 4,
            norm_num_groups=8, class_embed_type='identity', resnet_time_scale_shift='scale_shift')

    def forward(self, noisy, timestep, latent):
        return self.unet(noisy, timestep, class_labels=self.condition(latent)).sample


class DiffusionUltrasound(nn.Module):
    def __init__(self, world_config, base=32, sample_steps=40, guidance=1.5, noise_seed=42, prediction_type='epsilon'):
        super().__init__()
        if prediction_type not in ('epsilon', 'v'):
            raise ValueError('prediction_type must be epsilon or v')
        self.world = MotionWorldModel(**world_config).requires_grad_(False)
        self.denoiser = Denoiser(world_config['embed_dim'], world_config['image_size'], base)
        self.ema_denoiser = copy.deepcopy(self.denoiser).requires_grad_(False)
        self.sample_steps, self.guidance, self.noise_seed = sample_steps, guidance, noise_seed
        self.prediction_type = prediction_type
        self.image_size = world_config['image_size']
        # Cosine noise schedule (Nichol & Dhariwal) with 1000 training steps.
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                       prediction_type='v_prediction' if prediction_type == 'v' else 'epsilon')

    def train(self, mode=True):
        super().train(mode)
        self.world.eval()
        self.ema_denoiser.eval()
        return self

    def loss(self, source, target, action):
        with torch.no_grad():
            latent = self.world.predict_latent(source, action)
            # Classifier-free guidance: the condition is dropped (zeroed) for 10% of the samples.
            latent = latent * (torch.rand(len(source), 1, device=source.device) > .1)
        clean = 2 * target - 1                                   # images rescaled to [-1, 1]
        noise = torch.randn_like(clean)
        timestep = torch.randint(self.scheduler.config.num_train_timesteps, (len(source),), device=source.device)
        noisy = self.scheduler.add_noise(clean, noise, timestep)
        objective = noise if self.prediction_type == 'epsilon' else self.scheduler.get_velocity(clean, noise, timestep)
        return F.mse_loss(self.denoiser(noisy, timestep, latent).float(), objective.float())

    @torch.no_grad()
    def update_target(self, decay=.999):
        for p, q in zip(self.denoiser.parameters(), self.ema_denoiser.parameters()):
            q.lerp_(p, 1 - decay)

    @torch.no_grad()
    def predict(self, source, action, shared_noise=False):
        # Deterministic generation: seeded initial noise, DDIM (eta = 0) with the EMA denoiser and
        # classifier-free guidance, output = uncond + w (cond - uncond). Motion and zero-motion
        # predictions share the same noise, so their difference is due to the conditioning only.
        # shared_noise=True uses one noise image for the whole batch (videos along a trajectory),
        # so that consecutive frames differ only through their conditioning.
        latent = self.world.predict_latent(source, action)
        generator = torch.Generator(device=source.device).manual_seed(self.noise_seed)
        shape = (1 if shared_noise else len(source), 1, self.image_size, self.image_size)
        x = torch.randn(shape, device=source.device, generator=generator).expand(len(source), -1, -1, -1).contiguous()
        sampler = DDIMScheduler.from_config(self.scheduler.config, timestep_spacing='trailing')
        sampler.set_timesteps(self.sample_steps, device=source.device)
        for t in sampler.timesteps:
            timestep = t.expand(len(source))
            if self.guidance == 1:
                output = self.ema_denoiser(x, timestep, latent)
            else:
                unconditional, conditional = self.ema_denoiser(torch.cat([x, x]), timestep.repeat(2),
                    torch.cat([torch.zeros_like(latent), latent])).chunk(2)
                output = unconditional + self.guidance * (conditional - unconditional)
            x = sampler.step(output.float(), int(t), x).prev_sample
        return ((x + 1) / 2).clamp(0, 1)


def main():
    from .evaluate import evaluate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    device = torch.device(args.device)
    torch.set_num_threads(config.get('cpu_threads', 4))
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('high')
    world_state = torch.load(config['world_checkpoint'], map_location='cpu', weights_only=False)
    # Resolved config enables the shared evaluation and generation CLIs.
    config['model'] = world_state['config']['model']
    # The decoder learns to generate uncropped frames: world-model training augmentation is not inherited.
    config['data'] = {k: v for k, v in world_state['config']['data'].items() if k != 'crop_scale'}
    manifest_hash = hashlib.sha256(Path(config['data']['manifest']).read_bytes()).hexdigest()
    world_hash = hashlib.sha256(Path(config['world_checkpoint']).read_bytes()).hexdigest()
    model = DiffusionUltrasound(config['model'], **config['diffusion']).to(device)
    model.world.load_state_dict(world_state['model'], strict=True)
    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=config['lr'], weight_decay=.01)
    # best: lowest validation image MAE; stale: validations since it last improved (for `patience`).
    step, best, stale = 0, float('inf'), 0
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'last.pt').exists() and args.resume is None:
        raise FileExistsError('Existing run: supply --resume')
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if (not same_run(state['config'], config) or state['manifest_sha256'] != manifest_hash or
                state['world_sha256'] != world_hash):
            raise ValueError('Diffusion resume inputs changed')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, best, stale = state['step'], state['best_val_mae'], state.get('stale_validations', 0)
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        print(f'Resumed diffusion at step {step}', flush=True)
    train = MocapPairs(split='train', image_size=config['model']['image_size'], **config['data'])
    val = MocapPairs(split='val', image_size=config['model']['image_size'],
                     **dict(config['data'], max_pairs_per_episode=config.get('val_pairs_per_episode', 8)))
    stop_at = min(config['steps'], args.max_steps) if args.max_steps else config['steps']
    if stop_at <= step:
        raise ValueError('Stop step must exceed saved step')
    loader = DataLoader(train, batch_sampler=StepBatches(len(train), config['batch_size'], step, stop_at, config['seed']),
                        num_workers=config.get('workers', 2), pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(config['seed']))
    stopping = []
    def stop(signum, _):
        stopping.append(signum)
        print(f'Received {signum}; saving diffusion after this step', flush=True)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    (args.output / 'pid.txt').write_text(str(__import__('os').getpid()))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    def save(name):
        state = dict(kind='diffusion', model=model.state_dict(), optimizer=optimizer.state_dict(),
            step=step, best_val_mae=best, stale_validations=stale, config=config, manifest_sha256=manifest_hash, world_sha256=world_hash,
            preprocessing=world_state['preprocessing'], torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        temp = args.output / (name + '.tmp')
        torch.save(state, temp)
        temp.replace(args.output / name)

    model.train()
    early_stop = False
    started, start_step = time.monotonic(), step
    with (args.output / 'train.jsonl').open('a') as log:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = lr_at(step, config)
            source, target, action = (batch[k].to(device) for k in ['source', 'target', 'action'])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = model.loss(source, target, action)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite diffusion loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            model.update_target()
            step += 1
            if step % config['log_every'] == 0:
                row = dict(step=step, denoising_mse=float(loss.detach()), prediction_type=model.prediction_type,
                           grad_norm=float(grad), lr=optimizer.param_groups[0]['lr'], seconds=time.monotonic() - started,
                           steps_per_second=(step - start_step) / (time.monotonic() - started))
                log.write(json.dumps(row) + '\n'); log.flush()
                print(json.dumps(row), flush=True)
            if (step % config['eval_every'] == 0 or step == stop_at) and not stopping:
                metrics = evaluate(model, val, device, args.output / f'val_{step:06d}', batch_size=config.get('eval_batch_size', 8))
                print('Validation ' + json.dumps({k: v for k, v in metrics.items() if k != 'displayed_pairs'}), flush=True)
                # Selection and early stopping on held-out image MAE against the true target frame.
                if metrics['model_mae'] < best:
                    best, stale = metrics['model_mae'], 0
                    save('best.pt')
                else:
                    stale += 1
                if config.get('patience') and stale >= config['patience']:
                    early_stop = True
                    print(f'Early stopping: validation MAE did not improve for {stale} validations', flush=True)
            if step % config['save_every'] == 0 or step == stop_at or stopping or early_stop:
                save('last.pt')
            if stopping or early_stop:
                break
    (args.output / 'status.json').write_text(json.dumps(dict(step=step, stop_signal=stopping[0] if stopping else None,
                                                             early_stopping=early_stop, best_val_mae=best), indent=2) + '\n')
    print(f'Diffusion stopped cleanly at {step}', flush=True)


if __name__ == '__main__':
    main()
