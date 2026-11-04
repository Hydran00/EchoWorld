"""Single-GPU training with atomic checkpoints and graceful SIGINT/SIGTERM stop.

Per step: sample B pairs -> MotionWorldModel.forward (spatial + motion losses) -> AdamW
update of encoder, predictor (with policy_net) and projector -> EMA update of the target
encoder/projector. Periodic validation in representation space selects best.pt and, with
`patience`, stops training once the selection metric has not improved for that many validations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import torch
from torch.utils.data import DataLoader, Sampler
from .dataset import MocapPairs
from .model import MotionWorldModel
from .evaluate import evaluate_world


class StepBatches(Sampler):
    # Batches are sampled with replacement; the generator is reseeded from the global step, so
    # batch content depends only on (seed, step) and not on how many times training restarted.
    """Resume-identical samples and pair choices, independent of worker prefetch."""
    def __init__(self, length, batch_size, start, stop, seed):
        self.length, self.batch_size, self.start, self.stop, self.seed = length, batch_size, start, stop, seed

    def __iter__(self):
        for step in range(self.start, self.stop):
            g = torch.Generator().manual_seed(self.seed + step)
            indices = torch.randint(self.length, (self.batch_size,), generator=g).tolist()
            seeds = torch.randint(2**31 - 1, (self.batch_size,), generator=g).tolist()
            yield list(zip(indices, seeds))

    def __len__(self):
        return self.stop - self.start


def save_checkpoint(path, model, optimizer, config, step, best, manifest_hash, device, stale=0):
    manifest_path = config.get('data', {}).get('manifest')
    preprocessing = json.loads(Path(manifest_path).read_text()).get('settings', {}) if manifest_path else {}
    # best_score is "higher is better" (negated for val_loss); stale counts validations without improvement.
    state = dict(format_version=3, config=config, step=step, best_score=best, stale_validations=stale,
                 preprocessing=preprocessing,
                 manifest_sha256=manifest_hash, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def lr_at(step, config):
    # Linear warm-up followed by cosine decay to 10% of the peak learning rate.
    warmup, total = config['warmup_steps'], config['steps']
    if step < warmup:
        return config['lr'] * (step + 1) / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return config['lr'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


def wd_at(step, config):
    # Cosine weight-decay schedule of the released I-JEPA/EchoWorld code (CosineWDSchedule):
    # weight_decay -> final_weight_decay over the run, increasing regularization towards the end.
    # Without final_weight_decay the decay stays constant.
    start = config.get('weight_decay', .05)
    final = config.get('final_weight_decay', start)
    progress = min(1., step / max(1, config['steps']))
    return final + (start - final) * .5 * (1 + math.cos(math.pi * progress))


def same_run(saved, config):
    # Resume requires an identical configuration, except for `patience`, which only decides when
    # to stop and does not alter the training trajectory (it may be added to a running experiment).
    strip = lambda c: {k: v for k, v in c.items() if k != 'patience'}
    return strip(saved) == strip(config)


def selection_score(metrics, config):
    # Model selection metric, oriented so that higher is better.
    name = config.get('selection', 'val_loss')
    return -metrics[name] if name.startswith('val_') else metrics[name]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True, type=Path)
    checkpoint = p.add_mutually_exclusive_group()
    checkpoint.add_argument('--resume', type=Path)
    checkpoint.add_argument('--init-from', type=Path, help='Initialize weights for a new run with fresh optimizer/schedule')
    p.add_argument('--max-steps', type=int, help='Stop at this global step without changing the configured LR schedule')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(config.get('cpu_threads', 4))
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('high')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and args.resume is None:
        raise FileExistsError('Run already has a checkpoint; use --resume or a new output directory')
    manifest_hash = hashlib.sha256(Path(config['data']['manifest']).read_bytes()).hexdigest()
    train = MocapPairs(split='train', image_size=config['model']['image_size'], **config['data'])
    val_data = dict(config['data'], max_pairs_per_episode=config.get('val_pairs_per_episode', 32))
    val = MocapPairs(split='val', image_size=config['model']['image_size'], **val_data)
    model = MotionWorldModel(**config['model']).to(device)
    # As in the released helper.init_opt, biases and 1-D parameters (norms, tokens) are not decayed.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        dict(params=[p for n, p in trainable if 'bias' not in n and p.ndim != 1], weight_decay=wd_at(0, config)),
        dict(params=[p for n, p in trainable if 'bias' in n or p.ndim == 1], weight_decay=0., wd_exclude=True),
    ], lr=config['lr'])
    step, best, stale = 0, -float('inf'), 0
    if args.init_from:
        initial = torch.load(args.init_from, map_location='cpu', weights_only=False)
        if initial.get('kind') == 'diffusion' or initial['config']['model'] != config['model']:
            raise ValueError('Initialization requires a matching world-model architecture')
        model.load_state_dict(initial['model'], strict=True)
        (output / 'initialization.json').write_text(json.dumps(dict(checkpoint=str(args.init_from.resolve()),
            step=initial['step'], source_manifest_sha256=initial['manifest_sha256']), indent=2) + '\n')
        print(f'Initialized weights from {args.init_from}; optimizer and step reset', flush=True)
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if not same_run(state['config'], config) or state['manifest_sha256'] != manifest_hash:
            raise ValueError('Resume config or dataset manifest changed; use the exact saved configuration')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, best, stale = state['step'], state['best_score'], state['stale_validations']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        print(f'Resumed global step {step}', flush=True)
    stop_at = min(config['steps'], args.max_steps) if args.max_steps else config['steps']
    if stop_at <= step:
        raise ValueError('Stop step must exceed the saved step')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'run_info.json').write_text(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        train_pairs=len(train), val_pairs=len(val), manifest_sha256=manifest_hash,
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)), indent=2) + '\n')
    # Each loader item is a full batch of dicts with source/target images and both motion directions.
    loader = DataLoader(train, batch_sampler=StepBatches(len(train), config['batch_size'], step, stop_at, config['seed']),
        num_workers=config.get('workers', 2), pin_memory=device.type == 'cuda',
        generator=torch.Generator().manual_seed(config['seed']), persistent_workers=config.get('workers', 2) > 0)
    stopping = []

    def stop(signum, _):
        stopping.append(signum)
        print(f'Received signal {signum}; saving after the current optimizer step.', flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    (output / 'pid.txt').write_text(str(__import__('os').getpid()) + '\n')
    amp = config.get('amp', True) and device.type == 'cuda'
    model.train()
    early_stop = False
    started = time.monotonic()
    start_step = step
    print(f'Training {len(train)} pairs, validating {len(val)} pairs on {device}; steps {step} -> {stop_at}', flush=True)
    with (output / 'train.jsonl').open('a') as log:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = lr_at(step, config)
                if not group.get('wd_exclude'):
                    group['weight_decay'] = wd_at(step, config)
            source, target, action, reverse = (batch[k].to(device, non_blocking=True)
                                               for k in ['source', 'target', 'action', 'reverse_action'])
            # bfloat16 autocast for the transformer computation; losses are evaluated in float32.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss, values = model(source, target, action, reverse)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            # EMA momentum schedule 0.996 -> 1.0 (cosine), as in the paper.
            momentum = .996 + .004 * .5 * (1 - math.cos(math.pi * step / max(1, config['steps'])))
            model.update_target(momentum)
            step += 1
            if step % config.get('log_every', 10) == 0 or step == start_step + 1:
                row = dict(step=step, **{k: float(v.detach()) for k, v in values.items()},
                           grad_norm=float(grad_norm), lr=optimizer.param_groups[0]['lr'],
                           weight_decay=optimizer.param_groups[0]['weight_decay'],
                           seconds=time.monotonic() - started, steps_per_second=(step - start_step) / (time.monotonic() - started))
                if device.type == 'cuda':
                    row['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 1e9
                log.write(json.dumps(row) + '\n'); log.flush()
                print(json.dumps(row), flush=True)
            do_eval = step % config['eval_every'] == 0 or step == stop_at
            if do_eval and not stopping:
                metrics = evaluate_world(model, val, device, output / f'val_{step:06d}',
                                         batch_size=config.get('eval_batch_size', 32), amp=amp)
                print('Validation ' + json.dumps(dict(step=step, **metrics)), flush=True)
                # Model selection on the validation split only; the test split is never used here.
                score = selection_score(metrics, config)
                if score > best:
                    best, stale = score, 0
                    save_checkpoint(output / 'best.pt', model, optimizer, config, step, best, manifest_hash, device, stale)
                else:
                    stale += 1
                if config.get('patience') and stale >= config['patience']:
                    early_stop = True
                    print(f'Early stopping: {config.get("selection", "val_loss")} did not improve for '
                          f'{stale} validations', flush=True)
            if step % config['save_every'] == 0 or step == stop_at or stopping or early_stop:
                save_checkpoint(output / 'last.pt', model, optimizer, config, step, best, manifest_hash, device, stale)
            if stopping or early_stop:
                break
    print(f'Stopped cleanly at step {step}; checkpoint {output / "last.pt"}', flush=True)
    (output / 'status.json').write_text(json.dumps(dict(step=step, stop_signal=stopping[0] if stopping else None,
                                                      early_stopping=early_stop, best_score=best,
                                                      elapsed_seconds=time.monotonic() - started), indent=2) + '\n')


if __name__ == '__main__':
    main()
