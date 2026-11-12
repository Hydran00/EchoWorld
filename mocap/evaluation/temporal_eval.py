"""Evaluate complete anchored trajectories, including constant-copy temporal baselines.

Temporal errors compare predicted and recorded changes at the same pixel coordinates.
They measure motion *and* spatial fidelity; they are not ground-truth optical flow.
Low-pass versions reduce sensitivity to unpredictable ultrasound speckle. Every clip
uses one fixed initial noise image and no future ultrasound as model input.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mocap.processing.dataset import MocapPairs
from mocap.evaluation.evaluate import image_metrics, load_model
from mocap.processing.geometry import relative_action
from mocap.visualization.visualize import compose, motion_panel, save_video, to_pixels


def temporal_metrics(prediction, target):
    """Sequences [T,1,H,W], intensities [0,1]. Perfect=0 error/1 gain, static=0 gain."""
    if prediction.shape != target.shape or prediction.ndim != 4 or len(target) < 2:
        raise ValueError('Expected matching [T,1,H,W] sequences with T >= 2')
    prediction, target = prediction.float(), target.float()
    mask = (target.mean(dim=0, keepdim=True) > .05).float()
    result = {}
    for kernel in (1, 7, 15):
        if kernel == 1:
            p, t = prediction, target
        else:
            p = F.avg_pool2d(prediction, kernel, 1, kernel // 2)
            t = F.avg_pool2d(target, kernel, 1, kernel // 2)
        dp, dt = p.diff(dim=0), t.diff(dim=0)
        count = mask.sum().clamp_min(1) * len(dp)
        target_motion = (dt.abs() * mask).sum() / count
        error = ((dp - dt).abs() * mask).sum() / count
        model_motion = (dp.abs() * mask).sum() / count
        a, b = (dp * mask).flatten(), (dt * mask).flatten()
        cosine = (a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)
        prefix = f'temporal_k{kernel}'
        result.update({f'{prefix}_mae': float(error), f'{prefix}_target_change': float(target_motion),
            f'{prefix}_prediction_change': float(model_motion), f'{prefix}_cosine': float(cosine),
            f'{prefix}_gain_over_static': float(1 - error / target_motion) if target_motion > 1e-8 else None})
    return result


def select_clips(dataset, count=1, horizon=3., step=.1):
    """Fixed evenly spaced valid sources in each episode, independent of model outputs."""
    if count < 1 or horizon < 2 * step or step <= 0:
        raise ValueError('Invalid clip count or time interval')
    clips = []
    for ei, episode in enumerate(dataset.episodes):
        times, segments = episode['times'], episode['segments']
        candidates = []
        for item_ei, source, _ in dataset.items:
            if item_ei != ei:
                continue
            end = int(np.searchsorted(times, times[source] + round(horizon * 1e9)))
            if end < len(times) and segments[end] == segments[source]:
                candidates.append(source)
        if not candidates:
            continue
        slots = sorted(set(int((j + .5) * len(candidates) / min(count, len(candidates)))
                           for j in range(min(count, len(candidates)))))
        for slot in slots:
            source, indices = candidates[slot], []
            for k in range(1, int(round(horizon / step)) + 1):
                wanted = times[source] + round(k * step * 1e9)
                right = int(np.searchsorted(times, wanted))
                near = [j for j in (right - 1, right) if 0 <= j < len(times)]
                j = min(near, key=lambda j: abs(int(times[j]) - wanted))
                if (segments[j] != segments[source] or abs(int(times[j]) - wanted) > step / 2 * 1e9
                        or j <= source or (indices and j <= indices[-1])):
                    indices = []
                    break
                indices.append(j)
            if len(indices) >= 2:
                clips.append((ei, source, indices))
    return clips


def sequence_metrics(prediction, target, horizons, min_dt):
    result = {k: float(v.mean()) for k, v in image_metrics(prediction, target).items()}
    result.update(temporal_metrics(prediction, target))
    inside = horizons >= min_dt
    if int(inside.sum()) >= 2:
        result.update({f'in_range_{k}': v for k, v in temporal_metrics(prediction[inside], target[inside]).items()})
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, help='Explicit evaluation cache for matched comparisons')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--clips-per-episode', type=int, default=1)
    p.add_argument('--horizon', type=float, default=3.)
    p.add_argument('--step', type=float, default=.1)
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--videos', type=int, default=3, help='First N clips per seed, never cherry-picked')
    args = p.parse_args()
    if (args.output / 'metrics.json').exists():
        raise FileExistsError('Use a new output directory to preserve prior evaluations')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    model, state = load_model(args.checkpoint, device)
    if state.get('kind') != 'diffusion':
        raise ValueError('Temporal image evaluation requires a diffusion checkpoint')
    data = dict(state['config']['data'], max_pairs_per_episode=0)
    if args.split == 'val' and state['config'].get('validation_manifest'):
        data['manifest'] = state['config']['validation_manifest']
    if args.manifest:
        data['manifest'] = args.manifest
    dataset = MocapPairs(split=args.split, image_size=model.image_size, **data)
    clips = select_clips(dataset, args.clips_per_episode, args.horizon, args.step)
    if not clips:
        raise ValueError('No complete valid clips found')
    rows = []
    for seed in args.seeds:
        model.noise_seed = seed
        for clip_id, (ei, source, indices) in enumerate(clips):
            episode = dataset.episodes[ei]
            context = dataset.image(ei, source)[None].to(device)
            targets = torch.stack([dataset.image(ei, i) for i in indices]).to(device)
            actions = torch.from_numpy(relative_action(episode['poses'][source], episode['poses'][indices])).to(device)
            horizons = torch.tensor((episode['times'][indices] - episode['times'][source]) / 1e9, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == 'cuda'):
                zero = model.predict(context, torch.zeros_like(actions[:1]), shared_noise=True).float()
                predictions = torch.cat([model.predict(context.expand(len(a), -1, -1, -1), a, shared_noise=True).float()
                                         for a in actions.split(args.batch_size)])
            row = dict(episode=episode['episode_id'], source_index=source, target_indices=indices,
                       seed=seed, frames=len(indices), mean_translation_mm=float(actions[:, :3].norm(dim=1).mean()))
            for name, images in [('model', predictions), ('copy_context', context.expand_as(targets)),
                                 ('zero_action', zero.expand_as(targets))]:
                row[name] = sequence_metrics(images, targets, horizons, dataset.min_dt)
            row['zero_action_source_mae'] = float((zero - context).abs().mean())
            stem = f'clip_{clip_id:03d}_seed_{seed}'
            np.savez_compressed(args.output / f'{stem}.npz', prediction=predictions.cpu().numpy(),
                target=targets.cpu().numpy(), source=context.cpu().numpy(), zero=zero.cpu().numpy(),
                action=actions.cpu().numpy(), horizons=horizons.cpu().numpy())
            if clip_id < args.videos:
                frames = []
                for i, (prediction, target) in enumerate(zip(predictions.cpu(), targets.cpu())):
                    panels = [to_pixels(zero[0].cpu()), to_pixels(context[0].cpu()), to_pixels(prediction), to_pixels(target),
                              to_pixels(3 * (prediction - target).abs()),
                              motion_panel(episode['poses'], episode['times'], source, indices[-1], model.image_size, current=indices[i])]
                    frames.append(np.asarray(compose([panels], header=f'{episode["episode_id"]} seed={seed} t={float(horizons[i]):.2f}s')))
                save_video(frames, args.output / f'{stem}.mp4', fps=10)
            rows.append(row)
            print(json.dumps(dict(clip=clip_id, seed=seed, episode=episode['episode_id'],
                mae=row['model']['mae'], copy_mae=row['copy_context']['mae'],
                temporal_gain=row['model']['temporal_k7_gain_over_static'])), flush=True)
    aggregates = {}
    # Equal clip weighting; every episode has a fixed number of sources when sufficient data exist.
    for name in ('model', 'copy_context', 'zero_action'):
        aggregates[name] = {k: float(np.mean([r[name][k] for r in rows if r[name].get(k) is not None]))
                            for k in rows[0][name] if any(r[name].get(k) is not None for r in rows)}
    result = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_step=state['step'], split=args.split,
        manifest_sha256=hashlib.sha256(dataset.manifest_path.read_bytes()).hexdigest(),
        horizon=args.horizon, step=args.step, seeds=args.seeds, amp=args.amp, clips=len(clips),
        aggregates=aggregates, per_clip=rows)
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(aggregates, indent=2), flush=True)


if __name__ == '__main__':
    main()
