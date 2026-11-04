"""Evaluate held-out pairs: representation metrics (world model) or image comparisons (diffusion)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .dataset import MocapPairs
from .geometry import relative_action
from .model import MotionWorldModel
from .visualize import compose, motion_panel, save_grids, save_video, to_pixels


def load_model(checkpoint, device):
    # Only load locally produced, trusted checkpoints.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('kind') == 'diffusion':
        from .diffusion import DiffusionUltrasound
        model = DiffusionUltrasound(state['config']['model'], **state['config']['diffusion']).to(device)
    else:
        model = MotionWorldModel(**state['config']['model']).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    return model, state


def image_metrics(prediction, target):
    prediction, target = prediction.float(), target.float()
    error = prediction - target
    mse = error.square().mean(dim=(1, 2, 3))
    fg = (target > .05).float()
    fg_mae = (error.abs() * fg).sum(dim=(1, 2, 3)) / fg.sum(dim=(1, 2, 3)).clamp_min(1)
    # Local SSIM, 7x7 population moments, [0,1] dynamic range.
    mu_x, mu_y = F.avg_pool2d(prediction, 7, 1), F.avg_pool2d(target, 7, 1)
    vx = (F.avg_pool2d(prediction.square(), 7, 1) - mu_x.square()).clamp_min(0)
    vy = (F.avg_pool2d(target.square(), 7, 1) - mu_y.square()).clamp_min(0)
    cov = F.avg_pool2d(prediction * target, 7, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + .01**2) * (2 * cov + .03**2) /
            ((mu_x.square() + mu_y.square() + .01**2) * (vx + vy + .03**2)))
    return dict(mae=error.abs().mean(dim=(1, 2, 3)), mse=mse,
                psnr=-10 * mse.clamp_min(1e-12).log10(), foreground_mae=fg_mae,
                ssim=ssim.mean(dim=(1, 2, 3)))


def displayed_indices(dataset, count):
    """Dataset indices of the pairs shown in the comparison grids.

    The `count` rows are distributed round-robin over the episodes of the split, and within each
    episode evenly over its pairs, so that the grids cover subjects, scans and time.
    """
    by_episode = {}
    for index, (ei, _, _) in enumerate(dataset.items):
        by_episode.setdefault(ei, []).append(index)
    counts, remaining = dict.fromkeys(by_episode, 0), min(count, len(dataset.items))
    while remaining:
        for ei, group in by_episode.items():
            if remaining and counts[ei] < len(group):
                counts[ei] += 1
                remaining -= 1
    chosen = []
    for ei, group in by_episode.items():
        k = counts[ei]
        chosen.extend(group[int((j + .5) * len(group) / k)] for j in range(k))
    return chosen


@torch.no_grad()
def save_videos(model, dataset, device, output, count=10, step=.1, fps=5, batch_size=16, amp=True):
    """Anchored generation along a recorded trajectory, one MP4 per selected validation episode.

    The context frame is fixed (as in generate.py, anchored mode) and video frame k shows the
    prediction for the measured motion from the context to the recorded frame closest to
    t0 + k * step, up to the maximum pair interval. All frames share one initial noise image, so
    the changes along the video are due to the motion conditioning only. Frames with a horizon
    shorter than the minimum training interval are flagged, since such pairs are never seen in
    training. Playback is at `fps`; with step = 0.1 s and fps = 5 the video runs at half speed.
    """
    output = Path(output)
    episodes = sorted({ei for ei, _, _ in dataset.items})
    chosen = sorted({episodes[i] for i in np.linspace(0, len(episodes) - 1, min(count, len(episodes))).round().astype(int)})
    written = []
    for n, ei in enumerate(chosen):
        items = [item for item in dataset.items if item[0] == ei]
        _, source, _ = items[len(items) // 2]
        episode = dataset.episodes[ei]
        times, segments, poses = episode['times'], episode['segments'], episode['poses']
        frames = []
        for k in range(1, int(round(dataset.max_dt / step)) + 1):
            wanted = times[source] + round(k * step * 1e9)
            j = int(np.argmin(abs(times - wanted)))
            # Stop at the end of the episode or of the tracking segment; never cross a gap.
            # Instants without a recorded frame within step / 2 are skipped.
            if wanted > times[-1] or segments[j] != segments[source]:
                break
            if abs(int(times[j]) - wanted) <= step / 2 * 1e9 and j > source and (not frames or j != frames[-1]):
                frames.append(j)
        if not frames:
            continue
        context = dataset.image(ei, source)[None].to(device)
        actions = torch.from_numpy(relative_action(poses[source], poses[np.array(frames)])).to(device)
        predictions = []
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
            zero = model.predict(context, torch.zeros_like(actions[:1]), shared_noise=True)[0].float().cpu()
            for chunk in actions.split(batch_size):
                predictions.append(model.predict(context.expand(len(chunk), -1, -1, -1), chunk, shared_noise=True).float().cpu())
        predictions = torch.cat(predictions)
        video = []
        for j, prediction in zip(frames, predictions):
            target = dataset.image(ei, j)
            dt = (times[j] - times[source]) / 1e9
            note = '' if dataset.min_dt <= dt <= dataset.max_dt else f'  (outside training range {dataset.min_dt:g}-{dataset.max_dt:g} s)'
            panels = [to_pixels(zero), to_pixels(context[0]), to_pixels(prediction), to_pixels(target),
                      to_pixels(3 * (prediction - target).abs()),
                      motion_panel(poses, times, source, frames[-1], dataset.image_size, current=j)]
            video.append(np.asarray(compose([panels], header=f'{episode["episode_id"]}  frame {source}   t = +{dt:.2f} s{note}')))
        path = output / f'trajectory{n + 1}.mp4'
        save_video(video, path, fps)
        written.append(dict(file=path.name, episode=episode['episode_id'], source_index=source,
                            target_indices=frames, step_seconds=step, fps=fps))
    return written


@torch.no_grad()
def evaluate(model, dataset, device, output=None, batch_size=16, max_batches=0, amp=True,
             grid_rows=36, videos=10):
    # Pixel-space evaluation for models exposing predict(source, action) -> image (the diffusion
    # decoder). The prediction is compared with the true target and with two baselines: the
    # unchanged context image and the prediction under zero motion. With `output`, the pairs
    # selected by displayed_indices are drawn in comparison1.png, comparison2.png, ... (12 rows
    # each) and `videos` trajectory videos are written (save_videos).
    was_training = model.training
    model.eval()
    results, rows, details, pair_results = {}, [], [], []
    count = 0
    shown = {index: slot for slot, index in enumerate(displayed_indices(dataset, grid_rows))} if output else {}
    rows = [None] * len(shown)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        source, target, action = (batch[k].to(device) for k in ['source', 'target', 'action'])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
            prediction = model.predict(source, action)
            zero = model.predict(source, torch.zeros_like(action))
        batch_metrics = {}
        for name, value in [('model', prediction), ('copy_context', source), ('zero_motion', zero)]:
            for metric, values in image_metrics(value, target).items():
                key = f'{name}_{metric}'
                results[key] = results.get(key, 0.) + values.sum().item()
                batch_metrics[key] = values.cpu().tolist()
        sensitivity = (prediction.float() - zero.float()).abs().mean(dim=(1, 2, 3))
        results['motion_sensitivity_mae'] = results.get('motion_sensitivity_mae', 0.) + sensitivity.sum().item()
        results['mean_pair_dt_seconds'] = results.get('mean_pair_dt_seconds', 0.) + batch['dt_seconds'].sum().item()
        results['mean_translation_mm'] = results.get('mean_translation_mm', 0.) + action[:, :3].norm(dim=1).sum().item()
        results['mean_rotation_rad'] = results.get('mean_rotation_rad', 0.) + action[:, 3:].norm(dim=1).sum().item()
        # Per-pair metrics for every pair; panels only for the rows selected by displayed_indices.
        for i in range(len(source)):
            episode_id = dataset.episodes[int(batch['episode'][i])]['episode_id']
            pair_results.append(dict(
                episode=episode_id, source_index=int(batch['source_index'][i]),
                target_index=int(batch['target_index'][i]), dt_seconds=float(batch['dt_seconds'][i]),
                translation_mm=float(action[i, :3].norm()), rotation_rad=float(action[i, 3:].norm()),
                motion_sensitivity_mae=float(sensitivity[i]),
                **{key: values[i] for key, values in batch_metrics.items()}))
            index = batch_index * batch_size + i
            if index in shown:
                ei, a, b = int(batch['episode'][i]), int(batch['source_index'][i]), int(batch['target_index'][i])
                episode = dataset.episodes[ei]
                panels = [to_pixels(x[i]) for x in [zero, source, prediction, target, 3 * (prediction - target).abs()]]
                panels.append(motion_panel(episode['poses'], episode['times'], a, b, dataset.image_size,
                                           caption=f'{episode_id}  #{a}->{b}'))
                rows[shown[index]] = panels
                details.append(dict(file=f'comparison{shown[index] // 12 + 1}.png', row=shown[index] % 12 + 1,
                                    episode=episode_id, source_index=a, target_index=b,
                                    dt_seconds=float(batch['dt_seconds'][i]), action_mm_rotvec_rad=action[i].cpu().tolist()))
        count += len(source)
    if not count:
        raise ValueError('No evaluation samples')
    results = {k: v / count for k, v in results.items()}
    results.update(pairs=count, split=dataset.split, displayed_pairs=sorted(details, key=lambda d: (d['file'], d['row'])))
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'per_pair_metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_results[0]))
            writer.writeheader()
            writer.writerows(pair_results)
        rows = [row for row in rows if row is not None]   # fewer rows if max_batches stopped early
        save_grids(rows, output, per_file=12)
        for i, row in enumerate(rows):
            Image.fromarray(row[2][..., 0]).save(output / f'prediction_{i:02d}.png')
        if videos:
            results['videos'] = save_videos(model, dataset, device, output, count=videos, batch_size=batch_size, amp=amp)
        (output / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    model.train(was_training)
    return results


def recall_at_k(queries, keys, ks=(1, 5, 10)):
    # queries[i] and keys[i] belong to the same held-out pair. rank = number of keys more similar
    # to query i than its own key; recall@k is the fraction of pairs with rank < k. The search
    # space is the whole split, including other frames of the same episode, which makes the
    # task substantially harder than in-batch retrieval.
    """Fraction of queries whose own key is among the k most cosine-similar keys of the split."""
    similarity = F.normalize(queries.float(), dim=1) @ F.normalize(keys.float(), dim=1).T
    rank = (similarity > similarity.diagonal()[:, None]).sum(dim=1)
    return {k: (rank < k).float().mean().item() for k in ks}


@torch.no_grad()
def evaluate_world(model, dataset, device, output=None, batch_size=32, amp=True):
    """Representation metrics of the pre-training objective; no pixels are decoded.

    Validation loss uses fixed masks. Motion retrieval searches the projected target of every
    held-out pair: copy-context queries with the context image's own target projection,
    zero-motion queries with the prediction for a zero action.
    """
    # Three queries for the same keys (projected pooled target features of I_b):
    #   model         prediction from I_a and the measured motion a -> b
    #   zero_motion   prediction from I_a with a zero action (ablation of motion conditioning)
    #   copy_context  target feature of I_a itself, i.e. "assume the image does not change"
    # Motion modeling is informative only if `model` outperforms both baselines.
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals, count = {}, 0
    queries = {'model': [], 'zero_motion': [], 'copy_context': []}
    keys = []
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(0)  # identical masks at every validation, so losses are comparable across steps
        for batch in loader:
            source, target, action, reverse = (batch[k].to(device) for k in ['source', 'target', 'action', 'reverse_action'])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
                _, values = model(source, target, action, reverse)
                queries['model'].append(model.projector(model.predict_latent(source, action)).float())
                queries['zero_motion'].append(model.projector(model.predict_latent(source, torch.zeros_like(action))).float())
                queries['copy_context'].append(model.target_projector(model.target_latent(source).mean(1)).float())
                keys.append(model.target_projector(model.target_latent(target).mean(1)).float())
            for name, value in values.items():
                totals[f'val_{name}'] = totals.get(f'val_{name}', 0.) + float(value) * len(source)
            count += len(source)
    if not count:
        raise ValueError('No evaluation samples')
    results = {k: v / count for k, v in totals.items()}
    keys = torch.cat(keys)
    for name, values in queries.items():
        for k, recall in recall_at_k(torch.cat(values), keys).items():
            results[f'{name}_recall@{k}'] = recall
    results.update(pairs=count, split=dataset.split)
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    model.train(was_training)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest')
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-pairs-per-episode', type=int, default=64)
    p.add_argument('--min-dt', type=float)
    p.add_argument('--max-dt', type=float)
    p.add_argument('--target-dt', type=float, default=1., help='Preferred validation horizon in seconds')
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, state = load_model(args.checkpoint, device)
    data = dict(state['config']['data'])
    if args.manifest:
        data['manifest'] = args.manifest
    data['max_pairs_per_episode'] = args.max_pairs_per_episode
    data['eval_dt'] = args.target_dt
    if args.min_dt is not None:
        data['min_dt'] = args.min_dt
    if args.max_dt is not None:
        data['max_dt'] = args.max_dt
    dataset = MocapPairs(split=args.split, image_size=state['config']['model']['image_size'], **data)
    if state.get('kind') == 'diffusion':
        metrics = evaluate(model, dataset, device, args.output, args.batch_size)
    else:
        metrics = evaluate_world(model, dataset, device, args.output, args.batch_size)
    metrics.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_step=state['step'],
                   checkpoint_kind=state.get('kind', 'world_model'),
                   manifest=str(dataset.manifest_path), requested_horizon_seconds=args.target_dt,
                   manifest_sha256=hashlib.sha256(dataset.manifest_path.read_bytes()).hexdigest(),
                   image_size=state['config']['model']['image_size'],
                   max_pairs_per_episode=args.max_pairs_per_episode,
                   min_dt=data.get('min_dt', .4), max_dt=data.get('max_dt', 3.))
    (Path(args.output) / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metrics.items() if k != 'displayed_pairs'}, indent=2))


if __name__ == '__main__':
    main()
