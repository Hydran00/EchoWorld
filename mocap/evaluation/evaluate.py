"""Evaluate held-out pairs: representation metrics (world model) or image comparisons (diffusion)."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from mocap.processing.dataset import MocapPairs
from mocap.processing.geometry import relative_action
from mocap.pretraining.model import MotionWorldModel
from mocap.visualization.visualize import ROLLOUT_COLUMNS, combine_videos, compose, motion_panel, save_grids, save_video, to_pixels


def load_model(checkpoint, device):
    # Only load locally produced, trusted checkpoints.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('kind') == 'diffusion':
        from mocap.decoder.diffusion import DiffusionUltrasound
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


VIDEO_SECONDS = 3.   # minimum time covered by the validation videos


def attachment_of(dataset, episode, index, device):
    """(face [1], barycentric [1, 3]) of one frame for anatomy datasets, else None."""
    if not getattr(dataset, 'anatomy', False):
        return None
    face, barycentric = dataset.attachment(episode, index)
    return face[None].to(device), barycentric[None].to(device)


def expand_attachment(attachment, n):
    return None if attachment is None else tuple(x.expand(n, *x.shape[1:]) for x in attachment)


def rollout_dt(dataset):
    # Default fixed rollout Δt: the shortest training interval (configs override it with rollout_dt).
    return dataset.min_dt


@torch.no_grad()
def save_videos(model, dataset, device, output, count=10, step=.1, fps=5, batch_size=16, amp=True, reverse=False):
    """Anchored generation along a recorded trajectory, one MP4 per selected validation episode.

    The context frame is fixed (as in generate.py, anchored mode) and video frame k shows the
    prediction for the measured motion from the context to the recorded frame closest to
    t0 + k * step, up to the maximum pair interval (at least VIDEO_SECONDS). All frames share one initial noise image, so
    the changes along the video are due to the motion conditioning only. Frames with a horizon
    shorter than the minimum training interval are flagged, since such pairs are never seen in
    training. Playback is at `fps`; with step = 0.1 s and fps = 5 the video runs at half speed.
    With `reverse`, the targets are the recorded frames at t0 - k * step (backward probe motion,
    seen in training through the random temporal reversal of pairs); files trajectory_reverseN.mp4.
    """
    output = Path(output)
    sign, name = (-1, 'trajectory_reverse') if reverse else (1, 'trajectory')
    episodes = sorted({ei for ei, _, _ in dataset.items})
    chosen = sorted({episodes[i] for i in np.linspace(0, len(episodes) - 1, min(count, len(episodes))).round().astype(int)})
    written = []
    for n, ei in enumerate(chosen):
        items = [item for item in dataset.items if item[0] == ei]
        _, source, _ = items[len(items) // 2]
        episode = dataset.episodes[ei]
        times, segments, poses = episode['times'], episode['segments'], episode['poses']
        frames = []
        for k in range(1, int(round(max(dataset.max_dt, VIDEO_SECONDS) / step)) + 1):
            wanted = times[source] + sign * round(k * step * 1e9)
            j = int(np.argmin(abs(times - wanted)))
            # Stop at the end (start, if reverse) of the episode or of the tracking segment; never
            # cross a gap. Instants without a recorded frame within step / 2 are skipped.
            if not times[0] <= wanted <= times[-1] or segments[j] != segments[source]:
                break
            if abs(int(times[j]) - wanted) <= step / 2 * 1e9 and (j - source) * sign > 0 and (not frames or j != frames[-1]):
                frames.append(j)
        if not frames:
            continue
        context = dataset.image(ei, source)[None].to(device)
        actions = torch.from_numpy(relative_action(poses[source], poses[np.array(frames)])).to(device)
        here = attachment_of(dataset, ei, source, device)
        predictions = []
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
            zero = model.predict(context, torch.zeros_like(actions[:1]), shared_noise=True, anatomy=here)[0].float().cpu()
            for chunk in actions.split(batch_size):
                predictions.append(model.predict(context.expand(len(chunk), -1, -1, -1), chunk, shared_noise=True,
                                                 anatomy=expand_attachment(here, len(chunk))).float().cpu())
        predictions = torch.cat(predictions)
        video = []
        for j, prediction in zip(frames, predictions):
            target = dataset.image(ei, j)
            dt = (times[j] - times[source]) / 1e9
            note = '' if dataset.min_dt <= abs(dt) <= dataset.max_dt else f'  (outside training range {dataset.min_dt:g}-{dataset.max_dt:g} s)'
            note += '   REVERSE (backward motion)' if reverse else ''
            panels = [to_pixels(zero), to_pixels(context[0]), to_pixels(prediction), to_pixels(target),
                      to_pixels(3 * (prediction - target).abs()),
                      motion_panel(poses, times, source, frames[-1], dataset.image_size, current=j)]
            video.append(np.asarray(compose([panels], header=f'{episode["episode_id"]}  frame {source}   t = {dt:+.2f} s{note}')))
        path = output / f'{name}{n + 1}.mp4'
        save_video(video, path, fps)
        written.append(dict(file=path.name, episode=episode['episode_id'], source_index=source,
                            target_indices=frames, step_seconds=step, fps=fps))
    return written


@torch.no_grad()
def save_rollout_videos(model, dataset, device, output, count=10, step=None, horizon=None, fps=5, amp=True, reverse=False):
    """Autoregressive rollout along a recorded trajectory with a fixed Δt, one MP4 per episode.

    Same episodes and start frames as save_videos. Step k predicts the frame at t0 + k * step from
    the step k-1 prediction (step 1: the real context image), conditioned on the measured motion
    between the recorded frames at t0 + (k-1) * step and t0 + k * step. Every prediction therefore
    has the horizon `step` (default rollout_dt: inside the training interval), while errors
    accumulate along the rollout. Generated images are quantised to 8 bits before being fed back,
    like the recorded inputs, and every step uses the same seeded initial noise. The rollout has
    round(horizon / step) steps (horizon default: max(max_dt, VIDEO_SECONDS), as save_videos; 8 steps
    = 3.2 s for 3 s and 0.4 s, 30 steps for 0.1 s). Each step is
    held for 2 * step seconds of video (half speed, as save_videos at its default fps).
    With `reverse`, the rollout goes backward in time (t0 - k * step); files simulation_reverseN.mp4.
    """
    output = Path(output)
    sign, name = (-1, 'simulation_reverse') if reverse else (1, 'simulation')
    step = rollout_dt(dataset) if step is None else step
    horizon = max(dataset.max_dt, VIDEO_SECONDS) if horizon is None else horizon
    episodes = sorted({ei for ei, _, _ in dataset.items})
    chosen = sorted({episodes[i] for i in np.linspace(0, len(episodes) - 1, min(count, len(episodes))).round().astype(int)})
    written = []
    for n, ei in enumerate(chosen):
        items = [item for item in dataset.items if item[0] == ei]
        _, source, _ = items[len(items) // 2]
        episode = dataset.episodes[ei]
        times, segments, poses = episode['times'], episode['segments'], episode['poses']
        frames = [source]
        for k in range(1, int(round(horizon / step)) + 1):
            wanted = times[source] + sign * round(k * step * 1e9)
            j = int(np.argmin(abs(times - wanted)))
            # Stop at the end (start, if reverse) of the episode or tracking segment, or when no recorded
            # frame lies within min(0.1 s, step / 4) of the instant (a skipped step would change Δt).
            if (not times[0] <= wanted <= times[-1] or segments[j] != segments[source] or
                    abs(int(times[j]) - wanted) > min(.1, step / 4) * 1e9 or (j - frames[-1]) * sign <= 0):
                break
            frames.append(j)
        if len(frames) < 2:
            continue
        initial = dataset.image(ei, source)[None].to(device)
        current, predictions, dts = initial, [], []
        for a, b in zip(frames[:-1], frames[1:]):
            action = torch.from_numpy(relative_action(poses[a], poses[b])[None]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
                # The current position is the measured pose of the recorded frame a (known at inference).
                prediction = model.predict(current, action, anatomy=attachment_of(dataset, ei, a, device)).float()
            predictions.append((current[0].cpu(), prediction[0].cpu()))
            dts.append(abs(float(times[b] - times[a])) / 1e9)
            current = (prediction.clamp(0, 1) * 255).round() / 255
        video, maes = [], []
        repeat = max(1, int(round(2 * step * fps)))
        for k, (b, (context, prediction), dt) in enumerate(zip(frames[1:], predictions, dts)):
            target = dataset.image(ei, b)
            maes.append(float((prediction - target).abs().mean()))
            t = (times[b] - times[source]) / 1e9
            # Flag on the nominal Δt: recorded intervals jitter by a few ms around it.
            note = '' if dataset.min_dt <= step <= dataset.max_dt else f'  (dt outside training range {dataset.min_dt:g}-{dataset.max_dt:g} s)'
            note += '   REVERSE' if reverse else ''
            panels = [to_pixels(context), to_pixels(initial[0]), to_pixels(prediction), to_pixels(target),
                      to_pixels(3 * (prediction - target).abs()),
                      motion_panel(poses, times, source, frames[-1], dataset.image_size, current=b)]
            header = (f'{episode["episode_id"]}  frame {source}   rollout step {k + 1}/{len(dts)}   '
                      f't = {t:+.2f} s   dt = {dt:.2f} s{note}')
            video.extend([np.asarray(compose([panels], header=header, columns=ROLLOUT_COLUMNS))] * repeat)
        path = output / f'{name}{n + 1}.mp4'
        save_video(video, path, fps)
        written.append(dict(file=path.name, episode=episode['episode_id'], source_index=source,
                            target_indices=frames[1:], step_seconds=step, dt_seconds=[round(d, 4) for d in dts],
                            mae_per_step=maes, fps=fps, video_frames_per_step=repeat))
    return written


def save_all_videos(model, dataset, device, output, count=10, batch_size=16, amp=True, rollout_step=None, reverse=False):
    """Trajectory and rollout videos in one direction, plus their combination in one shareable file:
    trajectoryN | simulationN per row, episodes stacked vertically (videos_all.mp4, or
    videos_all_reverse.mp4 for the backward motion)."""
    suffix = '_reverse' if reverse else ''
    results = {f'videos{suffix}': save_videos(model, dataset, device, output, count=count, batch_size=batch_size,
                                              amp=amp, reverse=reverse),
               f'simulation_videos{suffix}': save_rollout_videos(model, dataset, device, output, count=count,
                                                                 step=rollout_step, amp=amp, reverse=reverse)}
    combined = combine_videos(output, f'videos_all{suffix}.mp4',
                              prefixes=(f'trajectory{suffix}', f'simulation{suffix}'))
    if combined:
        results[f'combined_video{suffix}'] = combined.name
    return results


@torch.no_grad()
def evaluate(model, dataset, device, output=None, batch_size=16, max_batches=0, amp=True,
             grid_rows=36, videos=10, rollout_step=None, reverse_videos=False, reverse=False):
    # Pixel-space evaluation for models exposing predict(source, action) -> image (the diffusion
    # decoder). The prediction is compared with the true target and with two baselines: the
    # unchanged context image and the prediction under zero motion. With `output`, the pairs
    # selected by displayed_indices are drawn in comparison1.png, comparison2.png, ... (12 rows
    # each), `videos` trajectory videos are written (save_videos, fixed context, growing horizon)
    # and as many autoregressive rollout videos (save_rollout_videos, fixed Δt = rollout_step,
    # default rollout_dt(dataset)), plus videos_all.mp4 with all of them (combine_videos). With
    # reverse_videos, the same for the backward probe motion (videos_all_reverse.mp4).
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
        if reverse:
            # Backward motion on the same pairs: predict I_a from I_b with the inverse motion b -> a.
            source, target, action = target, source, batch['reverse_action'].to(device)
        here = MotionWorldModel.batch_attachment(batch, 'target' if reverse else 'source', device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
            prediction = model.predict(source, action, anatomy=here)
            zero = model.predict(source, torch.zeros_like(action), anatomy=here)
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
    results.update(pairs=count, split=dataset.split, reverse=reverse, displayed_pairs=sorted(details, key=lambda d: (d['file'], d['row'])))
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
            for reverse in (False, True) if reverse_videos else (False,):
                results.update(save_all_videos(model, dataset, device, output, videos, batch_size, amp, rollout_step, reverse))
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
            # Anatomy models: attachment of the context frame only (the target's is used solely by
            # the reverse direction of the training loss, where b is the context).
            here, there = model.batch_attachment(batch, 'source', device), model.batch_attachment(batch, 'target', device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
                _, values = model(source, target, action, reverse, here, there)
                queries['model'].append(model.projector(model.predict_latent(source, action, here)).float())
                queries['zero_motion'].append(model.projector(model.predict_latent(source, torch.zeros_like(action), here)).float())
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


REPORT_METRICS = [('mae', 'MAE ↓', 4), ('foreground_mae', 'MAE fg ↓', 4), ('mse', 'MSE ↓', 4),
                  ('psnr', 'PSNR ↑', 2), ('ssim', 'SSIM ↑', 3)]
REPORT_METHODS = [('model', 'Model'), ('copy_context', 'Copy context image'), ('zero_motion', 'Zero motion')]


def write_report(path, sections, info):
    """Markdown tables of the image metrics, one per evaluation (title, metrics dict)."""
    lines = ['# Decoder evaluation', '',
             f"Checkpoint `{info['checkpoint']}` (step {info['checkpoint_step']}), split mode "
             f"`{info['split_mode']}`, training dt {info['min_dt']:g}-{info['max_dt']:g} s, "
             f"preferred horizon {info['requested_horizon_seconds']:g} s, "
             f"at most {info['max_pairs_per_episode']} pairs per episode.", '',
             '- **Copy context image**: the input image returned unchanged; **Zero motion**: the model '
             'with a null motion. Both are baselines.',
             '- **MAE fg**: MAE on the pixels where the true target is above 0.05 (ultrasound sector only).',
             '- **Reverse**: the same pairs with source and target swapped and the inverse motion.']
    if info['split_mode'] == 'random':
        lines.append('- Random split: validation/test episodes are unseen, but their subjects appear in training.')
    lines.append('')
    header = '| | ' + ' | '.join(label for _, label, _ in REPORT_METRICS) + ' |'
    rule = '|---|' + '---|' * len(REPORT_METRICS)
    lines += ['## Summary (model)', '', header.replace('| |', '| Evaluation |', 1), rule]
    for title, m in sections:
        lines.append(f'| {title} | ' + ' | '.join(f"{m[f'model_{k}']:.{d}f}" for k, _, d in REPORT_METRICS) + ' |')
    for title, m in sections:
        lines += ['', f'## {title}', '',
                  f"{m['pairs']} pairs, mean dt {m['mean_pair_dt_seconds']:.2f} s, mean translation "
                  f"{m['mean_translation_mm']:.1f} mm, mean rotation {np.degrees(m['mean_rotation_rad']):.1f} deg.", '',
                  header, rule]
        for method, label in REPORT_METHODS:
            lines.append(f'| {label} | ' + ' | '.join(f"{m[f'{method}_{k}']:.{d}f}" for k, _, d in REPORT_METRICS) + ' |')
    Path(path).write_text('\n'.join(lines) + '\n')


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
    p.add_argument('--split-mode', choices=['random', 'leave-out'],
                   help='Override the train/val/test split of the checkpoint (default: the one it was trained with)')
    p.add_argument('--no-reverse-videos', action='store_true',
                   help='Diffusion only: skip the backward-motion videos (videos_all_reverse.mp4)')
    p.add_argument('--no-test', action='store_true',
                   help='Diffusion with --split val only: skip the test-split evaluation (test/, videos_all_test.mp4)')
    p.add_argument('--videos-only', choices=['forward', 'reverse', 'both'],
                   help='Diffusion only: write only these videos into --output (no metrics), e.g. for an existing eval folder')
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, state = load_model(args.checkpoint, device)
    data = dict(state['config']['data'])
    if args.split == 'val' and state['config'].get('validation_manifest'):
        data['manifest'] = state['config']['validation_manifest']
    if args.manifest:
        data['manifest'] = args.manifest
    data['max_pairs_per_episode'] = args.max_pairs_per_episode
    data['eval_dt'] = args.target_dt
    if args.min_dt is not None:
        data['min_dt'] = args.min_dt
    if args.max_dt is not None:
        data['max_dt'] = args.max_dt
    if args.split_mode:
        data['split_mode'] = args.split_mode
    dataset = MocapPairs(split=args.split, image_size=state['config']['model']['image_size'], **data)
    if args.videos_only:
        if state.get('kind') != 'diffusion':
            raise ValueError('--videos-only requires a diffusion checkpoint')
        Path(args.output).mkdir(parents=True, exist_ok=True)
        info = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_step=state['step'],
                    split=args.split, split_mode=dataset.split_mode)
        for reverse in {'forward': (False,), 'reverse': (True,), 'both': (False, True)}[args.videos_only]:
            with torch.no_grad():
                info.update(save_all_videos(model, dataset, device, args.output, batch_size=args.batch_size, reverse=reverse))
        name = 'videos_reverse.json' if args.videos_only == 'reverse' else 'videos.json'
        (Path(args.output) / name).write_text(json.dumps(info, indent=2) + '\n')
        print(json.dumps({k: v for k, v in info.items() if not isinstance(v, list)}, indent=2))
        return
    output = Path(args.output)
    info = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_step=state['step'],
                checkpoint_kind=state.get('kind', 'world_model'), requested_horizon_seconds=args.target_dt,
                image_size=state['config']['model']['image_size'], max_pairs_per_episode=args.max_pairs_per_episode,
                min_dt=data.get('min_dt', .4), max_dt=data.get('max_dt', 3.), split_mode=dataset.split_mode)

    def finish(metrics, dataset, folder):
        metrics.update(info, manifest=str(dataset.manifest_path),
                       manifest_sha256=hashlib.sha256(dataset.manifest_path.read_bytes()).hexdigest())
        (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        return metrics

    if state.get('kind') != 'diffusion':
        metrics = finish(evaluate_world(model, dataset, device, output, args.batch_size), dataset, output)
        print(json.dumps({k: v for k, v in metrics.items() if k != 'displayed_pairs'}, indent=2))
        return
    # Decoder: forward evaluation with videos (videos_all.mp4, videos_all_reverse.mp4), the same pairs
    # with backward motion (metrics only), and with --split val also the test split (test/,
    # videos_all_test.mp4); metrics.md collects the tables.
    metrics = evaluate(model, dataset, device, output, args.batch_size, reverse_videos=not args.no_reverse_videos)
    reverse = evaluate(model, dataset, device, None, args.batch_size, reverse=True)
    metrics['reverse_motion'] = {k: v for k, v in reverse.items() if k != 'displayed_pairs'}
    metrics = finish(metrics, dataset, output)
    name = dict(val='Validation', test='Test')[args.split]
    sections = [(f'{name}, forward motion', metrics), (f'{name}, reverse motion', reverse)]
    if args.split == 'val' and not args.no_test:
        test_data = dict(data)
        test_data['manifest'] = args.manifest or state['config']['data']['manifest']
        test = MocapPairs(split='test', image_size=state['config']['model']['image_size'], **test_data)
        (output / 'test').mkdir(parents=True, exist_ok=True)
        test_metrics = finish(evaluate(model, test, device, output / 'test', args.batch_size), test, output / 'test')
        if (output / 'test' / 'videos_all.mp4').exists():
            shutil.copyfile(output / 'test' / 'videos_all.mp4', output / 'videos_all_test.mp4')
        metrics['test'] = {k: v for k, v in test_metrics.items()
                           if not isinstance(v, (list, dict)) or k == 'combined_video'}
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        sections.append(('Test, forward motion', test_metrics))
    write_report(output / 'metrics.md', sections, info)
    print((output / 'metrics.md').read_text())

if __name__ == '__main__':
    main()
