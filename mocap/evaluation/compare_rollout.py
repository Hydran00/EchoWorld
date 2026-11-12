"""Paired test: A predicts the endpoint directly; B rolls its own images forward.

Random recorded endpoints are near positive multiples of B's step, strictly below
max_horizon. Intermediate B poses follow the measured trajectory at fixed steps;
only the last step absorbs recorded timestamp jitter. No future image is input.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mocap.processing.dataset import MocapPairs
from mocap.evaluation.evaluate import image_metrics, load_model
from mocap.processing.geometry import PoseSeries, relative_action
from mocap.visualization.visualize import compose, to_pixels, GENERATED, GROUND_TRUTH, TEXT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sample_pairs(dataset, pairs_per_episode=32, seed=42, max_horizon=3., max_gap=.1,
                 step=.2, tolerance=.01):
    """Uniform eligible starts without replacement, then a uniform available grid horizon.

    Each horizon uses its nearest valid recorded target within tolerance. Sampling
    is conditional on available future; short segments remain eligible.
    Additional gap barriers prevent interpolation across dropped cached frames.
    """
    if pairs_per_episode < 1 or not np.isfinite(max_horizon) or max_horizon <= 0:
        raise ValueError('Expected positive sample count and finite positive horizon')
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError('Expected finite positive pose gap')
    validate_grid(step, tolerance)
    step_ns, tolerance_ns = round(step * 1e9), round(tolerance * 1e9)
    horizon_ns = round(max_horizon * 1e9)
    rng = np.random.default_rng(seed)
    samples, coverage = [], []
    for ei, episode in enumerate(dataset.episodes):
        times, segments = episode['times'], episode['segments']
        breaks = (segments[1:] != segments[:-1]) | (np.diff(times) > round(max_gap * 1e9))
        contiguous = np.r_[0, np.cumsum(breaks)]
        candidates = []
        for item_ei, source, targets in dataset.items:
            if item_ei != ei:
                continue
            targets = np.asarray(targets)
            delta = times[targets] - times[source]
            valid = ((delta > 0) & (delta < round(max_horizon * 1e9)) &
                     (contiguous[targets] == contiguous[source]))
            targets, delta = targets[valid], delta[valid]
            chosen = []
            for nominal in range(step_ns, horizon_ns, step_ns):
                if not len(delta):
                    break
                nearest = int(np.argmin(abs(delta - nominal)))
                if abs(int(delta[nearest]) - nominal) <= tolerance_ns:
                    chosen.append(int(targets[nearest]))
            if chosen:
                candidates.append((source, np.asarray(chosen)))
        count = min(pairs_per_episode, len(candidates))
        coverage.append(dict(episode=episode['episode_id'], eligible_sources=len(candidates), sampled=count))
        for slot in rng.choice(len(candidates), size=count, replace=False):
            source, targets = candidates[int(slot)]
            target = int(rng.choice(targets))
            actual_ns = int(times[target] - times[source])
            nominal_ns = ((actual_ns + step_ns // 2) // step_ns) * step_ns
            samples.append(dict(pair_id=len(samples), episode_index=ei, episode=episode['episode_id'],
                                source_index=int(source), target_index=target,
                                source_timestamp_ns=int(times[source]), target_timestamp_ns=int(times[target]),
                                dt_seconds=actual_ns / 1e9, nominal_dt_seconds=nominal_ns / 1e9,
                                target_offset_seconds=(actual_ns - nominal_ns) / 1e9))
    if not samples:
        raise ValueError('No valid pairs in the requested split')
    return samples, coverage


def validate_grid(step, tolerance):
    if not np.isfinite(step) or round(step * 1e9) < 1:
        raise ValueError('Rollout step must be finite and positive')
    if not np.isfinite(tolerance) or not 0 <= tolerance < step / 2:
        raise ValueError('Target tolerance must be finite, nonnegative and below half a step')


def rollout_path(episode, source, target, step=.2, max_gap=.1, tolerance=.01):
    """Exact intermediate steps; the last absorbs only the target timestamp jitter."""
    validate_grid(step, tolerance)
    times, poses, segments = episode['times'], episode['poses'], episode['segments']
    if not 0 <= source < target < len(times):
        raise ValueError('Expected a strictly future target')
    if (np.any(segments[source:target + 1] != segments[source]) or
            np.any(np.diff(times[source:target + 1]) > round(max_gap * 1e9))):
        raise ValueError('Rollout cannot cross a tracking or recording gap')
    duration = int(times[target] - times[source])
    step_ns = round(step * 1e9)
    steps = (duration + step_ns // 2) // step_ns
    if steps < 1 or abs(duration - steps * step_ns) > round(tolerance * 1e9):
        raise ValueError('Target is not near a positive multiple of the rollout step')
    offsets = np.r_[np.arange(steps, dtype=np.int64) * step_ns, duration]
    # Interpolate cached poses, never images; use local timestamps to avoid epoch rounding.
    local_times = times[source:target + 1] - times[source]
    series = PoseSeries(zip(local_times, poses[source:target + 1]), max_gap_ns=round(max_gap * 1e9))
    path = np.stack([series.at(int(t), tolerance_ns=round(max_gap * 1e9))[0] for t in offsets])
    return offsets, path


@torch.inference_mode()
def predict_batch(model_a, model_b, source, direct_actions, paths, amp=False):
    """Ragged rollout plus zero-action controls, with identical seeded noise per call.

    Intermediate feedback is quantised to uint8, matching existing rollout videos.
    Final outputs are scored as floats for both models. Future images are absent
    from this interface, so teacher forcing cannot enter the autoregressive loop.
    """
    device = source.device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
        direct = model_a.predict(source, direct_actions, shared_noise=True).float()
        zero_a = model_a.predict(source, torch.zeros_like(direct_actions), shared_noise=True).float()
        current, zero_current = source.clone(), source.clone()
        final, zero_final = torch.empty_like(source), torch.empty_like(source)
        for k in range(max(len(path) for path in paths)):
            indices = [i for i, path in enumerate(paths) if k < len(path)]
            actions = torch.as_tensor(np.stack([paths[i][k] for i in indices]), device=device)
            inputs = torch.cat([current[indices], zero_current[indices]])
            prediction, zero = model_b.predict(
                inputs, torch.cat([actions, torch.zeros_like(actions)]), shared_noise=True).float().chunk(2)
            final[indices], zero_final[indices] = prediction, zero
            current[indices] = (prediction.clamp(0, 1) * 255).round() / 255
            zero_current[indices] = (zero.clamp(0, 1) * 255).round() / 255
    return dict(a_direct=direct, b_rollout=final, copy_context=source,
                a_zero_action=zero_a, b_zero_action=zero_final)


METRICS = ('mae', 'mse', 'psnr', 'foreground_mae', 'ssim')
METHODS = ('a_direct', 'b_rollout', 'copy_context', 'a_zero_action', 'b_zero_action')


def summarize(rows, bootstrap=5000, seed=42):
    """Equal episode means and paired episode bootstrap; no IID-frame assumption."""
    if not rows or bootstrap < 1:
        raise ValueError('Expected nonempty rows and positive bootstrap count')
    episodes = sorted({row['episode'] for row in rows})
    per_episode = []
    keys = [f'{method}_{metric}' for method in METHODS for metric in METRICS]
    for episode in episodes:
        subset = [row for row in rows if row['episode'] == episode]
        per_episode.append(dict(episode=episode, pairs=len(subset),
                                **{key: float(np.mean([row[key] for row in subset])) for key in keys}))
    means = {key: float(np.mean([row[key] for row in per_episode])) for key in keys}
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(episodes), size=(bootstrap, len(episodes)))
    paired = {}
    for metric in METRICS:
        differences = np.array([row[f'b_rollout_{metric}'] - row[f'a_direct_{metric}'] for row in per_episode])
        ci = np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist() if len(episodes) > 1 else None
        lower = metric not in ('psnr', 'ssim')
        wins = [row[f'b_rollout_{metric}'] < row[f'a_direct_{metric}'] if lower else
                row[f'b_rollout_{metric}'] > row[f'a_direct_{metric}'] for row in rows]
        paired[metric] = dict(b_minus_a=float(differences.mean()), ci95=ci,
                              lower_is_better=lower, b_pair_win_fraction=float(np.mean(wins)))
    gains = {}
    for method, zero in [('a_direct', 'a_zero_action'), ('b_rollout', 'b_zero_action')]:
        gains[method] = {}
        for metric in ('mae', 'foreground_mae'):
            for baseline in ('copy_context', zero):
                denominator = means[f'{baseline}_{metric}']
                gains[method][f'{metric}_gain_over_{baseline}'] = (
                    1 - means[f'{method}_{metric}'] / denominator if denominator > 1e-12 else None)
    return dict(pairs=len(rows), episodes=len(episodes), means=means, paired=paired,
                gains=gains, per_episode=per_episode)


def report_text(result):
    summary = result['overall']
    lines = ['# Direct prediction versus autoregressive rollout', '',
             f"Split: {result['split']}; {summary['pairs']} pairs, {summary['episodes']} episodes.",
             'A uses the initial image and total motion once. B feeds back generated images at 200 ms '
             'steps (or the configured step), with only timestamp jitter in the last step to the same real target.',
             'Means weight episodes equally. Confidence intervals resample paired episodes; '
             'they do not estimate uncertainty across training seeds or unseen subjects.', '',
             '| Metric | A direct | B rollout | Copy | B − A [95% CI] |',
             '|---|---:|---:|---:|---:|']
    for metric in METRICS:
        p, m = summary['paired'][metric], summary['means']
        interval = 'n/a' if p['ci95'] is None else f"[{p['ci95'][0]:.5f}, {p['ci95'][1]:.5f}]"
        lines.append(f"| {metric} | {m[f'a_direct_{metric}']:.5f} | {m[f'b_rollout_{metric}']:.5f} | "
                     f"{m[f'copy_context_{metric}']:.5f} | {p['b_minus_a']:+.5f} {interval} |")
    lines.extend(['', 'Lower MAE/MSE/foreground MAE is better; higher PSNR/SSIM is better.', '',
                  '## Endpoint error by duration', '',
                  '| Nominal duration (s) | Pairs | A foreground MAE | B foreground MAE |',
                  '|---|---:|---:|---:|'])
    for label, subset in result['by_horizon'].items():
        m = subset['means']
        lines.append(f"| {label} | {subset['pairs']} | {m['a_direct_foreground_mae']:.5f} | {m['b_rollout_foreground_mae']:.5f} |")
    lines.extend(['', '![Endpoint error by horizon](error_by_horizon.png)', '',
                  '## Gains over baselines', '',
                  '| Method | MAE gain over copy | Foreground MAE gain over copy | Foreground MAE gain over zero action |',
                  '|---|---:|---:|---:|'])
    for method, zero in [('a_direct', 'a_zero_action'), ('b_rollout', 'b_zero_action')]:
        gains = summary['gains'][method]
        keys = ['mae_gain_over_copy_context', 'foreground_mae_gain_over_copy_context', f'foreground_mae_gain_over_{zero}']
        formatted = ['n/a' if gains[key] is None else f'{gains[key]:+.1%}' for key in keys]
        lines.append('| ' + method + ' | ' + ' | '.join(formatted) + ' |')
    lines.extend(['', 'Gain = 1 − error / baseline error; positive is better than the baseline.', '',
                  '## Protocol limits', '',
                  '- Source frames are sampled uniformly without replacement within each episode. '
                  'Target horizons are sampled uniformly among available positive step multiples below the maximum, '
                  'then matched to the nearest valid real frame within tolerance.',
                  '- All scored horizons are strictly below the configured maximum; no pair crosses a tracking/recording gap.',
                  '- Intermediate poses use linear translation and SLERP on the cached measured trajectory.',
                  '- No short residual steps: all B steps must lie within its training interval. '
                  'A’s out-of-training-range endpoints are flagged.',
                  '- Zero-action B repeats the same number of autoregressive steps as B, with zero action throughout.',
                  '- Fixed shared diffusion noise makes results independent of batch membership. '
                  'This comparison uses one noise seed and the supplied checkpoints, not matched retrainings.', ''])
    return '\n'.join(lines)



def save_plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    horizons = result['by_horizon']
    x = np.array([float(key) for key in horizons])
    values = list(horizons.values())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for method, label in [('a_direct', 'A: direct'), ('b_rollout', 'B: rollout'), ('copy_context', 'Copy context')]:
        axes[0].plot(x, [s['means'][f'{method}_foreground_mae'] for s in values], 'o-', label=label)
    axes[0].set_ylabel('Foreground MAE (lower is better)')
    axes[0].legend()
    differences = [s['paired']['foreground_mae'] for s in values]
    axes[1].plot(x, [s['b_minus_a'] for s in differences], 'o-', color='tab:purple')
    low = [s['ci95'][0] if s['ci95'] else np.nan for s in differences]
    high = [s['ci95'][1] if s['ci95'] else np.nan for s in differences]
    axes[1].fill_between(x, low, high, color='tab:purple', alpha=.2, label='95% paired episode bootstrap')
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_ylabel('Foreground MAE: B minus A (negative favours B)')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('Nominal target horizon (s)')
        ax.grid(alpha=.2)
    fig.suptitle(f"{result['split']}: {result['overall']['pairs']} pairs; equal episode weights")
    fig.tight_layout()
    fig.savefig(output / 'error_by_horizon.png', dpi=160)
    fig.savefig(output / 'error_by_horizon.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-a', type=Path, required=True)
    parser.add_argument('--checkpoint-b', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['test', 'val'], default='test')
    parser.add_argument('--split-mode', choices=['random', 'leave-out'],
                        help="Override checkpoint A's train/val/test split (e.g. leave-out on a custom manifest)")
    parser.add_argument('--pairs-per-episode', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-seed', type=int, default=42)
    parser.add_argument('--max-horizon', type=float, default=3.)
    parser.add_argument('--step', type=float, default=.2)
    parser.add_argument('--target-tolerance', type=float, default=.01)
    parser.add_argument('--max-pose-gap', type=float, default=.1)
    parser.add_argument('--min-mean', type=float, default=8.)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap < 1 or args.pairs_per_episode < 1:
        parser.error('batch-size, bootstrap and pairs-per-episode must be positive')
    if not np.isfinite(args.max_pose_gap) or args.max_pose_gap <= 0:
        parser.error('max-pose-gap must be finite and positive')
    if not np.isfinite(args.min_mean) or not 0 <= args.min_mean <= 255:
        parser.error('min-mean must be between 0 and 255')
    if not np.isfinite(args.max_horizon) or args.max_horizon <= 1e-9:
        parser.error('max-horizon must be finite and greater than one nanosecond')
    validate_grid(args.step, args.target_tolerance)
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve prior evaluations')
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    model_a, state_a = load_model(args.checkpoint_a, device)
    model_b, state_b = load_model(args.checkpoint_b, device)
    if state_a.get('kind') != 'diffusion' or state_b.get('kind') != 'diffusion':
        raise ValueError('Both checkpoints must include a diffusion image decoder')
    size = state_a['config']['model']['image_size']
    if size != state_b['config']['model']['image_size']:
        raise ValueError('Both models must use the same image resolution')
    model_a.noise_seed = model_b.noise_seed = args.noise_seed
    b_data = state_b['config']['data']
    if (args.step - args.target_tolerance < b_data['min_dt'] - 1e-9 or
            args.step + args.target_tolerance > b_data['max_dt'] + 1e-9):
        raise ValueError('Step and target tolerance must lie within B training interval')
    # Episodes of the requested split follow checkpoint A's train/val/test split unless --split-mode is given.
    split = {k: v for k, v in state_a['config']['data'].items() if k in ('split_mode', 'split_fractions', 'split_seed')}
    if args.split_mode:
        split['split_mode'] = args.split_mode
    dataset = MocapPairs(args.manifest, split=args.split, image_size=size, min_dt=1e-9,
                         max_dt=args.max_horizon, min_mean=args.min_mean, **split, max_pairs_per_episode=0)
    samples, coverage = sample_pairs(dataset, args.pairs_per_episode, args.seed, args.max_horizon,
                                     args.max_pose_gap, args.step, args.target_tolerance)
    actions, direct_actions = [], []
    for sample in samples:
        episode = dataset.episodes[sample['episode_index']]
        offsets, poses = rollout_path(episode, sample['source_index'], sample['target_index'],
                                     args.step, args.max_pose_gap, args.target_tolerance)
        intervals = np.diff(offsets) / 1e9
        actions.append(relative_action(poses[:-1], poses[1:]))
        direct = relative_action(poses[0], poses[-1])
        direct_actions.append(direct)
        a_data, b_data = state_a['config']['data'], state_b['config']['data']
        sample.update(rollout_steps=len(intervals), rollout_dt_seconds=intervals.tolist(),
                      last_step_seconds=float(intervals[-1]),
                      a_outside_training_range=not a_data['min_dt'] <= sample['dt_seconds'] <= a_data['max_dt'],
                      b_steps_outside_training_range=int(((intervals < b_data['min_dt']) | (intervals > b_data['max_dt'])).sum()),
                      translation_mm=float(np.linalg.norm(direct[:3])), rotation_rad=float(np.linalg.norm(direct[3:])),
                      direct_action=direct.tolist(), rollout_actions=actions[-1].tolist())
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema_version=1, split=args.split, manifest=str(args.manifest.resolve()),
                    software=dict(torch=torch.__version__, numpy=np.__version__, evaluator_sha256=sha256(__file__)),
                    aggregation='equal episode means; paired episode percentile bootstrap',
                    manifest_sha256=sha256(args.manifest), seed=args.seed, noise_seed=args.noise_seed,
                    max_horizon_exclusive=args.max_horizon, step=args.step, target_tolerance=args.target_tolerance,
                    endpoint_policy='nearest recorded frame to a positive step multiple; no short residual',
                    max_pose_gap=args.max_pose_gap, min_mean=args.min_mean, image_size=size,
                    amp=args.amp, batch_size=args.batch_size, bootstrap=args.bootstrap,
                    groups=sorted({e['group'] for e in dataset.episodes}), coverage=coverage,
                    sampling='uniform eligible source without replacement; uniform available grid horizon',
                    feedback='uint8 quantisation, no teacher forcing', shared_noise=True,
                    pose_interpolation='linear translation and SLERP; no extrapolation')
    for name, path, state in [('a', args.checkpoint_a, state_a), ('b', args.checkpoint_b, state_b)]:
        protocol[f'model_{name}'] = dict(checkpoint=str(path.resolve()), sha256=sha256(path),
                                        checkpoint_step=state['step'], config=state['config'])
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (args.output / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')
    rows, illustrated = [], set()
    started = time.monotonic()
    columns = [('Initial (GT)', GROUND_TRUTH), ('A direct', GENERATED), ('B rollout', GENERATED),
               ('Target (GT)', GROUND_TRUTH), ('A error x3', TEXT), ('B error x3', TEXT)]
    with (args.output / 'per_pair_metrics.csv').open('w', newline='') as handle:
        writer = None
        for start in range(0, len(samples), args.batch_size):
            selected = samples[start:start + args.batch_size]
            source = torch.stack([dataset.image(s['episode_index'], s['source_index']) for s in selected]).to(device)
            predictions = predict_batch(model_a, model_b, source,
                                        torch.as_tensor(np.stack(direct_actions[start:start + len(selected)]), device=device),
                                        actions[start:start + len(selected)], args.amp)
            target = torch.stack([dataset.image(s['episode_index'], s['target_index']) for s in selected]).to(device)
            batch_metrics = {f'{name}_{key}': value.cpu().tolist() for name, prediction in predictions.items()
                             for key, value in image_metrics(prediction, target).items()}
            if not all(np.isfinite(values).all() for values in batch_metrics.values()):
                raise FloatingPointError('Non-finite evaluation metrics')
            for i, sample in enumerate(selected):
                row = {key: value for key, value in sample.items() if key not in ('rollout_dt_seconds', 'direct_action', 'rollout_actions')}
                row.update({key: values[i] for key, values in batch_metrics.items()})
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if sample['episode'] not in illustrated:
                    illustrated.add(sample['episode'])
                    a, b = predictions['a_direct'][i], predictions['b_rollout'][i]
                    panels = [to_pixels(x) for x in [source[i], a, b, target[i], 3 * (a - target[i]).abs(), 3 * (b - target[i]).abs()]]
                    compose([panels], columns=columns,
                            header=f"{sample['episode']} #{sample['source_index']} -> #{sample['target_index']} "
                                   f"dt={sample['dt_seconds']:.3f}s; B steps={sample['rollout_steps']}").save(
                                       args.output / f"comparison_{sample['pair_id']:04d}.png")
            handle.flush()
            print(f'{len(rows)}/{len(samples)} pairs; elapsed {time.monotonic() - started:.1f}s', flush=True)
    result = dict(protocol, elapsed_seconds=time.monotonic() - started,
                  overall=summarize(rows, args.bootstrap, args.seed), by_horizon={}, by_translation={})
    for horizon in sorted({row['nominal_dt_seconds'] for row in rows}):
        subset = [row for row in rows if row['nominal_dt_seconds'] == horizon]
        result['by_horizon'][f'{horizon:g}'] = summarize(subset, args.bootstrap, args.seed)
    for lo, hi in [(0, 5), (5, 15), (15, float('inf'))]:
        subset = [row for row in rows if lo <= row['translation_mm'] < hi]
        if subset:
            result['by_translation'][f'[{lo:g}, {hi:g}) mm'] = summarize(subset, args.bootstrap, args.seed)
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output / 'report.md').write_text(report_text(result))
    save_plot(result, args.output)
    print(json.dumps(result['overall']['paired'], indent=2), flush=True)


if __name__ == '__main__':
    main()
