"""Single-step action ablation on identical held-out pairs for two diffusion models."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mocap.evaluation.compare_rollout import METRICS, sample_pairs, sha256
from mocap.processing.dataset import MocapPairs
from mocap.evaluation.evaluate import image_metrics, load_model
from mocap.processing.geometry import relative_action
from mocap.visualization.visualize import compose, to_pixels, GENERATED, GROUND_TRUTH

CONDITIONS = ('correct', 'zero', 'wrong')


def wrong_motion(actions):
    """Opposite translation direction and rotation vector, preserving both norms.

    This is a synthetic matched-magnitude control, not the SE(3) inverse action:
    an actual inverse would also rotate the translation into a different frame.
    Exactly zero actions cannot supply a distinct wrong-direction control.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
        raise ValueError('Expected finite [N, 6] actions')
    return -actions


@torch.inference_mode()
def predict_conditions(model, source, actions, amp=False):
    """All three conditions start from the same real image and share diffusion noise."""
    controls = dict(correct=actions, zero=torch.zeros_like(actions),
                    wrong=torch.as_tensor(wrong_motion(actions.cpu().numpy()), device=source.device))
    with torch.autocast(source.device.type, dtype=torch.bfloat16,
                        enabled=amp and source.device.type == 'cuda'):
        return {name: model.predict(source, action, shared_noise=True).float()
                for name, action in controls.items()}


def summary(rows, bootstrap=5000, seed=42):
    """Paired, equally weighted episode effects, with positive = correct action helps."""
    if not rows or bootstrap < 1:
        raise ValueError('Expected rows and positive bootstrap count')
    episodes = sorted({r['episode'] for r in rows})
    keys = [f'{model}_{condition}_{metric}' for model in ('a', 'b')
            for condition in CONDITIONS for metric in METRICS]
    per_episode = []
    for episode in episodes:
        subset = [r for r in rows if r['episode'] == episode]
        per_episode.append(dict(episode=episode, pairs=len(subset),
                                **{key: float(np.mean([r[key] for r in subset])) for key in keys}))
    means = {key: float(np.mean([r[key] for r in per_episode])) for key in keys}
    # A static ground-truth action is identical to its zero and negated controls.
    # Keep its reconstruction scores, but exclude it from action-dependence effects.
    active = [r for r in rows if r['action_nonzero']]
    active_episodes = sorted({r['episode'] for r in active})
    result = dict(pairs=len(rows), episodes=len(episodes), means=means, per_episode=per_episode,
                  action_test_pairs=len(active), action_test_episodes=len(active_episodes),
                  correct_advantage={}, b_minus_a_advantage={},
                  outside_training_pairs={m: sum(r[f'{m}_outside_training_range'] for r in rows)
                                          for m in ('a', 'b')})
    if not active:
        return result
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(active_episodes), size=(bootstrap, len(active_episodes)))

    def estimate(values):
        ci = np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist() if len(values) > 1 else None
        return dict(mean=float(values.mean()), ci95=ci)

    for control in ('zero', 'wrong'):
        result['correct_advantage'][control] = {}
        result['b_minus_a_advantage'][control] = {}
        for metric in METRICS:
            # Errors: control - correct. Scores: correct - control.
            sign = 1 if metric not in ('ssim', 'psnr') else -1
            episode_effects = {}
            for model in ('a', 'b'):
                pair_effects = [sign * (r[f'{model}_{control}_{metric}'] -
                                       r[f'{model}_correct_{metric}']) for r in active]
                values = np.array([np.mean([effect for r, effect in zip(active, pair_effects)
                                           if r['episode'] == episode]) for episode in active_episodes])
                episode_effects[model] = values
                result['correct_advantage'][control].setdefault(model, {})[metric] = dict(
                    estimate(values), correct_pair_win_fraction=float(np.mean(np.array(pair_effects) > 0)))
            result['b_minus_a_advantage'][control][metric] = estimate(
                episode_effects['b'] - episode_effects['a'])
    return result


def report(result):
    overall = result['overall']
    lines = ['# Single-step motion dependence', '',
             f"Split: {result['split']}; {overall['pairs']} pairs from {overall['episodes']} episodes.",
             'Both models predict directly from the SAME real source to the SAME real target. No rollout.',
             'Wrong action negates translation and rotation vectors: identical translation norm and rotation angle.',
             'Every condition uses identical seeded diffusion noise. Scores are in image space after diffusion.', '',
             '| Model / condition | MAE | Foreground MAE | SSIM |',
             '|---|---:|---:|---:|']
    for model in ('a', 'b'):
        for condition in CONDITIONS:
            values = [overall['means'][f'{model}_{condition}_{m}'] for m in ('mae', 'foreground_mae', 'ssim')]
            lines.append(f"| {model.upper()} / {condition} | " + ' | '.join(f'{v:.5f}' for v in values) + ' |')
    lines += ['', '## Correct-action advantage on foreground MAE', '',
              'Positive = the correct action reduces error relative to the control. '
              'Intervals use paired episode bootstrap with equal episode weights.',
              '| Control | A advantage [95% CI] | B advantage [95% CI] | B minus A advantage [95% CI] |',
              '|---|---:|---:|---:|']

    def fmt(value):
        ci = value['ci95']
        return f"{value['mean']:+.5f}" + (f" [{ci[0]:+.5f}, {ci[1]:+.5f}]" if ci else ' [CI unavailable]')

    for control, models in overall['correct_advantage'].items():
        values = [models[m]['foreground_mae'] for m in ('a', 'b')]
        values.append(overall['b_minus_a_advantage'][control]['foreground_mae'])
        lines.append('| ' + control + ' | ' + ' | '.join(fmt(v) for v in values) + ' |')
    lines += ['', '## By nominal horizon: correct-action advantage over wrong action', '',
              '| Horizon (s) | Pairs | A foreground MAE advantage | B foreground MAE advantage | A / B out of training range |',
              '|---|---:|---:|---:|---:|']
    for horizon, subset in result['by_horizon'].items():
        effects = subset['correct_advantage'].get('wrong')
        values = [fmt(effects[m]['foreground_mae']) if effects else 'n/a' for m in ('a', 'b')]
        outside = subset['outside_training_pairs']
        lines.append(f"| {horizon} | {subset['pairs']} | {values[0]} | {values[1]} | {outside['a']} / {outside['b']} |")
    lines += ['', '## Interpretation', '',
              '- Positive advantage over both controls supports useful action conditioning; output changes alone do not.',
              '- Wrong motion is a synthetic opposite-direction control, not a physically observed alternative trajectory.',
              '- Exact-zero actions are excluded from action effects but retained in reconstruction averages.',
              '- The current models have disjoint training horizons (A: 0.4–3 s; B: 0.19–0.21 s). '
              'Read each horizon and its range flags; pooled results cannot isolate intrinsic motion-learning ability.',
              '- These are complete encoder/predictor/decoder systems and one diffusion seed. '
              'Episode confidence intervals do not measure generalisation to new subjects or training seeds.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('checkpoint-a', 'checkpoint-b', 'manifest', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--split', choices=['test', 'val'], default='test')
    p.add_argument('--split-mode', choices=['random', 'leave-out'],
                        help="Override checkpoint A's train/val/test split (e.g. leave-out on a custom manifest)")
    p.add_argument('--pairs-per-episode', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise-seed', type=int, default=42)
    p.add_argument('--max-horizon', type=float, default=3.)
    p.add_argument('--step', type=float, default=.2)
    p.add_argument('--target-tolerance', type=float, default=.01)
    p.add_argument('--max-pose-gap', type=float, default=.1)
    p.add_argument('--min-mean', type=float, default=8.)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action='store_true')
    args = p.parse_args()
    if args.batch_size < 1 or args.bootstrap < 1 or args.pairs_per_episode < 1:
        p.error('Batch size, bootstrap count and pairs per episode must be positive')
    if not np.isfinite(args.min_mean) or not 0 <= args.min_mean <= 255:
        p.error('min-mean must be in [0, 255]')
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve evaluations')
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    models, states = {}, {}
    for name, path in [('a', args.checkpoint_a), ('b', args.checkpoint_b)]:
        models[name], states[name] = load_model(path, device)
        if states[name].get('kind') != 'diffusion':
            raise ValueError('Both checkpoints must include diffusion decoders')
        models[name].noise_seed = args.noise_seed
    size = states['a']['config']['model']['image_size']
    if size != states['b']['config']['model']['image_size']:
        raise ValueError('Models must have the same image size')
    # Episodes of the requested split follow checkpoint A's train/val/test split unless --split-mode is given.
    split = {k: v for k, v in states['a']['config']['data'].items() if k in ('split_mode', 'split_fractions', 'split_seed')}
    if args.split_mode:
        split['split_mode'] = args.split_mode
    dataset = MocapPairs(args.manifest, split=args.split, image_size=size, min_dt=1e-9,
                         max_dt=args.max_horizon, min_mean=args.min_mean, **split)
    samples, coverage = sample_pairs(dataset, args.pairs_per_episode, args.seed, args.max_horizon,
                                     args.max_pose_gap, args.step, args.target_tolerance)
    actions = []
    for sample in samples:
        episode = dataset.episodes[sample['episode_index']]
        action = relative_action(episode['poses'][sample['source_index']], episode['poses'][sample['target_index']])
        actions.append(action)
        sample.update(correct_action=action.tolist(), wrong_action=wrong_motion(action[None])[0].tolist(),
                      action_nonzero=bool(np.any(action != 0)),
                      translation_mm=float(np.linalg.norm(action[:3])), rotation_rad=float(np.linalg.norm(action[3:])))
        for model in ('a', 'b'):
            data = states[model]['config']['data']
            sample[f'{model}_outside_training_range'] = not data['min_dt'] <= sample['dt_seconds'] <= data['max_dt']
    protocol = dict(schema_version=1, mode='both_models_single_step_action_ablation', split=args.split,
                    manifest=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
                    evaluator_sha256=sha256(__file__), sampler_sha256=sha256(Path(__file__).with_name('compare_rollout.py')),
                    seed=args.seed, noise_seed=args.noise_seed, amp=args.amp, batch_size=args.batch_size,
                    bootstrap=args.bootstrap, pairs_per_episode=args.pairs_per_episode,
                    max_horizon_exclusive=args.max_horizon, step=args.step, target_tolerance=args.target_tolerance,
                    max_pose_gap=args.max_pose_gap, min_mean=args.min_mean,
                    wrong_action='negated translation and rotation vector; exact separate norm matching',
                    shared_noise=True, coverage=coverage, groups=sorted({e['group'] for e in dataset.episodes}))
    for name, path in [('a', args.checkpoint_a), ('b', args.checkpoint_b)]:
        protocol[f'model_{name}'] = dict(checkpoint=str(path.resolve()), sha256=sha256(path),
                                        checkpoint_step=states[name]['step'], config=states[name]['config'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (args.output / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')
    rows, illustrated = [], set()
    with (args.output / 'per_pair_metrics.csv').open('w', newline='') as handle:
        writer = None
        for start in range(0, len(samples), args.batch_size):
            selected = samples[start:start + args.batch_size]
            source = torch.stack([dataset.image(s['episode_index'], s['source_index']) for s in selected]).to(device)
            target = torch.stack([dataset.image(s['episode_index'], s['target_index']) for s in selected]).to(device)
            action = torch.as_tensor(np.stack(actions[start:start + len(selected)]), device=device)
            predictions = {name: predict_conditions(model, source, action, args.amp) for name, model in models.items()}
            metrics = {f'{name}_{condition}_{key}': values.cpu().tolist()
                       for name, conditions in predictions.items() for condition, prediction in conditions.items()
                       for key, values in image_metrics(prediction, target).items()}
            if not all(np.isfinite(values).all() for values in metrics.values()):
                raise FloatingPointError('Non-finite metrics')
            for i, sample in enumerate(selected):
                row = {k: v for k, v in sample.items() if k not in ('correct_action', 'wrong_action')}
                row.update({k: v[i] for k, v in metrics.items()})
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if sample['episode'] not in illustrated:
                    illustrated.add(sample['episode'])
                    for name in models:
                        panels = [to_pixels(source[i])] + [to_pixels(predictions[name][c][i]) for c in CONDITIONS] + [to_pixels(target[i])]
                        columns = [('Initial GT', GROUND_TRUTH)] + [(c, GENERATED) for c in CONDITIONS] + [('Target GT', GROUND_TRUTH)]
                        compose([panels], columns=columns,
                                header=f"{name.upper()} {sample['episode']} dt={sample['dt_seconds']:.3f}s").save(
                                    args.output / f"comparison_{sample['pair_id']:04d}_{name}.png")
            handle.flush()
            print(f'{len(rows)}/{len(samples)} pairs', flush=True)
    result = dict(protocol, overall=summary(rows, args.bootstrap, args.seed), by_horizon={})
    for horizon in sorted({r['nominal_dt_seconds'] for r in rows}):
        result['by_horizon'][f'{horizon:g}'] = summary([r for r in rows if r['nominal_dt_seconds'] == horizon],
                                                     args.bootstrap, args.seed)
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output / 'report.md').write_text(report(result))
    print(json.dumps(result['overall']['correct_advantage'], indent=2), flush=True)


if __name__ == '__main__':
    main()
