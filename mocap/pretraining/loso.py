"""Leave-one-subject-out (LOSO) cross-validation of world-model pre-training.

Every recording group outside the test split is held out once as the validation subject; the
remaining groups are used for training. The test group keeps its 'test' split in every fold and
is never used for training or selection.

    prepare   writes data/loso/<group>/manifest.json and configs/loso/<group>.json
    run       trains the given folds sequentially on one device (runs/loso/<group>)
    summary   aggregates the validation curves of all folds into runs/loso/summary.json

Retrieval recall depends on the number of candidate targets (validation pairs), so the number
of validation pairs per episode is chosen per fold to give approximately the same total
(target_val_pairs) in every fold; chance level k/N is reported alongside each fold.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def prepare(args):
    source = (ROOT / args.manifest).resolve()
    manifest = json.loads(source.read_text())
    test_groups = set(manifest.get('test_groups', []))
    groups = sorted({e['group'] for e in manifest['episodes']} - test_groups)
    base = json.loads((ROOT / args.base_config).read_text())
    for group in groups:
        folder = ROOT / 'data' / 'loso' / group
        folder.mkdir(parents=True, exist_ok=True)
        episodes = []
        for entry in manifest['episodes']:
            entry = dict(entry)
            # Episode folders stay in the source cache; paths are rewritten relative to the fold manifest.
            entry['path'] = os.path.relpath(source.parent / entry['path'], folder.resolve())
            if entry['group'] not in test_groups:
                entry['split'] = 'val' if entry['group'] == group else 'train'
            episodes.append(entry)
        fold = dict(manifest, episodes=episodes, val_groups=[group],
                    loso=dict(source_manifest=str(source), held_out=group))
        (folder / 'manifest.json').write_text(json.dumps(fold, indent=2) + '\n')
        count = sum(e['group'] == group for e in episodes)
        config = json.loads(json.dumps(base))
        config['data']['manifest'] = str((folder / 'manifest.json').relative_to(ROOT))
        config.pop('validation_manifest', None)
        # Approximately constant retrieval gallery across folds (see module docstring).
        config['val_pairs_per_episode'] = math.ceil(args.target_val_pairs / count)
        out = ROOT / 'configs' / 'loso' / f'{group}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(config, indent=2) + '\n')
        print(f'{group}: {count} validation episodes, {config["val_pairs_per_episode"]} pairs each')


def run(args):
    for group in args.groups:
        output = ROOT / 'runs' / 'loso' / group
        if (output / 'status.json').exists():
            print(f'{group}: already finished, skipped', flush=True)
            continue
        command = [sys.executable, '-u', '-m', 'mocap.pretraining.train', '--config', f'configs/loso/{group}.json',
                   '--output', str(output), '--device', args.device]
        if (output / 'last.pt').exists():
            command += ['--resume', str(output / 'last.pt')]
        output.mkdir(parents=True, exist_ok=True)
        print(f'{group}: {" ".join(command)}', flush=True)
        with open(output / 'console.log', 'a') as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f'{group}: exit code {code}', flush=True)


def summary(args):
    folds = {}
    for folder in sorted((ROOT / 'runs' / 'loso').glob('*/')):
        curve = {}
        for metrics in sorted(folder.glob('val_*/metrics.json')):
            curve[int(metrics.parent.name[4:])] = json.loads(metrics.read_text())
        if not curve:
            continue
        best = max(curve, key=lambda s: curve[s]['model_recall@5'])
        m = curve[best]
        folds[folder.name] = dict(
            finished=(folder / 'status.json').exists(), pairs=m['pairs'], best_step=best,
            chance_recall5=5 / m['pairs'],
            **{k: m[k] for k in ['model_recall@5', 'zero_motion_recall@5', 'copy_context_recall@5', 'val_loss']},
            curve_recall5={s: curve[s]['model_recall@5'] for s in sorted(curve)})
    keys = ['model_recall@5', 'zero_motion_recall@5', 'copy_context_recall@5', 'best_step']
    aggregate = {k: dict(mean=float(np.mean([f[k] for f in folds.values()])),
                         std=float(np.std([f[k] for f in folds.values()]))) for k in keys} if folds else {}
    # Mean validation curve over the folds that reached each step: used to choose the step count
    # of the final pre-training on all groups.
    steps = sorted({s for f in folds.values() for s in f['curve_recall5']})
    mean_curve = {s: float(np.mean([f['curve_recall5'][s] for f in folds.values() if s in f['curve_recall5']]))
                  for s in steps}
    result = dict(folds=folds, aggregate=aggregate, mean_curve_recall5=mean_curve)
    (ROOT / 'runs' / 'loso' / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{"fold":<10}{"pairs":>6}{"step":>6}{"R@5":>8}{"zero":>8}{"copy":>8}{"chance":>8}')
    for name, f in folds.items():
        print(f'{name:<10}{f["pairs"]:>6}{f["best_step"]:>6}{f["model_recall@5"]:>8.3f}'
              f'{f["zero_motion_recall@5"]:>8.3f}{f["copy_context_recall@5"]:>8.3f}{f["chance_recall5"]:>8.3f}'
              + ('' if f['finished'] else '  (running)'))
    for k, v in aggregate.items():
        print(f'{k}: {v["mean"]:.3f} +- {v["std"]:.3f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--manifest', default='data/ifl_aligned/manifest.json')
    a.add_argument('--base-config', default='configs/loso_base.json')
    a.add_argument('--target-val-pairs', type=int, default=480)
    b = sub.add_parser('run')
    b.add_argument('--groups', nargs='+', required=True)
    b.add_argument('--device', default='cuda:0')
    sub.add_parser('summary')
    args = p.parse_args()
    dict(prepare=prepare, run=run, summary=summary)[args.command](args)


if __name__ == '__main__':
    main()
