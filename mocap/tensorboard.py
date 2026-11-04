"""Mirror a run's JSON logs into TensorBoard events, live or after the fact.

    python -m mocap.tensorboard runs/mocap_full --follow
    python -m tensorboard.main --logdir runs

The training scripts remain the single source of truth: train.jsonl and val_<step>/metrics.json
are read, and events are rebuilt from scratch in <run>/tensorboard at every start, so that
restarting this command never duplicates points. Works for world-model and diffusion runs.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Overlaid charts: the model is informative only where it separates from its baselines.
LAYOUT = {'Baselines': {
    **{f'motion retrieval recall@{k}': ['Multiline', [f'val/{name}_recall@{k}' for name in
                                                     ('model', 'zero_motion', 'copy_context')]] for k in (1, 5, 10)},
    **{f'image {metric}': ['Multiline', [f'val/{name}_{metric}' for name in ('model', 'zero_motion', 'copy_context')]]
       for metric in ('mae', 'foreground_mae', 'ssim', 'psnr')},
    'loss': ['Multiline', ['train/loss', 'val/val_loss']],
}}


class RunMirror:
    def __init__(self, run):
        self.run = Path(run)
        target = self.run / 'tensorboard'
        shutil.rmtree(target, ignore_errors=True)
        self.writer = SummaryWriter(target)
        self.writer.add_custom_scalars(LAYOUT)
        self.offset, self.metrics_done, self.images_done = 0, set(), set()

    def train_rows(self):
        path = self.run / 'train.jsonl'
        if not path.exists():
            return 0
        with path.open('rb') as f:
            f.seek(self.offset)
            data = f.read()
        complete = data[:data.rfind(b'\n') + 1]   # a partially written last line waits for the next poll
        self.offset += len(complete)
        rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
        for row in rows:
            for key, value in row.items():
                if key != 'step' and isinstance(value, (int, float)):
                    self.writer.add_scalar(f'train/{key}', value, row['step'])
        return len(rows)

    def validations(self):
        new = 0
        for folder in sorted(self.run.glob('val_*')):
            step = int(folder.name.split('_')[1])
            if step not in self.metrics_done and (folder / 'metrics.json').exists():
                try:
                    metrics = json.loads((folder / 'metrics.json').read_text())
                except json.JSONDecodeError:   # still being written
                    continue
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self.writer.add_scalar(f'val/{key}', value, step)
                self.metrics_done.add(step)
                new += 1
            # Diffusion runs also export comparison1.png, comparison2.png, ... with the columns
            # zero motion | context | generated | target | error | probe motion.
            grids = sorted(folder.glob('comparison*.png'))
            if step not in self.images_done and grids:
                try:
                    images = [(path.stem, np.asarray(Image.open(path).convert('RGB'))) for path in grids]
                except OSError:   # still being written
                    continue
                for tag, grid in images:
                    self.writer.add_image(f'val/{tag}', grid, step, dataformats='HWC')
                self.images_done.add(step)
                new += 1
        return new

    def sync(self):
        new = self.train_rows() + self.validations()
        self.writer.flush()
        return new


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--follow', action='store_true', help='Keep polling until each run writes status.json')
    p.add_argument('--interval', type=float, default=15.)
    args = p.parse_args()
    mirrors = [RunMirror(run) for run in args.runs]
    while True:
        for mirror in mirrors:
            new = mirror.sync()
            if new:
                print(f'{mirror.run}: {new} new records', flush=True)
        if not args.follow:
            break
        # A run is finished once status.json is newer than its last log line (resume rewrites both).
        finished = [(m.run / 'status.json').exists() and (m.run / 'train.jsonl').exists() and
                    (m.run / 'status.json').stat().st_mtime >= (m.run / 'train.jsonl').stat().st_mtime
                    for m in mirrors]
        if all(finished):
            for mirror in mirrors:
                mirror.sync()
            break
        time.sleep(args.interval)
    for mirror in mirrors:
        mirror.writer.close()


if __name__ == '__main__':
    main()
