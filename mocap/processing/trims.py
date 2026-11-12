"""Manual episode trimming: suggestions, the persistent trim file and the trimmed cache.

Trim file (annotations/episode_trims.json), one record per episode of the source cache:

    start_frame / end_frame    inclusive frame indices in the source cache (convenience)
    start_header_ns / end_...  image header timestamps of those frames: the authoritative cut
    excluded                   drop the whole episode
    status                     'suggested' (never touched) or 'manual' (edited or confirmed in the editor)
    suggested                  the automatic suggestion, kept for "reset"

Suggestion: first and last *ultrasound* frame, i.e. mean intensity >= min_mean (probe in contact,
the pre-training filter) and black upper corners outside the fan (rejects scanner screens such as
the wallpaper recorded at the end of nardi6).

Delay compensation. The source cache (data/ifl_aligned) already pairs every image with the probe
pose at image_header_time - 200 ms (manifest pose_alignment). Cuts are image times, and `apply`
slices images and poses together, so every kept image keeps its compensated pose; the alignment
metadata is copied to the trimmed manifest.

Split. The random split permutes the sorted episode ids, so removing an episode would reshuffle
val/test relative to the world models trained on the source cache. `apply` therefore stores the
source random assignment in the trimmed manifest (frozen_random_split), used by assign_splits.

    python -m mocap.processing.trims suggest --manifest data/ifl_aligned/manifest.json --trims annotations/episode_trims.json
    python -m mocap.processing.trims apply --trims annotations/episode_trims.json --output data/ifl_trimmed [--overwrite]
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
import numpy as np
from mocap.processing.dataset import assign_splits

ROOT = Path(__file__).resolve().parents[2]   # EchoWorld/
DEFAULT_TRIMS = ROOT / 'annotations' / 'episode_trims.json'
SPLIT_DEFAULTS = dict(split_fractions=[.7, .15, .15], split_seed=42)


def outside_fan_mean(images):
    """[T, S, S] uint8 -> [T] mean intensity of the two upper corners, outside the ultrasound fan.

    Bands at 18-40% of the height and 13% of the width on each side, below the scanner icon.
    """
    s = images.shape[-1]
    rows, cols = slice(int(.18 * s), int(.4 * s)), int(.134 * s)
    return np.concatenate([images[:, rows, :cols], images[:, rows, -cols:]], axis=2).mean(axis=(1, 2))


def ultrasound_mask(images, min_mean=8., max_outside_fan_mean=20.):
    return (images.mean(axis=(1, 2)) >= min_mean) & (outside_fan_mean(images) <= max_outside_fan_mean)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_episode(manifest_path, entry):
    folder = Path(manifest_path).parent / entry['path']
    poses = dict(np.load(folder / 'poses.npz', allow_pickle=False))
    images = np.load(folder / 'images.npy', mmap_mode='r', allow_pickle=False)
    return folder, poses, images


def suggestion(images, poses, **filters):
    valid = np.flatnonzero(ultrasound_mask(images, **filters))
    if not len(valid):
        return dict(start_frame=0, end_frame=len(images) - 1, excluded=True)
    return dict(start_frame=int(valid[0]), end_frame=int(valid[-1]), excluded=False)


class TrimStore:
    """Thread-safe trim file; every change is written atomically and appended to a .log.jsonl."""
    def __init__(self, path=DEFAULT_TRIMS, manifest=None):
        self.path = Path(path).resolve()
        self.lock = threading.Lock()
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if manifest and Path(manifest).resolve() != self.manifest_path:
                raise ValueError(f'Trim file refers to {self.manifest_path}, not {manifest}')
        else:
            if manifest is None:
                raise FileNotFoundError(f'{self.path} does not exist; supply the source manifest')
            source = json.loads(Path(manifest).read_text())
            self.data = dict(schema_version=1, source_manifest=os.path.relpath(Path(manifest).resolve(), self.path.parent),
                             source_manifest_sha256=sha256(manifest), pose_alignment=source.get('pose_alignment'),
                             settings=source.get('settings'), filters=dict(min_mean=8., max_outside_fan_mean=20.),
                             convention='Inclusive cut [start, end] on image header timestamps of the source cache; '
                                        'each kept image keeps its cached pose (already at header time + pose offset).',
                             episodes={})
        self.manifest = json.loads(self.manifest_path.read_text())

    @property
    def manifest_path(self):
        return (self.path.parent / self.data['source_manifest']).resolve()

    def entries(self):
        return self.manifest['episodes']

    def entry(self, episode_id):
        return next(e for e in self.entries() if e['episode_id'] == episode_id)

    def record(self, episode_id, start, end, excluded, poses, status, note='', suggested=None):
        header = poses.get('image_header_timestamps_ns', poses['timestamps_ns'])
        times = poses['timestamps_ns']
        return dict(start_frame=int(start), end_frame=int(end), excluded=bool(excluded), status=status,
                    start_header_ns=int(header[start]), end_header_ns=int(header[end]),
                    kept_seconds=float((times[end] - times[start]) / 1e9), frames=int(len(times)),
                    note=note, suggested=suggested, updated_at=now())

    def suggest_missing(self, refresh=False):
        """Compute suggestions for new episodes (and, with refresh, for untouched ones)."""
        added = 0
        for entry in self.entries():
            current = self.data['episodes'].get(entry['episode_id'])
            if current and (current['status'] == 'manual' or not refresh):
                continue
            _, poses, images = load_episode(self.manifest_path, entry)
            s = suggestion(images, poses, **self.data['filters'])
            self.data['episodes'][entry['episode_id']] = self.record(
                entry['episode_id'], s['start_frame'], s['end_frame'], s['excluded'], poses, 'suggested', suggested=s)
            added += 1
        if added:
            self.save(dict(action='suggest', episodes=added))
        return added

    def update(self, episode_id, start_frame=None, end_frame=None, excluded=None, note=None, reset=False,
               fit=None, clear_fit=False):
        """Change the cut, exclusion or note; `fit` sets the RGB-D fit marker, `clear_fit` removes it.

        The fit marker ({frame, header_ns, rgbd_time_ns, depth_index, ...}, built by the trim editor) is
        kept across every other change, including a reset of the cut to the suggestion.
        """
        with self.lock:
            current = self.data['episodes'][episode_id]
            _, poses, _ = load_episode(self.manifest_path, self.entry(episode_id))
            if reset:
                s = current['suggested']
                new = self.record(episode_id, s['start_frame'], s['end_frame'], s['excluded'], poses, 'suggested',
                                  current.get('note', ''), s)
            else:
                start = current['start_frame'] if start_frame is None else int(start_frame)
                end = current['end_frame'] if end_frame is None else int(end_frame)
                if not 0 <= start <= end < len(poses['timestamps_ns']):
                    raise ValueError(f'Invalid cut [{start}, {end}]')
                new = self.record(episode_id, start, end, current['excluded'] if excluded is None else excluded, poses,
                                  'manual', current.get('note', '') if note is None else note, current['suggested'])
            marker = None if clear_fit else fit if fit is not None else current.get('fit')
            if marker is not None:
                new['fit'] = marker
                if fit is not None:
                    new['status'] = 'manual'
            self.data['episodes'][episode_id] = new
            self.save(dict(action='reset' if reset else 'update', episode=episode_id,
                           **{k: new[k] for k in ('start_frame', 'end_frame', 'excluded', 'note')},
                           fit=new.get('fit')))
            return new

    def save(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data['updated_at'] = now()
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.data, indent=1) + '\n')
        temp.replace(self.path)
        with self.path.with_suffix('.log.jsonl').open('a') as log:
            log.write(json.dumps(dict(time=now(), **event)) + '\n')


def apply(trims_path, output, overwrite=False):
    store = TrimStore(trims_path)
    source_path = store.manifest_path
    if sha256(source_path) != store.data['source_manifest_sha256']:
        print('Warning: the source manifest changed since the trims were created; cuts are matched by image timestamp', flush=True)
    output = Path(output).resolve()
    if (output / 'manifest.json').exists():
        if not overwrite:
            raise FileExistsError(f'{output} exists; pass --overwrite to regenerate it')
        shutil.rmtree(output / 'episodes', ignore_errors=True)
    source = store.manifest
    episodes, dropped = [], []
    for entry in source['episodes']:
        trim = store.data['episodes'].get(entry['episode_id'])
        if trim is None:
            raise ValueError(f'No trim for {entry["episode_id"]}; run suggest first')
        if trim['excluded']:
            dropped.append(entry['episode_id'])
            continue
        folder, poses, images = load_episode(source_path, entry)
        header = poses.get('image_header_timestamps_ns', poses['timestamps_ns'])
        # Image timestamps are authoritative; indices only if the timestamps are unchanged.
        start, end = (int(np.flatnonzero(header == trim[k])[0]) if (header == trim[k]).any() else None
                      for k in ('start_header_ns', 'end_header_ns'))
        if start is None or end is None:
            raise ValueError(f'Cut timestamps of {entry["episode_id"]} not found in the source cache')
        keep = slice(start, end + 1)
        target = output / entry['path']
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / 'images.npy', np.ascontiguousarray(images[keep]))
        np.savez(target / 'poses.npz', **{k: v[keep] if v.ndim and len(v) == len(images) else v for k, v in poses.items()})
        times = poses['timestamps_ns'][keep]
        info = dict(source_start_frame=start, source_end_frame=end, source_frames=len(images),
                    start_header_ns=trim['start_header_ns'], end_header_ns=trim['end_header_ns'],
                    status=trim['status'], note=trim.get('note', ''))
        new = dict(entry, frames=int(end - start + 1), duration_s=float((times[-1] - times[0]) / 1e9), trim=info)
        metadata = folder / 'metadata.json'
        if metadata.exists():
            meta = json.loads(metadata.read_text())
            meta.update(frames=new['frames'], duration_s=new['duration_s'], trim=info)
            (target / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        episodes.append(new)
    frozen = dict(SPLIT_DEFAULTS, source_manifest=str(source_path), source_manifest_sha256=sha256(source_path),
                  assignments=assign_splits(source, 'random', SPLIT_DEFAULTS['split_fractions'], SPLIT_DEFAULTS['split_seed']))
    manifest = dict(source, episodes=episodes, frozen_random_split=frozen,
                    trims=dict(file=str(Path(trims_path).resolve()), sha256=sha256(trims_path), dropped=dropped,
                               source_manifest=str(source_path), pose_alignment_preserved=source.get('pose_alignment')))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    kept = sum(e['frames'] for e in episodes)
    total = sum(e['frames'] for e in source['episodes'])
    print(f'{len(episodes)} episodes, {kept}/{total} frames kept; dropped {dropped}; manifest {output / "manifest.json"}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('suggest')
    s.add_argument('--manifest', default=str(ROOT / 'data' / 'ifl_aligned' / 'manifest.json'))
    s.add_argument('--trims', default=str(DEFAULT_TRIMS))
    s.add_argument('--refresh', action='store_true', help='recompute untouched (non-manual) suggestions')
    a = sub.add_parser('apply')
    a.add_argument('--trims', default=str(DEFAULT_TRIMS))
    a.add_argument('--output', default=str(ROOT / 'data' / 'ifl_trimmed'))
    a.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.command == 'suggest':
        store = TrimStore(args.trims, args.manifest)
        print(f'{store.suggest_missing(args.refresh)} suggestions written to {store.path}')
    else:
        apply(args.trims, args.output, args.overwrite)


if __name__ == '__main__':
    main()
