"""Canonical-frame cache: the trimmed episodes with probe poses mapped onto the canonical SMPL body.

Canonical_US (process_recordings --trims) fits SKEL on each recording at the RGB-D fit marker of the
trim editor, projects every probe pose on the fitted skin along the probe Z axis and transfers it to
the canonical body (theta = 0, beta = 0): canonical_xyz (skin point, metres) and canonical_rotations
(probe axes) in trajectory_full.npz. This module builds a cache identical to data/ifl_trimmed except
that world_T_probe is replaced by that canonical pose, interpolated at every image time:

    world_T_probe (canonical) = [canonical_rotation | canonical_xyz]   at timestamps_ns (image header + offset)

Everything downstream (pairs, relative actions inv(T_a) T_b, splits, guidance labels) is unchanged; the
actions are now motions on the canonical body instead of motions in the tracker frame.

Frames without a canonical pose (projection missed the skin, or no canonical sample within
--max-gap-ms) are dropped; segments break wherever consecutive kept frames are not bridged by
canonical samples. The random split of the source manifest (frozen_random_split) is kept, so the
episodes keep the split of every previous run.

    python -m mocap.processing.canonical build --source data/ifl_trimmed/manifest.json \
        --reconstructions ../Canonical_US/outputs/probe_reconstruction_trims --output data/ifl_canonical
    python -m mocap.processing.canonical loso --manifest data/ifl_canonical/manifest.json --val fengxhou --test xuesong \
        --output data/loso_canonical/fengxhou
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]


def find_reconstructions(roots):
    """Recording folder name -> reconstruction folder (searched recursively, e.g. per-shard outputs)."""
    found = {}
    for root in roots:
        for trajectory in Path(root).rglob('trajectory_full.npz'):
            folder = trajectory.parent
            if folder.name in found:
                raise ValueError(f'Recording {folder.name} found twice: {found[folder.name]} and {folder}')
            found[folder.name] = folder
    return found


def canonical_poses(times, samples, max_gap_ns):
    """Canonical 4x4 poses at `times` (linear + SLERP between bracketing samples); NaN where unavailable.

    Returns (poses [N, 4, 4], valid [N], bracket [N]) where bracket is the index of the left sample
    (used to break segments where two kept frames are not bridged by consecutive samples).
    """
    st, xyz, rot = samples['timestamps_ns'], samples['canonical_xyz'], samples['canonical_rotations']
    order = np.argsort(st)
    st, xyz, rot = st[order], xyz[order], rot[order]
    poses = np.full((len(times), 4, 4), np.nan)
    valid = np.zeros(len(times), bool)
    bracket = np.full(len(times), -1)
    for i, t in enumerate(times):
        right = int(np.searchsorted(st, t))
        if right < len(st) and st[right] == t:
            left = right
        elif 0 < right < len(st) and st[right] - st[right - 1] <= max_gap_ns:
            left = right - 1
        else:
            continue
        if left == right:
            p, r = xyz[left], rot[left]
        else:
            a = (t - st[left]) / (st[right] - st[left])
            p = (1 - a) * xyz[left] + a * xyz[right]
            r = Slerp([0., 1.], Rotation.from_matrix(rot[[left, right]]))([a]).as_matrix()[0]
        poses[i] = np.eye(4)
        poses[i, :3, :3], poses[i, :3, 3] = r, p
        valid[i], bracket[i] = True, left
    return poses, valid, bracket


def build(source, reconstructions, output, max_gap_ms=100., overwrite=False, min_frames=10):
    source = Path(source).resolve()
    manifest = json.loads(source.read_text())
    output = Path(output).resolve()
    if (output / 'manifest.json').exists():
        if not overwrite:
            raise FileExistsError(f'{output} exists; pass --overwrite')
        shutil.rmtree(output / 'episodes', ignore_errors=True)
    found = find_reconstructions(reconstructions)
    max_gap = int(max_gap_ms * 1e6)
    episodes, skipped, report = [], {}, []
    for entry in manifest['episodes']:
        name = Path(entry['path']).name
        folder = found.get(name)
        if folder is None:
            skipped[entry['episode_id']] = 'no reconstruction'
            continue
        status = json.loads((folder / 'report.json').read_text()).get('status') if (folder / 'report.json').exists() else None
        if status != 'complete':
            skipped[entry['episode_id']] = f'reconstruction status {status}'
            continue
        src = source.parent / entry['path']
        poses = dict(np.load(src / 'poses.npz'))
        images = np.load(src / 'images.npy', mmap_mode='r')
        with np.load(folder / 'trajectory_full.npz') as f:
            samples = {k: f[k] for k in ('timestamps_ns', 'canonical_xyz', 'canonical_rotations', 'surface_distance')}
        canonical, valid, bracket = canonical_poses(poses['timestamps_ns'], samples, max_gap)
        keep = np.flatnonzero(valid)
        if len(keep) < min_frames:
            skipped[entry['episode_id']] = f'{len(keep)} frames with a canonical pose'
            continue
        # New segments: original tracking segment changes, or kept frames not bridged by consecutive samples.
        seg = poses['segment_ids'][keep]
        br = bracket[keep]
        breaks = np.concatenate([[False], (seg[1:] != seg[:-1]) | (br[1:] - br[:-1] > 1)])
        segments = np.cumsum(breaks)
        out = {k: v[keep] if v.ndim and len(v) == len(images) else v for k, v in poses.items()}
        out['tracker_T_probe'] = out['world_T_probe']          # original tracker pose, for reference
        out['world_T_probe'] = canonical[keep]
        out['segment_ids'] = segments
        target = output / entry['path']
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / 'images.npy', np.ascontiguousarray(images[keep]))
        np.savez(target / 'poses.npz', **out)
        fit = json.loads((folder / 'fit_frame.json').read_text()) if (folder / 'fit_frame.json').exists() else None
        sd = np.abs(samples['surface_distance']) * 1000
        info = dict(reconstruction=str(folder), frames_kept=int(len(keep)), frames_source=int(len(images)),
                    surface_distance_median_mm=float(np.median(sd)), fit_frame=fit)
        report.append(dict(episode_id=entry['episode_id'], **info))
        times = out['timestamps_ns']
        episodes.append(dict(entry, frames=int(len(keep)), duration_s=float((times[-1] - times[0]) / 1e9), canonical=info))
    result = dict(manifest, episodes=episodes,
                  canonical=dict(source_manifest=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                 reconstructions=[str(Path(r).resolve()) for r in reconstructions], max_gap_ms=max_gap_ms,
                                 pose='canonical SMPL body (theta=0, beta=0): probe projected on the fitted SKEL skin along '
                                      'its Z axis, transferred by Canonical_US; metres', skipped=skipped))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'canonical_report.json').write_text(json.dumps(report, indent=2) + '\n')
    kept = sum(e['frames'] for e in episodes)
    print(f'{len(episodes)}/{len(manifest["episodes"])} episodes, {kept}/{sum(e["frames"] for e in manifest["episodes"])} '
          f'frames with a canonical pose -> {output / "manifest.json"}')
    for episode_id, reason in skipped.items():
        print(f'  skipped {episode_id}: {reason}')
    return result


def loso(manifest_path, val_group, test_group, output):
    """Leave-one-subject-out fold manifest (split field) over the same episode folders."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for entry in manifest['episodes']:
        split = 'val' if entry['group'] == val_group else 'test' if entry['group'] == test_group else 'train'
        episodes.append(dict(entry, split=split, path=os.path.relpath(manifest_path.parent / entry['path'], output)))
    fold = dict(manifest, episodes=episodes, val_groups=[val_group], test_groups=[test_group],
                loso=dict(source_manifest=str(manifest_path), held_out=val_group))
    (output / 'manifest.json').write_text(json.dumps(fold, indent=2) + '\n')
    counts = {s: sum(e['split'] == s for e in episodes) for s in ('train', 'val', 'test')}
    print(f'LOSO fold {val_group} (test {test_group}): {counts} -> {output / "manifest.json"}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command')
    b = sub.add_parser('build')
    b.add_argument('--source', default=str(ROOT / 'data' / 'ifl_trimmed' / 'manifest.json'))
    b.add_argument('--reconstructions', nargs='+', required=True)
    b.add_argument('--output', default=str(ROOT / 'data' / 'ifl_canonical'))
    b.add_argument('--max-gap-ms', type=float, default=100.)
    b.add_argument('--overwrite', action='store_true')
    f = sub.add_parser('loso')
    f.add_argument('--manifest', default=str(ROOT / 'data' / 'ifl_canonical' / 'manifest.json'))
    f.add_argument('--val', required=True)
    f.add_argument('--test', default='xuesong')
    f.add_argument('--output', required=True)
    args = p.parse_args()
    if args.command == 'loso':
        loso(args.manifest, args.val, args.test, args.output)
    elif args.command == 'build':
        build(args.source, args.reconstructions, args.output, args.max_gap_ms, args.overwrite)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
