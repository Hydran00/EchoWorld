"""python -m mocap.processing.prepare --root /path/to/ifl_recording --output data/ifl_raw_aligned_200ms --fps 100

Offline export of the raw recordings into a compact, reproducible training cache:

    MCAP episode --decode--> ultrasound frame (1080x1920 RGB) + tracker/calibration messages
                 --sample--> at most `fps` frames per second (all frames for fps above the sensor rate)
                 --crop/gray/pad--> uint8 [size, size]
                 --synchronize--> world_T_probe at the corrected acquisition timestamp, or rejection
    episodes/<group>/<bag>/ images.npy [N, size, size], poses.npz (timestamps, 4x4 poses,
                            segment ids), metadata.json (settings, statistics, calibration)
    manifest.json           all episodes with their train/val/test split
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps
from rosbags.highlevel import AnyReader
from mocap.processing.bags import PoseCollector, ultrasound_frames


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def preprocess(pixels, crop, size):
    # Screen capture -> grayscale -> crop of the ultrasound field of view (left, top, right,
    # bottom in pixels), removing the scanner interface -> letterbox to size x size. Padding
    # with black preserves the aspect ratio, so anatomical proportions are not distorted.
    if hasattr(pixels, 'to_ndarray'):
        pixels = pixels.to_ndarray(format='rgb24')
    image = Image.fromarray(pixels).convert('L')
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError(f'Crop {crop} outside image {image.size}')
    return np.asarray(ImageOps.pad(image.crop(crop), (size, size), method=Image.Resampling.BILINEAR, color=0))


def export_episode(bag, root, output, settings):
    bag, root, output = Path(bag), Path(root), Path(output)
    episode_id = bag.relative_to(root).as_posix()
    destination = output / 'episodes' / episode_id
    destination.mkdir(parents=True, exist_ok=True)
    # The cache signature combines export settings and source file sizes/mtimes. An existing
    # cache is reused only if it was produced from the same data with the same settings.
    fingerprint = [dict(name=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                   for p in sorted(bag.glob('*.mcap')) + [bag / 'metadata.yaml']]
    identity = dict(settings=settings, source_files=fingerprint, source=str(bag.resolve()))
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    meta_path = destination / 'metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('signature') != signature:
            raise ValueError(f'{episode_id}: cache settings/source changed; use a new output directory')
        if (destination / 'images.npy').exists() and (destination / 'poses.npz').exists():
            return meta
    start = time.monotonic()
    pose_time_offset_ms = float(settings.get('pose_time_offset_ms', 0.))
    if not np.isfinite(pose_time_offset_ms):
        raise ValueError('Pose time offset must be finite')
    pose_time_offset_ns = round(pose_time_offset_ms * 1e6)
    images, transforms, stamps, headers, logs, segments, offsets = [], [], [], [], [], [], []
    rejected, decoded, sampled, next_time, previous = 0, 0, 0, None, None
    candidates = []
    video_stats = {}
    with AnyReader([bag]) as reader:
        collector = PoseCollector(settings['marker'], settings['probe'], round(settings['max_pose_gap_ms'] * 1e6))
        # One pass matters: RGB-D shares MCAP chunks with the requested topics.
        for sensor, recorded, pixels in ultrasound_frames(reader, collector, video_stats):
            decoded += 1
            if previous is not None and sensor < previous:
                raise ValueError('Ultrasound sensor clock moved backwards')
            previous = sensor
            # Temporal subsampling: keep a frame only if at least 1/fps s has elapsed since the
            # last kept frame. Skipped frames are still decoded, as required by the video codec.
            if next_time is not None and sensor < next_time:
                continue
            next_time = sensor + round(1e9 / settings['fps'])
            sampled += 1
            image = preprocess(pixels, tuple(settings['crop']), settings['size'])
            candidates.append((sensor, recorded, image))
        poses, metadata = collector.finish()
    # Synchronization is performed after the pass, once all tracker samples are known, so that
    # every image can be interpolated between the samples that bracket its acquisition time.
    # A negative offset compensates delayed ultrasound: image header t -> tracker at t + offset.
    # Interpolate the original tracker stream; never shift image pixels or relabel frames per episode.
    for sensor, recorded, image in candidates:
        try:
            pose_time = sensor + pose_time_offset_ns
            pose, segment, offset = poses.at(pose_time, round(settings['sync_ms'] * 1e6))
        except ValueError:
            rejected += 1
            continue
        images.append(image)
        transforms.append(pose)
        stamps.append(pose_time)
        headers.append(sensor)
        logs.append(recorded)
        segments.append(segment)
        offsets.append(offset)
    if len(images) < 3:
        raise ValueError(f'{episode_id}: only {len(images)} valid ultrasound/pose pairs')
    with (destination / 'images.tmp').open('wb') as f:
        np.save(f, np.stack(images), allow_pickle=False)
    (destination / 'images.tmp').replace(destination / 'images.npy')
    with (destination / 'poses.tmp').open('wb') as f:
        np.savez(f, timestamps_ns=np.array(stamps, dtype=np.int64),
                 image_header_timestamps_ns=np.array(headers, dtype=np.int64),
                 recorded_ns=np.array(logs, dtype=np.int64),
                 world_T_probe=np.stack(transforms), segment_ids=np.array(segments, dtype=np.int64),
                 nearest_pose_offset_ns=np.array(offsets, dtype=np.int64))
    (destination / 'poses.tmp').replace(destination / 'poses.npz')
    meta = dict(episode_id=episode_id, group=bag.parent.name, path=destination.relative_to(output).as_posix(),
                frames=len(images), decoded_frames=decoded, sampled_frames=sampled, rejected_sync=rejected,
                duration_s=(stamps[-1] - stamps[0]) / 1e9, signature=signature,
                video_stats=video_stats,
                nearest_pose_ms_p95=float(np.percentile(offsets, 95) / 1e6),
                export_seconds=round(time.monotonic() - start, 2), **identity, **metadata)
    if pose_time_offset_ns:
        meta['pose_alignment'] = dict(offset_ms=pose_time_offset_ms,
            convention='image_header_time + offset = pose_query_time',
            interpolation='linear translation + SLERP on original tracker samples')
    atomic_json(meta_path, meta)
    return meta


def split_episodes(episodes, val_groups, test_groups):
    # Split by recording group (folder), not by frame or episode, to evaluate generalization
    # to recordings that were never seen during training.
    if set(val_groups) & set(test_groups):
        raise ValueError('Validation and test groups overlap')
    known = {e['group'] for e in episodes}
    if (set(val_groups) | set(test_groups)) - known:
        raise ValueError('Requested held-out groups are absent')
    for e in episodes:
        e['split'] = 'val' if e['group'] in val_groups else 'test' if e['group'] in test_groups else 'train'
    if not any(e['split'] == 'train' for e in episodes) or not any(e['split'] == 'val' for e in episodes):
        raise ValueError('Need nonempty training and validation groups')
    return episodes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=float, default=5.)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--crop', type=int, nargs=4, default=[520, 110, 1400, 800])
    p.add_argument('--sync-ms', type=float, default=50.)
    p.add_argument('--max-pose-gap-ms', type=float, default=100.)
    p.add_argument('--pose-time-offset-ms', type=float, default=-200.,
                   help='Image header t uses tracker pose at t + offset (default: -200 ms for the IFL setup; 0 disables correction)')
    p.add_argument('--marker', default='marker')
    p.add_argument('--probe', default='probe')
    p.add_argument('--val-groups', nargs='+', default=['yuan'])
    p.add_argument('--test-groups', nargs='+', default=['xuesong'])
    p.add_argument('--groups', nargs='+')
    p.add_argument('--limit-per-group', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--skip-errors', action='store_true', help='Explicitly permit incomplete export; errors remain in manifest')
    args = p.parse_args()
    if min(args.fps, args.size, args.sync_ms, args.max_pose_gap_ms, args.workers) <= 0:
        p.error('Rate, size, tolerances and workers must be positive')
    if not np.isfinite(args.pose_time_offset_ms):
        p.error('Pose time offset must be finite')
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bags = []
    for group in sorted(root.iterdir()):
        if not group.is_dir() or (args.groups and group.name not in args.groups):
            continue
        candidates = sorted(x.parent for x in group.glob('*/metadata.yaml'))
        bags.extend(candidates[:args.limit_per_group] if args.limit_per_group else candidates)
    if not bags:
        p.error('No <group>/<episode>/metadata.yaml recordings found')
    settings = {k: getattr(args, k) for k in ['fps', 'size', 'crop', 'sync_ms', 'max_pose_gap_ms', 'marker', 'probe']}
    settings['version'] = 1
    # Preserve the signature of existing zero-offset caches. Nonzero offsets require a new cache.
    if args.pose_time_offset_ms:
        settings['pose_time_offset_ms'] = args.pose_time_offset_ms
    episodes, errors = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(export_episode, b, root, output, settings): b for b in bags}
        for future in as_completed(futures):
            bag = futures[future]
            try:
                meta = future.result()
                episodes.append(meta)
                print(f'[{len(episodes) + len(errors)}/{len(bags)}] {meta["episode_id"]}: {meta["frames"]} frames, {meta["rejected_sync"]} sync rejects', flush=True)
            except Exception as exc:
                errors.append(dict(episode=str(bag.relative_to(root)), error=str(exc)))
                print(f'ERROR {bag.name}: {exc}', flush=True)
            atomic_json(output / 'export_progress.json', dict(completed=episodes, errors=errors))
    atomic_json(output / 'export_errors.json', errors)
    if errors and not args.skip_errors:
        raise RuntimeError(f'{len(errors)} episodes failed; see export_errors.json. Successful caches can be reused.')
    episodes = split_episodes(sorted(episodes, key=lambda e: e['episode_id']), args.val_groups, args.test_groups)
    manifest = dict(schema_version=1, source_root=str(root), settings=settings, episodes=episodes, errors=errors,
                    val_groups=args.val_groups, test_groups=args.test_groups)
    if args.pose_time_offset_ms:
        manifest['pose_alignment'] = dict(offset_ms=args.pose_time_offset_ms,
            convention='image_header_time + offset = pose_query_time',
            interpolation='linear translation + SLERP on original tracker samples')
    atomic_json(output / 'manifest.json', manifest)
    print(json.dumps({s: sum(e['frames'] for e in episodes if e['split'] == s) for s in ['train', 'val', 'test']}), flush=True)


if __name__ == '__main__':
    main()
