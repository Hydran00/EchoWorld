"""Trim file, editor saves and the trimmed cache (pose association and frozen split)."""
import json
import numpy as np
import pytest
from mocap.processing.dataset import assign_splits
from mocap.processing.trims import TrimStore, apply


def make_cache(tmp_path, episodes=6, frames=30, size=32):
    entries = []
    for i in range(episodes):
        folder = tmp_path / 'cache' / 'episodes' / f'g{i}' / 'bag'
        folder.mkdir(parents=True)
        images = np.full((frames, size, size), 100, dtype=np.uint8)
        images[:, :, :8] = images[:, :, -8:] = 0
        images[:3] = 0                       # dark start
        images[-4:] = 220                    # scanner screen at the end
        np.save(folder / 'images.npy', images)
        poses = np.repeat(np.eye(4)[None], frames, axis=0)
        poses[:, 0, 3] = np.arange(frames) * .001 + i
        header = 5_000_000_000 + np.arange(frames) * 40_000_000
        np.savez(folder / 'poses.npz', timestamps_ns=header, image_header_timestamps_ns=header,
                 world_T_probe=poses, segment_ids=np.zeros(frames, dtype=int))
        (folder / 'metadata.json').write_text(json.dumps(dict(frames=frames)))
        entries.append(dict(episode_id=f'g{i}/bag', group=f'g{i}', split='train', path=f'episodes/g{i}/bag', frames=frames))
    path = tmp_path / 'cache' / 'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1, episodes=entries, pose_alignment=dict(offset_ms=-200.))))
    return path


def test_suggestion_update_reset_and_log(tmp_path):
    manifest = make_cache(tmp_path)
    store = TrimStore(tmp_path / 'trims.json', manifest)
    assert store.suggest_missing() == 6
    record = store.data['episodes']['g0/bag']
    assert (record['start_frame'], record['end_frame'], record['status']) == (3, 25, 'suggested')
    store.update('g0/bag', end_frame=20, note='lift')
    reloaded = TrimStore(tmp_path / 'trims.json')
    assert reloaded.data['episodes']['g0/bag']['end_frame'] == 20
    assert reloaded.data['episodes']['g0/bag']['status'] == 'manual'
    assert reloaded.suggest_missing(refresh=True) == 5              # manual edits are never overwritten
    with pytest.raises(ValueError):
        reloaded.update('g0/bag', start_frame=21)
    assert reloaded.update('g0/bag', reset=True)['end_frame'] == 25
    # suggest, update, refresh, reset (the rejected update is not logged)
    assert len((tmp_path / 'trims.log.jsonl').read_text().splitlines()) == 4


def test_apply_keeps_image_pose_pairs_and_source_split(tmp_path):
    manifest = make_cache(tmp_path)
    store = TrimStore(tmp_path / 'trims.json', manifest)
    store.suggest_missing()
    store.update('g1/bag', start_frame=5, end_frame=10)
    source = json.loads(manifest.read_text())
    dropped = next(e for e, s in assign_splits(source, 'random').items() if s == 'val')
    store.update(dropped, excluded=True)
    out = tmp_path / 'trimmed'
    apply(tmp_path / 'trims.json', out)
    with pytest.raises(FileExistsError):
        apply(tmp_path / 'trims.json', out)
    apply(tmp_path / 'trims.json', out, overwrite=True)
    trimmed = json.loads((out / 'manifest.json').read_text())
    assert dropped not in [e['episode_id'] for e in trimmed['episodes']]
    assert trimmed['pose_alignment'] == dict(offset_ms=-200.)
    original = assign_splits(source, 'random')
    assert assign_splits(trimmed, 'random') == {e['episode_id']: original[e['episode_id']] for e in trimmed['episodes']}
    folder = out / 'episodes' / 'g1' / 'bag'
    images, poses = np.load(folder / 'images.npy'), np.load(folder / 'poses.npz')
    src = np.load(tmp_path / 'cache' / 'episodes' / 'g1' / 'bag' / 'poses.npz')
    assert len(images) == 6 and all(len(poses[k]) == 6 for k in poses)
    np.testing.assert_array_equal(poses['world_T_probe'], src['world_T_probe'][5:11])
    np.testing.assert_array_equal(poses['image_header_timestamps_ns'], src['image_header_timestamps_ns'][5:11])
    assert json.loads((folder / 'metadata.json').read_text())['trim']['source_start_frame'] == 5
