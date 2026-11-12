"""Protocol checks: paired sampling, exact temporal grid, and no teacher forcing."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mocap.evaluation import compare_rollout as comparison
from mocap.processing.geometry import relative_action

torch.set_num_threads(2)


def episode(times=None):
    if times is None:
        times = np.arange(41, dtype=np.int64) * 100_000_000
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    seconds = (times - times[0]) / 1e9
    poses[:, 0, 3] = seconds ** 2 * .003
    poses[:, :3, :3] = Rotation.from_euler('z', seconds[:, None] * .1).as_matrix()
    return dict(episode_id='test/bag', group='test', times=times, poses=poses,
                segments=np.zeros(len(times), dtype=np.int64))


def dataset_for(ep):
    items = [(0, i, np.arange(i + 1, len(ep['times']))) for i in range(len(ep['times']) - 1)]
    return SimpleNamespace(episodes=[ep], items=items)


def test_random_pairs_reproducible_on_grid_strictly_below_three_seconds():
    ep = episode()
    data = dataset_for(ep)
    a, coverage = comparison.sample_pairs(data, 25, seed=42)
    b, _ = comparison.sample_pairs(data, 25, seed=42)
    c, _ = comparison.sample_pairs(data, 25, seed=99)
    assert a == b and a != c
    assert len({s['source_index'] for s in a}) == len(a) == 25
    assert len({s['nominal_dt_seconds'] for s in a}) > 3
    assert coverage[0]['sampled'] == 25
    for sample in a:
        assert 0 < sample['dt_seconds'] < 3
        assert sample['nominal_dt_seconds'] <= 2.8
        assert abs(sample['target_offset_seconds']) <= .01
        assert sample['target_index'] > sample['source_index']


def test_no_sampling_or_interpolation_across_recording_or_tracking_gaps():
    ep = episode()
    ep['times'][21:] += 200_000_000  # unlabelled recording gap
    ep['segments'][11:21] = 1  # repeated ID after the barrier must not reconnect it
    samples, _ = comparison.sample_pairs(dataset_for(ep), 100)
    for sample in samples:
        a, b = sample['source_index'], sample['target_index']
        assert np.all(ep['segments'][a:b + 1] == ep['segments'][a])
        assert np.max(np.diff(ep['times'][a:b + 1])) <= 100_000_000
    with pytest.raises(ValueError, match='gap'):
        comparison.rollout_path(ep, 9, 13)
    with pytest.raises(ValueError, match='gap'):
        comparison.rollout_path(ep, 19, 23)


@pytest.mark.parametrize('jitter', [-9_000_000, 0, 9_000_000])
def test_target_jitter_does_not_add_a_short_residual_step(jitter):
    ep = episode(np.array([0, 100_000_000, 200_000_000, 300_000_000,
                           400_000_000, 500_000_000, 600_000_000 + jitter]))
    offsets, poses = comparison.rollout_path(ep, 0, 6, max_gap=.11)
    np.testing.assert_array_equal(offsets, [0, 200_000_000, 400_000_000, 600_000_000 + jitter])
    intervals = np.diff(offsets) / 1e9
    assert len(intervals) == 3
    assert np.all((intervals >= .19) & (intervals <= .21))
    np.testing.assert_allclose(poses[-1], ep['poses'][6])
    with pytest.raises(ValueError, match='multiple'):
        comparison.rollout_path(ep, 0, 5, max_gap=.11)


def test_interpolation_follows_measured_curved_trajectory_and_composes():
    # Large epoch timestamps and an intermediate step between cached timestamps.
    ep = episode(1_790_000_000_000_000_000 + np.array([0, 90, 180, 270, 360, 405]) * 1_000_000)
    offsets, poses = comparison.rollout_path(ep, 0, 5)
    assert offsets.tolist() == [0, 200_000_000, 405_000_000]
    expected_translation = ep['poses'][2, 0, 3] * (7 / 9) + ep['poses'][3, 0, 3] * (2 / 9)
    assert poses[1, 0, 3] == pytest.approx(expected_translation)
    assert Rotation.from_matrix(poses[1, :3, :3]).as_rotvec()[2] == pytest.approx(.02)
    transforms = [np.linalg.inv(a) @ b for a, b in zip(poses[:-1], poses[1:])]
    np.testing.assert_allclose(transforms[0] @ transforms[1], np.linalg.inv(poses[0]) @ poses[-1], atol=1e-12)
    local_actions = relative_action(poses[:-1], poses[1:])
    assert not np.allclose(local_actions[0], relative_action(poses[0], poses[-1]) / 2)


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, source, action, shared_noise=False):
        assert shared_noise
        self.calls.append((source.clone(), action.clone()))
        return (source * .5 + .1 + action[:, :1, None, None] / 100).clamp(0, 1)


def test_autoregression_uses_predictions_and_local_motion_with_ragged_batches():
    source = torch.full((2, 1, 16, 16), .4)
    paths = [np.array([[1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]], np.float32),
             np.array([[4, 0, 0, 0, 0, 0]], np.float32)]
    a, b = FakePredictor(), FakePredictor()
    outputs = comparison.predict_batch(a, b, source, torch.tensor([[3., 0, 0, 0, 0, 0],
                                                                  [4., 0, 0, 0, 0, 0]]), paths)
    assert len(a.calls) == 2  # direct plus zero-action baseline
    assert len(b.calls) == 2  # two steps, shrinking the batch after the first
    assert b.calls[0][0].shape[0] == 4 and b.calls[1][0].shape[0] == 2
    expected_feedback = round(.31 * 255) / 255
    assert b.calls[1][0][0].mean().item() == pytest.approx(expected_feedback)
    assert b.calls[1][1][0, 0] == 2  # local movement, not accumulated movement
    assert torch.count_nonzero(b.calls[1][1][1]) == 0
    assert outputs['b_rollout'][0].mean().item() == pytest.approx(expected_feedback * .5 + .12)
    assert outputs['b_rollout'][1].mean().item() == pytest.approx(.34)
    assert outputs['a_direct'][0].mean().item() == pytest.approx(.33)
    # Batch membership must not change the result for the same sample.
    single = comparison.predict_batch(FakePredictor(), FakePredictor(), source[:1],
                                      torch.tensor([[3., 0, 0, 0, 0, 0]]), paths[:1])
    for method in comparison.METHODS:
        torch.testing.assert_close(single[method][0], outputs[method][0])


def metric_row(name, a, b):
    row = dict(episode=name)
    for method in comparison.METHODS:
        value = a if method == 'a_direct' else b if method == 'b_rollout' else 1.
        row.update({f'{method}_{metric}': value for metric in comparison.METRICS})
    return row


def test_summary_weights_episodes_equally_and_bootstraps_paired_differences():
    rows = [metric_row('short', .2, .1)] + [metric_row('long', .4, .1)] * 9
    summary = comparison.summarize(rows, bootstrap=1000)
    assert summary['means']['a_direct_mae'] == pytest.approx(.3)  # not .38
    assert summary['paired']['mae']['b_minus_a'] == pytest.approx(-.2)
    assert summary['paired']['mae']['ci95'] == pytest.approx([-.3, -.1])
    assert summary['paired']['mae']['b_pair_win_fraction'] == 1
    assert comparison.summarize(rows[:1])['paired']['mae']['ci95'] is None


def test_cli_writes_reproducible_paired_test_artifacts_and_preserves_existing_output(tmp_path, monkeypatch):
    ep = episode()
    folder = tmp_path / 'test'
    folder.mkdir()
    np.save(folder / 'images.npy', np.full((len(ep['times']), 16, 16), 100, np.uint8))
    np.savez(folder / 'poses.npz', timestamps_ns=ep['times'], world_T_probe=ep['poses'], segment_ids=ep['segments'])
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, episodes=[
        dict(episode_id='test/bag', group='test', split='test', path='test')])))
    checkpoint = tmp_path / 'fake.pt'
    checkpoint.write_bytes(b'test fake; model loader is patched')
    state = dict(kind='diffusion', step=123, config=dict(model=dict(image_size=16),
                                                       data=dict(min_dt=.19, max_dt=.21)))
    monkeypatch.setattr(comparison, 'load_model', lambda path, device: (FakePredictor(), state))
    output = tmp_path / 'comparison'
    monkeypatch.setattr(sys, 'argv', ['compare_rollout', '--checkpoint-a', str(checkpoint),
                                     '--checkpoint-b', str(checkpoint), '--manifest', str(manifest),
                                     '--output', str(output), '--device', 'cpu', '--pairs-per-episode', '3',
                                     '--batch-size', '2', '--bootstrap', '10'])
    comparison.main()
    result = json.loads((output / 'metrics.json').read_text())
    samples = json.loads((output / 'samples.json').read_text())
    assert result['split'] == 'test' and result['overall']['pairs'] == 3
    assert all(s['b_steps_outside_training_range'] == 0 for s in samples)
    assert (output / 'per_pair_metrics.csv').exists()
    assert (output / 'report.md').exists()
    assert len(list(output.glob('comparison_*.png'))) == 1
    with pytest.raises(FileExistsError):
        comparison.main()
