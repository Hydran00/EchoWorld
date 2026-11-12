import json
import sys

import numpy as np
import pytest
import torch

from mocap.evaluation import compare_motion as evaluation


def test_wrong_motion_preserves_separate_norms_and_reverses_directions():
    action = np.array([[3, 4, 0, .1, -.2, .3], [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, .2, 0, 0]], dtype=np.float32)
    wrong = evaluation.wrong_motion(action)
    for block in (slice(0, 3), slice(3, 6)):
        np.testing.assert_array_equal(np.linalg.norm(wrong[:, block], axis=1),
                                      np.linalg.norm(action[:, block], axis=1))
        np.testing.assert_allclose((wrong[:, block] * action[:, block]).sum(1),
                                   -(action[:, block] ** 2).sum(1))
    np.testing.assert_array_equal(action[0], [3, 4, 0, np.float32(.1), np.float32(-.2), np.float32(.3)])
    np.testing.assert_array_equal(wrong[1], np.zeros(6))
    with pytest.raises(ValueError):
        evaluation.wrong_motion(np.array([[np.nan] * 6]))
    with pytest.raises(ValueError):
        evaluation.wrong_motion(np.zeros((2, 5)))


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, source, action, shared_noise=False):
        assert shared_noise
        self.calls.append((source.clone(), action.clone()))
        return (source + action[:, :1, None, None] / 100).clamp(0, 1)


def test_all_conditions_use_original_source_not_previous_prediction():
    model = FakeModel()
    source = torch.full((2, 1, 16, 16), .5)
    action = torch.tensor([[10., 0, 0, .1, 0, 0], [0, 0, 0, 0, 0, 0]])
    result = evaluation.predict_conditions(model, source, action)
    assert len(model.calls) == 3
    for image, _ in model.calls:
        torch.testing.assert_close(image, source)
    torch.testing.assert_close(model.calls[0][1], action)
    torch.testing.assert_close(model.calls[1][1], torch.zeros_like(action))
    torch.testing.assert_close(model.calls[2][1], -action)
    assert result['correct'][0].mean().item() == pytest.approx(.6)
    assert result['zero'][0].mean().item() == pytest.approx(.5)
    assert result['wrong'][0].mean().item() == pytest.approx(.4)
    torch.testing.assert_close(result['correct'][1], result['wrong'][1])


def row(episode, scale=1., active=True):
    r = dict(episode=episode, action_nonzero=active, a_outside_training_range=False,
             b_outside_training_range=True)
    errors = {'a': {'correct': 1, 'zero': 3, 'wrong': 4},
              'b': {'correct': 2, 'zero': 3, 'wrong': 6}}
    for model, conditions in errors.items():
        for condition, error in conditions.items():
            for metric in evaluation.METRICS:
                sign = -1 if metric in ('psnr', 'ssim') else 1
                r[f'{model}_{condition}_{metric}'] = sign * scale * error
    return r


def test_paired_effects_have_consistent_sign_and_equal_episode_weights():
    rows = [row('short', 1)] + [row('long', 3)] * 9 + [row('static', 100, active=False)]
    s = evaluation.summary(rows, bootstrap=1000)
    assert s['action_test_pairs'] == 10 and s['action_test_episodes'] == 2
    for metric in evaluation.METRICS:
        assert s['correct_advantage']['zero']['a'][metric]['mean'] == pytest.approx(4.)
        assert s['correct_advantage']['wrong']['a'][metric]['mean'] == pytest.approx(6.)
        assert s['b_minus_a_advantage']['zero'][metric]['mean'] == pytest.approx(-2.)
        assert s['b_minus_a_advantage']['wrong'][metric]['mean'] == pytest.approx(2.)
        assert s['correct_advantage']['wrong']['a'][metric]['ci95'] == pytest.approx([3., 9.])
    assert evaluation.summary([row('one')])['correct_advantage']['wrong']['a']['mae']['ci95'] is None
    assert evaluation.summary([row('static', active=False)])['correct_advantage'] == {}


def test_cli_saves_actions_all_conditions_horizons_and_range_flags(tmp_path, monkeypatch):
    count = 41
    np.save(tmp_path / 'images.npy', np.full((count, 16, 16), 100, dtype=np.uint8))
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, 0, 3] = np.arange(count) * .001
    np.savez(tmp_path / 'poses.npz', timestamps_ns=np.arange(count) * 100_000_000,
             segment_ids=np.zeros(count, dtype=np.int64), world_T_probe=poses)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, episodes=[
        dict(episode_id='subject/bag', group='subject', split='test', path='.')])))
    checkpoint = tmp_path / 'fake.pt'
    checkpoint.write_bytes(b'fake checkpoint')
    state = dict(kind='diffusion', step=1, config=dict(model=dict(image_size=16),
                                                     data=dict(min_dt=.19, max_dt=.21)))
    monkeypatch.setattr(evaluation, 'load_model', lambda path, device: (FakeModel(), state))
    output = tmp_path / 'results'
    monkeypatch.setattr(sys, 'argv', ['compare_motion', '--checkpoint-a', str(checkpoint),
                                     '--checkpoint-b', str(checkpoint), '--manifest', str(manifest),
                                     '--output', str(output), '--pairs-per-episode', '3',
                                     '--batch-size', '2', '--device', 'cpu', '--bootstrap', '10'])
    evaluation.main()
    metrics = json.loads((output / 'metrics.json').read_text())
    samples = json.loads((output / 'samples.json').read_text())
    assert metrics['overall']['pairs'] == 3
    assert metrics['mode'] == 'both_models_single_step_action_ablation'
    assert len(metrics['overall']['means']) == 30
    assert sum(s['pairs'] for s in metrics['by_horizon'].values()) == 3
    for s in samples:
        np.testing.assert_array_equal(s['wrong_action'], -np.array(s['correct_action']))
        assert s['b_outside_training_range'] == (not .19 <= s['dt_seconds'] <= .21)
    assert (output / 'report.md').exists() and (output / 'per_pair_metrics.csv').exists()
    assert len(list(output.glob('comparison_*.png'))) == 2
    with pytest.raises(FileExistsError):
        evaluation.main()
