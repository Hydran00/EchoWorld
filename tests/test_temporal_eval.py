from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mocap.evaluation.temporal_eval import select_clips, temporal_metrics


def moving_gaussian():
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
    return torch.stack([torch.exp(-((x - 12 - i)**2 + (y - 16)**2) / 30.) for i in range(5)])[:, None]


def test_temporal_metrics_reward_correct_motion_not_stillness():
    target = moving_gaussian()
    perfect = temporal_metrics(target, target)
    copy = temporal_metrics(target[:1].expand_as(target), target)
    reverse = temporal_metrics(target.flip(0), target)
    for kernel in (1, 7, 15):
        prefix = f'temporal_k{kernel}'
        assert perfect[f'{prefix}_mae'] == 0
        assert perfect[f'{prefix}_gain_over_static'] == 1
        assert perfect[f'{prefix}_cosine'] == pytest.approx(1., abs=1e-6)
        assert copy[f'{prefix}_gain_over_static'] == 0
        assert copy[f'{prefix}_cosine'] == 0
        assert reverse[f'{prefix}_gain_over_static'] < 0


def test_no_motion_has_no_undefined_or_fabricated_gain():
    target = moving_gaussian()[:1].expand(5, -1, -1, -1)
    metrics = temporal_metrics(target, target)
    assert metrics['temporal_k7_gain_over_static'] is None
    assert metrics['temporal_k7_mae'] == 0
    with pytest.raises(ValueError):
        temporal_metrics(target[:1], target[:1])


def test_clips_never_cross_gaps_and_do_not_depend_on_images():
    episode = dict(times=np.arange(12, dtype=np.int64) * 100_000_000,
                   segments=np.array([0] * 6 + [1] * 6))
    dataset = SimpleNamespace(episodes=[episode], items=[(0, i, np.array([i + 1])) for i in range(11)])
    clips = select_clips(dataset, count=10, horizon=.3, step=.1)
    assert clips
    for _, source, targets in clips:
        assert len(targets) == 3
        assert all(episode['segments'][j] == episode['segments'][source] for j in targets)
        assert all(b > a for a, b in zip([source] + targets[:-1], targets))
