"""Guidance fine-tuning: labels, causal histories, ensemble, weight transfer and CLI resume."""
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from mocap.processing.geometry import matrix, relative_action
from mocap.finetuning.guidance import (GuidanceModel, GuidanceSequences, ensemble, exponential_history, param_groups,
                            to_label, to_matrix)
from mocap.pretraining.model import MotionWorldModel
from mocap.pretraining.train import save_checkpoint

torch.set_num_threads(2)
SMALL_WORLD = dict(image_size=32, embed_dim=24, encoder_depth=2, predictor_depth=1, num_heads=3,
                   projector_dim=16, mask=dict(min_keep=1))
SEQ = dict(num_frames=4, exp_alpha=.4, seq_prob=.5, color_jitter=.2, goal_offset_s=0., min_episode_s=.5, eval_stride=3)


def make_cache(tmp_path, frames=20, size=32):
    entries = []
    for i, (group, split) in enumerate([('a', 'train'), ('b', 'val'), ('c', 'test')]):
        folder = tmp_path / group
        folder.mkdir()
        images = np.full((frames, size, size), 120, dtype=np.uint8)
        images[:, :, :8] = images[:, :, -8:] = 0             # black outside the fan, as in real frames
        images[-3] = 200                                     # scanner wallpaper: bright, but not ultrasound
        images[-2:] = 0                                      # probe lifted: not a valid goal
        np.save(folder / 'images.npy', images)
        g = np.random.default_rng(i)
        poses = np.stack([matrix(g.normal(0, .05, 3), Rotation.from_rotvec(g.normal(0, .3, 3)).as_quat())
                          for _ in range(frames)])
        np.savez(folder / 'poses.npz', timestamps_ns=1_000_000_000 + np.arange(frames) * 100_000_000,
                 world_T_probe=poses, segment_ids=np.zeros(frames, dtype=int))
        entries.append(dict(episode_id=group + '/bag', group=group, split=split, path=group))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1, episodes=entries)))
    return path


def test_exponential_history_matches_released_sampler():
    def released(t, K, alpha):            # finetune/echo_datasets_new.py, verbatim logic
        lambd = alpha * K / (t + 1)
        q = (np.arange(1, K + 1) - .5) / K
        return np.round(np.maximum(t - (-(1 / lambd) * np.log(1 - q)), 0)).astype(int)[::-1]
    for t in [0, 3, 50, 400]:
        np.testing.assert_array_equal(exponential_history(t, 7, .4), released(t, 7, .4))
        assert (np.diff(exponential_history(t, 7, .4)) >= 0).all() and exponential_history(t, 7, .4).max() <= t


def test_labels_are_goal_in_probe_frame_and_histories_are_causal(tmp_path):
    path = make_cache(tmp_path)
    data = GuidanceSequences(path, split='train', image_size=32, split_mode='leave-out', **SEQ)
    episode = data.episodes[0]
    assert episode['goal'] == 16                              # last ultrasound frame: wallpaper and dark tail excluded
    poses = episode['poses']
    for seed in range(20):
        item = data[(0, seed)]
        assert item['images'].shape == (4, 1, 32, 32) and item['rel'].shape == (4, 4, 6)
        np.testing.assert_allclose(item['labels'][-1], to_label(relative_action(poses[item['frame']], poses[16])), atol=1e-4)
        # Composition: (last -> node n) then (node n -> goal) equals (last -> goal), for every node.
        composed = to_matrix(to_label(item['rel'][-1].numpy())) @ to_matrix(item['labels'].numpy())
        np.testing.assert_allclose(composed, np.broadcast_to(to_matrix(item['labels'][-1].numpy()), composed.shape), atol=1e-3)
        np.testing.assert_allclose(item['rel'].diagonal(dim1=0, dim2=1).numpy(), 0, atol=1e-4)
        assert item['time_to_goal_s'] >= 0
    val = GuidanceSequences(path, split='val', image_size=32, split_mode='leave-out', **SEQ)
    assert len(val) == len(range(0, 17, 3))
    a, b = val[2], val[2]                                     # deterministic evaluation history, no jitter
    torch.testing.assert_close(a['images'], b['images'])
    assert a['frame'] == 6 and a['labels'].shape == (4, 6)


def test_ensemble_of_exact_predictions_is_exact(tmp_path):
    data = GuidanceSequences(make_cache(tmp_path), split='train', image_size=32, split_mode='leave-out', **SEQ)
    items = [data[(0, s)] for s in range(5)]
    labels = np.stack([i['labels'].numpy() for i in items])
    rel = np.stack([i['rel'].numpy() for i in items])
    np.testing.assert_allclose(ensemble(labels, rel), labels[:, -1], atol=1e-3)


def test_weight_transfer_layer_decay_and_motion_conditioning():
    world = MotionWorldModel(**SMALL_WORLD)
    model = GuidanceModel(SMALL_WORLD, drop_path=0.)
    model.load_world(world.state_dict())
    for k, v in world.target_encoder.state_dict().items():
        torch.testing.assert_close(model.feature_model.state_dict()[k], v)
    for k, v in world.predictor.policy_net.state_dict().items():
        torch.testing.assert_close(model.action_encoder.state_dict()[k], v)
    images, rel = torch.rand(2, 3, 1, 32, 32), torch.randn(2, 3, 3, 6)
    out = model.eval()(images, rel)
    assert out.shape == (2, 3, 6)
    assert not torch.allclose(out, model(images, rel, zero_motion=True))
    assert GuidanceModel(SMALL_WORLD, fusion='none').eval()(images, rel).shape == (2, 3, 6)
    scales = {round(g['lr_scale'], 6) for g in param_groups(model, .05, .5)}
    assert scales == {1., .5, .25, .125}                     # head/top, block 1, block 0, patch embedding


def test_cli_train_resume_evaluate_and_goals(tmp_path):
    path = make_cache(tmp_path)
    world = MotionWorldModel(**SMALL_WORLD)
    world_path = tmp_path / 'world.pt'
    save_checkpoint(world_path, world, torch.optim.AdamW(world.parameters()),
                    dict(model=SMALL_WORLD, data=dict(manifest=str(path), min_mean=8., split_mode='leave-out')),
                    1, 0., 'test', torch.device('cpu'))
    config = dict(seed=0, world_checkpoint=str(world_path), backbone='target_encoder', data={}, sequence=SEQ,
                  guidance=dict(fusion='geomlp', drop_path=.1, num_planes=1, num_heads=4), batch_size=2, workers=0,
                  cpu_threads=2, amp=False, lr=1e-4, min_lr=1e-6, warmup_steps=1, steps=3, weight_decay=.05,
                  layer_decay=.65, log_every=1, eval_every=2, save_every=1, eval_batch_size=4)
    config_path = tmp_path / 'guidance.json'
    config_path.write_text(json.dumps(config))
    run = tmp_path / 'run'
    base = [sys.executable, '-m', 'mocap.finetuning.guidance']
    for extra in [['--max-steps', '1'], ['--resume', str(run / 'last.pt')]]:
        result = subprocess.run(base + ['train', '--config', str(config_path), '--output', str(run), '--device', 'cpu'] + extra,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((run / 'status.json').read_text())['step'] == 3
    metrics = json.loads((run / 'val_000002' / 'metrics.json').read_text())
    assert {'ensemble_mae', 'policy_trans_err_mm', 'baseline_stay_mae', 'baseline_train_mean_mae'} <= set(metrics)
    result = subprocess.run(base + ['evaluate', '--checkpoint', str(run / 'best.pt'), '--split', 'test', '--device', 'cpu',
                                    '--workers', '0'], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / 'eval_best_test' / 'metrics.md').exists()
    result = subprocess.run(base + ['goals', '--config', str(config_path), '--output', str(tmp_path / 'goals')],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'goals' / 'goals_val.png').exists()
