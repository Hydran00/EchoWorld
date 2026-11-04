"""Focused correctness checks for the new data/geometry/model path."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from mocap.bags import decode_image, static_chain
from mocap.dataset import MocapPairs
from mocap.geometry import matrix, relative_action, PoseSeries
from mocap.model import MotionWorldModel
from mocap.prepare import split_episodes, preprocess
from mocap.train import StepBatches, save_checkpoint
from mocap.evaluate import load_model, image_metrics
from mocap.generate import validate_trajectory

torch.set_num_threads(2)


def test_relative_motion_is_invariant_to_tracker_frame():
    a = matrix([1, 2, 3], Rotation.from_euler('z', 90, degrees=True).as_quat())
    local = matrix([.01, 0, 0], Rotation.from_euler('x', .2).as_quat())
    b = a @ local
    global_change = matrix([5, -2, 1], Rotation.from_euler('xyz', [.1, .4, .5]).as_quat())
    np.testing.assert_allclose(relative_action(a, b), [10, 0, 0, .2, 0, 0], atol=1e-5)
    np.testing.assert_allclose(relative_action(global_change @ a, global_change @ b), relative_action(a, b), atol=1e-5)


def test_pose_interpolation_and_tracking_barriers():
    a = np.eye(4)
    b = matrix([.02, 0, 0], Rotation.from_euler('z', .2).as_quat())
    series = PoseSeries([(100, a), (200, b)], max_gap_ns=110)
    pose, _, offset = series.at(150, tolerance_ns=60)
    np.testing.assert_allclose(relative_action(a, pose), [10, 0, 0, 0, 0, .1], atol=1e-6)
    assert offset == 50
    for time in [99, 201]:
        with pytest.raises(ValueError):
            series.at(time)
    with pytest.raises(ValueError):
        series.at(150, tolerance_ns=40)
    broken = PoseSeries([(100, a), (150, None), (200, b)])
    with pytest.raises(ValueError):
        broken.at(175)
    assert broken.at(100)[1] != broken.at(200)[1]
    with pytest.raises(ValueError):
        PoseSeries([(100, a), (300, b)], max_gap_ns=100).at(200)


def test_calibration_chain_and_inverse():
    ab = matrix([1, 0, 0], [0, 0, 0, 1])
    bc = matrix([0, 2, 0], [0, 0, 0, 1])
    edges = {('a', 'b'): ab, ('b', 'c'): bc}
    np.testing.assert_allclose(static_chain(edges, 'a', 'c'), ab @ bc)
    np.testing.assert_allclose(static_chain(edges, 'c', 'a'), np.linalg.inv(ab @ bc))
    with pytest.raises(ValueError):
        static_chain(edges, 'a', 'missing')


def test_raw_decoding_honors_stride_endian_and_channels():
    msg = SimpleNamespace(encoding='bgr8', width=1, height=2, step=4, is_bigendian=False,
                          data=bytes([1, 2, 3, 0, 4, 5, 6, 0]))
    np.testing.assert_array_equal(decode_image(msg), [[[3, 2, 1]], [[6, 5, 4]]])
    msg = SimpleNamespace(encoding='mono16', width=2, height=1, step=4, is_bigendian=True,
                          data=bytes([0, 0, 255, 255]))
    np.testing.assert_array_equal(decode_image(msg), [[0, 255]])
    msg.step = 1
    with pytest.raises(ValueError):
        decode_image(msg)


def test_crop_bounds():
    assert preprocess(np.ones((20, 30, 3), dtype=np.uint8), (2, 2, 25, 18), 16).shape == (16, 16)
    with pytest.raises(ValueError):
        preprocess(np.zeros((20, 30), dtype=np.uint8), (0, 0, 40, 20), 16)


def test_group_splits_reject_leakage():
    episodes = [dict(group='a'), dict(group='a'), dict(group='b'), dict(group='c')]
    split_episodes(episodes, ['b'], ['c'])
    assert [e['split'] for e in episodes] == ['train', 'train', 'val', 'test']
    with pytest.raises(ValueError):
        split_episodes(episodes, ['b'], ['b'])


def make_cache(tmp_path, size=32):
    entries = []
    for group, split in [('a', 'train'), ('b', 'val')]:
        folder = tmp_path / group
        folder.mkdir()
        np.save(folder / 'images.npy', np.full((12, size, size), 120, dtype=np.uint8))
        poses = np.repeat(np.eye(4)[None], 12, axis=0)
        poses[:, 0, 3] = np.arange(12) * .001
        np.savez(folder / 'poses.npz', timestamps_ns=1_000_000_000 + np.arange(12) * 200_000_000,
                 world_T_probe=poses, segment_ids=np.repeat([0, 1], 6))
        entries.append(dict(episode_id=group + '/bag', group=group, split=split, path=group))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1, episodes=entries)))
    return path


def test_pairs_never_cross_episode_or_tracking_gap(tmp_path):
    path = make_cache(tmp_path)
    data = MocapPairs(path, image_size=32, min_dt=.2, max_dt=2)
    for ei, source, targets in data.items:
        assert all(data.episodes[ei]['segments'][source] == data.episodes[ei]['segments'][targets])
    sample = data[(0, 123)]
    assert sample['source'].shape == (1, 32, 32)
    torch.testing.assert_close(sample['action'], data[(0, 123)]['action'])
    assert data.episodes[0]['group'] == 'a'
    assert MocapPairs(path, split='val', image_size=32).episodes[0]['group'] == 'b'
    metadata = json.loads(path.read_text())
    metadata['episodes'][1]['group'] = 'a'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='leakage'):
        MocapPairs(path, image_size=32)


def test_pair_crop_is_shared_reproducible_and_train_only(tmp_path):
    path = make_cache(tmp_path, size=64)
    for folder in ['a', 'b']:
        np.save(tmp_path / folder / 'images.npy', np.arange(12 * 64 * 64, dtype=np.uint32).reshape(12, 64, 64).astype(np.uint8))
    plain = MocapPairs(path, image_size=64, min_dt=.2, max_dt=.8, min_mean=0)
    cropped = MocapPairs(path, image_size=64, min_dt=.2, max_dt=.8, min_mean=0, crop_scale=(.3, .5))
    a, b = cropped[(0, 7)], cropped[(0, 7)]
    torch.testing.assert_close(a['source'], b['source'])
    reference = plain[(0, 7)]
    assert a['source_index'] == reference['source_index'] and a['target_index'] == reference['target_index']
    assert not torch.allclose(a['source'], reference['source'])
    box = cropped.crop_box(torch.Generator().manual_seed(3))
    torch.testing.assert_close(cropped.crop(reference['target'], box),
                               cropped.crop(reference['target'], cropped.crop_box(torch.Generator().manual_seed(3))))
    val = MocapPairs(path, split='val', image_size=64, min_dt=.2, max_dt=.8, min_mean=0, crop_scale=(.3, .5))
    plain_val = MocapPairs(path, split='val', image_size=64, min_dt=.2, max_dt=.8, min_mean=0)
    torch.testing.assert_close(val[0]['source'], plain_val[0]['source'])
    with pytest.raises(ValueError):
        MocapPairs(path, image_size=64, crop_scale=(.5, 1.5))


def test_weight_decay_schedule_and_selection():
    from mocap.train import wd_at, selection_score
    config = dict(steps=100, weight_decay=.04, final_weight_decay=.4)
    assert wd_at(0, config) == pytest.approx(.04) and wd_at(100, config) == pytest.approx(.4)
    assert wd_at(50, config) == pytest.approx(.22)
    assert wd_at(70, dict(steps=100)) == pytest.approx(.05)
    metrics = dict(val_loss=.5, **{'model_recall@5': .3})
    assert selection_score(metrics, {}) == -.5
    assert selection_score(metrics, dict(selection='model_recall@5')) == .3


def test_resumed_batch_sampling_is_identical():
    full = list(StepBatches(30, 4, 0, 8, 42))
    assert full[5:] == list(StepBatches(30, 4, 5, 8, 42))


SMALL_WORLD = dict(image_size=128, embed_dim=24, encoder_depth=1, predictor_depth=1, num_heads=3,
                   projector_dim=16, mask=dict(min_keep=4))


def test_info_nce_pairs_prediction_with_its_own_target():
    from mocap.model import info_nce
    a = torch.eye(4)
    assert info_nce(torch.cat([a, a]), temperature=.01) < 1e-3
    assert info_nce(torch.cat([a, a.roll(1, 0)]), temperature=.01) > 10


def test_model_motion_gradients_and_checkpoint(tmp_path):
    from mocap.evaluate import evaluate_world
    config = dict(model=SMALL_WORLD)
    model = MotionWorldModel(**config['model'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    source, target = torch.rand(3, 1, 128, 128), torch.rand(3, 1, 128, 128)
    actions = torch.randn(3, 6)
    loss, values = model(source, target, actions, -actions)
    loss.backward()
    assert torch.isfinite(loss) and set(values) == {'loss', 'spatial', 'motion'}
    assert model.predictor.policy_net[0].weight.grad.norm() > 0
    assert model.projector[0].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert all(p.grad is None for p in model.target_projector.parameters())
    optimizer.step()
    model.update_target()
    model.eval()
    with torch.no_grad():
        expected = model.predict_latent(source, actions)
        assert expected.shape == (3, 24)
        assert not torch.allclose(expected, model.predict_latent(source, torch.zeros_like(actions)))
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, model, optimizer, config, 1, .2, 'test', torch.device('cpu'))
    loaded, state = load_model(path, torch.device('cpu'))
    assert state['step'] == 1 and state['best_score'] == .2 and state['stale_validations'] == 0
    with torch.no_grad():
        torch.testing.assert_close(expected, loaded.predict_latent(source, actions))
    dataset = MocapPairs(make_cache(tmp_path, size=128), split='val', image_size=128)
    first = evaluate_world(loaded, dataset, torch.device('cpu'), batch_size=4)
    assert first == evaluate_world(loaded, dataset, torch.device('cpu'), batch_size=4)
    assert 0 <= first['model_recall@1'] <= 1 and np.isfinite(first['val_loss'])
    with pytest.raises(ValueError):
        validate_trajectory(np.zeros((3, 4, 4)))


def test_evaluation_metrics_identity():
    x = torch.rand(2, 1, 32, 32)
    metrics = image_metrics(x, x)
    assert metrics['mae'].max() == 0
    torch.testing.assert_close(metrics['ssim'], torch.ones(2))


def test_evaluation_exports_traceable_pair_metrics(tmp_path):
    import csv
    from mocap.evaluate import evaluate

    class ActionOffset(torch.nn.Module):
        def predict(self, source, action, shared_noise=False):
            return source + action[:, 0, None, None, None] / 20.

    dataset = MocapPairs(make_cache(tmp_path), split='val', image_size=32)
    output = tmp_path / 'evaluation'
    result = evaluate(ActionOffset(), dataset, torch.device('cpu'), output, batch_size=3)
    with (output / 'per_pair_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result['pairs'] == len(dataset)
    assert all(row['episode'] == 'b/bag' for row in rows)
    for i, row in enumerate(rows):
        sample = dataset[i]
        assert int(row['source_index']) == sample['source_index']
        assert int(row['target_index']) == sample['target_index']
        assert float(row['model_mae']) == pytest.approx(float(sample['action'][0]) / 20.)
        assert float(row['zero_motion_mae']) == 0
        assert float(row['copy_context_mae']) == 0
    assert np.mean([float(row['model_mae']) for row in rows]) == pytest.approx(result['model_mae'])
    # Headless figures: one grid per 12 displayed pairs, plus anchored trajectory videos.
    grids = sorted(output.glob('comparison*.png'))
    assert [p.name for p in grids] == ['comparison1.png'] and len(result['displayed_pairs']) == len(dataset)
    assert result['videos'] and all((output / v['file']).stat().st_size > 0 for v in result['videos'])


def test_ffmpeg_decoder_timestamp_mapping_and_flush():
    import av
    from fractions import Fraction
    from mocap.bags import ultrasound_frames, US_FFMPEG
    encoder = av.CodecContext.create('libx264', 'w')
    encoder.width = encoder.height = 32
    encoder.pix_fmt = 'yuv420p'
    encoder.time_base = Fraction(1, 30)
    encoder.options = {'preset': 'fast', 'crf': '18'}
    packets = []
    for i in range(6):
        frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 30 + i * 20, dtype=np.uint8), format='rgb24')
        frame.pts = i
        packets.extend(encoder.encode(frame))
    packets.extend(encoder.encode(None))
    messages = [SimpleNamespace(encoding='h264', flags=int(packet.is_keyframe), pts=packet.pts,
                               data=bytes(packet), header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=packet.pts * 100)))
                for packet in packets]
    connection = SimpleNamespace(topic=US_FFMPEG, msgtype='ffmpeg')
    class Reader:
        connections = [connection]
        def messages(self, connections):
            return iter((connection, 2_000_000_000 + m.pts, m) for m in messages)
        def deserialize(self, raw, msgtype):
            return raw
    decoded = list(ultrasound_frames(Reader()))
    assert [sensor for sensor, _, _ in decoded] == [1_000_000_000 + i * 100 for i in range(6)]
    for i, (_, _, frame) in enumerate(decoded):
        assert abs(frame.to_ndarray(format='rgb24').mean() - (30 + 20 * i)) < 4


def test_cli_training_resume_matches_uninterrupted(tmp_path):
    import subprocess
    import sys
    path = make_cache(tmp_path, size=128)
    config = dict(seed=42, model=SMALL_WORLD,
                  data=dict(manifest=str(path), min_dt=.2, max_dt=.8, min_mean=8., crop_scale=[.3, 1.]),
                  batch_size=2, workers=0, cpu_threads=2, amp=False, lr=.0003,
                  warmup_steps=1, steps=4, log_every=1, eval_every=2, save_every=1,
                  weight_decay=.04, final_weight_decay=.4, selection='model_recall@5', patience=1,
                  val_pairs_per_episode=2, eval_batch_size=2)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    base = [sys.executable, '-m', 'mocap.train', '--config', str(config_path), '--device', 'cpu']
    split, full = tmp_path / 'split', tmp_path / 'full'
    for args in [base + ['--output', str(split), '--max-steps', '2'],
                 base + ['--output', str(split), '--resume', str(split / 'last.pt')],
                 base + ['--output', str(full)]]:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    a = torch.load(split / 'last.pt', weights_only=False)
    b = torch.load(full / 'last.pt', weights_only=False)
    assert a['step'] == b['step'] == 4
    # Constant test images give perfect retrieval at every validation: no improvement after step 2.
    assert json.loads((full / 'status.json').read_text())['early_stopping'] and a['stale_validations'] == 1
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)


@pytest.mark.parametrize('prediction_type', ['epsilon', 'v'])
def test_diffusion_freezes_world_and_matches_ablation_noise(tmp_path, prediction_type):
    from mocap.diffusion import DiffusionUltrasound
    world = SMALL_WORLD
    diffusion = dict(base=8, sample_steps=3, guidance=1.5, noise_seed=42, prediction_type=prediction_type)
    model = DiffusionUltrasound(world, **diffusion)
    source, target, action = torch.rand(2, 1, 128, 128), torch.rand(2, 1, 128, 128), torch.randn(2, 6)
    loss = model.loss(source, target, action)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.denoiser.condition[0].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.world.parameters())
    model.eval()
    a = model.predict(source, action)
    b = model.predict(source, action)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert a.min() >= 0 and a.max() <= 1 and torch.isfinite(a).all()
    path = tmp_path / 'diffusion.pt'
    torch.save(dict(kind='diffusion', config=dict(model=world, diffusion=diffusion), model=model.state_dict()), path)
    loaded, _ = load_model(path, torch.device('cpu'))
    torch.testing.assert_close(a, loaded.predict(source, action))


def test_diffusion_cli_resume_accepts_added_patience(tmp_path):
    import subprocess
    import sys
    from mocap.train import same_run
    assert same_run(dict(lr=1, patience=2), dict(lr=1)) and not same_run(dict(lr=1), dict(lr=2))
    path = make_cache(tmp_path, size=128)
    world = MotionWorldModel(**SMALL_WORLD)
    optimizer = torch.optim.AdamW(world.parameters())
    world_path = tmp_path / 'world.pt'
    save_checkpoint(world_path, world, optimizer, dict(model=SMALL_WORLD, data=dict(manifest=str(path), min_dt=.2,
                    max_dt=.8, min_mean=8.)), 1, 0., 'test', torch.device('cpu'))
    config = dict(seed=42, world_checkpoint=str(world_path),
                  diffusion=dict(base=8, sample_steps=2, guidance=1.5, noise_seed=42, prediction_type='v'),
                  batch_size=2, workers=0, cpu_threads=2, lr=.0002, warmup_steps=1, steps=3,
                  log_every=1, eval_every=1, save_every=1, val_pairs_per_episode=2)
    config_path = tmp_path / 'diffusion.json'
    base = [sys.executable, '-m', 'mocap.diffusion', '--config', str(config_path), '--device', 'cpu',
            '--output', str(tmp_path / 'run')]
    for extra, patience in [(['--max-steps', '1'], None), (['--resume', str(tmp_path / 'run' / 'last.pt')], 100)]:
        config_path.write_text(json.dumps(dict(config, **({'patience': patience} if patience else {}))))
        result = subprocess.run(base + extra, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    status = json.loads((tmp_path / 'run' / 'status.json').read_text())
    assert status['step'] == 3 and not status['early_stopping']


def test_diffusers_denoiser_uses_condition_and_matches_cosine_schedule():
    import math
    from mocap.diffusion import DiffusionUltrasound
    model = DiffusionUltrasound(SMALL_WORLD, base=8, sample_steps=3, prediction_type='v')
    x, t = torch.randn(2, 1, 128, 128), torch.tensor([10, 500])
    latent = torch.randn(2, SMALL_WORLD['embed_dim'])
    with torch.no_grad():
        assert not torch.allclose(model.denoiser(x, t, latent), model.denoiser(x, t, torch.zeros_like(latent)))
    # Cosine schedule of Nichol & Dhariwal (beta <= 0.999).
    s = torch.linspace(0, 1, 1001, dtype=torch.float64)
    cumulative = torch.cos((s + .008) / 1.008 * math.pi / 2).square()
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=.999)
    torch.testing.assert_close(model.scheduler.alphas_cumprod.double(), (1 - beta).cumprod(0), rtol=1e-4, atol=1e-6)
