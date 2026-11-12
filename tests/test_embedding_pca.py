"""Timestamp sampling, full split coverage, and reproducible PCA/t-SNE coordinates."""
import json

import numpy as np
import pytest
import torch

from mocap.evaluation.embedding_pca import extract_embeddings, fit_pca, fit_tsne, sample_frames
from mocap.pretraining.model import MotionWorldModel


def test_timestamp_sampling_keeps_actual_endpoint_and_epoch_precision():
    times = 1_800_000_000_000_000_000 + np.array([0, 430, 990, 1450, 2020, 2310]) * 1_000_000
    parts = sample_frames(times, np.zeros(len(times), dtype=int))
    assert [part.tolist() for part in parts] == [[0, 2, 4, 5]]
    assert sample_frames(times[:2], [0, 0])[0].tolist() == [0, 1]
    assert sample_frames(times[:1], [0])[0].tolist() == [0]


def test_sampling_breaks_at_tracking_changes_and_missing_recordings():
    times = np.array([0, 400, 800, 900, 1300, 4000, 4300]) * 1_000_000
    parts = sample_frames(times, [0, 0, 0, 1, 1, 1, 1])
    assert [part.tolist() for part in parts] == [[0, 2], [3, 4], [5, 6]]
    parts = sample_frames(times[:5], [0, 0, 1, 1, 0])
    assert [part.tolist() for part in parts] == [[0, 1], [2, 3], [4]]


@pytest.mark.parametrize('times,segments,interval', [([0, 0], [0, 0], 1),
    ([1, 0], [0, 0], 1), ([], [], 1), ([0], [], 1), ([0], [0], 0),
    ([0], [0], float('nan')), ([0], [0], float('inf'))])
def test_sampling_rejects_invalid_inputs(times, segments, interval):
    with pytest.raises(ValueError):
        sample_frames(np.array(times, dtype=np.int64), segments, interval)


def test_full_test_split_includes_dark_short_episodes_and_matches_encoder(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = MotionWorldModel(image_size=32, embed_dim=24, encoder_depth=1,
        predictor_depth=1, num_heads=3, projector_dim=16, mask=dict(min_keep=1))
    entries, images = [], {}
    for name, split, pixels in [('a', 'test', [0, 60, 120, 180]),
                                 ('b', 'test', [0]), ('c', 'train', [240, 200])]:
        folder = tmp_path / name
        folder.mkdir()
        images[name] = np.broadcast_to(np.array(pixels, dtype=np.uint8)[:, None, None], (len(pixels), 32, 32)).copy()
        np.save(folder / 'images.npy', images[name])
        np.savez(folder / 'poses.npz', timestamps_ns=np.arange(len(pixels), dtype=np.int64) * 400_000_000,
                 segment_ids=np.zeros(len(pixels), dtype=int))
        entries.append(dict(episode_id=name, group=name, split=split, path=name))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1, episodes=entries)))
    device = torch.device('cpu')
    embeddings, rows, episodes = extract_embeddings(model, path, 'test', device, batch_size=2)
    assert [e['episode_id'] for e in episodes] == ['a', 'b']
    assert [(r['episode_id'], r['frame_index']) for r in rows] == [('a', 0), ('a', 2), ('a', 3), ('b', 0)]
    assert sum(r['is_end'] for r in rows) == 2
    pixels = np.concatenate([images['a'][[0, 2, 3]], images['b']])
    batch = torch.from_numpy(pixels).float()[:, None] / 255.
    with torch.inference_mode():
        expected = model.encoder(model.encoder_input(batch)).mean(1).numpy()
    np.testing.assert_allclose(embeddings, expected, atol=1e-6)
    target, target_rows, _ = extract_embeddings(model, path, 'test', device, batch_size=1, representation='target')
    with torch.inference_mode():
        expected_target = model.target_latent(batch).mean(1).numpy()
    np.testing.assert_allclose(target, expected_target, atol=1e-6)
    assert target_rows == rows


def test_saved_pca_basis_reproduces_one_shared_coordinate_system():
    embeddings = np.random.default_rng(12).normal(size=(19, 8)) + np.arange(8) * 10
    coordinates, pca = fit_pca(embeddings)
    np.testing.assert_allclose((embeddings - pca.mean_) @ pca.components_.T, coordinates, atol=1e-12)
    np.testing.assert_allclose(coordinates.mean(axis=0), 0, atol=1e-12)
    assert 0 < pca.explained_variance_ratio_.sum() <= 1
    with pytest.raises(ValueError, match='identical'):
        fit_pca(np.ones((3, 8)))


def test_tsne_is_reproducible_and_keeps_separated_clusters_as_neighbors():
    rng = np.random.default_rng(12)
    labels = np.repeat(np.arange(3), 20)
    centers = rng.normal(size=(3, 64)) * 10
    embeddings = centers[labels] + rng.normal(scale=.05, size=(60, 64))
    coordinates, info = fit_tsne(embeddings, perplexity=8, seed=17)
    repeated, _ = fit_tsne(embeddings, perplexity=8, seed=17)
    np.testing.assert_allclose(coordinates, repeated, atol=1e-6)
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = distances.argsort(axis=1)[:, :5]
    assert (labels[neighbors] == labels[:, None]).mean() > .95
    assert info['input_dimensions'] == 50
    assert np.isfinite(info['kl_divergence'])


def test_tsne_rejects_undefined_neighborhoods_and_constant_embeddings():
    embeddings = np.random.default_rng(12).normal(size=(10, 4))
    for perplexity in (0, -1, 10, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='perplexity'):
            fit_tsne(embeddings, perplexity=perplexity)
    with pytest.raises(ValueError, match='identical'):
        fit_tsne(np.ones((40, 4)))


def test_three_point_sampling_uses_episode_time_and_preserves_gaps():
    times = 1_800_000_000_000_000_000 + np.array([0, 100, 200, 300, 4900, 5100, 6000, 10000]) * 1_000_000
    parts = sample_frames(times, np.zeros(len(times), dtype=int), sampling='start-middle-end')
    assert [part.tolist() for part in parts] == [[0], [4], [7]]
    times = np.array([0, 100, 200, 600, 800]) * 1_000_000
    parts = sample_frames(times, [0, 1, 0, 0, 0], sampling='start-middle-end')
    assert [part.tolist() for part in parts] == [[0], [2, 4]]
    for count in (1, 2, 3):
        parts = sample_frames(np.arange(count) * 100_000_000, np.zeros(count), sampling='start-middle-end')
        assert np.concatenate(parts).tolist() == list(range(count))
