"""Participant boundaries and trajectory continuity in the read-only viewer."""
import json

import numpy as np
import pytest

from mocap.visualization.canonical_dataset import display_indices, prepare_groups
from mocap.visualization.participants import load_mapping, recording_identity


@pytest.mark.parametrize('bag,doctor,subject', [
    ('bag_expert1_2026-10-28', 'Amir', 'Yuan'),
    ('bag_expert18_2026-10-28', 'Amir', 'Yuan'),
    ('bag_expert19_2026-10-28', 'Amir', 'Davide'),
    ('bag_expert38_2026-10-28', 'Amir', 'Davide'),
    ('bag_Yuan9b_2026-10-24', 'Davide', 'Yuan'),
    ('bag_nardi12_2026-10-27', 'Feng', 'Davide'),
    ('bag_davide11_2026-10-24', 'Yuan', 'Davide'),
    ('bag_test_2026-10-24', 'Davide', 'Test'),
    ('bag_feng13_2026-10-27', 'Davide', 'Feng'),
    ('bag_fengxhou25_2026-10-26', 'Davide', 'Fengxhou'),
])
def test_user_participant_mapping(bag, doctor, subject):
    identity = recording_identity('unrelated-folder/' + bag, load_mapping())
    assert (identity['doctor'], identity['subject']) == (doctor, subject)


def test_expert_outside_user_range_and_overlapping_rules_are_not_guessed():
    mapping = load_mapping()
    with pytest.raises(ValueError, match='Missing explicit'):
        recording_identity('bag_expert39_2026', mapping)
    mapping['rules'].append(dict(prefix='expert', first=18, last=19, doctor='Amir', subject='Davide'))
    with pytest.raises(ValueError, match='Conflicting'):
        recording_identity('bag_expert18_2026', mapping)


def test_sampling_keeps_real_endpoints_and_segment_boundaries():
    np.testing.assert_array_equal(display_indices(np.array([0, 0, 1, 1, 1, 1, 1]), 3), [0, 1, 2, 3, 6])
    np.testing.assert_array_equal(display_indices(np.array([0]), 3), [0])


def test_groups_keep_doctors_distinct_on_same_patient_and_do_not_change_training_data(tmp_path):
    entries = []
    source_poses = np.tile(np.eye(4), (8, 1, 1))
    source_poses[:, 0, 3] = np.arange(8) / 100
    for name in ('davide1', 'nardi1', 'expert19', 'expert18'):
        folder = tmp_path / name
        folder.mkdir()
        np.savez(folder / 'poses.npz', world_T_probe=source_poses, segment_ids=[0, 0, 1, 1, 1, 1, 1, 1])
        entries.append(dict(episode_id=f'original/bag_{name}_2026', group='unchanged', path=name))
    manifest = tmp_path / 'manifest.json'
    original_manifest = json.dumps(dict(episodes=entries))
    manifest.write_text(original_manifest)
    groups, count = prepare_groups(manifest, load_mapping(), stride=3)
    assert count == 4
    assert set(groups) == {('Davide', 'Yuan'), ('Davide', 'Feng'), ('Davide', 'Amir'), ('Yuan', 'Amir')}
    group = groups[('Davide', 'Yuan')]
    assert group['points'][2] == [None, None, None]  # Gap between segment 0 and segment 1.
    np.testing.assert_allclose(group['ends'][0], [.07, 0, -.004])
    assert 'Medico: Yuan' in group['hover'][0] and 'Soggetto: Davide' in group['hover'][0]
    assert manifest.read_text() == original_manifest
    for entry in entries:
        with np.load(tmp_path / entry['path'] / 'poses.npz') as saved:
            np.testing.assert_array_equal(saved['world_T_probe'], source_poses)
