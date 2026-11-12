import json
import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mocap.visualization.tensorboard import refresh_mirrors


def test_discovers_new_runs_and_follows_completed_run_without_duplicate_points(tmp_path):
    first = tmp_path / 'first'
    first.mkdir()
    (first / 'train.jsonl').write_text(json.dumps(dict(step=1, loss=2.)) + '\n')
    (first / 'status.json').write_text('{}')
    mirrors = {}
    try:
        refresh_mirrors(mirrors, [], tmp_path)
        original = mirrors[first]
        original.sync()
        second = tmp_path / 'nested' / 'second'
        second.mkdir(parents=True)
        (second / 'train.jsonl').write_text(json.dumps(dict(step=1, loss=3.)) + '\n')
        with (first / 'train.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(step=2, loss=1.)) + '\n')
        refresh_mirrors(mirrors, [], tmp_path)
        assert mirrors[first] is original
        assert set(mirrors) == {first, second}
        for mirror in mirrors.values():
            mirror.sync()
        events = EventAccumulator(str(first / 'tensorboard')).Reload().Scalars('train/loss')
        assert [(e.step, e.value) for e in events] == [(1, 2.), (2, 1.)]
        events = EventAccumulator(str(second / 'tensorboard')).Reload().Scalars('train/loss')
        assert [(e.step, e.value) for e in events] == [(1, 3.)]
        shutil.rmtree(second)
        refresh_mirrors(mirrors, [], tmp_path)
        assert set(mirrors) == {first}
    finally:
        for mirror in mirrors.values():
            mirror.writer.close()
