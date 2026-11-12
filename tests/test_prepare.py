"""Raw extraction applies temporal calibration once, before training reads the cache."""
import numpy as np
import pytest


@pytest.mark.parametrize('offset_ms,first_index', [(0.,0),(-200.,2)])
def test_raw_export_pairs_image_with_acquisition_pose_and_preserves_header(tmp_path, monkeypatch, offset_ms, first_index):
    import contextlib
    from types import SimpleNamespace
    import mocap.processing.prepare as prepare
    from mocap.processing.geometry import PoseSeries
    root = tmp_path / 'raw'
    bag = root / 'group' / 'episode'
    bag.mkdir(parents=True)
    (bag / 'metadata.yaml').write_text('fixture')
    times = 1_000_000_000 + np.arange(5,dtype=np.int64)*100_000_000
    poses = np.repeat(np.eye(4)[None],5,axis=0)
    poses[:,0,3] = np.arange(5)*.01
    series = PoseSeries(list(zip(times,poses)),max_gap_ns=100_000_000)
    monkeypatch.setattr(prepare,'AnyReader',lambda paths:contextlib.nullcontext(None))
    monkeypatch.setattr(prepare,'PoseCollector',lambda *args:SimpleNamespace(finish=lambda:(series,{})))
    frames = [(int(t),int(t)+50_000_000,np.full((16,16),20+i,np.uint8)) for i,t in enumerate(times)]
    monkeypatch.setattr(prepare,'ultrasound_frames',lambda *args:iter(frames))
    settings=dict(marker='marker',probe='probe',max_pose_gap_ms=100.,sync_ms=50.,fps=100.,
                  crop=[0,0,16,16],size=16,version=1,pose_time_offset_ms=offset_ms)
    output=tmp_path/'cache'
    meta=prepare.export_episode(bag,root,output,settings)
    folder=output/meta['path']
    with np.load(folder/'poses.npz') as cache:
        np.testing.assert_array_equal(cache['image_header_timestamps_ns'],times[first_index:])
        np.testing.assert_array_equal(cache['timestamps_ns'],times[first_index:]+round(offset_ms*1e6))
        np.testing.assert_array_equal(cache['recorded_ns'],times[first_index:]+50_000_000)
        np.testing.assert_allclose(cache['world_T_probe'],poses[:5-first_index])
    np.testing.assert_array_equal(np.load(folder/'images.npy')[:,0,0],np.arange(20+first_index,25))
    assert meta['rejected_sync']==first_index
    if offset_ms:
        assert meta['pose_alignment']['offset_ms']==offset_ms
    else:
        assert 'pose_alignment' not in meta

    # Re-exporting an unchanged source reuses the prepared cache without decoding
    # or applying the offset a second time. A changed offset needs a new cache.
    def unexpected_decode(*args):
        raise AssertionError('An unchanged cache must not be decoded again')
    monkeypatch.setattr(prepare, 'ultrasound_frames', unexpected_decode)
    repeated = prepare.export_episode(bag, root, output, settings)
    assert repeated['signature'] == meta['signature']
    with np.load(folder / 'poses.npz') as cache:
        np.testing.assert_array_equal(cache['timestamps_ns'], times[first_index:] + round(offset_ms * 1e6))
    with pytest.raises(ValueError, match='cache settings/source changed'):
        prepare.export_episode(bag, root, output, dict(settings, pose_time_offset_ms=offset_ms - 100.))
