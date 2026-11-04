"""ROS-free MCAP decoding, following Canonical_US/rgbd_bag.py.

Only ultrasound, rigid poses and static probe calibration are deserialized.
FFmpeg packets must be replayed in order, including frames skipped for sampling.

Topics read from each MCAP episode:
    /ultrasound/image_raw or /ultrasound_image/ffmpeg   ultrasound frames (1080x1920 screen capture)
    /rigid_poses                                         optical tracker: polaris_frame_T_marker
    /tf_static                                           recorded calibration: marker_T_probe
RGB-D, force and point-cloud topics are never deserialized.
"""
from collections import deque
import numpy as np
from .geometry import matrix, PoseSeries

US = '/ultrasound/image_raw'
US_FFMPEG = '/ultrasound_image/ffmpeg'


def stamp(msg):
    # Sensor (header) timestamp in nanoseconds. Synchronization relies on header stamps, not on
    # the time at which the message was logged, because logging adds transport latency.
    t = msg.header.stamp
    result = int(t.sec) * 1_000_000_000 + int(t.nanosec)
    if result <= 0:
        raise ValueError('Missing/zero sensor timestamp; refusing implicit clock substitution')
    return result


def decode_image(msg):
    # Raw sensor_msgs/Image -> numpy array, honouring row stride, endianness and channel order.
    formats = {'mono8': ('u1', 1), '8uc1': ('u1', 1), 'rgb8': ('u1', 3),
               'bgr8': ('u1', 3), 'rgba8': ('u1', 4), 'bgra8': ('u1', 4),
               'mono16': ('u2', 1), '16uc1': ('u2', 1)}
    enc = msg.encoding.lower()
    if enc not in formats:
        raise ValueError(f'Unsupported ultrasound encoding: {enc}')
    kind, channels = formats[enc]
    dtype = np.dtype(kind).newbyteorder('>' if msg.is_bigendian else '<')
    row_bytes = msg.width * channels * dtype.itemsize
    if msg.step < row_bytes or len(msg.data) < msg.height * msg.step:
        raise ValueError('Invalid image stride or buffer length')
    rows = np.frombuffer(msg.data, np.uint8, count=msg.height * msg.step).reshape(msg.height, msg.step)
    array = rows[:, :row_bytes].copy().view(dtype).reshape(msg.height, msg.width, channels)
    if kind == 'u2':
        array = (array.astype(np.float32) / 257).round().astype(np.uint8)
    if channels == 1:
        return array[:, :, 0]
    array = array[:, :, :3]
    return array[:, :, ::-1].copy() if enc.startswith('bgr') else array


def static_chain(edges, parent, child):
    # Breadth-first composition of static transforms (inverting edges when needed) to obtain
    # parent_T_child, here marker_T_probe: the rigid offset from the tracked marker to the probe.
    queue = deque([(parent, np.eye(4))])
    visited = {parent}
    while queue:
        current, transform = queue.popleft()
        if current == child:
            return transform
        for (a, b), part in sorted(edges.items()):
            for start, end, value in [(a, b, part), (b, a, np.linalg.inv(part))]:
                if start == current and end not in visited:
                    visited.add(end)
                    queue.append((end, transform @ value))
    raise ValueError(f'Missing recorded static calibration {parent} -> {child}')


class PoseCollector:
    # Accumulates tracker samples and static calibration during the single pass over the bag.
    # finish() returns probe poses: world_T_probe = polaris_frame_T_marker @ marker_T_probe.
    def __init__(self, marker='marker', probe='probe', max_gap_ns=100_000_000):
        self.marker, self.probe, self.max_gap_ns = marker, probe, max_gap_ns
        self.edges, self.samples, self.parents = {}, [], set()

    def add(self, topic, msg):
        if topic == '/tf_static':
            for tf in msg.transforms:
                t, q = tf.transform.translation, tf.transform.rotation
                edge = (tf.header.frame_id.lstrip('/'), tf.child_frame_id.lstrip('/'))
                value = matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
                if edge in self.edges and not np.allclose(self.edges[edge], value, atol=1e-7):
                    raise ValueError(f'Static calibration changed: {edge}')
                self.edges[edge] = value
        else:
            if not (len(msg.frames) == len(msg.poses) == len(msg.inbound)):
                raise ValueError('Malformed RigidArray')
            self.parents.add(msg.header.frame_id.lstrip('/'))
            # A sample without a valid marker pose is kept as None: it becomes a barrier that
            # splits the episode into tracking segments (see geometry.PoseSeries).
            pose = None
            for name, valid, p in zip(msg.frames, msg.inbound, msg.poses):
                if name.lstrip('/') == self.marker and valid:
                    t, q = p.position, p.orientation
                    try:
                        pose = matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
                    except ValueError:
                        pass
            self.samples.append((stamp(msg), pose))

    def finish(self):
        if not self.samples or len(self.parents) != 1:
            raise ValueError(f'Need rigid poses in one reference frame; found {self.parents}')
        calibration = static_chain(self.edges, self.marker, self.probe)
        series = PoseSeries([(t, None if p is None else p @ calibration) for t, p in self.samples], self.max_gap_ns)
        return series, dict(parent_frame=next(iter(self.parents)), probe_frame=self.probe,
                            marker_frame=self.marker, marker_T_probe=calibration.tolist(),
                            pose_samples=len(self.samples), valid_pose_samples=sum(p is not None for _, p in self.samples))


def read_poses(reader, marker='marker', probe='probe', max_gap_ns=100_000_000):
    collector = PoseCollector(marker, probe, max_gap_ns)
    connections = [c for c in reader.connections if c.topic in ('/rigid_poses', '/tf_static')]
    for c, _, raw in reader.messages(connections=connections):
        collector.add(c.topic, reader.deserialize(raw, c.msgtype))
    return collector.finish()


def ultrasound_frames(reader, pose_collector=None, video_stats=None):
    """Yield (sensor_ns, log_ns, pixels-or-AV-frame); pixels stay lazy for video."""
    raw = [c for c in reader.connections if c.topic == US]
    connections = raw or [c for c in reader.connections if c.topic == US_FFMPEG]
    if not connections:
        raise ValueError('No supported ultrasound topic')
    if len(connections) != 1:
        raise ValueError('Multiple ultrasound publishers are ambiguous')
    if pose_collector is not None:
        connections = connections + [c for c in reader.connections if c.topic in ('/rigid_poses', '/tf_static')]
    # H.264 frames depend on previous frames, so every packet is decoded in order, starting at
    # the first keyframe. Decoders may return frames with a delay; `pending` maps each packet
    # PTS to its timestamps so that each decoded picture receives the stamp of its own packet.
    decoder, codec, pending = None, None, {}
    stats = video_stats if video_stats is not None else {}
    stats.update(skipped_before_keyframe=0, unmatched_packet_timestamps_ns=[])

    def emit(frames):
        for frame in frames:
            if frame.pts not in pending:
                raise ValueError('Decoded frame lacks a matching FFmpeg packet timestamp')
            sensor, recorded = pending.pop(frame.pts)
            yield sensor, recorded, frame

    for c, recorded, payload in reader.messages(connections=connections):
        msg = reader.deserialize(payload, c.msgtype)
        if c.topic in ('/rigid_poses', '/tf_static'):
            pose_collector.add(c.topic, msg)
            continue
        if raw:
            yield stamp(msg), recorded, decode_image(msg)
            continue
        import av
        packet_codec = msg.encoding.split(';')[0]
        if decoder is None:
            if not msg.flags & 1:
                stats['skipped_before_keyframe'] += 1
                continue
            codec = packet_codec
            decoder = av.CodecContext.create(codec, 'r')
            decoder.thread_count = 2
        if codec != packet_codec:
            raise ValueError('Video codec changed within an episode')
        pts = int(msg.pts)
        if pts in pending:
            raise ValueError('Duplicate pending FFmpeg PTS')
        pending[pts] = stamp(msg), recorded
        packet = av.Packet(bytes(msg.data))
        packet.pts = pts
        yield from emit(decoder.decode(packet))
    if decoder is not None:
        yield from emit(decoder.decode(None))
        # Recording may end with incomplete pictures or include non-picture
        # packets. Retain only decoded frames with a known stamp; report drops.
        stats['unmatched_packet_timestamps_ns'] = [sensor for sensor, _ in pending.values()]
