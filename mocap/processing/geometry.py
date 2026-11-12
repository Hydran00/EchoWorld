"""SE(3) conventions: parent_T_child, metres, ROS xyzw quaternions.

A pose A_T_B is a 4x4 homogeneous matrix mapping coordinates expressed in frame B to frame A.
Here the world frame is the optical tracker (polaris_frame) and the child frame is the
calibrated ultrasound probe, so world_T_probe locates the imaging plane in space.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def matrix(translation, quaternion):
    translation, quaternion = np.asarray(translation), np.asarray(quaternion)
    if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
        raise ValueError('Non-finite pose')
    if np.linalg.norm(quaternion) < 1e-8:
        raise ValueError('Zero quaternion')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:3, 3] = translation
    return result


def relative_action(source, target):
    """inv(world_T_source) @ world_T_target, [translation mm, rotvec radians].

    The local source-probe frame removes the arbitrary global tracker frame.
    This explicitly differs from the paper's Euler/spatial-difference encoding.
    """
    # delta = source_T_target: the target probe pose expressed in the source probe frame.
    # Any rigid change of the world frame G cancels out, since inv(G A) (G B) = inv(A) B.
    # The action is therefore defined with respect to the probe itself (e.g. "advance 5 mm
    # along the probe axis, rotate 0.1 rad about its long axis"), not with respect to the room.
    delta = np.linalg.inv(source) @ target
    # Encoding: translation in millimetres, rotation as an axis-angle vector (direction = axis,
    # norm = angle in radians). Unlike Euler angles, this is continuous and free of gimbal lock
    # for the small inter-frame rotations considered here.
    return np.concatenate((delta[..., :3, 3] * 1000,
                           Rotation.from_matrix(delta[..., :3, :3]).as_rotvec()), axis=-1).astype(np.float32)


class PoseSeries:
    """Interpolate only adjacent, valid measurements; never extrapolate."""
    def __init__(self, samples, max_gap_ns=100_000_000):
        # Preserve invalid samples as barriers, with last publication winning.
        # samples: (timestamp_ns, world_T_probe or None). None marks a tracker sample in which
        # the marker was not visible ("inbound" false). Consecutive valid samples belong to the
        # same segment; a new segment starts at every invalid sample or temporal gap. Downstream,
        # training pairs are formed only within a segment, so motion is never inferred across
        # periods in which the probe position is unknown.
        samples = dict(samples)
        self.times = np.array(sorted(samples), dtype=np.int64)
        self.poses = [samples[t] for t in self.times]
        self.max_gap_ns = max_gap_ns
        self.segments = []
        segment = 0
        for i, pose in enumerate(self.poses):
            if i and (pose is None or self.poses[i - 1] is None or
                      self.times[i] - self.times[i - 1] > max_gap_ns):
                segment += 1
            self.segments.append(segment)

    def at(self, timestamp, tolerance_ns=50_000_000):
        # Pose at an ultrasound timestamp, interpolated between the two bracketing tracker
        # samples. Both brackets must be valid, at most tolerance_ns from the image and at most
        # max_gap_ns apart; otherwise the image is rejected rather than assigned an uncertain pose.
        right = int(np.searchsorted(self.times, timestamp))
        if right < len(self.times) and self.times[right] == timestamp:
            if self.poses[right] is None:
                raise ValueError('Invalid tracking')
            return self.poses[right], self.segments[right], 0
        if right == 0 or right == len(self.times):
            raise ValueError('Pose extrapolation required')
        left = right - 1
        gap = int(self.times[right] - self.times[left])
        distance = max(timestamp - int(self.times[left]), int(self.times[right]) - timestamp)
        if (gap > self.max_gap_ns or distance > tolerance_ns or
                self.poses[left] is None or self.poses[right] is None):
            raise ValueError('Tracking gap or synchronization tolerance exceeded')
        # Linear interpolation of translation and spherical linear interpolation (SLERP) of
        # rotation, i.e. constant velocity motion along the shortest rotation between brackets.
        alpha = (timestamp - int(self.times[left])) / gap
        a, b = self.poses[left], self.poses[right]
        result = np.eye(4)
        result[:3, 3] = (1 - alpha) * a[:3, 3] + alpha * b[:3, 3]
        result[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))([alpha]).as_matrix()[0]
        return result, self.segments[left], min(timestamp - int(self.times[left]), int(self.times[right]) - timestamp)
