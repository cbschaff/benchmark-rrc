from random import shuffle
from itertools import islice
import time

INF = float('inf')

RRT_ITERATIONS = 20
RRT_RESTARTS = 2
RRT_SMOOTHING = 20

# INCR_RRT_RESTARTS = 10
INCR_RRT_ITERATIONS = 30


def irange(start, stop=None, step=1):  # np.arange
    if stop is None:
        stop = start
        start = 0
    while start < stop:
        yield start
        start += step


def negate(test):
    return lambda *args, **kwargs: not test(*args, **kwargs)


def argmin(function, sequence):
    # TODO: use min
    values = list(sequence)
    scores = [function(x) for x in values]
    return values[scores.index(min(scores))]


def pairs(lst):
    return zip(lst[:-1], lst[1:])


def merge_dicts(*args):
    result = {}
    for d in args:
        result.update(d)
    return result
    # return dict(reduce(operator.add, [d.items() for d in args]))


def flatten(iterable_of_iterables):
    return (item for iterables in iterable_of_iterables for item in iterables)


def randomize(sequence):
    shuffle(sequence)
    return sequence


def take(iterable, n=INF):
    if n == INF:
        n = None  # NOTE - islice takes None instead of INF
    elif n == None:
        n = 0  # NOTE - for some of the uses
    return islice(iterable, n)


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    enums['names'] = sorted(enums.keys(), key=lambda k: enums[k])
    return type('Enum', (), enums)


def elapsed_time(start_time):
    return time.time() - start_time


def weighted_position_error(pose_diff):
    import numpy as np
    _CUBE_WIDTH = 0.065
    _ARENA_RADIUS = 0.195
    _min_height = _CUBE_WIDTH / 2
    _max_height = 0.1

    range_xy_dist = _ARENA_RADIUS * 2
    range_z_dist = _max_height

    xy_dist = np.linalg.norm(
        pose_diff[:2]
    )
    z_dist = abs(pose_diff[2])

    # weight xy- and z-parts by their expected range
    return (xy_dist / range_xy_dist + z_dist / range_z_dist) / 2


def weighted_paired_position_error(q1, q2):
    import numpy as np
    _CUBE_WIDTH = 0.065
    _ARENA_RADIUS = 0.195
    _min_height = _CUBE_WIDTH / 2
    _max_height = 0.1

    range_xy_dist = _ARENA_RADIUS * 2
    range_z_dist = _max_height

    xy_dist = np.linalg.norm(
        np.asarray(q2[:2]) - np.asarray(q1[:2])
    )
    z_dist = abs(q2[2] - q1[2])

    # weight xy- and z-parts by their expected range
    return (xy_dist / range_xy_dist + z_dist / range_z_dist) / 2


def position_error(pose_diff):
    import numpy as np
    return np.linalg.norm(pose_diff[:3])


def weighted_euler_rot_error(pose_diff):
    import numpy as np
    from scipy.spatial.transform import Rotation
    euler_rot_diff = pose_diff[3:]
    error_rot = Rotation.from_euler('xyz', euler_rot_diff)
    orientation_error = error_rot.magnitude()

    # scale both position and orientation error to be within [0, 1] for
    # their expected ranges
    scaled_orientation_error = orientation_error / np.pi

    return scaled_orientation_error


def weighted_paired_euler_rot_error(q1, q2):
    import pybullet as p
    import numpy as np
    from scipy.spatial.transform import Rotation

    # https://stackoverflow.com/a/21905553
    goal_rot = Rotation.from_quat(p.getQuaternionFromEuler(q2[3:]))
    actual_rot = Rotation.from_quat(p.getQuaternionFromEuler(q1[3:]))
    error_rot = goal_rot.inv() * actual_rot
    orientation_error = error_rot.magnitude()

    # scale both position and orientation error to be within [0, 1] for
    # their expected ranges
    scaled_orientation_error = orientation_error / np.pi
    return scaled_orientation_error


def weighted_pose_error(pose_diff):
    scaled_pos_error = weighted_position_error(pose_diff)
    scaled_rot_error = weighted_euler_rot_error(pose_diff)
    # scaled_error = (scaled_pos_error + scaled_rot_error) / 2

    # This may require some tuning:
    scaled_error = (scaled_pos_error + scaled_rot_error) / 2
    return scaled_error


def weighted_paired_pose_error(q1, q2):
    scaled_pos_error = weighted_paired_position_error(q1, q2)
    scaled_rot_error = weighted_paired_euler_rot_error(q1, q2)

    scaled_error = (scaled_pos_error + scaled_rot_error) / 2
    return scaled_error


def pose_competition_reward_error(pose_diff):
    scaled_pos_error = weighted_position_error(pose_diff)
    scaled_rot_error = weighted_euler_rot_error(pose_diff)
    # scaled_error = (scaled_pos_error + scaled_rot_error) / 2

    # This may require some tuning:
    scaled_error = (scaled_pos_error + scaled_rot_error) / 2
    return scaled_error
