import numpy as np
from scipy.spatial.transform import Rotation as R


def get_most_vertical_axis(orientation):
    axes = np.eye(3)
    axes_in_base_frame = R.from_quat(orientation).apply(axes)
    z = axes_in_base_frame[:, 2]  # dot product with z axis
    ind = np.argmax(np.abs(z))
    sign = np.sign(z[ind])
    return sign * axes[ind], sign * axes_in_base_frame[ind]


def project_cube_xy_plane(orientation):
    ax_cube_frame, ax_base_frame = get_most_vertical_axis(orientation)
    rot_align = vector_align_rotation(ax_base_frame, np.array([0, 0, 1]))
    return (rot_align * R.from_quat(orientation)).as_quat()


def vector_align_rotation(a, b):
    """
    return Rotation that transform vector a to vector b

    input
    a : np.array(3)
    b : np.array(3)

    return
    rot : scipy.spatial.transform.Rotation
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    assert norm_a != 0 and norm_b != 0

    a = a / norm_a
    b = b / norm_b

    cross = np.cross(a, b)
    norm_cross = np.linalg.norm(cross)
    cross = cross / norm_cross
    dot = np.dot(a, b)

    if norm_cross < 1e-8 and dot > 0:
        '''same direction, no rotation a == b'''
        return R.from_quat([0, 0, 0, 1])
    elif norm_cross < 1e-8 and dot < 0:
        '''opposite direction a == -b'''
        c = np.eye(3)[np.argmax(np.linalg.norm(np.eye(3) - a, axis=1))]
        cross = np.cross(a, c)
        norm_cross = np.linalg.norm(cross)
        cross = cross / norm_cross

        return R.from_rotvec(cross * np.pi)

    rot = R.from_rotvec(cross * np.arctan2(norm_cross, dot))

    assert np.linalg.norm(rot.apply(a) - b) < 1e-7
    return rot


def roll_and_pitch_aligned(cube_orientation, goal_orientation):
    ax_cube, _ = get_most_vertical_axis(cube_orientation)
    ax_goal, _ = get_most_vertical_axis(goal_orientation)
    return np.allclose(ax_cube, ax_goal)


def get_yaw_diff(ori, goal_ori):
    # NOTE: This assertion should be here only when the object is a cube
    # assert roll_and_pitch_aligned(ori, goal_ori)
    proj_goal_ori = project_cube_xy_plane(goal_ori)
    rot = R.from_quat(proj_goal_ori) * R.from_quat(ori).inv()
    return rot.as_rotvec()[2]


def get_roll_pitch_axis_and_angle(cube_orientation, goal_orientation):
    if roll_and_pitch_aligned(cube_orientation, goal_orientation):
        return None, None
    rot_cube = R.from_quat(cube_orientation)
    rot_goal = R.from_quat(project_cube_xy_plane(goal_orientation))
    ax_up_cube, _ = get_most_vertical_axis(cube_orientation)

    diffs = []
    axis_angles = []
    for i, axis in enumerate(np.eye(3)):
        if ax_up_cube[i] != 0:
            continue
        for angle in [np.pi / 2, -np.pi / 2]:
            rot_align = R.from_rotvec(axis * angle)
            rot_diff = (rot_goal * (rot_cube * rot_align).inv()).magnitude()
            diffs.append(rot_diff)
            axis_angles.append((axis, angle))

    return axis_angles[np.argmin(diffs)]
