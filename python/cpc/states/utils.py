import numpy as np
from scipy.spatial.transform import Rotation as R


def _quat_mult(q1, q2):
    x0, y0, z0, w0 = q2
    x1, y1, z1, w1 = q1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0])


def _quat_conj(q):
    ret = np.copy(q)
    ret[:3] *= -1
    return ret


def _get_angle_axis(current, target):
    # Return:
    # (1) angle err between orientations
    # (2) Unit rotation axis
    rot = R.from_quat(_quat_mult(current, _quat_conj(target)))

    rotvec = rot.as_rotvec()
    norm = np.linalg.norm(rotvec)

    if norm > 1E-8:
        return norm, (rotvec / norm)
    else:
        return 0, np.zeros(len(rotvec))


def _get_angle_axis_top_only(current, target):
    # Return:
    # (1) angle err between orientations
    # (2) Unit rotation axis

    # Faces = +/-x, +/-y, +/-z
    axes = np.eye(3)
    axes = np.concatenate((axes, -1 * np.eye(3)), axis=0)

    # See which face is on top of the goal cube
    z_axis = [0., 0., 1.]
    target_axis = None
    min_angle = 100.
    min_axis_id = 0
    for i, axis in enumerate(axes):
        qg_i = R.from_quat(target).apply(axis)
        angle = np.arccos(np.dot(qg_i, z_axis))
        if angle < min_angle:
            min_angle = angle
            min_axis_id = i
            target_axis = qg_i

    # See where that face is on the current cube
    current_axis = R.from_quat(current).apply(axes[min_axis_id])

    # Angle calculation
    angle = np.arccos(np.dot(current_axis, target_axis))
    if angle > 1.0:
        angle = 1.0
    if angle < -1.0:
        angle = -1.0

    # Get rotvec only for top of cube
    rotvec = np.cross(current_axis, target_axis)
    norm = np.linalg.norm(rotvec)

    # Don't try if angle is too small or too big
    if (norm <= 1E-8) or (angle > np.pi/2.0):
        return 0, np.zeros(len(rotvec))
    else:
        return angle, (rotvec / norm)


def pitch_orient(observation):
    manip_angle = 0
    manip_axis = np.zeros(3)
    manip_arm = 0
    current = observation["object_orientation"]
    target = observation["goal_object_orientation"]

    # Faces = +/-x, +/-y, +/-z
    axes = np.eye(3)
    axes = np.concatenate((axes, -1 * np.eye(3)), axis=0)

    # See which face is on top of the goal cube
    z_axis = [0., 0., 1.]
    min_angle = 100.
    min_axis_id = 0
    for i, axis in enumerate(axes):
        qg_i = R.from_quat(target).apply(axis)
        angle = np.arccos(np.dot(qg_i, z_axis))
        if angle < min_angle:
            min_angle = angle
            min_axis_id = i

    print("Min Axis ID: ", min_axis_id)

    # See where that face is on the current cube
    manip_axis = R.from_quat(current).apply(axes[min_axis_id])
    print("Manip Axis: ", manip_axis)
    if np.abs(1 - manip_axis[2]) < 0.3:
        # On top, no manipulation
        manip_angle = 0
    elif np.abs(-1 - manip_axis[2]) < 0.3:
        # On bottom, need 2 90deg rotations
        manip_angle = 180
    else:
        # On side, need 1 90deg rotation
        manip_angle = 90

    # For 180deg rotation, determine which side gets manipulated
    # Since there are 2 options
    if manip_angle == 180:
        test_axis_id = (min_axis_id + 1) % len(axes)
        test_manip_axis = R.from_quat(current).apply(axes[test_axis_id])
        test_target_axis = R.from_quat(target).apply(axes[test_axis_id])

        angle = np.arccos(np.dot(test_manip_axis, test_target_axis))

        # If angle is small, manip the other axis
        if angle < np.pi/2.0:
            test_axis_id = (min_axis_id + 2) % len(axes)
            test_manip_axis = R.from_quat(current).apply(axes[test_axis_id])

        # Update manip axis to the correct side
        manip_axis = test_manip_axis

    # Determine which arm gets the 90deg rotation
    if manip_angle > 0:
        arm_angle = np.rad2deg(np.arctan2(
            manip_axis[1], manip_axis[0]))

        print("arm angle: ", arm_angle, " ax: ", manip_axis)

        if arm_angle > -90.0 and arm_angle <= 30.0:
            manip_arm = 1
        elif arm_angle > 30.0 and arm_angle <= 150:
            manip_arm = 0
        else:
            manip_arm = 2

    # Clear any weird "z" angle from manip axis
    manip_axis[2] = 0

    print("Manip angle: ", manip_angle, " Manip arm: ", manip_arm)
    if manip_angle != 0:
        return manip_angle, manip_axis, manip_arm, False
    else:
        manip_angle, manip_axis, manip_arm = yaw_orient(observation)
        return yaw_orient_diff(observation['object_orientation'],
            observation['goal_object_orientation']), manip_axis, manip_arm, True


def yaw_orient(observation):
    manip_angle = 0
    manip_axis = np.zeros(3)
    manip_arm = 0
    current = observation["object_orientation"]
    target = observation["goal_object_orientation"]

    # Faces = +/-x, +/-y, +/-z
    axes = np.eye(3)
    axes = np.concatenate((axes, -1 * np.eye(3)), axis=0)

    # See which face is on top of the goal cube
    z_axis = [1., 0., 0.]
    min_angle = 100.
    min_axis_id = 0
    for i, axis in enumerate(axes):
        qg_i = R.from_quat(target).apply(axis)
        angle = np.arccos(np.dot(qg_i, z_axis))
        if angle < min_angle:
            min_angle = angle
            min_axis_id = i

    print("Min Axis ID: ", min_axis_id)

    # See where that face is on the current cube
    manip_axis = R.from_quat(current).apply(axes[min_axis_id])
    print("Manip Axis: ", manip_axis)
    if np.abs(1 - manip_axis[2]) < 0.1:
        # On top, no manipulation
        manip_angle = 0
    elif np.abs(-1 - manip_axis[2]) < 0.1:
        # On bottom, need 2 90deg rotations
        manip_angle = 180
    else:
        # On side, need 1 90deg rotation
        manip_angle = 90

    # For 180deg rotation, determine which side gets manipulated
    # Since there are 2 options
    if manip_angle == 180:
        test_axis_id = (min_axis_id + 1) % len(axes)
        test_manip_axis = R.from_quat(current).apply(axes[test_axis_id])
        test_target_axis = R.from_quat(target).apply(axes[test_axis_id])

        angle = np.arccos(np.dot(test_manip_axis, test_target_axis))

        # If angle is small, manip the other axis
        if angle < np.pi/2.0:
            test_axis_id = (min_axis_id + 2) % len(axes)
            test_manip_axis = R.from_quat(current).apply(axes[test_axis_id])

        # Update manip axis to the correct side
        manip_axis = test_manip_axis

    # Determine which arm gets the 90deg rotation
    if manip_angle > 0:
        arm_angle = np.rad2deg(np.arctan2(
            manip_axis[1], manip_axis[0]))

        print("arm angle: ", arm_angle, " ax: ", manip_axis)

        if arm_angle > -90.0 and arm_angle <= 30.0:
            manip_arm = 1
        elif arm_angle > 30.0 and arm_angle <= 150:
            manip_arm = 0
        else:
            manip_arm = 2

    # Clear any weird "z" angle from manip axis
    manip_axis[2] = 0

    print("For YAW, Manip angle: ", manip_angle, " Manip arm: ", manip_arm)
    return manip_angle, manip_axis, manip_arm


def yaw_orient_diff(current, target):
    rot = R.from_quat(current)
    # target - current
    diff_rot = (R.from_quat(target) * R.from_quat(current).inv())
    yaw_diff = diff_rot.as_euler('xyz')[-1]

    return np.abs(yaw_diff)
