from random import randint


def smooth_path(path, extend, collision, iterations=200):
    """smooth a trajectory path, randomly replace jigged subpath with shortcuts

    Parameters
    ----------
    path : list
        [description]
    extend : function
        [description]
    collision : function
        [description]
    iterations : int, optional
        number of iterations for the random smoothing procedure, by default 50

    Returns
    -------
    [type]
        [description]
    """
    smoothed_path = path
    for _ in range(iterations):
        if len(smoothed_path) <= 2:
            return smoothed_path
        i = randint(0, len(smoothed_path) - 1)
        j = randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(smoothed_path[i], smoothed_path[j]))
        if (len(shortcut) < (j - i)) and all(not collision(q) for q in shortcut):
            smoothed_path = smoothed_path[:i + 1] + shortcut + smoothed_path[j + 1:]
    return smoothed_path

# TODO: sparsify path to just waypoints


def wholebody_smooth_path(path, joint_conf_path, extend, collision, ik, calc_tippos_fn, sample_joint_conf_fn, iterations=50):
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision
    smoothed_path = path
    smoothed_jconf_path = joint_conf_path
    for _ in range(iterations):
        if len(smoothed_path) <= 2:
            return smoothed_path

        i = randint(0, len(smoothed_path) - 1)
        j = randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue

        if j < i:
            i, j = j, i
        shortcut = list(extend(smoothed_path[i], smoothed_path[j]))

        failed = False
        shortcut_jconfs = []
        if len(shortcut) < (j - i):
            for cube_pose in shortcut:
                tip_positions = calc_tippos_fn(cube_pose)
                # Keep num_samples=1!!!!! This reallly slows down at larger values (and we only use the first one anyway)
                ik_sols = sample_multiple_ik_with_collision(ik, functools.partial(collision, cube_pose),
                                                            sample_joint_conf_fn, tip_positions, num_samples=1)
                if len(ik_sols) == 0:
                    failed = True
                    break
                shortcut_jconfs.append(ik_sols[0])

            if failed:
                continue

            print(f'shortcut is found!!: {i}:{j} {len(shortcut)}')
            smoothed_path = smoothed_path[:i + 1] + shortcut + smoothed_path[j + 1:]
            smoothed_jconf_path = smoothed_jconf_path[:i + 1] + shortcut_jconfs + smoothed_jconf_path[j + 1:]
    return smoothed_path, smoothed_jconf_path
