import random
import numpy as np
from itertools import product

from pybullet_planning.utils import CIRCULAR_LIMITS, DEFAULT_RESOLUTION, MAX_DISTANCE
from pybullet_planning.interfaces.env_manager.pose_transformation import circular_difference, get_unit_vector

from pybullet_planning.interfaces.env_manager.user_io import wait_for_user
from pybullet_planning.interfaces.debug_utils import add_line
from pybullet_planning.interfaces.robots.joint import get_custom_limits, get_joint_positions
from pybullet_planning.interfaces.robots.collision import get_collision_fn, get_cube_tip_collision_fn

from pybullet_planning.motion_planners import birrt, wholebody_rrt, wholebody_incremental_rrt, wholebody_best_effort_direct_path, wholebody_birrt, lazy_prm, direct_path
from pybullet_planning.interfaces.planner_interface.utils import preserve_pos_and_vel, preserve_pos_and_vel_wholebody
import pybullet as p

#####################################

# Joint motion planning

def uniform_generator(d):
    while True:
        yield np.random.uniform(size=d)

def halton_generator(d):
    import ghalton
    seed = random.randint(0, 1000)
    #sequencer = ghalton.Halton(d)
    sequencer = ghalton.GeneralizedHalton(d, seed)
    #sequencer.reset()
    while True:
        [weights] = sequencer.get(1)
        yield np.array(weights)

def unit_generator(d, use_halton=False):
    if use_halton:
        try:
            import ghalton
        except ImportError:
            print('ghalton is not installed (https://pypi.org/project/ghalton/)')
            use_halton = False
    return halton_generator(d) if use_halton else uniform_generator(d)

def interval_generator(lower, upper, **kwargs):
    assert len(lower) == len(upper)
    assert np.less(lower, upper).all() # TODO: equality
    extents = np.array(upper) - np.array(lower)
    for scale in unit_generator(len(lower), **kwargs):
        point = np.array(lower) + scale*extents
        yield point

def get_sample_fn(body, joints, custom_limits={}, **kwargs):
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits, circular_limits=CIRCULAR_LIMITS)
    generator = interval_generator(lower_limits, upper_limits, **kwargs)
    def fn():
        return tuple(next(generator))
    return fn

def get_halton_sample_fn(body, joints, **kwargs):
    return get_sample_fn(body, joints, use_halton=True, **kwargs)

def get_difference_fn(body, joints):
    from pybullet_planning.interfaces.robots.joint import is_circular
    circular_joints = [is_circular(body, joint) for joint in joints]

    def fn(q2, q1):
        return tuple(circular_difference(value2, value1) if circular else (value2 - value1)
                     for circular, value2, value1 in zip(circular_joints, q2, q1))
    return fn

def get_distance_fn(body, joints, weights=None, weight_fn=None): #, norm=2):
    # TODO: use the energy resulting from the mass matrix here?
    if weights is None:
        weights = 1*np.ones(len(joints)) # TODO: use velocities here
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        if weight_fn is not None:
            return weight_fn(diff)
        return np.sqrt(np.dot(weights, diff * diff))
        #return np.linalg.norm(np.multiply(weights * diff), ord=norm)
    return fn

def get_paired_distance_fn(body, joints, weight_fn):
    '''This calls weighted_pose calc function with 2 inputs'''
    from pybullet_planning.motion_planners.utils import weighted_paired_pose_error
    def fn(q1, q2):
        return weight_fn(q1, q2)
    return fn

def get_refine_fn(body, joints, num_steps=0):
    difference_fn = get_difference_fn(body, joints)
    num_steps = num_steps + 1
    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            positions = (1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            q = tuple(positions)
            #q = tuple(wrap_positions(body, joints, positions))
            yield q
    return fn

def refine_path(body, joints, waypoints, num_steps):
    refine_fn = get_refine_fn(body, joints, num_steps)
    refined_path = []
    for v1, v2 in zip(waypoints, waypoints[1:]):
        refined_path += list(refine_fn(v1, v2))
    return refined_path

def get_extend_fn(body, joints, resolutions=None, norm=2):
    # norm = 1, 2, INF
    if resolutions is None:
        resolutions = DEFAULT_RESOLUTION*np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        #steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions))))
        steps = int(np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm))
        refine_fn = get_refine_fn(body, joints, num_steps=steps)
        return refine_fn(q1, q2)
    return fn

def get_calc_tippos_fn(current_tip_pos, cube_tip_pos, cube_pos, cube_qt):
    from pybullet_planning.utils.transformations import apply_transform, assign_positions_to_fingers
    # target_tip_positions = apply_transform(cube_pos, cube_qt, cube_tip_pos)
    # target_tip_positions, tip_assignments = assign_positions_to_fingers(current_tip_pos, target_tip_positions)
    # sorted_cube_tip_positions = cube_tip_pos[tip_assignments, :]

    def fn(cube_pose):
        cube_pos = cube_pose[:3]
        cube_qt = p.getQuaternionFromEuler(cube_pose[3:])
        target_tip_positions = apply_transform(cube_pos, cube_qt, cube_tip_pos)
        return target_tip_positions
    return fn


def remove_redundant(path, tolerance=1e-3):
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        difference = np.array(new_path[-1]) - np.array(conf)
        if not np.allclose(np.zeros(len(difference)), difference, atol=tolerance, rtol=0):
            new_path.append(conf)
    return new_path

def waypoints_from_path(path, tolerance=1e-3):
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path
    difference_fn = lambda q2, q1: np.array(q2) - np.array(q1)
    #difference_fn = get_difference_fn(body, joints)

    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=tolerance, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints

def adjust_path(robot, joints, path):
    start_positions = get_joint_positions(robot, joints)
    difference_fn = get_difference_fn(robot, joints)
    differences = [difference_fn(q2, q1) for q1, q2 in zip(path, path[1:])]
    adjusted_path = [np.array(start_positions)]
    for difference in differences:
        adjusted_path.append(adjusted_path[-1] + difference)
    return adjusted_path


def plan_waypoints_joint_motion(body, joints, waypoints, start_conf=None, obstacles=[], attachments=[],
                                self_collisions=True, disabled_collisions=set(),
                                resolutions=None, custom_limits={}, max_distance=MAX_DISTANCE):
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance)
    if start_conf is None:
        start_conf = get_joint_positions(body, joints)
    else:
        assert len(start_conf) == len(joints)

    for i, waypoint in enumerate([start_conf] + list(waypoints)):
        if collision_fn(waypoint):
            #print("Warning: waypoint configuration {}/{} is in collision".format(i, len(waypoints)))
            return None
    path = [start_conf]
    for waypoint in waypoints:
        assert len(joints) == len(waypoint)
        for q in extend_fn(path[-1], waypoint):
            if collision_fn(q):
                return None
            path.append(q)
    return path

# def plan_direct_joint_motion(body, joints, end_conf, **kwargs):
#     """plan a joint trajectory connecting the robot's current conf to the end_conf

#     Parameters
#     ----------
#     body : [type]
#         [description]
#     joints : [type]
#         [description]
#     end_conf : [type]
#         [description]

#     Returns
#     -------
#     [type]
#         [description]
#     """
#     return plan_waypoints_joint_motion(body, joints, [end_conf], **kwargs)

def check_initial_end(start_conf, end_conf, collision_fn, diagnosis=False):
    if collision_fn(start_conf, diagnosis):
        # print("Warning: initial configuration is in collision")
        return False
    if collision_fn(end_conf, diagnosis):
        # print("Warning: end configuration is in collision")
        return False
    return True


@preserve_pos_and_vel
def plan_joint_motion(body, joints, end_conf, obstacles=[], attachments=[],
                      self_collisions=True, disabled_collisions=set(), extra_disabled_collisions=set(),
                      weights=None, resolutions=None, max_distance=MAX_DISTANCE, ignore_collision_steps=0, custom_limits={}, diagnosis=False,
                      constraint_fn=None, max_dist_on=None, **kwargs):
    """call birrt to plan a joint trajectory from the robot's **current** conf to ``end_conf``.
    """
    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions,
                                    disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance, max_dist_on=max_dist_on)
    if constraint_fn is not None:
        old_collision_fn = collision_fn

        def _collision_and_constraint(q, diagnosis=False):
            return old_collision_fn(q, diagnosis) or constraint_fn(q, diagnosis)
        collision_fn = _collision_and_constraint

    start_conf = get_joint_positions(body, joints)

    if ignore_collision_steps == 0 and not check_initial_end(start_conf, end_conf, collision_fn, diagnosis=diagnosis):
        return None
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, ignore_collision_steps=ignore_collision_steps, **kwargs)
    #return plan_lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn)


@preserve_pos_and_vel
def plan_direct_joint_motion(body, joints, end_conf, obstacles=[], attachments=[],
                             self_collisions=True, disabled_collisions=set(), extra_disabled_collisions=set(),
                             weights=None, resolutions=None, max_distance=MAX_DISTANCE, ignore_collision_steps=0, custom_limits={}, diagnosis=False,
                             constraint_fn=None, max_dist_on=None, **kwargs):
    assert len(joints) == len(end_conf)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions,
                                    disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance, max_dist_on=max_dist_on)
    if constraint_fn is not None:
        old_collision_fn = collision_fn

        def _collision_and_constraint(q, diagnosis=False):
            return old_collision_fn(q, diagnosis) or constraint_fn(q, diagnosis)
        collision_fn = _collision_and_constraint

    start_conf = get_joint_positions(body, joints)

    if ignore_collision_steps == 0 and not check_initial_end(start_conf, end_conf, collision_fn, diagnosis=diagnosis):
        return None
    return direct_path(start_conf, end_conf, extend_fn, collision_fn, ignore_collision_steps=ignore_collision_steps)


@preserve_pos_and_vel_wholebody
def plan_wholebody_motion(cube_body, joints, finger_body, finger_joints, end_conf, current_tip_pos, cube_tip_pos, ik, obstacles=[], attachments=[],
                          init_joint_conf=None, disabled_collisions=set(), weights=None, resolutions=None, max_distance=MAX_DISTANCE, custom_limits={}, diagnosis=False,
                          vis_fn=None, use_rrt=False, use_incremental_rrt=False, use_ori=False, goal_threshold=0.1, additional_collision_fn=None, **kwargs):
    from pybullet_planning.interfaces.env_manager.pose_transformation import get_pose
    from pybullet_planning.motion_planners.utils import weighted_pose_error, weighted_position_error, weighted_paired_pose_error
    assert len(joints) == len(end_conf)

    assert additional_collision_fn is None or callable(additional_collision_fn)

    def collision_fn(*args, **kwargs):
        org_collision_fn = get_cube_tip_collision_fn(cube_body, finger_body, finger_joints, obstacles=obstacles, attachments=attachments,
                                                     disabled_collisions=disabled_collisions, diagnosis=diagnosis, vis_fn=vis_fn, max_distance=max_distance)
        if additional_collision_fn is None:
            return org_collision_fn(*args, **kwargs)
        return org_collision_fn(*args, **kwargs) or additional_collision_fn(*args, **kwargs)

    sample_fn = get_sample_fn(cube_body, joints, custom_limits=custom_limits)
    sample_joint_conf_fn = get_sample_fn(finger_body, finger_joints, custom_limits={})
    if use_ori:
        weight_fn = weighted_paired_pose_error
        distance_fn = get_paired_distance_fn(cube_body, joints, weight_fn=weight_fn)
    else:
        weight_fn = weighted_position_error
        distance_fn = get_distance_fn(cube_body, joints, weights=None, weight_fn=weight_fn)
    extend_fn = get_extend_fn(cube_body, joints, resolutions=resolutions)


    # obtain start configuration
    start_pos, start_quat = get_pose(cube_body)
    start_ori = p.getEulerFromQuaternion(start_quat)
    start_conf = np.concatenate([start_pos, start_ori]).reshape(-1)
    calc_tippos_fn = get_calc_tippos_fn(current_tip_pos, cube_tip_pos, start_pos, start_quat)

    if use_rrt:
        def goal_test(q):
            d = distance_fn(end_conf, q)
            return d <= goal_threshold

        from pybullet_planning.motion_planners.wholebody_rrt_connect import rrt_goal_sample
        def sample_goal():
            sample = rrt_goal_sample(goal_test, end_conf, goal_threshold, use_ori)
            return sample
        return wholebody_rrt(start_conf, end_conf, sample_goal, goal_test, distance_fn,
                             sample_fn, extend_fn, collision_fn, calc_tippos_fn,
                             sample_joint_conf_fn, ik, **kwargs)
    elif use_incremental_rrt:
        from pybullet_planning.motion_planners.wholebody_rrt_connect import rrt_goal_sample
        return wholebody_incremental_rrt(start_conf, end_conf, use_ori, distance_fn,
                                         sample_fn, extend_fn, collision_fn, calc_tippos_fn,
                                         sample_joint_conf_fn, ik, **kwargs)
    else:
        return wholebody_birrt(
            start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn,
            calc_tippos_fn, sample_joint_conf_fn, ik,
            init_joint_conf=init_joint_conf, **kwargs
        )


def plan_lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn, **kwargs):
    # TODO: cost metric based on total robot movement (encouraging greater distances possibly)
    path, samples, edges, colliding_vertices, colliding_edges = lazy_prm(
        start_conf, end_conf, sample_fn, extend_fn, collision_fn, num_samples=200, **kwargs)
    if path is None:
        return path

    #lower, upper = get_custom_limits(body, joints, circular_limits=CIRCULAR_LIMITS)
    def draw_fn(q): # TODO: draw edges instead of vertices
        return np.append(q[:2], [1e-3])
        #return np.array([1, 1, 0.25])*(q + np.array([0., 0., np.pi]))
    handles = []
    for q1, q2 in zip(path, path[1:]):
        handles.append(add_line(draw_fn(q1), draw_fn(q2), color=(0, 1, 0)))
    for i1, i2 in edges:
        color = (0, 0, 1)
        if any(colliding_vertices.get(i, False) for i in (i1, i2)) or colliding_vertices.get((i1, i2), False):
            color = (1, 0, 0)
        elif not colliding_vertices.get((i1, i2), True):
            color = (0, 0, 0)
        handles.append(add_line(draw_fn(samples[i1]), draw_fn(samples[i2]), color=color))
    wait_for_user()
    return path
