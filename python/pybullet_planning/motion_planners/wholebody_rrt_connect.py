import time
from random import random

from itertools import takewhile
from .smoothing import wholebody_smooth_path
from .rrt import TreeNode, configs, extract_ik_solutions
from .utils import irange, argmin, INCR_RRT_ITERATIONS, RRT_ITERATIONS, RRT_RESTARTS, RRT_SMOOTHING, INF, elapsed_time, negate

__all__ = [
    'wholebody_rrt_connect',
    'wholebody_birrt',
    'wholebody_direct_path',
    'wholebody_best_effort_rrt',
    'wholebody_best_effort_direct_path',
    'wholebody_rrt',
    'wholebody_incremental_rrt'
    ]


def asymmetric_extend(q1, q2, extend_fn, backward=False):
    """directional extend_fn
    """
    if backward:
        return reversed(list(extend_fn(q2, q1)))
    return extend_fn(q1, q2)


def wholebody_extend_towards(tree, target, distance_fn, extend_fn, collision_fn,
                             calc_tippos_fn, sample_joint_conf_fn, ik, swap,
                             tree_frequency, goal_test_fn=None):
    import functools
    import numpy as np
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision, sample_no_collision_ik
    last = argmin(lambda n: distance_fn(n.config, target), tree)
    # print('selected last:', str(last)[:40])
    extend = list(asymmetric_extend(last.config, target, extend_fn, swap))
    # safe = list(takewhile(negate(collision_fn), extend))
    safe = []

    # for each pose in extend, it checks if any IK solution exists that is not in collison.
    # while such solution exist, it keeps appending the cube_pose to 'safe'
    # once it reaches a point where no such solution exist, it exits the loop.
    ik_solutions = []
    at_goal = False
    for i, cube_pose in enumerate(extend):
        tip_positions = calc_tippos_fn(cube_pose)
        ik_sols = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, cube_pose),
                                                    sample_joint_conf_fn, tip_positions, num_samples=1)
        # NOTE: this doesn't work but Idk why..
        # ik_sols = sample_no_collision_ik(ik, functools.partial(collision_fn, cube_pose),
        #                                             sample_joint_conf_fn, tip_positions, n_samples=20)
        if len(ik_sols) == 0:
            break
        ik_solutions.append(ik_sols)
        safe.append(cube_pose)

        if goal_test_fn is not None and goal_test_fn(cube_pose):
            at_goal = True
            break

    # print('safe length', len(safe))  # DEBUG

    # add the sequence of safe nodes to the Tree.
    # each node in safe has a corresponding set of IK solutions.
    # We create nodes for each of those IK solutions, and regard them as in a same group.
    # for i, q in enumerate(safe):
    #     if (i % tree_frequency == 0) or (i == len(safe) - 1):
    #         # append node for every ik solution
    #         group = []
    #         for ik_sol in ik_solutions[i]:
    #             # find the argmin over ik solutions in the parent group
    #             ik_last = argmin(lambda node: distance_fn(node.ik_solution, ik_sol), last.group)
    #             last = TreeNode(q, parent=ik_last, ik_solution=ik_sol)
    #             group.append(last)
    #             tree.append(last)
    #
    #         # add the nodes in the same group to node.group
    #         for node in group:
    #             node.group = group
    for i, q in enumerate(safe):
        if (i % tree_frequency == 0) or (i == len(safe) - 1):
            # append node for every ik solution
            ik_sol = ik_solutions[i][0]
            last = TreeNode(q, parent=last, ik_solution=ik_sol)
            tree.append(last)
    if goal_test_fn:
        return last, at_goal
    else:
        success = len(extend) == len(safe)
        return last, success


def rrt_goal_sample(goal_test, end_pose, goal_threshold, ori_matters):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pybullet_planning.motion_planners.utils import weighted_position_error

    def _sample():
        pos_noise = (np.random.rand(3) - 0.5) * goal_threshold
        if weighted_position_error(pos_noise) > goal_threshold:
            pos_noise = goal_threshold * pos_noise / np.linalg.norm(pos_noise)
        pos = end_pose[:3] + pos_noise
        axis = np.random.rand(3) - 0.5
        axis /= np.linalg.norm(axis)
        if ori_matters:
            angle = np.pi * np.random.rand() * goal_threshold
            rot_noise = Rotation.from_rotvec(angle * axis)
            rot = Rotation.from_euler('zyx', end_pose[3:])
            ori = (rot_noise * rot).as_euler('zyx')
        else:
            angle = np.pi * np.random.rand()
            ori = Rotation.from_rotvec(angle * axis).as_euler('zyx')
        return np.concatenate([pos, ori])
    s = _sample()
    it = 0
    max_it = 100
    while not goal_test(s) and it < max_it:
        s = _sample()
        it += 1
    if it == max_it:
        return end_pose
    return s


def wholebody_rrt(q1, q2, goal_sample_fn, goal_test, distance_fn, sample_fn,
                  extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn,
                  ik, goal_probability=0.66, iterations=RRT_ITERATIONS,
                  tree_frequency=1, max_time=INF, restarts=RRT_RESTARTS,
                  smoothing=RRT_SMOOTHING):

    start_time = time.time()
    path, joint_path = __wholebody_direct_path(q1, q2, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik)
    if path is not None:
        return path, joint_path

    for _ in range(restarts):
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_time:
            return None, None
        path, joint_path, _ = _rrt(q1, goal_sample_fn, goal_test, distance_fn,
                                sample_fn, extend_fn, collision_fn,
                                calc_tippos_fn, sample_joint_conf_fn, ik,
                                goal_probability, iterations, tree_frequency,
                                max_time=max_time - elapsed_time)
        if path is not None:
            if smoothing is None:
                return path, joint_path
            return wholebody_smooth_path(path, joint_path, extend_fn,
                                         collision_fn, ik, calc_tippos_fn,
                                         sample_joint_conf_fn,
                                         iterations=smoothing)
    return None, None


def wholebody_incremental_rrt(q1, q2, use_ori, distance_fn, sample_fn,
                              extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn,
                              ik, goal_probability=0.66, min_goal_threshold=0.0, max_goal_threshold=0.6,
                              iterations=INCR_RRT_ITERATIONS, tree_frequency=1, max_time=INF, restarts=RRT_RESTARTS,
                              n_goal_sets=20,
                              smoothing=RRT_SMOOTHING, **kwargs):
    # additional params:
    end_conf = q2
    # print('-------------------------------')
    # print('max_goal_threshold', max_goal_threshold)
    # print('min_goal_threshold', min_goal_threshold)
    # print('goal_probability', goal_probability)
    # print('restarts', restarts)
    # print('rrt_iterations', iterations)
    # print('-------------------------------')

    def get_goal_test_fn(end_conf, goal_threshold):
        def goal_test_fn(q):
            d = distance_fn(end_conf, q)
            return d <= goal_threshold
        return goal_test_fn

    def get_goal_sample_fn(goal_test, end_conf, use_ori, goal_threshold):
        def goal_sample_fn():
            sample = rrt_goal_sample(goal_test, end_conf, goal_threshold, use_ori)
            return sample
        return goal_sample_fn

    start_time = time.time()

    # check if there's a direct path
    path, joint_path = __wholebody_direct_path(q1, q2, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik)
    if path is not None:
        return path, joint_path

    preserved_tree = []
    prev_goal_threshold = max_goal_threshold
    for i in range(n_goal_sets):
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_time:
            return None, None
        goal_threshold = ((1 - i / n_goal_sets)
                          * (max_goal_threshold - min_goal_threshold)
                          + min_goal_threshold)
        print('trying goal threshold', goal_threshold)
        goal_test = get_goal_test_fn(end_conf, goal_threshold)
        goal_sample_fn = get_goal_sample_fn(goal_test, end_conf, use_ori, goal_threshold)

        path, joint_path, preserved_tree = _rrt(q1, goal_sample_fn,
                                                goal_test,
                                                distance_fn,
                                                sample_fn, extend_fn, collision_fn,
                                                calc_tippos_fn, sample_joint_conf_fn, ik,
                                                goal_probability, iterations, tree_frequency,
                                                max_time=max_time - elapsed_time, preserved_tree=preserved_tree, **kwargs)
        if path is None:
            if i == 0:
                break
            else:
                print('** [done] path is found with goal_threshold:', prev_goal_threshold)
                return wholebody_smooth_path(prev_path, prev_joint_path, extend_fn,
                                            collision_fn, ik, calc_tippos_fn,
                                            sample_joint_conf_fn,
                                             iterations=smoothing)

        print('path is found with goal_threshold:', goal_threshold)
        prev_goal_threshold = goal_threshold
        prev_path, prev_joint_path = path, joint_path

    # NOTE: Very rare, but there's a case that does not have a direct path but can achieve zero error!
    if path is not None:
        print('** [done] path is found with goal_threshold:', prev_goal_threshold)
        return wholebody_smooth_path(prev_path, prev_joint_path, extend_fn,
                                    collision_fn, ik, calc_tippos_fn,
                                    sample_joint_conf_fn,
                                        iterations=smoothing)

    return None, None


def _rrt(q1, goal_sample_fn, goal_test, distance_fn, sample_fn,
         extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn,
         ik, goal_probability=0.2,
         iterations=RRT_ITERATIONS, tree_frequency=1, max_time=INF,
         preserved_tree=[]):
    """[summary]

    Parameters
    ----------
    q1 : [type]
        [description]
    q2 : [type]
        [description]
    distance_fn : [type]
        [description]
    sample_fn : [type]
        [description]
    extend_fn : [type]
        [description]
    collision_fn : [type]
        [description]
    iterations : [type], optional
        [description], by default RRT_ITERATIONS
    tree_frequency : int, optional
        the frequency of adding tree nodes when extending. For example, if tree_freq=2, then a tree node is added every three nodes,
        by default 1
    max_time : [type], optional
        [description], by default INF

    Returns
    -------
    [type]
        [description]
    """
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision

    start_time = time.time()
    assert tree_frequency >= 1

    # create a node for each configuration
    tip_positions1 = calc_tippos_fn(q1)
    # NOTE: very rare, but num_samples=1 fails to get a solution, and that is critical for initial configuration.
    ik_solutions1 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q1),
                                                      sample_joint_conf_fn, tip_positions1, num_samples=1)

    if len(ik_solutions1) == 0:
        print('No valid IK solution for Initial configuration found')
        return None, None, None

    endpose_in_collision = True
    for _ in range(10):
        q2 = goal_sample_fn()
        ik_sol_q2 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q1),
                                                      sample_joint_conf_fn, tip_positions1, num_samples=1)
        if len(ik_sol_q2) > 0:
            endpose_in_collision = False
            break
    if endpose_in_collision:
        print('No valid IK solution for end configuration found')
        return None, None, None

    ik_sol1 = ik_solutions1[0]

    if not callable(goal_sample_fn):
        g = goal_sample_fn
        goal_sample_fn = lambda: g

    if len(preserved_tree) > 0:
        nodes = preserved_tree
    else:
        nodes = [TreeNode(q1, ik_solution=ik_sol1)]
    # print('iterations', iterations)
    for iteration in irange(iterations):
        if max_time <= elapsed_time(start_time):
            break

        goal = random() < goal_probability or iteration == 0
        s = goal_sample_fn() if goal else sample_fn()
        last, success = wholebody_extend_towards(nodes, s, distance_fn, extend_fn,
                                                 collision_fn, calc_tippos_fn,
                                                 sample_joint_conf_fn, ik,
                                                 False, tree_frequency,
                                                 goal_test)

        # print('sequence length:', len(last.retrace()))
        if success:
            path, joint_conf_path = last.retrace_all()
            return configs(path), joint_conf_path, nodes
    return None, None, None


def wholebody_rrt_connect(q1, q2, init_joint_conf, end_joint_conf, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                          iterations=RRT_ITERATIONS, tree_frequency=1, max_time=INF):
    """[summary]

    Parameters
    ----------
    q1 : [type]
        [description]
    q2 : [type]
        [description]
    distance_fn : [type]
        [description]
    sample_fn : [type]
        [description]
    extend_fn : [type]
        [description]
    collision_fn : [type]
        [description]
    iterations : [type], optional
        [description], by default RRT_ITERATIONS
    tree_frequency : int, optional
        the frequency of adding tree nodes when extending. For example, if tree_freq=2, then a tree node is added every three nodes,
        by default 1
    max_time : [type], optional
        [description], by default INF

    Returns
    -------
    [type]
        [description]
    """
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision

    start_time = time.time()
    assert tree_frequency >= 1

    # create a node for each configuration
    tip_positions1 = calc_tippos_fn(q1)
    # ik_solutions1 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q1),
    #                                                   sample_joint_conf_fn, tip_positions1, num_samples=1)

    tip_positions2 = calc_tippos_fn(q2)
    # ik_solutions2 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q2),
    #                                                   sample_joint_conf_fn, tip_positions2, num_samples=1)

    nodes1, nodes2 = [TreeNode(q1, ik_solution=init_joint_conf)], [TreeNode(q2, ik_solution=end_joint_conf)]
    for iteration in irange(iterations):
        if max_time <= elapsed_time(start_time):
            break
        swap = len(nodes1) > len(nodes2)
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1
        sample = sample_fn()
        last1, _ = wholebody_extend_towards(tree1, sample_fn(), distance_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                                  swap, tree_frequency)
        last2, success = wholebody_extend_towards(tree2, last1.config, distance_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                                        not swap, tree_frequency)

        if success:
             path1, joint_conf_path1 = last1.retrace_all()
             path2, joint_conf_path2 = last2.retrace_all()
             if swap:
                 path1, path2 = path2, path1
                 joint_conf_path1, joint_conf_path2 = joint_conf_path2, joint_conf_path1
             entire_path = path1[:-1] + path2[::-1]
             return configs(entire_path), extract_ik_solutions(entire_path)
    return None, None

def __wholebody_direct_path(q1, q2, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik):
    """
    I know it's confusing to have 2 functions both named wholebody_direct_path...
    The difference is very tiny:
    This one simply calculates joint configurations for q1 and q2,
    and the other one doesn't but expects init_joint_conf and end_joint_conf as its input.
    """
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision

    # collision and IK check on initial and end configuration
    tip_positions1 = calc_tippos_fn(q1)
    ik_solutions1 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q1, diagnosis=False),
                                                      sample_joint_conf_fn, tip_positions1, num_samples=1)
    if len(ik_solutions1) == 0:
        # print('Initial Configuration in collision')
        return None, None

    tip_positions2 = calc_tippos_fn(q2)
    ik_solutions2 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q2, diagnosis=False),
                                                      sample_joint_conf_fn, tip_positions2, num_samples=1)
    if len(ik_solutions2) == 0:
        # print('End Configuration in collision')
        return None, None

    ik_sol1 = ik_solutions1[0]
    ik_sol2 = ik_solutions2[0]

    path, joint_conf_path = [], []
    for q in extend_fn(q1, q2):
        tip_positions = calc_tippos_fn(q)
        ik_solutions = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q, diagnosis=False),
                                                         sample_joint_conf_fn, tip_positions, num_samples=1)
        if len(ik_solutions) == 0:  # or collision_fn(q, joint_conf=ik_solutions[0]):
            return None, None

        path.append(q)
        joint_conf_path.append(ik_solutions[0])
    print('DIRECT PATH is found!!')
    return path, joint_conf_path


def wholebody_direct_path(q1, q2, init_joint_conf, end_joint_conf, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik, **kwargs):
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision
    # TEMP
    # if collision_fn(q1) or collision_fn(q2):
    #     return None
    path = [q1]
    joint_conf_path = [init_joint_conf]

    for q in extend_fn(q1, q2):
        tip_positions = calc_tippos_fn(q)
        if (q == q2).all():
            ik_solutions = [end_joint_conf]
        else:
            ik_solutions = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q, diagnosis=False),
                                                            sample_joint_conf_fn, tip_positions, num_samples=1)
        if len(ik_solutions) == 0:
            return None, None
        else:
            ik_solution = ik_solutions[0]
            if collision_fn(q, joint_conf=ik_solution):
                return None, None

        path.append(q)
        joint_conf_path.append(ik_solution)
    print('DIRECT PATH is found!!')
    return path, joint_conf_path


def wholebody_birrt(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                    init_joint_conf=None, restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING, max_time=INF, **kwargs):
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision

    # collision and IK check on initial and end configuration
    tip_positions1 = calc_tippos_fn(q1)
    if init_joint_conf is None:
        ik_solutions1 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q1),
                                                        sample_joint_conf_fn, tip_positions1, num_samples=1)
    else:
        ik_solutions1 = [] if collision_fn(q1, init_joint_conf) else [init_joint_conf]

    if len(ik_solutions1) == 0:
        print('Initial Configuration in collision')
        return None, None

    tip_positions2 = calc_tippos_fn(q2)
    ik_solutions2 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q2),
                                                      sample_joint_conf_fn, tip_positions2, num_samples=1)
    if len(ik_solutions2) == 0:
        print('End Configuration in collision')
        return None, None

    ik_sol1 = ik_solutions1[0]
    ik_sol2 = ik_solutions2[0]

    start_time = time.time()
    path, joint_conf_path = wholebody_direct_path(q1, q2, ik_sol1, ik_sol2, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik, **kwargs)
    if path is not None:
        return path, joint_conf_path

    for _ in irange(restarts + 1):
        if max_time <= elapsed_time(start_time):
            break
        path, joint_conf_path = wholebody_rrt_connect(q1, q2, ik_sol1, ik_sol2, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                                                      max_time=max_time - elapsed_time(start_time), **kwargs)
        if path is not None:
            if smooth is None:
                return path, joint_conf_path
            return wholebody_smooth_path(path, joint_conf_path, extend_fn, collision_fn, ik, calc_tippos_fn, sample_joint_conf_fn, iterations=smooth)
    return None, None


def wholebody_best_effort_rrt(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                  goal_sample_fn=None, reward_dist_fn=None, n_goal_samples=100,
                  init_joint_conf=None, restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING, iterations=RRT_ITERATIONS, max_time=INF, **kwargs):
    import functools
    import numpy as np
    from random import random
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision

    if goal_sample_fn is None:
        goal_sample_fn = sample_fn

    # collision and IK check on initial and end configuration
    tip_positions1 = calc_tippos_fn(q1)
    if init_joint_conf is None:
        ik_solutions1 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q1, diagnosis=False),
                                                          sample_joint_conf_fn, tip_positions1, num_samples=1)
    else:
        ik_solutions1 = [] if collision_fn(q1, init_joint_conf) else [init_joint_conf]

    if len(ik_solutions1) == 0:
        print('Initial Configuration in collision')
        return None, None
    ik_sol1 = ik_solutions1[0]

    # Check if q2 is feasible
    tip_positions2 = calc_tippos_fn(q2)
    ik_solutions2 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q2, diagnosis=False),
                                                      sample_joint_conf_fn, tip_positions2, num_samples=1)

    # q2 is feasible! we just run wholebody_birrt.
    if len(ik_solutions2) > 0:
        return wholebody_birrt(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                               init_joint_conf=init_joint_conf, restarts=restarts, smooth=smooth, max_time=max_time, **kwargs)

    # sample from goal_sample_fn() and check if the pose is feasible
    candidates = sample_feasible_goals(goal_sample_fn, ik, collision_fn, sample_joint_conf_fn, calc_tippos_fn, n_goal_samples=100)

    # search for the end pose that will have the highest reward
    new_goal, goal_joint_conf = argmin(lambda cand: reward_dist_fn(cand[0], q2), candidates)
    new_goal = np.asarray(new_goal)

    return wholebody_birrt(q1, new_goal, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                           init_joint_conf=init_joint_conf, restarts=restarts, smooth=smooth, max_time=max_time, **kwargs)



def wholebody_best_effort_direct_path(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                                      goal_sample_fn=None, reward_dist_fn=None, n_goal_samples=100, n_trials=10,
                                      init_joint_conf=None, restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING, iterations=RRT_ITERATIONS, max_time=INF, **kwargs):
    import functools
    import numpy as np
    from random import random
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision

    if goal_sample_fn is None:
        goal_sample_fn = sample_fn

    # collision and IK check on initial and end configuration
    tip_positions1 = calc_tippos_fn(q1)
    if init_joint_conf is None:
        ik_solutions1 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q1, diagnosis=False),
                                                          sample_joint_conf_fn, tip_positions1, num_samples=1)
    else:
        ik_solutions1 = [] if collision_fn(q1, init_joint_conf) else [init_joint_conf]

    if len(ik_solutions1) == 0:
        print('Initial Configuration in collision')
        return None, None
    ik_sol1 = ik_solutions1[0]

    # Check if q2 is feasible
    tip_positions2 = calc_tippos_fn(q2)
    ik_solutions2 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q2, diagnosis=False),
                                                      sample_joint_conf_fn, tip_positions2, num_samples=1)

    # q2 is feasible! we just run wholebody_birrt.
    if len(ik_solutions2) > 0:
        return wholebody_birrt(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                               init_joint_conf=init_joint_conf, restarts=restarts, smooth=smooth, max_time=max_time, **kwargs)

    # sample from goal_sample_fn() and check if the pose is feasible
    for j in range(n_trials):
        candidates = sample_feasible_goals(goal_sample_fn, ik, collision_fn, sample_joint_conf_fn, calc_tippos_fn)

        # search for the end pose that will have the highest reward
        new_goal, goal_joint_conf = argmin(lambda cand: reward_dist_fn(cand[0], q2), candidates)
        new_goal = np.asarray(new_goal)

        # Obtain the end pose
        tip_positions2 = calc_tippos_fn(new_goal)
        ik_solutions2 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, new_goal, diagnosis=False),
                                                          sample_joint_conf_fn, tip_positions2, num_samples=1)
        ik_sol2 = ik_solutions2[0]
        path, joint_conf_path = wholebody_direct_path(q1, new_goal, ik_sol1, ik_sol2, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik, **kwargs)

        if path is None:
            print('Direct path is not found. Sampling another goal...')
            continue

        return path, joint_conf_path


def sample_feasible_goals(goal_sample_fn, ik, collision_fn, sample_joint_conf_fn, calc_tippos_fn, n_goal_samples=100):
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision
    candidates = []
    for i in range(n_goal_samples):
        sampled_goal = goal_sample_fn()
        tip_positions = calc_tippos_fn(sampled_goal)
        ik_solutions = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, sampled_goal, diagnosis=False),
                                                        sample_joint_conf_fn, tip_positions, num_samples=1)
        if len(ik_solutions) > 0:
            candidates.append((sampled_goal, ik_solutions[0]))

    if len(candidates) == 0:
        raise ValueError('No feasible goal is sampled.')
    return candidates
