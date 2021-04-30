import time

from itertools import takewhile
from .smoothing import smooth_path
from .rrt import TreeNode, configs
from .utils import irange, argmin, RRT_ITERATIONS, RRT_RESTARTS, RRT_SMOOTHING, INF, elapsed_time, negate

__all__ = [
    'rrt_connect',
    'birrt',
    'direct_path',
    ]

def asymmetric_extend(q1, q2, extend_fn, backward=False):
    """directional extend_fn
    """
    if backward:
        return reversed(list(extend_fn(q2, q1)))
    return extend_fn(q1, q2)

def extend_towards(tree, target, distance_fn, extend_fn, collision_fn, swap, tree_frequency, ignore_collision_steps=0, target_node_length=None):
    last = argmin(lambda n: distance_fn(n.config, target), tree)
    extend = list(asymmetric_extend(last.config, target, extend_fn, swap))
    tolerable_col_steps = max(0, ignore_collision_steps - target_node_length) if target_node_length is not None else 0
    col_count = 0
    maybe_safe = []
    safe = []
    for i, q in enumerate(extend):
        path_length = len(last.retrace()) + i + 1  # path lengths to reach the node in this tree
        if collision_fn(q):
            if path_length <= ignore_collision_steps:
                maybe_safe.append(q)
            elif col_count < tolerable_col_steps:  # experienced collisions, but not too many steps passed after that
                maybe_safe.append(q)
            else:
                break
            col_count += 1
        else:
            if len(maybe_safe) == len(safe):  # the path has been safe so far.
                safe.append(q)
            maybe_safe.append(q)

            if col_count > 0:  # if it experienced collision even once, the counter will be ticked every step after that
                col_count += 1

    success = len(extend) == len(maybe_safe)  # NOTE: maybe_safe was actually safe!!
    if success:
        safe = maybe_safe  # We had some collisions but reached to the other tree within ignore_collision_steps!!
    # safe = list(takewhile(negate(collision_fn), extend))
    for i, q in enumerate(safe):
        if (i % tree_frequency == 0) or (i == len(safe) - 1):
            last = TreeNode(q, parent=last)
            tree.append(last)
    return last, success

def rrt_connect(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn,
                iterations=RRT_ITERATIONS, tree_frequency=1, max_time=INF, ignore_collision_steps=0, **kwargs):
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
    # TODO: collision(q1, q2)
    start_time = time.time()
    assert tree_frequency >= 1
    if ignore_collision_steps == 0 and collision_fn(q1):
        return None
    if ignore_collision_steps == 0 and collision_fn(q2):
        return None

    nodes1, nodes2 = [TreeNode(q1)], [TreeNode(q2)]
    for iteration in irange(iterations):
        if max_time <= elapsed_time(start_time):
            break
        swap = len(nodes1) > len(nodes2)
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1
        # NOTE: at this point, tree 1 always has less nodes inside
        # This means that Tree 1 has possibly a large obstacle close to its leaves and cannot expand easily
        # Thus, as you can see from the below 2 lines, tree1 always grow toward a sampled point
        # whereas tree2 just expands toward the last added node in tree1

        last1, _ = extend_towards(tree1, sample_fn(), distance_fn, extend_fn, collision_fn,
                                  swap, tree_frequency, ignore_collision_steps=ignore_collision_steps)
        last2, success = extend_towards(tree2, last1.config, distance_fn, extend_fn, collision_fn,
                                        not swap, tree_frequency, ignore_collision_steps=ignore_collision_steps, target_node_length=len(last1.retrace()))

        if success:
            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            #print('{} iterations, {} nodes'.format(iteration, len(nodes1) + len(nodes2)))
            return configs(path1[:-1] + path2[::-1])
    return None

# TODO: version which checks whether the segment is valid

def direct_path(q1, q2, extend_fn, collision_fn, ignore_collision_steps=0):
    if collision_fn(q1) or collision_fn(q2):
        return None
    path = [q1]
    sequence = list(extend_fn(q1, q2))
    for i, q in enumerate(sequence):
        if ignore_collision_steps - 1 < i < len(sequence) - ignore_collision_steps \
           and collision_fn(q):
            return None
        path.append(q)
    return path


def birrt(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn,
          restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING, max_time=INF, ignore_collision_steps=0, **kwargs):
    """birrt [summary]

    TODO: add citation to the algorithm.
    See `pybullet_planning.interfaces.planner_interface.joint_motion_planning.plan_joint_motion` for an example
    of standard usage.

    Parameters
    ----------
    q1 : [type]
        [description]
    q2 : [type]
        [description]
    distance_fn : [type]
        see `pybullet_planning.interfaces.planner_interface.joint_motion_planning.get_difference_fn` for an example
    sample_fn : function handle
        configuration space sampler
        see `pybullet_planning.interfaces.planner_interface.joint_motion_planning.get_sample_fn` for an example
    extend_fn : function handle
        see `pybullet_planning.interfaces.planner_interface.joint_motion_planning.get_extend_fn` for an example
    collision_fn : function handle
        collision checking function
        see `pybullet_planning.interfaces.robots.collision.get_collision_fn` for an example
    restarts : int, optional
        [description], by default RRT_RESTARTS
    iterations : int, optional
        [description], by default RRT_ITERATIONS
    smooth : int, optional
        smoothing iterations, by default RRT_SMOOTHING

    Returns
    -------
    [type]
        [description]
    """
    start_time = time.time()
    if ignore_collision_steps == 0 and collision_fn(q1):
        return None
    if ignore_collision_steps == 0 and collision_fn(q2):
        return None

    path = direct_path(q1, q2, extend_fn, collision_fn, ignore_collision_steps=ignore_collision_steps)
    if path is not None:
        return path
    for _ in irange(restarts + 1):
        if max_time <= elapsed_time(start_time):
            break
        path = rrt_connect(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn,
                           max_time=max_time - elapsed_time(start_time), ignore_collision_steps=ignore_collision_steps, **kwargs)
        if path is not None:
            if smooth is None:
                return path
            return smooth_path(path, extend_fn, collision_fn, iterations=smooth)
    return None
