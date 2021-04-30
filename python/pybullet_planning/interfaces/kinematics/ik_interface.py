
from pybullet_planning.interfaces.env_manager.pose_transformation import multiply, invert, quat_from_matrix, matrix_from_quat, \
    point_from_pose, quat_from_pose, get_distance, unit_pose
from pybullet_planning.interfaces.robots.joint import joints_from_names, get_joint_positions, violates_limits
from pybullet_planning.interfaces.robots.link import get_link_pose, link_from_name

def get_ik_tool_link_pose(fk_fn, robot, ik_joints, base_link, \
                          joint_values=None, use_current=False):
    """Use the given forward_kinematics function to compute ik_tool_link pose
    based on current joint configurations in pybullet.

    Note: The resulting FK pose is relative to the world frame, not the robot's base frame.

    Parameters
    ----------
    fk_fn : function handle
        fk(a 6-list) : point, rot_matrix
    robot : pybullet robot
    ik_joint : list of int
        a list of pybullet joint index that is registered for IK/FK
        can be obtained by e.g.: ``ik_joints = joints_from_names(robot, ik_joint_names)``
    base_link : int
        robot base link index, can be obtained from ``link_from_name(robot, base_link_name)``
    joint_values : list of float
        robot joint values for FK computation, value correponds to ik_joint_names, default to None
    use_current: bool
        if true, use the current configuration in pybullet env for FK, default
        to False

    Returns
    -------
    pybullet Pose
        Pose = (point, quat) = ([x,y,z], [4-list])
    """
    if use_current:
        conf = get_joint_positions(robot, ik_joints)
    else:
        assert joint_values
        conf = joint_values

    base_from_tool = compute_forward_kinematics(fk_fn, conf)
    world_from_base = get_link_pose(robot, base_link)
    return multiply(world_from_base, base_from_tool)


def get_ik_generator(ik_fn, robot, base_link, world_from_tcp, ik_tool_link_from_tcp=None, **kwargs):
    """get an ik generator

    Parameters
    ----------
    ik_fn : function handle
        get_ik(point, rot) : list of jt solutions
        point = [x,y,z]
        rot = 3x3 rotational matrix as a row-major list
    robot : pybullet robot
    base_link : int
        robot base link index, can be obtained from ``link_from_name(robot, base_link_name)``
    world_from_tcp : pybullet pose
        tcp pose in the world frame
    ik_tool_link_from_tcp : pybullet pose
        tcp pose in ik tool link's frame, optional, defaults = None

    Returns
    -------
    generator
        use next() to get solutions
    """
    world_from_base = get_link_pose(robot, base_link)
    base_from_tcp = multiply(invert(world_from_base), world_from_tcp)
    if ik_tool_link_from_tcp:
        base_from_ik_tool_link = multiply(base_from_tcp, invert(ik_tool_link_from_tcp))
    else:
        base_from_ik_tool_link = base_from_tcp
    yield compute_inverse_kinematics(ik_fn, base_from_ik_tool_link, **kwargs)


def sample_tool_ik(ik_fn, robot, ik_joints, world_from_tcp, base_link,
                   ik_tool_link_from_tcp=None,  closest_only=False, get_all=False, **kwargs):
    """ sample ik joints for a given tcp pose in the world frame

    Parameters
    ----------
    ik_fn : function handle
        get_ik(point, rot) : list of jt solutions
        point = [x,y,z]
        rot = 3x3 rotational matrix as a row-major list
    robot : pybullet robot
    ik_joint : list of int
        a list of pybullet joint index that is registered for IK/FK
        can be obtained by e.g.: ``ik_joints = joints_from_names(robot, ik_joint_names)``
    base_link : int
        robot base link index, can be obtained from ``link_from_name(robot, base_link_name)``
    world_from_tcp : pybullet pose
        tcp pose in the world frame
    ik_tool_link_from_tcp : pybullet pose
        tcp pose in ik tool link's frame, optional, defaults to None
    get_all : bool
        get all ik sol computed
    **kwargs : optional arguments
        for solution selection, nearby_conf=True to select the one that's closest
        to current conf in pybullet env

    Returns
    -------
    a list of 6-list
        computed IK solutions that satisfy joint limits
    """
    generator = get_ik_generator(ik_fn, robot, base_link, world_from_tcp, ik_tool_link_from_tcp, **kwargs)
    sols = next(generator)
    if closest_only and sols:
        current_conf = get_joint_positions(robot, ik_joints)
        sols = [min(sols, key=lambda conf: get_distance(current_conf, conf))]
    sols = list(filter(lambda conf: not violates_limits(robot, ik_joints, conf), sols))
    return sols if get_all else select_solution(robot, ik_joints, sols, **kwargs)


def compute_forward_kinematics(fk_fn, conf):
    """use the given fk function to compute FK

    Parameters
    ----------
    fk_fn : function handle
    conf : list
        joint values

    Returns
    -------
    pos: 3-list
        [x,y,z] of the FK solution pose
    quat: 4-list
        quaternion of the FK solution pose
    """
    pose = fk_fn(list(conf))
    pos, rot = pose
    quat = quat_from_matrix(rot) # [X,Y,Z,W]
    return pos, quat


def compute_inverse_kinematics(ik_fn, pose, sampled=[]):
    """compute ik solutions using the given ik function handle

    Parameters
    ----------
    ik_fn : function handle
        get_ik(point, rot) : list of jt solutions
        point = [x,y,z]
        rot = 3x3 rotational matrix as a row-major list
    pose : pybullet pose
        pose of the ik tool link
    sampled : list
        a list of externally sampled solutions that wants to be appended to the
        computed ik solutions

    Returns
    -------
    a list of 6-lists
        a list of ik solutions
    """
    pos = point_from_pose(pose)
    rot = matrix_from_quat(quat_from_pose(pose)).tolist()
    if sampled:
        solutions = ik_fn(list(pos), list(rot), sampled)
    else:
        solutions = ik_fn(list(pos), list(rot))
    if solutions is None:
        return []
    return solutions


def select_solution(body, joints, solutions, nearby_conf=True, random=False, **kwargs):
    """select one joint configuration given a list of them

    Parameters
    ----------
    body : pybullet body
        This can be any pybullet body, including the robot
    joints : a list of pybullet joint
    solutions : a list of float-lists
        joint values
    nearby_conf : bool
        return the joint conf that is closest to the current conf in pybullet env,
        defaults to True
    random : bool
        randomly chose one
    **kwargs : optional
        additional arguments for the cost function when ranking

    Returns
    -------
    a list of lists or one list
        the joint configuration(s)
    """
    if not solutions:
        return None
    if random and not nearby_conf:
        return random.choice(solutions)
    if nearby_conf:
        nearby_conf = get_joint_positions(body, joints)
        # TODO: sort by distance before collision checking
        # TODO: search over neighborhood of sampled joints when nearby_conf != None
        return min(solutions, key=lambda conf: get_distance(nearby_conf, conf, **kwargs))
    else:
        return random.choice(solutions)
