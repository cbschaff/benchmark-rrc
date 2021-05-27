import random
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
import time
from mp.align_rotation import get_yaw_diff


def set_seed(seed=0):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def get_rotation_between_vecs(v1, v2):
    """Rotation from v1 to v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    if np.isclose(np.linalg.norm(axis), 0):
        if np.dot(v1, v2) > 0:
            # zero rotation
            return np.array([0, 0, 0, 1])
        else:
            # 180 degree rotation

            # get perp vec
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            v1 /= np.linalg.norm(v1)
            axis -= np.dot(axis, v1) * v1
            axis /= np.linalg.norm(axis)
            assert np.isclose(np.dot(axis, v1), 0)
            assert np.isclose(np.dot(axis, v2), 0)
            return np.array([axis[0], axis[1], axis[2], 0])
    axis /= np.linalg.norm(axis)
    angle = np.arccos(v1.dot(v2))
    quat = np.zeros(4)
    quat[:3] = axis * np.sin(angle / 2)
    quat[-1] = np.cos(angle / 2)
    return quat


# ref: https://en.wikipedia.org/wiki/Slerp
def slerp(v0, v1, t_array):
    '''This performs Spherical linear interpolation.
    v0 and v1 are quaternions.
    ex: slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
    '''
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)

    if dot < 0.0:
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = v0[np.newaxis,:] + t_array[:,np.newaxis] * (v1 - v0)[np.newaxis,:]
        return (result.T / np.linalg.norm(result, axis=1)).T

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])


class Transform(object):
    def __init__(self, pos=None, ori=None, T=None):
        if pos is not None and ori is not None:
            self.R = np.array(p.getMatrixFromQuaternion(ori)).reshape((3, 3))
            self.pos = pos
            self.T = np.eye(4)
            self.T[:3, :3] = self.R
            self.T[:3, -1] = self.pos
        elif T is not None:
            self.T = T
            self.R = T[:3, :3]
            self.pos = T[:3, -1]
        else:
            raise ValueError("You must specify T or both pos and ori.")

    def adjoint(self):
        def _skew(p):
            return np.array([
                [0, -p[2], p[1]],
                [p[2], 0, -p[0]],
                [-p[1], p[0], 0],
            ])

        adj = np.zeros((6, 6))
        adj[:3, :3] = self.R
        adj[3:, 3:] = self.R
        adj[3:, :3] = _skew(self.pos).dot(self.R)
        return adj

    def inverse(self):
        T = np.eye(4)
        T[:3, :3] = self.R.T
        T[:3, -1] = -self.R.T.dot(self.pos)
        return Transform(T=T)

    def __call__(self, x):
        if isinstance(x, Transform):
            return Transform(T=self.T.dot(x.T))
        else:
            # check for different input forms
            one_dim = len(x.shape) == 1
            homogeneous = x.shape[-1] == 4
            if one_dim:
                x = x[None]
            if not homogeneous:
                x_homo = np.ones((x.shape[0], 4))
                x_homo[:, :3] = x
                x = x_homo

            # transform points
            x = self.T.dot(x.T).T

            # create output to match input form
            if not homogeneous:
                x = x[:, :3]
            if one_dim:
                x = x[0]
            return x


def is_valid_action(action, action_type='position'):
    from trifinger_simulation.trifinger_platform import TriFingerPlatform
    spaces = TriFingerPlatform.spaces

    if action_type == 'position':
        action_space = spaces.robot_position
    elif action_type == 'torque':
        action_space = spaces.robot_position

    return (action_space.low <= action).all() and (action <= action_space.high).all()


def repeat(sequence, num_repeat=3):
    '''
    [1,2,3] with num_repeat = 3  --> [1,1,1,2,2,2,3,3,3]
    '''
    return list(e for e in sequence for _ in range(num_repeat))


def ease_out(sequence, in_rep=1, out_rep=5):
    '''
    create "ease out" motion where an action is repeated for *out_rep* times at the end.
    '''
    in_seq_length = len(sequence[:-len(sequence) // 3])
    out_seq_length = len(sequence[-len(sequence) // 3:])
    x = [0, out_seq_length - 1]
    rep = [in_rep, out_rep]
    out_repeats = np.interp(list(range(out_seq_length)), x, rep).astype(int).tolist()

    #in_repeats = np.ones(in_seq_length).astype(int).tolist()
    in_repeats = np.ones(in_seq_length) * in_rep
    in_repeats = in_repeats.astype(int).tolist()
    repeats = in_repeats + out_repeats
    assert len(repeats) == len(sequence)

    seq = [repeat([e], n_rep) for e, n_rep in zip(sequence, repeats)]
    seq = [y for x in seq for y in x]  # flatten it

    return seq


class keep_state:
    '''
    A Context Manager that preserves the state of the simulator
    '''
    def __init__(self, env):
        self.finger_id = env.platform.simfinger.finger_id
        self.joints = env.platform.simfinger.pybullet_link_indices
        self.cube_id = env.platform.cube._object_id

    def __enter__(self):
        self.state_id = p.saveState()

    def __exit__(self, type, value, traceback):
        p.restoreState(stateId=self.state_id)
        p.removeState(stateUniqueId=self.state_id)


def get_body_state(body_id):
    position, orientation = p.getBasePositionAndOrientation(
        body_id
    )
    velocity = p.getBaseVelocity(body_id)
    return list(position), list(orientation), list(velocity)


def set_body_state(body_id, position, orientation, velocity=None):
    p.resetBasePositionAndOrientation(
        body_id,
        position,
        orientation,
    )
    if velocity is not None:
        linear_vel, angular_vel = velocity
        p.resetBaseVelocity(body_id, linear_vel, angular_vel)


class AssertNoStateChanges:
    def __init__(self, env):
        self.cube_id = env.platform.cube._object_id
        self.finger_id = env.platform.simfinger.finger_id
        self.finger_links = env.platform.simfinger.pybullet_link_indices

    def __enter__(self):
        from pybullet_planning.interfaces.robots.joint import get_joint_velocities, get_joint_positions
        org_obj_pos, org_obj_ori, org_obj_vel = get_body_state(self.cube_id)
        self.org_obj_pos = org_obj_pos
        self.org_obj_ori = org_obj_ori
        self.org_obj_vel = org_obj_vel

        self.org_joint_pos = get_joint_positions(self.finger_id, self.finger_links)
        self.org_joint_vel = get_joint_velocities(self.finger_id, self.finger_links)

    def __exit__(self, type, value, traceback):
        from pybullet_planning.interfaces.robots.joint import get_joint_velocities, get_joint_positions
        obj_pos, obj_ori, obj_vel = get_body_state(self.cube_id)
        np.testing.assert_array_almost_equal(self.org_obj_pos, obj_pos)
        np.testing.assert_array_almost_equal(self.org_obj_ori, obj_ori)
        np.testing.assert_array_almost_equal(self.org_obj_vel[0], obj_vel[0])
        np.testing.assert_array_almost_equal(self.org_obj_vel[1], obj_vel[1])

        joint_pos = get_joint_positions(self.finger_id, self.finger_links)
        joint_vel = get_joint_velocities(self.finger_id, self.finger_links)
        np.testing.assert_array_almost_equal(self.org_joint_pos, joint_pos)
        np.testing.assert_array_almost_equal(self.org_joint_vel, joint_vel)


def complete_joint_configs(start, goal, unit_rad=0.008):
    start = np.asarray(start)
    goal = np.asarray(goal)
    assert start.shape == goal.shape == (9, )

    max_diff = max(goal - start)

    num_keypoints = int(max_diff / unit_rad)
    joint_configs = [start + (goal - start) * i / num_keypoints for i in range(num_keypoints)]
    return joint_configs


def sample_uniform_from_circle(radius):
    # copied from move_cube.sample_goal()
    def random_xy(radius):
        # sample uniform position in circle (https://stackoverflow.com/a/50746409)
        radius = radius * np.sqrt(random.random())
        theta = random.uniform(0, 2 * np.pi)

        # x,y-position of the cube
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        return x, y
    return np.asarray(random_xy(radius))


def filter_none_elements(joint_conf_sequence):
    '''
    This function removes None in the sequence, and also returns indices that correspond to non-None elements
    '''
    valid_indices = []
    valid_joint_confs = []
    for idx, jconf in enumerate(joint_conf_sequence):
        if jconf is None:
            continue
        valid_joint_confs.append(jconf)
        valid_indices.append(idx)

    return valid_indices, valid_joint_confs


class SphereMarker:
    def __init__(self, radius, position, color=(0, 1, 0, 0.5)):
        """
        Create a sphere marker for visualization

        Args:
            width (float): Length of one side of the cube.
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            color: Color of the cube as a tuple (r, b, g, q)
            """
        self.shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        self.body_id = p.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
        )

    def set_state(self, position):
        """Set pose of the marker.

        Args:
            position: Position (x, y, z)
        """
        orientation = [0, 0, 0, 1]
        p.resetBasePositionAndOrientation(
            self.body_id, position, orientation
        )

    def __del__(self):
        """
        Removes the visual object from the environment
        """
        # At this point it may be that pybullet was already shut down. To avoid
        # an error, only remove the object if the simulation is still running.
        if p.isConnected():
            p.removeBody(self.body_id)
