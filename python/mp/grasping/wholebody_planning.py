#!/usr/bin/env python3
import pybullet as p
import numpy as np
import time
from mp.utils import Transform, filter_none_elements, repeat
from pybullet_planning import plan_wholebody_motion
from collections import namedtuple
from trifinger_simulation.tasks.move_cube import _ARENA_RADIUS, _min_height, _max_height
from mp.const import COLLISION_TOLERANCE

from .ik import IKUtils
from .grasp_sampling import GraspSampler

dummy_links = [-2, -3, -4, -100, -101, -102]  # link indices <=100 denotes circular joints
custom_limits = {
    -2: (-_ARENA_RADIUS, _ARENA_RADIUS),
    -3: (-_ARENA_RADIUS, _ARENA_RADIUS),
    -4: (_min_height, _max_height),
    -100: (-np.pi, np.pi),
    -101: (-np.pi, np.pi),
    -102: (-np.pi, np.pi)
}


class Path(object):
    def __init__(self, cube, joint_conf, tip_path, grasp):
        self.cube = cube
        self.joint_conf = joint_conf
        self.tip_path = tip_path
        self.grasp = grasp

    def repeat(self, n):
        return Path(
            repeat(self.cube, n),
            repeat(self.joint_conf, n),
            repeat(self.tip_path, n),
            self.grasp
        )

    def set_min_height(self, env, h):
        for i, tips in enumerate(self.tip_path):
            if np.any(tips[:, 2] < h):
                tips[:, 2] = np.maximum(tips[:, 2], h)
        self.joint_conf = IKUtils(env).sample_iks(self.tip_path)

    @classmethod
    def tighten(cls, env, path, coef=0.9):
        ik_utils = IKUtils(env)
        cube_tip_pos = path.grasp.cube_tip_pos * coef
        tip_path = _get_tip_path(cube_tip_pos, path.cube)

        print('wholebody planning path length:', len(tip_path))
        jconf_sequence = ik_utils.sample_iks(tip_path)
        inds, joint_conf = filter_none_elements(jconf_sequence)

        # if two or more tip positions are invalid (no ik solution), just use the original grasp
        num_no_iksols = len(jconf_sequence) - len(joint_conf)
        if num_no_iksols > 0:
            print(f'warning: {num_no_iksols} IK solutions are not found in WholebodyPlanning.get_tighter_path')
        if num_no_iksols > 1:
            # print(f'warning: num_no_iksols > 1 --> not using tighter grasp path')
            print(f'warning: num_no_iksols > 1')
            # return path
        if len(joint_conf) == 0:
            print(f'warning: No IK solution is found in WholebodyPlanning.get_tighter_path!! --> not using tighter grasp path')
            return path

        cube_path = [path.cube[idx] for idx in inds]
        tip_path = [tip_path[idx] for idx in inds]
        grasp = path.grasp
        grasp.cube_tip_pos *= coef
        grasp.base_tip_pos = grasp.T_cube_to_base(grasp.cube_tip_pos)
        grasp.q = joint_conf[0]

        return cls(cube_path, joint_conf, tip_path, grasp)


def get_joint_states(robot_id, link_indices):
    joint_states = [joint_state[0] for joint_state in p.getJointStates(robot_id, link_indices)]
    return np.asarray(joint_states)


def disable_tip_collisions(env):
    disabled_collisions = set()
    for tip_link in env.platform.simfinger.pybullet_tip_link_indices:
        disabled_collisions.add(((env.platform.cube._object_id, -1), (env.platform.simfinger.finger_id, tip_link)))
    return disabled_collisions


def _get_tip_path(cube_tip_positions, cube_path):
    def get_quat(euler):
        return p.getQuaternionFromEuler(euler)
    return [
        Transform(cube_pose[:3], get_quat(cube_pose[3:]))(cube_tip_positions)
        for cube_pose in cube_path
    ]


class WholeBodyPlanner:
    """A class for wholebody planning.

    'wholebody' refers to the fact that we virtually attach all fingertips on the object
    with spherical joints, and consider them to be a single object with multiple links.
    We perform path planning of the 'whole' object while checking collision between the finger links etc.

    Very roughly speaking, `plan` method does:
    1. Sample a set of grasps from 'heuristic grasps' and 'random grasps'
    2. For each grasp
        a. Virtually attach fingertips to the object with spherical joints
           (In practice, we just independently solve IK for each fingertip position, without hard constraints)
        b. run a path planning and check if the object's goal pose is reachable with the grasp,
           rejecting paths that has a collision (e.g., between finger links).
        c. if collision path is found, exit the loop
    3. Once a valid grasp and path is found, it returns `Path` object that contains
       both of grasp and path information
    """

    def __init__(self, env):
        self.env = env

        # disable collision check for tip
        self._disabled_collisions = disable_tip_collisions(self.env)

    def _get_disabled_colilsions(self):
        disabled_collisions = set()
        for tip_link in self.env.platform.simfinger.pybullet_tip_link_indices:
            disabled_collisions.add(((self.env.platform.cube._object_id, -1), (self.env.platform.simfinger.finger_id, tip_link)))
        return disabled_collisions

    def get_fingers_collision_fn(self):
        """returns a collision function for finger-finger collision
        NOTE: USE IT FOR YAW-GRASP PLANNING ONLY!! The collision config is slightly different from one for MPFC, and it can cause false-positive collisions.
        """
        from .collision_config import CollisionConfig
        collision_fn = CollisionConfig(self.env).get_collision_fn(config_type="yaw_flip_path", diagnosis=False)
        def _fingers_collision_fn(cube_pose, joint_conf, diagnosis=False):
            """a function for finger-finger collision detection"""
            if collision_fn(joint_conf):
                print('finger-finger collision!')
                return True
            return False
        return _fingers_collision_fn

    def plan(self, pos, quat, goal_pos, goal_quat, heuristic_grasps=None, retry_grasp=10,
             use_rrt=False, use_incremental_rrt=False, min_goal_threshold=0.01,
             max_goal_threshold=0.8, use_ori=False, avoid_edge_faces=True,
             yawing_grasp=False, collision_tolerance=-COLLISION_TOLERANCE * 10,
             path_min_height=0.01, direct_path=False):
        resolutions = 0.03 * np.array([0.3, 0.3, 0.3, 1, 1, 1])  # roughly equiv to the lengths of one step.

        goal_ori = p.getEulerFromQuaternion(goal_quat)
        target_pose = np.concatenate([goal_pos, goal_ori])
        grasp_sampler = GraspSampler(self.env, pos, quat, slacky_collision=True, avoid_edge_faces=avoid_edge_faces, yawing_grasp=yawing_grasp)
        if heuristic_grasps is None:
            grasps = [g for g in grasp_sampler.get_heuristic_grasps()]
        else:
            grasps = heuristic_grasps

        print("WHOLEBODY PLANNING")
        print(f"num heuristic grasps: {len(grasps)}")

        # sample random grasp if no heuristic grasp is available
        if len(grasps) == 0:
            grasps = (grasp_sampler() for _ in range(10))

        counter = 0
        cube_path = None
        if not use_ori:
            print("CHANGING GOAL ORIENTATION...")
            use_ori = True
            goal_ori = p.getEulerFromQuaternion(quat)
            target_pose = np.concatenate([goal_pos, goal_ori])
        from mp.utils import keep_state
        while cube_path is None and counter < retry_grasp + 1:
            retry_count = max(0, counter)
            goal_threshold = ((retry_count / (retry_grasp + 1))
                              * (max_goal_threshold - min_goal_threshold)
                              + min_goal_threshold)
            print(counter, goal_threshold)

            for grasp in grasps:
                with keep_state(self.env):
                    self.env.platform.simfinger.reset_finger_positions_and_velocities(grasp.q)
                    cube_path, joint_conf_path = plan_wholebody_motion(
                        self.env.platform.cube._object_id,
                        dummy_links,
                        self.env.platform.simfinger.finger_id,
                        self.env.platform.simfinger.pybullet_link_indices,
                        target_pose,
                        grasp.base_tip_pos,
                        grasp.cube_tip_pos,
                        init_joint_conf=grasp.q,
                        ik=self.env.pinocchio_utils.inverse_kinematics,
                        obstacles=[self.env.platform.simfinger.finger_id],
                        disabled_collisions=self._disabled_collisions,
                        custom_limits=custom_limits,
                        resolutions=resolutions,
                        diagnosis=False,
                        max_distance=collision_tolerance,
                        # vis_fn=vis_cubeori.set_state,
                        iterations=20,
                        use_rrt=use_rrt,
                        use_incremental_rrt=use_incremental_rrt,
                        use_ori=use_ori,
                        goal_threshold=goal_threshold,
                        # NOTE:
                        # if restarts == -1 --> only checks direct path
                        # otherwise, it tries direct path once and then try find a random path restarts + 1 times
                        restarts=-1 if counter == 0 or direct_path else 1,  # only check for direct path the first time
                        additional_collision_fn=self.get_fingers_collision_fn() if yawing_grasp else None
                    )
                if cube_path is not None:
                    break
            counter += 1

        if cube_path is None:
            raise RuntimeError('wholebody planning failed')

        one_pose = (len(np.shape(cube_path)) == 1)
        if one_pose:
            cube_path = [cube_path]
        tip_path = _get_tip_path(grasp.cube_tip_pos, cube_path)
        path = Path(cube_path, joint_conf_path, tip_path, grasp)
        path.set_min_height(self.env, path_min_height)
        return path


if __name__ == '__main__':
    from trifinger_simulation.tasks import move_cube
    from env.make_env import make_env

    reward_fn = 'competition_reward'
    termination_fn = 'position_close_to_goal'
    initializer = 'small_rot_init'
    env = make_env(move_cube.sample_goal(-1).to_dict(), 4,
                   reward_fn=reward_fn,
                   termination_fn=termination_fn,
                   initializer=initializer,
                   action_space='position',
                   sim=True, visualization=True)

    for i in range(1):
        obs = env.reset()

        pos = obs["object_position"]
        quat = obs["object_orientation"]
        goal_pos = obs["goal_object_position"]
        goal_quat = obs["goal_object_orientation"]
        planner = WholeBodyPlanner(env)
        path = planner.plan(pos, quat, goal_pos, goal_quat, use_rrt=True,
                            use_ori=True, min_goal_threshold=0.01,
                            max_goal_threshold=0.3)

        if path.cube is None:
            print('PATH is NOT found...')
            quit()

        # clear some windows in GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # change camera parameters # You can also rotate the camera by CTRL + drag
        p.resetDebugVisualizerCamera( cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])
        # p.startStateLogging( p.STATE_LOGGING_VIDEO_MP4, f'wholebody_planning_{args.seed}.mp4')
        for cube_pose, joint_conf in zip(path.cube, path.joint_conf):
            point, ori = cube_pose[:3], cube_pose[3:]
            quat = p.getQuaternionFromEuler(ori)
            for i in range(3):
                p.resetBasePositionAndOrientation(env.platform.cube._object_id, point, quat)
                env.platform.simfinger.reset_finger_positions_and_velocities(joint_conf)
                time.sleep(0.01)
