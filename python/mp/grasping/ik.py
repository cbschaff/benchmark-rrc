from mp.utils import keep_state
from mp.const import INIT_JOINT_CONF
import numpy as np


class IKUtils:
    def __init__(self, env, yawing_grasp=False):
        self.fk = env.pinocchio_utils.forward_kinematics
        self.ik = env.pinocchio_utils.inverse_kinematics
        self.finger_id = env.platform.simfinger.finger_id
        self.tip_ids = env.platform.simfinger.pybullet_tip_link_indices
        self.link_ids = env.platform.simfinger.pybullet_link_indices
        self.cube_id = env.platform.cube._object_id
        self.env = env
        self.sample_fn = self._get_sample_fn()
        self.yawing_grasp = yawing_grasp

    def sample_no_collision_ik(self, target_tip_positions, slacky_collision=False, diagnosis=False):
        with keep_state(self.env):
            ik_solution = self._sample_no_collision_ik(target_tip_positions, slacky_collision=slacky_collision, diagnosis=diagnosis)
        if ik_solution is None:
            return []
        return ik_solution

    def _sample_no_collision_ik(
        self, target_tip_positions, slacky_collision=False, diagnosis=False
    ):
        from pybullet_planning.interfaces.kinematics.ik_utils import sample_ik_solution

        num_samples = 3
        collision_fn = self._get_collision_fn(slacky_collision)

        ik_solution = sample_ik_solution(self.ik, self.sample_fn, target_tip_positions)
        if ik_solution is None:
            return None

        ik_solutions = [ik_solution]
        ik_solutions += [
            sample_ik_solution(self.ik, self.sample_fn, target_tip_positions)
            for _ in range(num_samples)
        ]
        for ik_sol in ik_solutions:
            if ik_sol is None:
                continue
            if not collision_fn(ik_sol, diagnosis=diagnosis):
                return ik_sol
        return None

    def sample_ik(self, target_tip_positions, allow_partial_sol=False):
        '''
        NOTE: calling this function again and again is very slow.
        This is due to keep_state context manager.
        Use sample_iks for such use cases
        '''
        with keep_state(self.env):
            ik_solution = self._sample_ik(target_tip_positions, allow_partial_sol=allow_partial_sol)
        if ik_solution is None:
            return []
        return [ik_solution]

    def _sample_ik(self, target_tip_positions, allow_partial_sol=False):
        from pybullet_planning.interfaces.kinematics.ik_utils import (
            sample_ik_solution,
            sample_partial_ik_solution,
        )
        if allow_partial_sol:
            ik_solution = sample_partial_ik_solution(self.ik, self.sample_fn,
                                                       target_tip_positions)
        else:
            ik_solution = sample_ik_solution(self.ik, self.sample_fn,
                                             target_tip_positions)
        if ik_solution is None:
            return None
        return ik_solution

    def sample_iks(self, target_tip_pos_seq):
        '''
        NOTE: return value "solutions" contains None if the corresponding IK solution is not found
        '''
        solutions = []
        with keep_state(self.env):
            for target_tip_pos in target_tip_pos_seq:
                ik_solution = self._sample_ik(target_tip_pos)
                solutions.append(ik_solution)
        return solutions

    def _get_collision_fn(self, slacky_collision, diagnosis=False):
        from . import CollisionConfig
        # from pybullet_planning.interfaces.robots.collision import get_collision_fn
        # import functools
        # return functools.partial(
        #     get_collision_fn(**self._get_collision_conf(slacky_collision)),
        #     diagnosis=diagnosis
        # )
        if slacky_collision:
            config_type = "mpfc"
        else:
            config_type = None

        # DIRTY: use a specific collision config if self.yawing_grasp is set
        if self.yawing_grasp:
            config_type = "yaw_flip_grasp"
        return CollisionConfig(self.env).get_collision_fn(config_type, diagnosis=diagnosis)

    # def _get_collision_conf(self, slacky_collision, _yaw_flip=False):
    #     from code.const import COLLISION_TOLERANCE
    #     workspace_id = 0
    #     if _yaw_flip:  # HACK: This flag is only used in yaw-flipping
    #         # NOTE: this config only cares about finger-finger collision
    #         disabled_collisions = [((self.finger_id, tip_id), (self.cube_id, -1))
    #                                for tip_id in self.tip_ids]
    #         config = {
    #             'body': self.finger_id,
    #             'joints': self.link_ids,
    #             'obstacles': [workspace_id],  # ignore collisions with cube
    #             'self_collisions': True,
    #             'extra_disabled_collisions': disabled_collisions,
    #             'max_distance': -COLLISION_TOLERANCE
    #         }
    #     elif slacky_collision:
    #         disabled_collisions = [((self.finger_id, tip_id), (self.cube_id, -1))
    #                                for tip_id in self.tip_ids]
    #         config = {
    #             'body': self.finger_id,
    #             'joints': self.link_ids,
    #             'obstacles': [self.cube_id, workspace_id],
    #             'self_collisions': True,
    #             'extra_disabled_collisions': disabled_collisions,
    #             'max_distance': -COLLISION_TOLERANCE
    #         }
    #     else:
    #         config = {
    #             'body': self.finger_id,
    #             'joints': self.link_ids,
    #             'obstacles': [self.cube_id, workspace_id],
    #             'self_collisions': False
    #         }

    #     return config

    def _get_sample_fn(self):
        space = self.env.platform.spaces.robot_position.gym
        def _sample_fn():
            s = np.random.rand(space.shape[0])
            return s * (space.high - space.low) + space.low
        return _sample_fn

    def get_joint_conf(self):
        obs = self.env.platform.simfinger._get_latest_observation()
        return obs.position, obs.velocity
