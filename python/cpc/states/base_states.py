import time

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from mp.states import State
from mp.utils import Transform
from cpc.states import utils
from mp.grasping import get_planned_grasp
from mp.grasping.grasp_sampling import GraspSampler

CUBE_SIZE = 0.0325
DAMP = 1E-6
EPS = 1E-2


class SimpleState(State):
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.start_time = None
        self.time_exceed_threshold = None
        self.interval = 100
        self.gain_increase_factor = 1.2
        self.t = 0
        self.interval_ctr = 0
        self.max_interval_ctr = None
        # self.reset()

    def reset(self):
        self.actions = None

    def connect(self, next_state, failure_state):
        self.next_state = next_state
        self.failure_state = failure_state

    def update_gain(self):
        if self.env.simulation:
            return
        self.t += 1
        if self.t % self.interval == 0 and self.interval_ctr < self.max_interval_ctr:
            self.k_p *= self.gain_increase_factor
            self.interval_ctr += 1

    def time_exceeded(self):
        return time.time() - self.start_time > self.time_exceed_threshold

    def _get_jacobian(self, observation):
        ret = []
        for tip in self.env.platform.simfinger.pybullet_tip_link_indices:
            J, _ = p.calculateJacobian(
                self.env.platform.simfinger.finger_id,
                tip,
                np.zeros(3).tolist(),
                observation["robot_position"].tolist(),
                observation["robot_velocity"].tolist(),
                np.zeros(len(observation["robot_position"])).tolist(),
                self.env.platform.simfinger._pybullet_client_id
            )
            ret.append(J)
        ret = np.vstack(ret)
        return ret

    def _get_gravcomp(self, observation):
        # Returns: 9 torques required for grav comp
        ret = p.calculateInverseDynamics(self.env.platform.simfinger.finger_id,
                                         observation["robot_position"].tolist(
                                         ),
                                         observation["robot_velocity"].tolist(
                                         ),
                                         np.zeros(
                                             len(observation["robot_position"])).tolist(),
                                         self.env.platform.simfinger._pybullet_client_id)

        ret = np.array(ret)
        return ret

    def _get_tip_poses(self, observation):
        return observation["robot_tip_positions"].flatten()

    def get_torque_action(self, observation, force):
        J = self._get_jacobian(observation)
        torque = J.T.dot(np.linalg.solve(
            J.dot(J.T) + DAMP * np.eye(9), force))

        ret = np.array(torque + self._get_gravcomp(observation),
                       dtype=np.float64)
        self.action_space_limits = self.env.action_space['torque']
        return ret

    def get_action_generator(self, obs, info):
        """Yields (action, info) tuples."""
        raise NotImplementedError

    def __call__(self, obs, info={}):
        raise NotImplementedError


class SimpleGoToInitState(SimpleState):
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.interval = 100
        self.gain_increase_factor = 1.1
        self.max_interval_ctr = 20
        self.init_gain()
        self.reset()

    def init_gain(self):
        if self.env.simulation:
            self.k_p = 0.5
        else:
            self.k_p = 1.5

    def reset(self):
        self.init_gain()
        self.interval_ctr = 0
        self.t = 0

    def __call__(self, obs, info={}):
        self.update_gain()
        current = self._get_tip_poses(obs)
        up_position = np.array([0.5, 1.2, -2.4] * 3)
        desired = np.array(
            self.env.pinocchio_utils.forward_kinematics(up_position)).flatten()
        err = desired - current
        torque = self.get_torque_action(obs, self.k_p * err)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=1)
        if np.linalg.norm(err) < 2 * EPS:
            info['force_offset'] = obs['tip_force']
            self.reset()
            return action, self.next_state, info
        else:
            return action, self, info


class GoToInitState(SimpleState):
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.interval = 100
        self.gain_increase_factor = 1.1
        self.max_interval_ctr = 20
        self.init_gain()
        self.reset()

    def init_gain(self):
        if self.env.simulation:
            self.k_p = 0.5
        else:
            self.k_p = 1.5

    def reset(self):
        self.init_gain()
        self.interval_ctr = 0
        self.t = 0

    def __call__(self, obs, info={}):
        self.update_gain()
        current = self._get_tip_poses(obs)
        up_position = np.array([0.5, 1.2, -2.4] * 3)
        desired = np.array(
            self.env.pinocchio_utils.forward_kinematics(up_position)).flatten()
        err = desired - current
        torque = self.get_torque_action(obs, self.k_p * err)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=1)
        if np.linalg.norm(err) < 2 * EPS:
            info['force_offset'] = obs['tip_force']
            info['path'] = None
            if self.env.difficulty == 4:
                self.reset()
                if 'do_premanip' not in info.keys():
                    return action, self.failure_state, info
                if info['do_premanip']:
                    return action, self.failure_state, info
                else:
                    return action, self.next_state, info
            else:
                self.reset()
                return action, self.next_state, info
        else:
            return action, self, info


class AlignState(SimpleState):
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.start_time = None
        self.time_exceed_threshold = 10.0
        self.interval = 100
        self.gain_increase_factor = 1.1
        self.max_interval_ctr = 20
        self.init_gain()

    def init_gain(self):
        if self.env.simulation:
            self.k_p = 0.5
        else:
            self.k_p = 1.25

    def reset(self):
        self.init_gain()
        self.interval_ctr = 0
        self.t = 0
        self.start_time = None

    def __call__(self, obs, info={}):
        if self.start_time is None:
            self.start_time = time.time()

        self.update_gain()
        current = self._get_tip_poses(obs)
        desired = np.tile(obs["object_position"], 3) + \
            CUBE_SIZE * np.array([0, 1.6, 2, 1.6 * 0.866, 1.6 *
                                  (-0.5), 2, 1.6 * (-0.866), 1.6 * (-0.5), 2])

        err = desired - current
        torque = self.get_torque_action(obs, self.k_p * err)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=1)
        if np.linalg.norm(err) < 2 * EPS:
            self.reset()
            return action, self.next_state, info
        else:
            if self.time_exceeded():
                self.reset()
                return action, self.failure_state, info
            else:
                return action, self, info


class LowerState(SimpleState):
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.start_time = None
        self.err_sum = np.zeros(9)
        self.time_exceed_threshold = 10.0
        self.interval = 100
        self.gain_increase_factor = 1.1
        self.max_interval_ctr = 20
        self.init_gain()

    def init_gain(self):
        if self.env.simulation:
            self.k_p = 0.5
        else:
            self.k_p = 1.25

    def reset(self):
        self.init_gain()
        self.err_sum = np.zeros(9)
        self.interval_ctr = 0
        self.t = 0
        self.start_time = None

    def __call__(self, obs, info={}):
        if self.start_time is None:
            self.start_time = time.time()

        self.update_gain()
        current = self._get_tip_poses(obs)

        desired = np.tile(obs["object_position"], 3) + \
            CUBE_SIZE * np.array([0, 1.6, 0, 1.6 * 0.866, 1.6 *
                                  (-0.5), 0, 1.6 * (-0.866), 1.6 * (-0.5), 0])

        err = desired - current
        err_mag = np.linalg.norm(err[:3])
        if err_mag < 0.1:
            self.err_sum += err

        torque = self.get_torque_action(
            obs, self.k_p * err + 0.001 * self.err_sum)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=1)
        if np.linalg.norm(err) < 1 * EPS:
            self.reset()
            return action, self.next_state, info
        else:
            if self.time_exceeded():
                self.reset()
                return action, self.failure_state, info
            else:
                return action, self, info


class IntoState(SimpleState):
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.start_time = None
        self.time_exceed_threshold = 10.0
        self.grasp_check_failed_count = 0
        self.t = 0
        self.interval_ctr = 0
        self.interval = 100
        self.gain_increase_factor = 1.2
        self.max_interval_ctr = 20
        self.init_gain()

    def init_gain(self):
        if self.env.simulation:
            self.k_p = 0.5
        else:
            self.k_p = 1.75

    def update_gain(self):
        # if self.env.simulation:
        #     return
        self.t += 1
        if self.t % self.interval == 0 and self.interval_ctr < self.max_interval_ctr:
            self.k_p *= self.gain_increase_factor
            self.interval_ctr += 1

    def reset(self):
        self.init_gain()
        self.interval_ctr = 0
        self.t = 0
        self.interval_ctr = 0
        self.start_time = None
        self.grasp_check_failed_count = 0
        self.success_ctr = 0

    def success(self):
        return self.success_ctr > 50

    def object_grasped(self, obs):
        current = self._get_tip_poses(obs)
        current_x = current[0::3]
        current_y = current[1::3]
        difference_x = [abs(p1 - p2)
                        for p1 in current_x for p2 in current_x if p1 != p2]
        difference_y = [abs(p1 - p2)
                        for p1 in current_y for p2 in current_y if p1 != p2]

        close_x = any(d < 0.0001 for d in difference_x)
        close_y = any(d < 0.0001 for d in difference_y)
        close = close_x and close_y

        if close:
            self.grasp_check_failed_count += 1

        return self.grasp_check_failed_count < 5

    def __call__(self, obs, info={}):
        if self.start_time is None:
            self.start_time = time.time()

        self.update_gain()
        current = self._get_tip_poses(obs)

        desired = np.tile(obs["object_position"], 3)

        err = desired - current
        torque = self.get_torque_action(obs, self.k_p * err)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=1)

        # Read Tip Force
        tip_forces = obs["tip_force"] - info["force_offset"]
        switch = True
        for f in tip_forces:
            if f < 0.04:
                switch = False
        if switch:
            self.success_ctr += 1

        if self.success():
            grasp_sampler = GraspSampler(
                self.env, obs['object_position'], obs['object_orientation'])
            custom_grasp = [grasp_sampler.get_custom_grasp(
                obs['robot_tip_positions'])]
            try:
                grasp, path = get_planned_grasp(self.env, obs['object_position'], obs['object_orientation'],
                                                obs['goal_object_position'], obs['goal_object_orientation'],
                                                tight=True, heuristic_grasps=custom_grasp)
            except Exception:
                grasp, path = custom_grasp[0], None
            info['grasp'] = grasp
            info['path'] = path
            self.reset()
            return action, self.next_state, info
        else:
            if self.time_exceeded() or not self.object_grasped(obs):
                self.reset()
                return action, self.failure_state, info
            else:
                return action, self, info


class MoveToGoalState(SimpleState):
    def __init__(self, env, k_p_goal=0.65, k_p_into=0.2, k_i_goal=0.004, gain_increase_factor=1.04, interval=1800, max_interval_ctr=1800):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.goal_err_sum = np.zeros(9)
        self.grasp_check_failed_count = 0
        self.start_time = None
        self.time_exceed_threshold = 20.0
        self.success_ctr = 0
        self.init_gain_increase_factor = gain_increase_factor
        self.max_interval_ctr = max_interval_ctr
        self.interval = interval
        self.init_k_p_goal = k_p_goal
        self.init_k_p_into = k_p_into
        self.init_k_i_goal = k_i_goal
        if self.env.simulation:
            self.frameskip = 1
            self.max_k_p = 1.5
        else:
            self.frameskip = 4
            self.max_k_p = 1.25
        self.init_gain(self.init_k_p_goal,
                       self.init_k_p_into, self.init_k_i_goal)

    def init_gain(self, k_p_goal=0.65, k_p_into=0.2, k_i_goal=0.004):
        if self.env.simulation:
            self.k_p_goal = k_p_goal
            self.k_p_into = k_p_into
            self.k_i_goal = k_i_goal
        else:
            self.k_p_goal = k_p_goal
            self.k_p_into = k_p_into
            self.k_i_goal = k_i_goal

    def update_gain(self):
        # if self.env.simulation:
        #     return
        self.t += 1
        if self.t % self.interval == 0 and self.interval_ctr < self.max_interval_ctr:
            self.k_p_goal *= self.gain_increase_factor
            self.interval_ctr += 1

    def reset(self):
        self.init_gain(self.init_k_p_goal,
                       self.init_k_p_into, self.init_k_i_goal)
        self.interval_ctr = 0
        self.success_ctr = 0
        self.t = 0
        self.start_time = None
        self.gain_increase_factor = self.init_gain_increase_factor
        self.grasp_check_failed_count = 0
        self.goal_err_sum = np.zeros(9)

    def success(self):
        return self.success_ctr > 20

    def object_grasped(self, obs, grasp):
        T_cube_to_base = Transform(obs['object_position'],
                                   obs['object_orientation'])
        target_tip_pos = T_cube_to_base(grasp.cube_tip_pos)
        center_of_tips = np.mean(target_tip_pos, axis=0)
        dist = np.linalg.norm(target_tip_pos - obs['robot_tip_positions'])
        center_dist = np.linalg.norm(
            center_of_tips - np.mean(obs['robot_tip_positions'], axis=0))
        object_is_grasped = center_dist < 0.07 and dist < 0.10
        if object_is_grasped:
            self.grasp_check_failed_count = 0
        else:
            self.grasp_check_failed_count += 1
            print('incremented grasp_check_failed_count')
            print(f'center_dist: {center_dist:.4f}\tdist: {dist:.4f}')

        return self.grasp_check_failed_count < 5

    def __call__(self, obs, info={}):
        self.update_gain()
        self.k_p_goal = min(self.max_k_p, self.k_p_goal)

        if self.start_time is None:
            self.start_time = time.time()

        current = self._get_tip_poses(obs)
        desired = np.tile(obs["object_position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        goal = np.tile(obs["goal_object_position"], 3)
        # TODO: Add difficulty param
        if self.env.difficulty == 1 and not self.success():
            goal[2] += 0.002  # Reduces friction with floor

        goal_err = goal - desired
        err_mag = np.linalg.norm(goal_err[:3])

        if err_mag < 0.1:
            self.goal_err_sum += goal_err

        force = self.k_p_into * into_err + self.k_p_goal * \
            goal_err + self.k_i_goal * self.goal_err_sum
        torque = self.get_torque_action(obs, force)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=self.frameskip)

        if err_mag < EPS:
            self.success_ctr += 1
        else:
            if self.success():
                self.gain_increase_factor = self.init_gain_increase_factor
                self.success_ctr = 0
                self.start_time = time.time()

        if self.success():
            self.gain_increase_factor = 1.0

        if self.object_grasped(obs, info['grasp']):
            return action, self, info
        else:
            self.reset()
            return self.get_action(frameskip=0), self.failure_state, info


class GoalWithOrientState(SimpleState):
    def __init__(self, env, k_p_goal=0.4, k_p_into=0.2, k_i_goal=0.004, k_p_ang=0.04, k_i_ang=0.001, gain_increase_factor=1.04, interval=1800, max_interval_ctr=1800):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.goal_err_sum = np.zeros(9)
        self.ang_err_sum = np.zeros(9)
        self.grasp_check_failed_count = 0
        self.start_time = None
        self.time_exceed_threshold = 20.0
        self.success_ctr = 0
        self.gain_increase_factor = gain_increase_factor
        self.max_interval_ctr = max_interval_ctr
        self.interval = interval
        self.init_k_p_goal = k_p_goal
        self.init_k_p_into = k_p_into
        self.init_k_i_goal = k_i_goal
        self.init_k_p_ang = k_p_ang
        self.init_k_i_ang = k_i_ang
        if self.env.simulation:
            self.frameskip = 3
            self.max_k_p = 1.5
        else:
            self.frameskip = 4
            self.max_k_p = 0.8
        self.init_gain(self.init_k_p_goal, self.init_k_p_into,
                       self.init_k_i_goal, self.init_k_p_ang, self.init_k_i_ang)

    def init_gain(self, k_p_goal=0.65, k_p_into=0.2, k_i_goal=0.004, k_p_ang=0.04, k_i_ang=0.001):
        if self.env.simulation:
            self.k_p_goal = k_p_goal
            self.k_p_into = k_p_into
            self.k_i_goal = k_i_goal
            self.k_p_ang = k_p_ang
            self.k_i_ang = k_i_ang
        else:
            self.k_p_goal = k_p_goal
            self.k_p_into = k_p_into
            self.k_i_goal = k_i_goal
            self.k_p_ang = k_p_ang
            self.k_i_ang = k_i_ang

    def update_gain(self):
        # if self.env.simulation:
        #     return
        self.t += 1
        if self.t % self.interval == 0 and self.interval_ctr < self.max_interval_ctr:
            self.k_p_goal *= self.gain_increase_factor
            self.interval_ctr += 1

    def reset(self):
        self.init_gain(self.init_k_p_goal, self.init_k_p_into,
                       self.init_k_i_goal, self.init_k_p_ang, self.init_k_i_ang)
        self.interval_ctr = 0
        self.success_ctr = 0
        self.t = 0
        self.start_time = None
        self.gain_increase_factor = 1.04
        self.grasp_check_failed_count = 0
        self.goal_err_sum = np.zeros(9)
        self.ang_err_sum = np.zeros(9)

    def success(self):
        return self.success_ctr > 20

    def object_grasped(self, obs, grasp):
        T_cube_to_base = Transform(obs['object_position'],
                                   obs['object_orientation'])
        target_tip_pos = T_cube_to_base(grasp.cube_tip_pos)
        center_of_tips = np.mean(target_tip_pos, axis=0)
        dist = np.linalg.norm(target_tip_pos - obs['robot_tip_positions'])
        center_dist = np.linalg.norm(
            center_of_tips - np.mean(obs['robot_tip_positions'], axis=0))
        object_is_grasped = center_dist < 0.07 and dist < 0.10
        if object_is_grasped:
            self.grasp_check_failed_count = 0
        else:
            self.grasp_check_failed_count += 1
            print('incremented grasp_check_failed_count')
            print(f'center_dist: {center_dist:.4f}\tdist: {dist:.4f}')

        return self.grasp_check_failed_count < 5

    def __call__(self, obs, info={}):
        self.update_gain()
        self.k_p_goal = min(self.max_k_p, self.k_p_goal)

        if self.start_time is None:
            self.start_time = time.time()

        current = self._get_tip_poses(obs)
        desired = np.tile(obs["object_position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        goal = np.tile(obs["goal_object_position"], 3)
        goal_err = goal - desired
        err_mag = np.linalg.norm(goal_err[:3])

        angle, axis = utils._get_angle_axis(
            obs["object_orientation"], obs["goal_object_orientation"])
        ang_err = np.zeros(9)
        ang_err[:3] = -angle * \
            np.cross(into_err[:3] / np.linalg.norm(into_err[:3]), axis)
        ang_err[3:6] = -angle * \
            np.cross(into_err[3:6] / np.linalg.norm(into_err[3:6]), axis)
        ang_err[6:] = -angle * \
            np.cross(into_err[6:] / np.linalg.norm(into_err[6:]), axis)

        if err_mag < 0.1:
            self.goal_err_sum += goal_err
            self.ang_err_sum += ang_err

        if err_mag < EPS:
            self.success_ctr += 1
        else:
            if self.success():
                self.gain_increase_factor = 1.04
                self.success_ctr = 0
                self.start_time = time.time()

        if self.success():
            self.gain_increase_factor = 1.0

        force = self.k_p_into * into_err + self.k_p_goal * goal_err + self.k_i_goal * \
            self.goal_err_sum + self.k_p_ang * ang_err + self.k_i_ang * self.ang_err_sum
        # force = 0.2 * into_err + self.k_p * goal_err + 0.2 * ang_err
        torque = self.get_torque_action(obs, force)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=self.frameskip)

        if self.success():
            self.gain_increase_factor = 1.0

        if self.object_grasped(obs, info['grasp']):
            return action, self, info
        else:
            self.reset()
            return self.get_action(frameskip=0), self.failure_state, info


class GetPreManipInfoState(SimpleState):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.reset()

    def __call__(self, obs, info={}):
        info['manip_angle'], info['manip_axis'], info['manip_arm'], info['is_yaw_orient'] = utils.pitch_orient(
            obs)
        if info['is_yaw_orient']:
            if info['manip_angle'] > 1.57:
                info['manip_angle'] = 3.14 - info['manip_angle']
        if info['manip_angle'] < 0.2:
            info['do_premanip'] = False
        else:
            info['do_premanip'] = True

        if info['do_premanip']:
            self.reset()
            return self.get_action(frameskip=0), self.next_state, info
        else:
            self.reset()
            return self.get_action(frameskip=0), self.failure_state, info


class PreAlignState(SimpleState):
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.start_time = None
        self.time_exceed_threshold = 2.0
        self.interval = 100
        self.gain_increase_factor = 1.1
        self.max_interval_ctr = 20
        self.init_gain()

    def init_gain(self):
        if self.env.simulation:
            self.k_p = 0.5
        else:
            self.k_p = 1.25

    def reset(self):
        self.init_gain()
        self.interval_ctr = 0
        self.t = 0
        self.start_time = None

    def __call__(self, obs, info={}):
        if self.start_time is None:
            self.start_time = time.time()

        self.update_gain()
        current = self._get_tip_poses(obs)

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (info['manip_arm'] + 1 - i) % 3
            locs[index] = 1.5 * \
                R.from_rotvec(
                    np.pi/2 * (i-1.0) * np.array([0, 0, 1])).apply(info['manip_axis'])
            locs[index][2] = 2

        desired = np.tile(obs['object_position'], 3) + \
            (CUBE_SIZE) * np.hstack(locs)

        # desired[3*info['manip_arm']:3*info['manip_arm']+3] = obs['object_position']
        # desired[3*info['manip_arm']+2] = 0.1

        err = desired - current
        torque = self.get_torque_action(obs, self.k_p * err)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=1)
        if np.linalg.norm(err) < 2 * EPS:
            self.reset()
            return action, self.next_state, info
        else:
            if self.time_exceeded():
                self.reset()
                return action, self.failure_state, info
            else:
                return action, self, info


class PreLowerState(SimpleState):
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.start_time = None
        self.err_sum = np.zeros(9)
        self.time_exceed_threshold = 10.0
        self.interval = 100
        self.gain_increase_factor = 1.1
        self.max_interval_ctr = 20
        self.init_gain()

    def init_gain(self):
        if self.env.simulation:
            self.k_p = 0.5
        else:
            self.k_p = 1.0

    def reset(self):
        self.init_gain()
        self.err_sum = np.zeros(9)
        self.interval_ctr = 0
        self.t = 0
        self.start_time = None

    def __call__(self, obs, info={}):
        if self.start_time is None:
            self.start_time = time.time()

        self.update_gain()
        current = self._get_tip_poses(obs)

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (info['manip_arm'] + 1 - i) % 3
            locs[index] = 1.5 * \
                R.from_rotvec(
                    np.pi/2 * (i-1.0) * np.array([0, 0, 1])).apply(info['manip_axis'])

        desired = np.tile(obs['object_position'], 3) + \
            CUBE_SIZE * np.hstack(locs)

        # desired[3*info['manip_arm']: 3*info['manip_arm'] + 2] -= 0.4*CUBE_SIZE
        # desired[3*info['manip_arm']+2] = 0.1
        # desired[3*info['manip_arm']:3*info['manip_arm']+3] = obs['object_position']
        # desired[3*info['manip_arm']+2] = 0.1

        err = desired - current
        err_mag = np.linalg.norm(err[:3])
        if err_mag < 0.1:
            self.err_sum += err

        torque = self.get_torque_action(
            obs, self.k_p * err + 0.001 * self.err_sum)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=1)

        if np.linalg.norm(err) < 2 * EPS:
            info['previous_state'] = obs["robot_position"]
            self.reset()
            return action, self.next_state, info
        else:
            if self.time_exceeded():
                self.reset()
                return action, self.failure_state, info
            else:
                return action, self, info


class PreIntoState(SimpleState):
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.start_time = None
        self.time_exceed_threshold = 10.0
        self.grasp_check_failed_count = 0
        self.t = 0
        self.interval_ctr = 0
        self.interval = 100
        self.gain_increase_factor = 1.2
        self.max_interval_ctr = 20
        self.init_gain()

    def init_gain(self):
        if self.env.simulation:
            self.k_p = 1.4
        else:
            self.k_p = 1.75

    def update_gain(self):
        # if self.env.simulation:
        #     return
        self.t += 1
        if self.t % self.interval == 0 and self.interval_ctr < self.max_interval_ctr:
            self.k_p *= self.gain_increase_factor
            self.interval_ctr += 1

    def reset(self):
        self.init_gain()
        self.t = 0
        self.interval_ctr = 0
        self.start_time = None
        self.grasp_check_failed_count = 0
        self.success_ctr = 0

    def success(self):
        return self.success_ctr > 50

    def object_grasped(self, obs):
        current = self._get_tip_poses(obs)
        current_x = current[0::3]
        current_y = current[1::3]
        difference_x = [abs(p1 - p2)
                        for p1 in current_x for p2 in current_x if p1 != p2]
        difference_y = [abs(p1 - p2)
                        for p1 in current_y for p2 in current_y if p1 != p2]

        close_x = any(d < 0.0001 for d in difference_x)
        close_y = any(d < 0.0001 for d in difference_y)
        close = close_x and close_y

        if close:
            self.grasp_check_failed_count += 1

        return self.grasp_check_failed_count < 5

    def __call__(self, obs, info={}):
        if self.start_time is None:
            self.start_time = time.time()

        self.update_gain()
        current = self._get_tip_poses(obs)

        desired = np.tile(obs["object_position"], 3)
        # desired[3*info['manip_arm']+2] += 0.2*CUBE_SIZE
        # desired[3*info['manip_arm']+2] = 0.1
        # desired[3*info['manip_arm']:3*info['manip_arm']+3] = obs['object_position']
        desired[3*info['manip_arm']+2] = 0.02

        err = desired - current

        # Lower force of manip arm
        err[3*info['manip_arm']:3*info['manip_arm'] + 3] *= 0.4

        # Read Tip Force
        tip_forces = obs["tip_force"] - info["force_offset"]
        switch = True
        for f in tip_forces:
            if f < 0.04:
                switch = False

        # Override with small diff
        diff = obs["robot_position"] - info['previous_state']
        info['previous_state'] = obs["robot_position"]

        if np.amax(diff) < 5e-5:
            switch = True

        desired = np.tile(obs["object_position"], 3)
        # desired[3*info['manip_arm']:3*info['manip_arm']+3] = obs['object_position']
        # desired[3*info['manip_arm']+2] = 0.1

        err = desired - current
        torque = self.get_torque_action(obs, self.k_p * err)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=1)

        if switch:
            self.success_ctr += 1

        if self.success():
            grasp_sampler = GraspSampler(
                self.env, obs['object_position'], obs['object_orientation'])
            custom_grasp = [grasp_sampler.get_custom_grasp(
                obs['robot_tip_positions'])]
            # grasp, path = get_planned_grasp(self.env, obs['object_position'], obs['object_orientation'],
            #                                 obs['goal_object_position'], obs['goal_object_orientation'],
            #                                 tight=True, heuristic_grasps=custom_grasp)
            info['grasp'] = custom_grasp[0]
            # info['path'] = path
            info['pregoal_state'] = [0., 0., 0.04]
            info['pregoal_state'][2] += 0.04
            self.reset()
            return action, self.next_state, info
        else:
            if self.time_exceeded() or not self.object_grasped(obs):
                self.reset()
                return action, self.failure_state, info
            else:
                return action, self, info


class PreGoalState(GoalWithOrientState):
    def __init__(self, env, k_p_goal=0.45, k_p_into=0.22, k_i_goal=0.004, k_p_ang=0.04, k_i_ang=0.001, gain_increase_factor=1.04, interval=1800, max_interval_ctr=1800, pitch_rot_factor=1.5, pitch_lift_factor=2.0):
        # super().__init__(env, **kwargs)
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.goal_err_sum = np.zeros(9)
        self.ang_err_sum = np.zeros(9)
        self.grasp_check_failed_count = 0
        self.start_time = None
        self.time_exceed_threshold = 20.0
        self.success_ctr = 0
        self.gain_increase_factor = gain_increase_factor
        self.max_interval_ctr = max_interval_ctr
        self.interval = interval
        self.init_k_p_goal = k_p_goal
        self.init_k_p_into = k_p_into
        self.init_k_i_goal = k_i_goal
        self.init_k_p_ang = k_p_ang
        self.init_k_i_ang = k_i_ang
        self.pitch_rot_factor = pitch_rot_factor
        self.pitch_lift_factor = pitch_lift_factor
        self.next_state = None
        self.failure_state = None
        if self.env.simulation:
            self.frameskip = 3
            self.max_k_p = 1.5
        else:
            self.frameskip = 4
            self.max_k_p = 0.8


    def init_gain(self, k_p_goal=0.65, k_p_into=0.2, k_i_goal=0.004, k_p_ang=0.04, k_i_ang=0.001):
        if self.env.simulation:
            self.k_p_goal = self.init_k_p_goal
            self.k_p_into = self.init_k_p_into
            self.k_i_goal = self.init_k_i_goal
            self.k_p_ang = self.init_k_p_ang
            self.k_i_ang = self.init_k_i_ang
        else:
            self.k_p_goal = k_p_goal
            self.k_p_into = k_p_into
            self.k_i_goal = k_i_goal
            self.k_p_ang = k_p_ang
            self.k_i_ang = k_i_ang

    def success(self):
        return self.success_ctr > 30

    def __call__(self, obs, info={}):
        if info['is_yaw_orient']:
            return self.get_action(frameskip=0), self.next_state, info
        self.update_gain()
        self.k_p_goal = min(self.max_k_p, self.k_p_goal)

        if self.start_time is None:
            self.start_time = time.time()

        current = self._get_tip_poses(obs)
        desired = np.tile(obs["object_position"], 3)
        # desired[3*info['manip_arm']:3*info['manip_arm']+3] = obs['object_position']
        # desired[3*info['manip_arm']+2] += 0.065

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)
        # Lower force of manip arm
        into_err[3*info['manip_arm']:3*info['manip_arm'] + 3] *= 0.02

        goal = info['pregoal_state']
        goal[2] = self.pitch_lift_factor * CUBE_SIZE
        goal = np.tile(goal, 3)
        goal_err = goal - desired
        goal_err[3*info['manip_arm']:3*info['manip_arm'] + 3] *= 0.
        goal_err[3*info['manip_arm']+2] *= 0.0

        rot_err = np.zeros(9)
        rot_err[3*info['manip_arm']:3*info['manip_arm'] + 3] = obs["object_position"] + \
            np.array([0, 0, self.pitch_rot_factor * CUBE_SIZE])
        rot_err[3*info['manip_arm']:3*info['manip_arm'] +
                3] -= current[3*info['manip_arm']:3*info['manip_arm'] + 3]

        err_mag = np.linalg.norm(goal_err[:3])
        if err_mag < 0.1:
            self.goal_err_sum += goal_err
            self.ang_err_sum += rot_err

        # Once manip arm is overhead, drop
        diff = np.linalg.norm(
            current[3*info['manip_arm']:3*info['manip_arm']+2] - obs["object_position"][:2])

        # 0.05 * into_err + 0.1 * goal_err + 0.25 * rot_err
        force = self.k_p_into * into_err + self.k_p_goal * goal_err + self.k_i_goal * \
            self.goal_err_sum + self.k_p_ang * rot_err + self.k_i_ang * self.ang_err_sum
        torque = self.get_torque_action(obs, force)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=self.frameskip)

        if diff < 0.01:
            self.success_ctr += 1
        # else:
        #     if self.success():
        #         self.gain_increase_factor = self.init_gain_increase_factor
        #         self.success_ctr = 0
        #         self.start_time = time.time()

        if self.success():
            self.gain_increase_factor = 1.0
            info['manip_angle'] -= 90
            self.reset()
            return action, self.next_state, info

        if self.object_grasped(obs, info['grasp']):
            return action, self, info
        else:
            self.reset()
            return self.get_action(frameskip=0), self.next_state, info


class PreGoalState2(SimpleState):
    def __init__(self, env, k_p_goal=0.4, k_p_into=0.2, k_i_goal=0.004, k_p_ang=0.04, k_i_ang=0.001, gain_increase_factor=1.04, interval=1800, max_interval_ctr=1800):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.goal_err_sum = np.zeros(9)
        self.ang_err_sum = np.zeros(9)
        self.grasp_check_failed_count = 0
        self.start_time = None
        self.time_exceed_threshold = 20.0
        self.success_ctr = 0
        self.gain_increase_factor = gain_increase_factor
        self.max_interval_ctr = max_interval_ctr
        self.interval = interval
        self.init_k_p_goal = k_p_goal
        self.init_k_p_into = k_p_into
        self.init_k_i_goal = k_i_goal
        self.init_k_p_ang = k_p_ang
        self.init_k_i_ang = k_i_ang
        if self.env.simulation:
            self.frameskip = 3
            self.max_k_p = 1.5
        else:
            self.frameskip = 4
            self.max_k_p = 0.8
        self.init_gain(self.init_k_p_goal, self.init_k_p_into,
                       self.init_k_i_goal, self.init_k_p_ang, self.init_k_i_ang)

    def init_gain(self, k_p_goal=0.05, k_p_into=0.2, k_i_goal=0.00005, k_p_ang=0.01, k_i_ang=0.0001):
        if self.env.simulation:
            self.k_p_goal = k_p_goal
            self.k_p_into = k_p_into
            self.k_i_goal = k_i_goal
            self.k_p_ang = k_p_ang
            self.k_i_ang = k_i_ang
        else:
            self.k_p_goal = k_p_goal
            self.k_p_into = k_p_into
            self.k_i_goal = k_i_goal
            self.k_p_ang = k_p_ang
            self.k_i_ang = k_i_ang

    def update_gain(self):
        if self.env.simulation:
            return
        self.t += 1
        if self.t % self.interval == 0 and self.interval_ctr < self.max_interval_ctr:
            self.k_p_goal *= self.gain_increase_factor
            self.interval_ctr += 1

    def reset(self):
        self.init_gain(self.init_k_p_goal, self.init_k_p_into,
                       self.init_k_i_goal, self.init_k_p_ang, self.init_k_i_ang)
        self.interval_ctr = 0
        self.success_ctr = 0
        self.t = 0
        self.start_time = None
        self.gain_increase_factor = 1.04
        self.grasp_check_failed_count = 0
        self.goal_err_sum = np.zeros(9)
        self.ang_err_sum = np.zeros(9)

    def success(self):
        return self.success_ctr > 20

    def object_grasped(self, obs, grasp):
        T_cube_to_base = Transform(obs['object_position'],
                                   obs['object_orientation'])
        target_tip_pos = T_cube_to_base(grasp.cube_tip_pos)
        center_of_tips = np.mean(target_tip_pos, axis=0)
        dist = np.linalg.norm(target_tip_pos - obs['robot_tip_positions'])
        center_dist = np.linalg.norm(
            center_of_tips - np.mean(obs['robot_tip_positions'], axis=0))
        object_is_grasped = center_dist < 0.07 and dist < 0.10
        if object_is_grasped:
            self.grasp_check_failed_count = 0
        else:
            self.grasp_check_failed_count += 1
            print('incremented grasp_check_failed_count')
            print(f'center_dist: {center_dist:.4f}\tdist: {dist:.4f}')

        return self.grasp_check_failed_count < 5

    def __call__(self, obs, info={}):
        if not info['is_yaw_orient']:
            return self.get_action(frameskip=0), self.failure_state, info
        self.update_gain()
        yaw_diff = utils.yaw_orient_diff(
            obs['object_orientation'], obs['goal_object_orientation'])
        print("YAW DIFF: ", yaw_diff)
        if yaw_diff < 0.1 or 3.14 - yaw_diff < 0.1:
            return self.get_action(frameskip=0), self.next_state, info

        self.k_p_goal = min(self.max_k_p, self.k_p_goal)

        if self.start_time is None:
            self.start_time = time.time()

        current = self._get_tip_poses(obs)
        desired = np.tile(obs["object_position"], 3)
        desired[3*info['manip_arm']+2] = 0.033

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        goal = np.tile([0., 0., 0.0325], 3)
        goal_err = goal - desired
        err_mag = np.linalg.norm(goal_err[:3])

        angle, axis = utils._get_angle_axis(
            obs["object_orientation"], obs["goal_object_orientation"])
        ang_err = np.zeros(9)
        ang_err[:3] = -angle * \
            np.cross(into_err[:3] / np.linalg.norm(into_err[:3]), axis)
        ang_err[3:6] = -angle * \
            np.cross(into_err[3:6] / np.linalg.norm(into_err[3:6]), axis)
        ang_err[6:] = -angle * \
            np.cross(into_err[6:] / np.linalg.norm(into_err[6:]), axis)

        if err_mag < 0.1:
            self.goal_err_sum += goal_err
            self.ang_err_sum += ang_err

        if err_mag < EPS:
            self.success_ctr += 1
        else:
            if self.success():
                self.gain_increase_factor = 1.04
                self.success_ctr = 0
                self.start_time = time.time()

        force = self.k_p_into * into_err + self.k_p_goal * goal_err + self.k_i_goal * \
            self.goal_err_sum + self.k_p_ang * ang_err + self.k_i_ang * self.ang_err_sum
        # force = 0.2 * into_err + self.k_p * goal_err + 0.2 * ang_err
        torque = self.get_torque_action(obs, force)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=self.frameskip)

        info['path'] = None
        if self.success():
            self.gain_increase_factor = 1.0
            self.reset()
            return self.get_action(frameskip=0), self.next_state, info

        if self.object_grasped(obs, info['grasp']):
            return action, self, info
        else:
            self.reset()
            return self.get_action(frameskip=0), self.failure_state, info


class PreOrientState(SimpleState):
    def __init__(self, env):
        self.env = env
        self.start_time = None
        self.next_state = None
        self.failure_state = None
        self.time_exceed_threshold = 2.0
        self.grasp_check_failed_count = 0
        self.t = 0
        self.interval_ctr = 0
        self.interval = 100
        self.gain_increase_factor = 1.2
        self.max_interval_ctr = 20
        self.init_gain()

    def init_gain(self):
        if self.env.simulation:
            self.k_p = 0.5
        else:
            self.k_p = 1.2

    def update_gain(self):
        # if self.env.simulation:
        #     return
        self.t += 1
        if self.t % self.interval == 0 and self.interval_ctr < self.max_interval_ctr:
            self.k_p *= self.gain_increase_factor
            self.interval_ctr += 1

    def reset(self):
        self.init_gain()
        self.interval_ctr = 0
        self.t = 0
        self.interval_ctr = 0
        self.start_time = None
        self.grasp_check_failed_count = 0
        self.success_ctr = 0

    def success(self):
        return self.success_ctr > 10

    def object_grasped(self, obs):
        current = self._get_tip_poses(obs)
        current_x = current[0::3]
        current_y = current[1::3]
        difference_x = [abs(p1 - p2)
                        for p1 in current_x for p2 in current_x if p1 != p2]
        difference_y = [abs(p1 - p2)
                        for p1 in current_y for p2 in current_y if p1 != p2]

        close_x = any(d < 0.0001 for d in difference_x)
        close_y = any(d < 0.0001 for d in difference_y)
        close = close_x and close_y

        if close:
            self.grasp_check_failed_count += 1

        return self.grasp_check_failed_count < 5

    def __call__(self, obs, info={}):
        if self.start_time is None:
            self.start_time = time.time()

        self.update_gain()
        current = self._get_tip_poses(obs)

        desired = np.tile(obs["object_position"], 3)

        err = desired - current
        torque = self.get_torque_action(obs, self.k_p * err)
        action = self.get_action(torque=np.clip(
            torque, self.action_space_limits.low, self.action_space_limits.high), frameskip=1)

        # Read Tip Force
        tip_forces = obs["tip_force"] - info["force_offset"]
        switch = True
        for f in tip_forces:
            if f < 0.01:
                switch = False
        if switch:
            self.success_ctr += 1

        if self.success():
            grasp_sampler = GraspSampler(
                self.env, obs['object_position'], obs['object_orientation'])
            custom_grasp = [grasp_sampler.get_custom_grasp(
                obs['robot_tip_positions'])]
            # grasp, path = get_planned_grasp(self.env, obs['object_position'], obs['object_orientation'],
            #                                 obs['goal_object_position'], obs['goal_object_orientation'],
            #                                 tight=True, heuristic_grasps=custom_grasp)
            info['grasp'] = custom_grasp[0]
            # info['path'] = path
            self.reset()
            return action, self.next_state, info
        else:
            if self.time_exceeded() or not self.object_grasped(obs):
                self.reset()
                return action, self.failure_state, info
            else:
                return action, self, info
