#!/usr/bin/env python3
import pybullet as p
import numpy as np
from mp.utils import get_rotation_between_vecs, slerp, Transform
from trifinger_simulation import TriFingerPlatform
from scipy.spatial.transform import Rotation

DEBUG = False
if DEBUG:
    color_set = ((1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5))


class PlanningAndForceControlPolicy:
    BO_num_tipadjust_steps = 50  # BO
    def __init__(self, env, obs, fc_policy, path, action_repeat=2*2,
                 adjust_tip=True, adjust_tip_ori=False):
        self.env = env
        self.fc_policy = fc_policy
        self.obs = obs
        self.path = path.repeat(action_repeat)
        self.grasp = self.path.grasp
        self.joint_sequence = self.path.joint_conf
        self.cube_sequence = self.path.cube
        self.adjust_tip = adjust_tip
        self._step = 0
        self._actions_in_progress = False
        self.adjust_tip_ori = adjust_tip_ori
        self.executing_tip_adjust = False
        if DEBUG:
            self.visual_markers = []
            self.vis_tip_center = None

    def at_end_of_sequence(self, step):
        return step >= len(self.cube_sequence)

    def add_tip_adjustments(self, obs):
        num_steps = self.BO_num_tipadjust_steps
        print("Appending to path....")
        # tip_pos = self.path.tip_path[-1]
        dir = 0.5 * (obs['goal_object_position'] - obs['object_position'])
        cube_pos = self.cube_sequence[-1][:3]
        cube_ori = p.getQuaternionFromEuler(self.cube_sequence[-1][3:])

        grasp = self.path.grasp
        if self.adjust_tip_ori:
            yaxis = np.array([0, 1, 0])
            goal_obj_yaxis = Rotation.from_quat(obs['goal_object_orientation']).apply(yaxis)
            obj_yaxis = Rotation.from_quat(cube_ori).apply(yaxis)
            diff_quat = get_rotation_between_vecs(obj_yaxis, goal_obj_yaxis)
            resolution = np.arange(0, 1, 1.0 / num_steps)
            interp_quat = slerp(np.array([0, 0, 0, 1]), diff_quat, resolution)

        warning_counter = 0
        warning_tips = []
        for i in range(num_steps):
            translation = cube_pos + i / num_steps * dir
            if self.adjust_tip_ori:
                rotation = (Rotation.from_quat(interp_quat[i]) * Rotation.from_quat(cube_ori)).as_quat()
            else:
                rotation = cube_ori
            goal_tip_pos = Transform(translation, rotation)(grasp.cube_tip_pos)
            q = obs['robot_position']

            for j, tip in enumerate(goal_tip_pos):
                q = self.env.pinocchio_utils.inverse_kinematics(j, tip, q)
                if q is None:
                    q = self.joint_sequence[-1]
                    # print(f'[tip adjustments] warning: IK solution is not found for tip {j}. Using the last joint conf')
                    warning_counter += 1
                    if j not in warning_tips:
                        warning_tips.append(j)
                    break
            if q is None:
                print('[tip adjustments] warning: IK solution is not found for all tip positions.')
                print(f'[tip adjustments] aborting tip adjustments (loop {i} / {num_steps})')
                break
            target_cube_pose = np.concatenate([
                translation,
                p.getEulerFromQuaternion(rotation)
            ])
            self.cube_sequence.append(target_cube_pose)
            self.joint_sequence.append(q)
            self.path.tip_path.append(goal_tip_pos)
        if warning_counter > 0:
            print(f'[tip adjustments] warning: IK solution is not found for {warning_counter} / {num_steps} times on tips {warning_tips}.')

    def __call__(self, obs):
        if not self._actions_in_progress:
            if np.linalg.norm(
                obs['robot_position'] - self.path.joint_conf[0]
            ).sum() > 0.25:
                print(
                    'large initial joint conf error:',
                    np.linalg.norm(obs['robot_position']
                                   - self.path.joint_conf[0])
                )
        self._actions_in_progress = True

        step = self._step
        if self.adjust_tip and self.at_end_of_sequence(step):
            self.add_tip_adjustments(obs)
        step = min(step, len(self.cube_sequence) - 1)
        target_cube_pose = self.cube_sequence[step]
        target_joint_conf = self.joint_sequence[step]

        torque = self.fc_policy(obs, target_cube_pose[:3],
                                p.getQuaternionFromEuler(target_cube_pose[3:]))
        action = {
            'position': np.asarray(target_joint_conf),
            'torque': torque
        }
        self._step += 1
        return self._clip_action(action)

    def _clip_action(self, action):
        tas = TriFingerPlatform.spaces.robot_torque.gym
        pas = TriFingerPlatform.spaces.robot_position.gym
        action['position'] = np.clip(action['position'], pas.low, pas.high)
        action['torque'] = np.clip(action['torque'], tas.low, tas.high)
        return action
