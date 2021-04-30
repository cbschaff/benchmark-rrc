import numpy as np
from scipy.spatial.transform import Rotation


def no_termination(observation):
    return False


def position_close_to_goal(observation):
    dist_to_goal = np.linalg.norm(
        observation["desired_goal"]["position"]
        - observation["achieved_goal"]["position"]
    )
    return dist_to_goal < 0.01


def pos_and_rot_close_to_goal(observation):
    rot_error_deg = _orientation_error(observation) * 180
    allowance = 5.0
    return position_close_to_goal(observation) and rot_error_deg < allowance


def _orientation_error(observation):
    '''copied from reward_fns.py'''
    goal_rot = Rotation.from_quat(observation['desired_goal']['orientation'])
    actual_rot = Rotation.from_quat(observation['achieved_goal']['orientation'])
    error_rot = goal_rot.inv() * actual_rot
    return error_rot.magnitude() / np.pi


class StayCloseToGoal(object):
    def __init__(self, success_steps=30, is_level_4=False):
        self.counter = 0
        self.success_steps = success_steps
        self.goal_check = pos_and_rot_close_to_goal if is_level_4 else position_close_to_goal

    def __call__(self, observation):
        if self.goal_check(observation):
            self.counter += 1
            if self.counter >= self.success_steps:
                self.counter = 0
                return True
        else:
            self.counter = 0
        return False


stay_close_to_goal = StayCloseToGoal(is_level_4=False)
stay_close_to_goal_level_4 = StayCloseToGoal(is_level_4=True)
