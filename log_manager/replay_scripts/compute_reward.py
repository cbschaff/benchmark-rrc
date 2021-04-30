#!/usr/bin/env python3

import os
import argparse
import robot_fingers
import numpy as np
from trifinger_simulation.tasks import move_cube
import json


def compute_reward(logdir):
    log = robot_fingers.TriFingerPlatformLog(os.path.join(logdir, "robot_data.dat"),
                                             os.path.join(logdir, "camera_data.dat"))
    with open(os.path.join(logdir, "goal.json"), 'r') as f:
        goal = json.load(f)
    difficulty = goal['difficulty']
    goal_pose = move_cube.Pose(position=np.array(goal['goal']['position']),
                               orientation=np.array(goal['goal']['orientation']))

    reward = 0.0
    for t in range(log.get_first_timeindex(), log.get_last_timeindex() + 1):
        camera_observation = log.get_camera_observation(t)
        reward -= move_cube.evaluate_state(
            goal_pose, camera_observation.filtered_object_pose, difficulty

        )
    return reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="path to the log directory")
    args = parser.parse_args()
    reward = compute_reward(args.logdir)
    with open(os.path.join(args.logdir, 'reward.json'), 'w') as f:
        json.dump({'reward': reward}, f)
