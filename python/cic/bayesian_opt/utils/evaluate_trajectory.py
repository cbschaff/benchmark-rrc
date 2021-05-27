#!/usr/bin/env python3

import os
import argparse
import robot_fingers
import numpy as np
from trifinger_simulation.tasks import move_cube
import json

"""Functions for sampling, validating and evaluating "move cube" goals."""
import json
import pickle as pkl

import numpy as np
from scipy.spatial.transform import Rotation



_ARENA_RADIUS = 0.195
_max_height = 0.1

def todegree(w):
    return w*180/np.pi


def torad(w):
    return w*np.pi/180

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def to_H(R, T=np.zeros(3)):
    H = np.eye(4)
    H[:-1,:-1] = R
    H[:-1,-1] = T
    return H

def closest_axis_2_userdefined(H, vec):
    # print (H)
    # print (np.linalg.inv(H[:-1,:-1]))
    min_angle = 190
    x_des = np.array(vec)
    index = 0
    sign = 0
    reverse = False
    for i in range(3):
        x = H[:-1, i]
        theta = todegree(angle(x, x_des))
        # print (theta)
        if theta > 90:
            theta = theta - 180
            if theta ==0:
                reverse = True
        if min_angle > np.abs(theta):
            min_angle = np.abs(theta)
            index = i
            if theta == 0.:
                if reverse:
                    sign = -1
                else:
                    sign = 1
            else:
                sign = np.sign(theta)
    return min_angle, index, sign

def evaluate_state(goal_pose, actual_pose, difficulty):
    """Compute cost of a given cube pose.  Less is better.

    Args:
        goal_pose:  Goal pose of the cube.
        actual_pose:  Actual pose of the cube.
        difficulty:  The difficulty level of the goal (see
            :func:`sample_goal`).  The metric for evaluating a state differs
            depending on the level.

    Returns:
        Cost of the actual pose w.r.t. to the goal pose.  Lower value means
        that the actual pose is closer to the goal.  Zero if actual == goal.
    """

    def weighted_position_error():
        range_xy_dist = _ARENA_RADIUS * 2
        range_z_dist = _max_height

        xy_dist = np.linalg.norm(
            goal_pose.position[:2] - actual_pose.position[:2]
        )
        z_dist = abs(goal_pose.position[2] - actual_pose.position[2])

        # weight xy- and z-parts by their expected range
        return (xy_dist / range_xy_dist + z_dist / range_z_dist) / 2

    if difficulty in (1, 2, 3):
        # consider only 3d position
        return weighted_position_error()
    elif difficulty == 4:
        # consider whole pose
        scaled_position_error = weighted_position_error()

        # https://stackoverflow.com/a/21905553
        goal_rot = Rotation.from_quat(goal_pose.orientation)
        actual_rot = Rotation.from_quat(actual_pose.orientation)
        error_rot = goal_rot.inv() * actual_rot
        orientation_error = error_rot.magnitude()

        # scale both position and orientation error to be within [0, 1] for
        # their expected ranges
        scaled_orientation_error = orientation_error / np.pi

        scaled_error = (scaled_position_error + scaled_orientation_error) / 2
        return scaled_error

        # Use DISP distance (max. displacement of the corners)
        # goal_corners = get_cube_corner_positions(goal_pose)
        # actual_corners = get_cube_corner_positions(actual_pose)
        # disp = max(np.linalg.norm(goal_corners - actual_corners, axis=1))
    else:
        raise ValueError("Invalid difficulty %d" % difficulty)


def evaluate_state_orientation(goal_pose, actual_pose, difficulty):
    goal_rot = Rotation.from_quat(goal_pose.orientation)
    actual_rot = Rotation.from_quat(actual_pose.orientation)
    error_rot = goal_rot.inv() * actual_rot
    orientation_error = error_rot.magnitude()

    # scale both position and orientation error to be within [0, 1] for
    # their expected ranges
    scaled_orientation_error = orientation_error / np.pi
    return scaled_orientation_error

def reward_rot_lift(goal_pose, cube_pose, difficulty):
    goal_orient = Rotation.from_quat(goal_pose.orientation)
    current_orient = Rotation.from_quat(cube_pose.orientation)
    theta1, index1, sign1 = closest_axis_2_userdefined(to_H(goal_orient.as_matrix()),[0,0,1])
    theta2, index2, sign2 = closest_axis_2_userdefined(to_H(current_orient.as_matrix()),[0,0,1])

    if ((index1==index2) and (sign1==sign2)):
        return 0
    else:
        return 1


def compute_reward_rot_lift(logdir, simulation):
    with open(os.path.join(logdir, "goal.json"), 'r') as f:
        goal = json.load(f)
    difficulty = goal['difficulty']
    goal_pose = move_cube.Pose(position=np.array(goal['goal']['position']),
                               orientation=np.array(goal['goal']['orientation']))

    indice = goal['reachstart']
    if (indice==-1):
        return -10000, False

    if (simulation==0):
        min_length = 40000
        log = robot_fingers.TriFingerPlatformWithObjectLog(
            os.path.join(logdir, "robot_data.dat"),
            os.path.join(logdir, "camera_data.dat"),
        )
        reward = 0.0
        count = 0
        for t in range(indice, log.get_last_timeindex() + 1):
            count += 1
            camera_observation = log.get_camera_observation(t)
            reward -= reward_rot_lift(
                goal_pose, camera_observation.filtered_object_pose, difficulty

            )
            if (count==min_length):
                break
        if (count==min_length):
            return reward, True
        else:
            return -10000, False

    else:
        min_length = 10000#45000 #less since less rate
        path = os.path.join(logdir, 'observations.pkl')
        with open(path, 'rb') as handle:
            observations = pkl.load(handle)
        reward = 0.0
        ex_state = move_cube.sample_goal(difficulty=-1)
        count = 0
        for i in range(indice,len(observations)):
            count += 1
            ex_state.position = observations[i]["achieved_goal"]["position"]
            ex_state.orientation = observations[i]["achieved_goal"]["orientation"]
            reward -= reward_rot_lift(
                goal_pose, ex_state, difficulty

            )
            if (count==min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False

def compute_reward_rot_ground(logdir, simulation):
    with open(os.path.join(logdir, "goal.json"), 'r') as f:
        goal = json.load(f)
    difficulty = goal['difficulty']
    goal_pose = move_cube.Pose(position=np.array(goal['goal']['position']),
                               orientation=np.array(goal['goal']['orientation']))

    indice = goal['reachstart']
    if (indice==-1):
        return -10000, False

    if (simulation==0):
        min_length = 40000
        log = robot_fingers.TriFingerPlatformWithObjectLog(
            os.path.join(logdir, "robot_data.dat"),
            os.path.join(logdir, "camera_data.dat"),
        )
        reward = 0.0
        count = 0
        for t in range(indice, log.get_last_timeindex() + 1):
            count += 1
            camera_observation = log.get_camera_observation(t)
            reward -= evaluate_state_orientation(
                goal_pose, camera_observation.filtered_object_pose, difficulty

            )
            if (count==min_length):
                break
        if (count==min_length):
            return reward, True
        else:
            return -10000, False

    else:
        min_length = 10000#45000 #less since less rate
        path = os.path.join(logdir, 'observations.pkl')
        with open(path, 'rb') as handle:
            observations = pkl.load(handle)
        reward = 0.0
        ex_state = move_cube.sample_goal(difficulty=-1)
        count = 0
        for i in range(indice,len(observations)):
            count += 1
            ex_state.position = observations[i]["achieved_goal"]["position"]
            ex_state.orientation = observations[i]["achieved_goal"]["orientation"]
            reward -= evaluate_state_orientation(
                goal_pose, ex_state, difficulty

            )
            if (count==min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False


def compute_reward_18(logdir, simulation):
    with open(os.path.join(logdir, "goal.json"), 'r') as f:
        goal = json.load(f)
    difficulty = goal['difficulty']
    goal_pose = move_cube.Pose(position=np.array(goal['goal']['position']),
                               orientation=np.array(goal['goal']['orientation']))

    indice = goal['reachstart']
    if (indice==-1):
        return -10000, False

    if (simulation==0):
        min_length = 18000
        log = robot_fingers.TriFingerPlatformWithObjectLog(
            os.path.join(logdir, "robot_data.dat"),
            os.path.join(logdir, "camera_data.dat"),
        )
        reward = 0.0
        count = 0
        for t in range(log.get_first_timeindex(), log.get_last_timeindex() + 1):
            count += 1
            camera_observation = log.get_camera_observation(t)
            reward -= evaluate_state(
                goal_pose, camera_observation.filtered_object_pose, difficulty

            )
            if (count == min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False

    else:
        min_length = 10000
        path = os.path.join(logdir, 'observations.pkl')
        with open(path, 'rb') as handle:
            observations = pkl.load(handle)
        reward = 0.0
        count = 0
        ex_state = move_cube.sample_goal(difficulty=-1)
        for i in range(len(observations)):
            count += 1
            ex_state.position = observations[i]["achieved_goal"]["position"]
            ex_state.orientation = observations[i]["achieved_goal"]["orientation"]
            reward -= evaluate_state(
                goal_pose, ex_state, difficulty

            )
            if (count == min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False

def compute_reward_adaptive(logdir, simulation, TOTALTIMESTEPS):
    with open(os.path.join(logdir, "goal.json"), 'r') as f:
        goal = json.load(f)
    difficulty = goal['difficulty']
    goal_pose = move_cube.Pose(position=np.array(goal['goal']['position']),
                               orientation=np.array(goal['goal']['orientation']))

    indice = goal['reachstart']
    if (indice==-1):
        return -10000, False

    if (simulation==0):
        min_length = TOTALTIMESTEPS
        log = robot_fingers.TriFingerPlatformWithObjectLog(
            os.path.join(logdir, "robot_data.dat"),
            os.path.join(logdir, "camera_data.dat"),
        )
        reward = 0.0
        count = 0
        for t in range(log.get_first_timeindex(), log.get_last_timeindex() + 1):
            count += 1
            camera_observation = log.get_camera_observation(t)
            reward -= evaluate_state(
                goal_pose, camera_observation.filtered_object_pose, difficulty

            )
            if (count == min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False

    else:
        min_length = TOTALTIMESTEPS
        path = os.path.join(logdir, 'observations.pkl')
        with open(path, 'rb') as handle:
            observations = pkl.load(handle)
        reward = 0.0
        count = 0
        ex_state = move_cube.sample_goal(difficulty=-1)
        for i in range(len(observations)):
            count += 1
            ex_state.position = observations[i]["achieved_goal"]["position"]
            ex_state.orientation = observations[i]["achieved_goal"]["orientation"]
            reward -= evaluate_state(
                goal_pose, ex_state, difficulty

            )
            if (count == min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False

def compute_reward_adaptive_ORIENT(logdir, simulation, TOTALTIMESTEPS):
    with open(os.path.join(logdir, "goal.json"), 'r') as f:
        goal = json.load(f)
    difficulty = goal['difficulty']
    goal_pose = move_cube.Pose(position=np.array(goal['goal']['position']),
                               orientation=np.array(goal['goal']['orientation']))

    indice = goal['reachstart']
    if (indice==-1):
        return -10000, False

    if (simulation==0):
        min_length = TOTALTIMESTEPS
        log = robot_fingers.TriFingerPlatformWithObjectLog(
            os.path.join(logdir, "robot_data.dat"),
            os.path.join(logdir, "camera_data.dat"),
        )
        reward = 0.0
        count = 0
        for t in range(log.get_first_timeindex(), log.get_last_timeindex() + 1):
            count += 1
            camera_observation = log.get_camera_observation(t)
            reward -= evaluate_state(
                goal_pose, camera_observation.filtered_object_pose, difficulty

            )
            if (count == min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False

    else:
        min_length = TOTALTIMESTEPS
        path = os.path.join(logdir, 'observations.pkl')
        with open(path, 'rb') as handle:
            observations = pkl.load(handle)
        reward = 0.0
        count = 0
        ex_state = move_cube.sample_goal(difficulty=-1)
        for i in range(len(observations)):
            count += 1
            ex_state.position = observations[i]["achieved_goal"]["position"]
            ex_state.orientation = observations[i]["achieved_goal"]["orientation"]
            reward -= evaluate_state(
                goal_pose, ex_state, difficulty

            )
            if (count == min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False


def compute_reward_adaptive_behind(logdir, simulation, TOTALTIMESTEPS):
    with open(os.path.join(logdir, "goal.json"), 'r') as f:
        goal = json.load(f)
    difficulty = goal['difficulty']
    goal_pose = move_cube.Pose(position=np.array(goal['goal']['position']),
                               orientation=np.array(goal['goal']['orientation']))

    indice = goal['reachstart']
    if (indice==-1):
        return -10000, False

    if (simulation==0):
        min_length = TOTALTIMESTEPS
        log = robot_fingers.TriFingerPlatformWithObjectLog(
            os.path.join(logdir, "robot_data.dat"),
            os.path.join(logdir, "camera_data.dat"),
        )
        reward = 0.0
        count = 0
        for t in range(log.get_last_timeindex() - TOTALTIMESTEPS, log.get_last_timeindex() + 1):
            count += 1
            camera_observation = log.get_camera_observation(t)
            reward -= evaluate_state(
                goal_pose, camera_observation.filtered_object_pose, difficulty

            )
            if (count == min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False

    else:
        min_length = TOTALTIMESTEPS
        path = os.path.join(logdir, 'observations.pkl')
        with open(path, 'rb') as handle:
            observations = pkl.load(handle)
        reward = 0.0
        count = 0
        ex_state = move_cube.sample_goal(difficulty=-1)
        for i in range(len(observations)-TOTALTIMESTEPS-1,len(observations)):
            count += 1
            ex_state.position = observations[i]["achieved_goal"]["position"]
            ex_state.orientation = observations[i]["achieved_goal"]["orientation"]
            reward -= evaluate_state(
                goal_pose, ex_state, difficulty

            )
            if (count == min_length):
                break
        if (count == min_length):
            return reward, True
        else:
            return -10000, False


def compute_reward(logdir, simulation):
    with open(os.path.join(logdir, "goal.json"), 'r') as f:
        goal = json.load(f)
    difficulty = goal['difficulty']
    goal_pose = move_cube.Pose(position=np.array(goal['goal']['position']),
                               orientation=np.array(goal['goal']['orientation']))

    if (simulation==0):
        log = robot_fingers.TriFingerPlatformWithObjectLog(
            os.path.join(logdir, "robot_data.dat"),
            os.path.join(logdir, "camera_data.dat"),
        )
        reward = 0.0
        for t in range(log.get_first_timeindex(), log.get_last_timeindex() + 1):
            camera_observation = log.get_camera_observation(t)
            reward -= evaluate_state(
                goal_pose, camera_observation.filtered_object_pose, difficulty

            )
        return reward, True
    else:
        path = os.path.join(logdir, 'observations.pkl')
        with open(path, 'rb') as handle:
            observations = pkl.load(handle)
        reward = 0.0
        ex_state = move_cube.sample_goal(difficulty=-1)
        for i in range(len(observations)):
            ex_state.position = observations[i]["achieved_goal"]["position"]
            ex_state.orientation = observations[i]["achieved_goal"]["orientation"]
            reward -= evaluate_state(
                goal_pose, ex_state, difficulty

            )
        return reward, True


if __name__ == '__main__':
    # we assume those 4 input args,...
    #simulation, path, decision_function
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="path to the log directory")
    parser.add_argument("simulation", help="path to the log directory", type=int)
    parser.add_argument("decisionfunction", help="path to the log directory")
    args = parser.parse_args()

    valid_eval = False
    if (args.decisionfunction=='standart'):
        reward, valid_eval = compute_reward(args.logdir, args.simulation)
    elif (args.decisionfunction=='standart_18'):
        reward, valid_eval = compute_reward_18(args.logdir, args.simulation)
    elif (args.decisionfunction=='rot_lift'):
        reward, valid_eval = compute_reward_rot_lift(args.logdir, args.simulation)
    elif (args.decisionfunction == 'rot_ground'):
        reward, valid_eval = compute_reward_rot_ground(args.logdir, args.simulation)
    elif (args.decisionfunction == 'standart_15'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,15000)
    elif (args.decisionfunction == 'standart_11'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,11000)
    elif (args.decisionfunction == 'standart_10'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,10000)
    elif (args.decisionfunction == 'standart_20'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,20000)
    elif (args.decisionfunction == 'standart_30'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,30000)
    elif (args.decisionfunction == 'standart_40'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,40000)
    elif (args.decisionfunction == 'standart_ORIENT_40'):
        reward, valid_eval = compute_reward_adaptive_ORIENT(args.logdir, args.simulation,40000)
    elif (args.decisionfunction == 'standart_45'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,45000)
    elif (args.decisionfunction == 'standart_50'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,50000)
    elif (args.decisionfunction == 'standart_60'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,60000)
    elif (args.decisionfunction == 'standart_70'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation,70000)
    elif (args.decisionfunction == 'standart_behind_10'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation, 10000)
    elif (args.decisionfunction == 'standart_behind_15'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation, 15000)
    elif (args.decisionfunction == 'standart_behind_20'):
        reward, valid_eval = compute_reward_adaptive(args.logdir, args.simulation, 20000)

        compute_reward_adaptive_behind



    with open(os.path.join(args.logdir, 'reward.json'), 'w') as f:
        json.dump({'reward': reward, 'valid': int(valid_eval)}, f)
