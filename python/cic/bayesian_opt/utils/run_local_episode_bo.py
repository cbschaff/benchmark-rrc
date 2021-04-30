#!/usr/bin/env python3
"""Run a single episode with our controller.
This script expects the following arguments in the given order:
 - Difficulty level (needed for reward computation)
 - goal pose of the object (as JSON string) (optional)
"""
import sys
import json

from trifinger_simulation.tasks import move_cube
from env.make_env import make_env
from mp.utils import set_seed

from cic import rotation_primitives

from cic.bayesian_opt.const import RELATIVE_MOVEMENT, EPISODE_LEN_SIM

from cic import parameters_new_grasp as cic_parameters_new_grasp
from cic.states import CICStateMachineLvl4 as CICwithCG

def _init_env(goal_pose_json, difficulty, path=None):
    eval_config = {
        'action_space': 'torque_and_position',
        'frameskip': 3,
        'reward_fn': 'competition_reward',
        'termination_fn': 'no_termination',
        'initializer': 'bo_init',
        'monitor': False,
        'visualization': False,
        'sim': True,
        'rank': 0,
        'episode_length': EPISODE_LEN_SIM
    }

    set_seed(0)
    goal_pose_dict = json.loads(goal_pose_json)
    env = make_env(goal_pose_dict, difficulty, path=path, **eval_config)
    return env


def main():
    with open("/ws/src/usercode/python/cic/bayesian_opt/content/diff.txt", 'r') as f:
        difficulty = int(f.readline())

    import numpy as np
    import pickle as pkl
    with open("/ws/src/usercode/python/cic/bayesian_opt/content/pos.pkl", 'rb') as f:
        init_arr_params = pkl.load(f)

    with open("/ws/src/usercode/python/cic/bayesian_opt/content/params.pkl", 'rb') as f:
        bo_params = np.asarray(pkl.load(f),dtype=float)

    with open("/ws/src/usercode/python/cic/bayesian_opt/content/iter_idx.txt", 'r') as f:
        curr_idx = int(f.readline())

    with open("/ws/src/usercode/python/cic/bayesian_opt/content/model.txt", 'r') as f:
        model_to_be_loaded = (f.readline())
    print ("The model to be loaded is: ", model_to_be_loaded)

    import os
    import shutil
    output_path = "/root_dir/" + sys.argv[2][1:]

    os.mkdir(output_path)

    print ("The current index is : " + str(curr_idx))
    init_information = np.asarray(init_arr_params[:, curr_idx], dtype=float)
    # Override goal pose:
    goal_pose_json = json.dumps({
        'position': init_information[7:10].tolist(),
        'orientation': init_information[10:14].tolist()
    })

    print ("The specified goal is: ", init_information[7:10])
    print ("The parameters to be optimized are: " + str(bo_params))

    # TODO: here: manipulate the params such that the right ones are optimized,...

    env = _init_env(goal_pose_json, difficulty, path=output_path)

    parameters = cic_parameters_new_grasp.CubeLvl4Params(env)
    parameters.orient_grasp_xy_lift += float(bo_params[0])
    parameters.orient_grasp_h_lift += float(bo_params[1])
    parameters.orient_gain_xy_lift_lift = float(bo_params[2])
    parameters.orient_gain_z_lift_lift = float(bo_params[3])
    parameters.orient_pos_gain_impedance_lift_lift = float(bo_params[4])
    parameters.orient_force_factor_lift = float(bo_params[5])
    parameters.orient_force_factor_rot_lift = float(bo_params[6])
    parameters.orient_int_orient_gain = float(bo_params[7])
    parameters.orient_int_pos_gain = float(bo_params[8])

    state_machine = CICwithCG(env, parameters=parameters)


    obs = env.reset()
    # TODO: eventually if we want to work with relative goal poses this should be accounted for here,...
    state_machine.reset()

    if (RELATIVE_MOVEMENT):
        # after the observation has been reset -> we now define the goal relative to the current position,...
        goal_list = rotation_primitives.calculate_mutltiple_goals(init_information, obs)
        env.goal_list = goal_list
        env.set_goal(orientation=goal_list[0][0], pos=goal_list[0][1], log_timestep=False)

    done = False
    while not done:

        if (RELATIVE_MOVEMENT):
            goal_list = rotation_primitives.calculate_mutltiple_goals(init_information, obs)
            env.goal_list = goal_list

        action = state_machine(obs)
        obs, _, done, _ = env.step(action)


if __name__ == "__main__":
    main()