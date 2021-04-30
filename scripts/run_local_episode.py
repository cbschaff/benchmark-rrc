#!/usr/bin/env python3
"""Run a single episode with a controller in simulation."""
import argparse

from env.make_env import make_env
from trifinger_simulation.tasks import move_cube
from mp.utils import set_seed
from combined_code import create_state_machine


def _init_env(goal_pose_dict, difficulty):
    eval_config = {
        'action_space': 'torque_and_position',
        'frameskip': 3,
        'reward_fn': 'competition_reward',
        'termination_fn': 'no_termination',
        'initializer': 'random_init',
        'monitor': False,
        'visualization': True,
        'sim': True,
        'rank': 0
    }

    set_seed(0)
    env = make_env(goal_pose_dict, difficulty, **eval_config)
    return env


def main():
    parser = argparse.ArgumentParser('args')
    parser.add_argument('difficulty', type=int)
    parser.add_argument('method', type=str, help="The method to run. One of 'mp-pg', 'cic-cg', 'cpc-tg'")
    parser.add_argument('--residual', default=False, action='store_true',
                        help="add to use residual policies. Only compatible with difficulties 3 and 4.")
    parser.add_argument('--bo', default=False, action='store_true',
                        help="add to use BO optimized parameters.")
    args = parser.parse_args()
    goal_pose = move_cube.sample_goal(args.difficulty)
    goal_pose_dict = {
        'position': goal_pose.position.tolist(),
        'orientation': goal_pose.orientation.tolist()
    }

    env = _init_env(goal_pose_dict, args.difficulty)
    state_machine = create_state_machine(args.difficulty, args.method, env,
                                         args.residual, args.bo)

    #####################
    # Run state machine
    #####################
    obs = env.reset()
    state_machine.reset()

    done = False
    while not done:
        action = state_machine(obs)
        obs, _, done, _ = env.step(action)


if __name__ == "__main__":
    main()
