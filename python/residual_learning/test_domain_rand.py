#!/usr/bin/env python3
from code.make_env import make_training_env
from trifinger_simulation.tasks import move_cube

import statistics
import argparse
import pybullet as p
import time

import dl
import numpy as np


def main(args):
    goal = move_cube.sample_goal(args.difficulty)
    goal_dict = {
        'position': goal.position,
        'orientation': goal.orientation
    }
    eval_config = {
        'cube_goal_pose': goal_dict,
        'goal_difficulty': args.difficulty,
        'action_space': 'torque' if args.policy == 'fc' else 'torque_and_position',
        'frameskip': 3,
        'reward_fn': 'competition_reward',
        'termination_fn': 'no_termination',
        'initializer': 'training_init',
        'sim': True,
        'monitor': False,
        'rank': args.seed,
        'training': True
    }
    env = make_training_env(visualization=True, **eval_config)

    acc_rewards = []
    wallclock_times = []
    aligning_steps = []
    env_steps = []
    avg_rewards = []
    for i in range(args.num_episodes):
        start = time.time()
        is_done = False
        observation = env.reset()
        accumulated_reward = 0
        aligning_steps.append(env.unwrapped.step_count)

        #clear some windows in GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        #change camera parameters # You can also rotate the camera by CTRL + drag
        p.resetDebugVisualizerCamera( cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])

        step = 0
        while not is_done and step < 2000:
            step += 1
            action = env.action_space.sample()
            if isinstance(action, dict):
                action['torque'] *= 0
                action['position'] *= 0
            else:
                action *= 0
            observation, reward, is_done, info = env.step(action)
            accumulated_reward += reward
        acc_rewards.append(accumulated_reward)
        env_steps.append(env.unwrapped.step_count - aligning_steps[-1])
        avg_rewards.append(accumulated_reward / env_steps[-1])
        print("Episode {}\tAccumulated reward: {}".format(i, accumulated_reward))
        print("Episode {}\tAlinging steps: {}".format(i, aligning_steps[-1]))
        print("Episode {}\tEnv steps: {}".format(i, env_steps[-1]))
        print("Episode {}\tAvg reward: {}".format(i, avg_rewards[-1]))
        end = time.time()
        print('Elapsed:', end - start)
        wallclock_times.append(end - start)

    env.close()

    def _print_stats(name, data):
        print('======================================')
        print(f'Mean   {name}\t{np.mean(data):.2f}')
        print(f'Max    {name}\t{max(data):.2f}')
        print(f'Min    {name}\t{min(data):.2f}')
        print(f'Median {name}\t{statistics.median(data):2f}')

    print('Total elapsed time\t{:.2f}'.format(sum(wallclock_times)))
    print('Mean elapsed time\t{:.2f}'.format(sum(wallclock_times) / len(wallclock_times)))
    _print_stats('acc reward', acc_rewards)
    _print_stats('aligning steps', aligning_steps)
    _print_stats('step reward', avg_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=int, default=1, help="difficulty")
    parser.add_argument("--num_episodes", default=10, type=int, help="number of episodes to record a video")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--policy", default='mpfc', choices=["fc", "mpfc"], help="which policy to run")
    args = parser.parse_args()

    print('For faster recording, run `git apply faster_recording_patch.diff`. This temporary changes episode length and window size.')

    dl.rng.seed(args.seed)
    main(args)
