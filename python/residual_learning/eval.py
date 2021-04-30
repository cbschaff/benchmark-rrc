"""Visualize a learned residual controller.
"""
from residual_learning.residual_sac import ResidualSAC
import dl
import os
import torch
import numpy as np
from dl import nest
import argparse
import yaml
from scipy.spatial.transform import Rotation as R


def _load_env_and_policy(logdir, t=None):

    gin_bindings = [
        "make_training_env.sim=True",
        "make_training_env.visualization=False",
        "make_training_env.monitor=True",
        "make_training_env.reward_fn='competition_reward'",
        "make_training_env.initializer='random_init'",
    ]

    config = os.path.join(logdir, 'config.gin')
    dl.load_config(config, gin_bindings)
    alg = ResidualSAC(logdir)
    alg.load(t)
    env = alg.env
    pi = alg.pi
    dl.rl.set_env_to_eval_mode(env)
    pi.eval()
    init_ob = alg.data_manager._ob
    if t is None:
        t = max(alg.ckptr.ckpts())
    return env, pi, alg.device, init_ob, t


def get_best_eval():
    if not os.path.exists('/logdir/eval/'):
        return None
    best_t = None
    best_r = -10 ** 9
    for eval in os.listdir('/logdir/eval/'):
        data = torch.load(os.path.join('/logdir/eval', eval))
        if best_r < data['mean_reward']:
            best_r = data['mean_reward']
            best_t = int(eval.split('.')[0])
            print(best_r, best_t)
    return best_t


def get_error(obs):
    pos_err = np.linalg.norm(obs['obs']['goal_object_position']
                             - obs['obs']['object_position'])
    r_goal = R.from_quat(obs['obs']['goal_object_orientation'])
    r_obj = R.from_quat(obs['obs']['object_orientation'])
    ori_err = (r_goal * r_obj.inv()).magnitude()
    return pos_err, ori_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=int, default=None, help="checkpoint timestep")
    parser.add_argument('-n', type=int, default=1, help="number of episodes")
    parser.add_argument('--base', default=False, action='store_true', help="visualize the base_policy")
    args = parser.parse_args()

    t = get_best_eval() if args.t is None else args.t
    env, pi, device, obs, ckpt = _load_env_and_policy('/logdir', t)

    def _to_torch(x):
        return torch.from_numpy(x).to(device)

    def _to_numpy(x):
        return x.cpu().numpy()

    eval_dir = '/logdir/test'
    os.makedirs(eval_dir, exist_ok=True)
    if args.base:
        output_path = os.path.join(eval_dir, 'base_policy.mp4')
    else:
        output_path = os.path.join(eval_dir, f'{ckpt:09d}.mp4')

    if os.path.exists(output_path):
        return

    episode_rewards = []
    pos_errs = []
    ori_errs = []
    drop_count = 0
    for i in range(args.n):
        obs = env.reset()
        reward = 0.0
        length = 0
        pos_err = None
        ori_err = None
        best_r = None

        done = False
        while not done:
            if args.base:
                action = np.zeros_like(obs['action']['torque'])
            else:
                obs = nest.map_structure(_to_torch, obs)
                with torch.no_grad():
                    action = pi(obs).action
                action = nest.map_structure(_to_numpy, action)
            obs, r, done, _ = env.step(action)
            if best_r is None or r > best_r:
                pos_err, ori_err = get_error(obs)
            length += 1
            reward += r.item()
        if length < 334:
            drop_count += 1
        else:
            pos_errs.append(pos_err)
            ori_errs.append(ori_err)
        episode_rewards.append(reward)

    data = {'episode_rewards': episode_rewards,
            'mean_reward': np.mean(episode_rewards).item(),
            'std_reward': np.std(episode_rewards).item(),
            'drop_frac': drop_count / args.n,
            'mean_pos_err': np.mean(pos_errs).item(),
            'mean_ori_err': np.mean(ori_errs).item(),
            'err_count': len(pos_errs)}

    if args.base:
        with open(os.path.join(eval_dir, 'base_policy.yaml'), 'w') as f:
            yaml.dump(data, f)
    else:
        with open(os.path.join(eval_dir, f'{ckpt}.yaml'), 'w') as f:
            yaml.dump(data, f)


if __name__ == "__main__":
    main()
