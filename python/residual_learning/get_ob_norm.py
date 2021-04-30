import argparse
from dl import nest
from residual_learning.make_training_env import make_training_env
from residual_learning.state_machines import MPPGStateMachine
import torch
import numpy as np
import os


def get_norm_params(n, difficulty, use_domain_rand):
    term_fn = 'position_close_to_goal' if difficulty < 4 else 'pos_and_rot_close_to_goal'
    env = make_training_env(32, MPPGStateMachine, difficulty, 'torque_and_position',
                            frameskip=3,
                            sim=True,
                            visualization=False,
                            reward_fn='competition_reward',
                            termination_fn=term_fn,
                            initializer='training_init',
                            episode_length=3750,
                            monitor=False,
                            seed=0,
                            norm_observations=True,
                            max_torque=0.0,
                            max_position=0.0,  # set all residual actions to 0
                            denylist_states=['FailureState'],
                            domain_randomization=use_domain_rand
                            )
    env.steps = n
    env.find_norm_params()

    def get_var(std):
        if std is not None:
            return std ** 2
    return env.mean, nest.map_structure(get_var, env.std), env


class RunningNorm(object):
    def __init__(self, eps=1e-5):
        self.count = 0
        self.eps = eps

    def _update(self, item):
        mean, var, batch_mean, batch_var = item
        if mean is not None:
            delta = batch_mean - mean
            new_mean = mean + delta * (self.batch_count / self.new_count)
            new_var = self.count * var + self.batch_count * batch_var
            new_var += (delta**2) * self.count * (self.batch_count / self.new_count)
            new_var /= self.new_count
            mean[:] = new_mean
            var[:] = new_var

    def update(self, mean, var, count):
        if self.count == 0:
            self.mean = mean
            self.var = var
            self.count = count

        else:
            self.batch_count = count
            self.new_count = count + self.count
            nest.map_structure(
                self._update, nest.zip_structure(self.mean, self.var, mean, var)
            )
            self.count = self.new_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent.')
    parser.add_argument('n', type=int, help='steps for each difficulty')
    parser.add_argument('--domain_rand', default=False, action='store_true',
                        help=' use domain randomization')
    args = parser.parse_args()

    means, stds = [], []
    rn = RunningNorm()
    for difficulty in range(1, 5):
        mean, var, env = get_norm_params(args.n, difficulty, args.domain_rand)
        rn.update(mean, var, args.n)

    env.mean = rn.mean

    def get_std(x):
        if x is not None:
            return np.sqrt(x)
    env.std = nest.map_structure(get_std, rn.var)

    state_dict = env.state_dict()
    dirpath = os.path.dirname(__file__)
    f = 'obs_norm.pt' if not args.domain_rand else 'obs_norm_rand.pt'
    torch.save(state_dict, os.path.join(dirpath, f))
    env.close()
