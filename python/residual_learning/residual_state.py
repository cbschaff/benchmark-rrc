

from mp.states import State
from .residual_sac import ResidualSAC
import dl
from dl import nest
import torch
import numpy as np
import os


def load_policy(logdir):
    config = os.path.join(logdir, 'logs/config.gin')
    dl.load_config(config)
    alg = ResidualSAC(logdir)
    pi = alg.pi
    pi.eval()
    frameskip = alg.env.unwrapped.envs[0].frameskip
    max_torque = alg.env.unwrapped.envs[0].max_torque
    return pi, alg.device, frameskip, max_torque


class TrainingUtil(object):
    def __init__(self, env, device, max_torque):
        self.env = env
        self.device = device
        self.max_torque = max_torque
        self.ob_norm = torch.load(
            os.path.join(os.path.dirname(__file__), 'obs_norm.pt')
        )

    def _add_action_to_obs(self, obs, ac):
        obs = {'obs': obs, 'action': {}}
        if ac['torque'] is None:
            obs['action']['torque_enabled'] = np.array([0])
            obs['action']['torque'] = np.zeros_like(self.env.action_space['torque'].low)
        else:
            obs['action']['torque_enabled'] = np.array([1])
            obs['action']['torque'] = ac['torque']
        if ac['position'] is None:
            obs['action']['position_enabled'] = np.array([0])
            obs['action']['position'] = np.zeros_like(self.env.action_space['position'].low)
            obs['action']['tip_positions'] = obs['obs']['robot_tip_positions']
        else:
            obs['action']['position_enabled'] = np.array([1])
            obs['action']['position'] = ac['position']
            obs['action']['tip_positions'] = np.array(
                self.env.pinocchio_utils.forward_kinematics(ac['position'])
            )
        return obs

    def get_observation(self, obs, base_action):
        # Add base action to obs
        obs = self._add_action_to_obs(obs, base_action)

        # Add batch dim
        obs = nest.map_structure(lambda x: x[None], obs)

        # Normalize observations
        def norm(item):
            ob, mean, std = item
            if mean is not None:
                return (ob - mean) / std
            else:
                return ob
        obs = nest.map_structure(norm, nest.zip_structure(
            obs, self.ob_norm['mean'], self.ob_norm['std']))

        # convert to torch tensors
        return nest.map_structure(lambda x: torch.from_numpy(x).to(self.device), obs)

    def combine_actions(self, base_action, residual_action):
        action = {}
        if base_action['torque'] is not None:
            action['torque'] = np.clip(
                base_action['torque'] + self.max_torque * residual_action['torque'],
                self.env.action_space['torque'].low,
                self.env.action_space['torque'].high
            )
        else:
            action['torque'] = None

        action['position'] = base_action['position']

        frameskip = min(base_action['frameskip'], residual_action['frameskip'])
        action['frameskip'] = frameskip
        return action


class ResidualState(State):
    def __init__(self, base_state, logdir):
        self.env = base_state.env
        self.pi, self.device, self.frameskip, self.max_torque = load_policy(logdir)
        self.base_state = base_state
        self.util = TrainingUtil(self.env, self.device, self.max_torque)
        self.residual_action = {'torque': None, 'position': None, 'frameskip': 0}
        self.base_action = {'torque': None, 'position': None, 'frameskip': 0}

    def reset(self):
        self.base_state.reset()
        self.residual_action = {'torque': None, 'position': None, 'frameskip': 0}
        self.base_action = {'torque': None, 'position': None, 'frameskip': 0}

    def connect(self, *args, **kwargs):
        self.base_state.connect(*args, **kwargs)

    def __call__(self, obs, info={}):
        if self.base_action['frameskip'] == 0:
            self.base_action, next_state, info = self.base_state(obs, info)
            if next_state is not self.base_state:
                return self.base_action, next_state, info
        if self.residual_action['frameskip'] == 0:
            obs = self.util.get_observation(obs, self.base_action)
            # to numpy and remove batch dimension
            action = self.pi(obs).action.cpu().numpy()[0]
            self.residual_action = {
                'torque': action, 'position': None, 'frameskip': self.frameskip
            }

        action = self.util.combine_actions(self.base_action, self.residual_action)
        self.base_action['frameskip'] -= action['frameskip']
        self.residual_action['frameskip'] -= action['frameskip']
        return action, self, info
