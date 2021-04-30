"""Define networks and ResidualPPO2."""
from dl.modules import TanhDiagGaussian, ProductDistribution
from dl import nest
import torch
import torch.nn as nn
import numpy as np


class ObservationFilter(object):
    """Filters information to policy and value networks for residual_learning
       and domain randomization."""

    def get_policy_ob_shape(self, ob_space):
        shapes = [
            ob_space['obs'][k].shape for k in ob_space['obs'].spaces
            if k not in ['clean', 'params', 'action']
        ]
        return np.sum([np.prod(s) for s in shapes])

    def get_value_fn_ob_shape(self, ob_space):
        if 'clean' not in ob_space['obs'].spaces:  # dict ob space with no domain randomization
            shapes = [ob_space['obs'][k].shape for k in ob_space['obs'].spaces]
            return np.sum([np.prod(s) for s in shapes])
        else:
            shapes = [ob_space['obs'][k].shape for k in ob_space['obs']['clean'].spaces]
            n = np.sum([np.prod(s) for s in shapes])
            return n + np.prod(ob_space['obs']['params'].shape)

    def get_policy_observation(self, ob):
        flat_obs = nest.flatten({
            k: v for k, v in ob['obs'].items() if k not in ['clean', 'params']
        })
        return {
            'obs': torch.cat([torch.flatten(ob, 1).float() for ob in flat_obs], dim=1),
            'action': ob['action']
        }

    def get_value_fn_observation(self, ob):
        if 'clean' not in ob['obs']:
            flat_obs = torch.cat([torch.flatten(o, 1)
                                  for o in nest.flatten(ob['obs'])],
                                 dim=1)
        else:
            flat_obs = nest.flatten(ob['obs']['clean']) + nest.flatten(ob['obs']['params'])
            flat_obs = torch.cat([torch.flatten(ob, 1).float() for ob in flat_obs],
                                 dim=1)
        return {
            'obs': flat_obs,
            'action': ob['action']
        }


class TorqueAndPositionAction(nn.Module):
    def __init__(self, n_in, n_out_torque, n_out_position, **kwargs):
        super().__init__()
        self.torque_dist = TanhDiagGaussian(n_in, n_out_torque, **kwargs)
        self.position_dist = TanhDiagGaussian(n_in, n_out_position, **kwargs)

    def forward(self, x):
        return ProductDistribution({
            'torque': self.torque_dist(x),
            'position': self.position_dist(x)
        })


class ObservationEmbedding(nn.Module):
    def __init__(self, obs_shape, embedding_size):
        super().__init__()

        def net(n_in):
            return nn.Sequential(
                nn.Linear(n_in, 2 * embedding_size),
                nn.LayerNorm(2 * embedding_size),
                nn.ReLU(),
                nn.Linear(2 * embedding_size, embedding_size),
                nn.LayerNorm(embedding_size)
            )

        self.torque_embedding = net(9)
        self.position_embedding = net(9)
        self.tip_pos_embedding = net(9)
        self.obs_embedding = net(obs_shape)
        self.activation = nn.ReLU()

    def forward(self, obs):
        action = obs['action']
        te = action['torque_enabled'] * self.torque_embedding(action['torque'])
        pe = action['position_enabled'] * (
            self.position_embedding(action['position'])
            + self.tip_pos_embedding(torch.flatten(action['tip_positions'], 1))
        )
        return self.activation(
            torch.cat([self.obs_embedding(obs['obs']), te + pe], dim=1)
        )
