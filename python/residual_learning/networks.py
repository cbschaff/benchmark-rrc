"""Define networks and ResidualPPO2."""
from dl.rl import PolicyBase, ContinuousQFunctionBase
from dl.rl import Policy, QFunction
from dl import nest, TanhDiagGaussian
import torch
import torch.nn as nn
import numpy as np
from residual_learning import modules
import gin


@gin.configurable
class NetworkParams(object):
    def __init__(self, size=256, embedding_size=64, pi_layers=2, vf_layers=3,
                 max_torque=0.1, init_std=1.0):
        self.size = size
        self.embedding_size = embedding_size
        self.pi_layers = pi_layers
        self.vf_layers = vf_layers
        self.max_torque = max_torque
        self.init_log_std = np.log(init_std)


class PolicyNet(PolicyBase):
    """Policy network."""

    def __init__(self, observation_space, action_space):
        self.params = NetworkParams()
        self.obs_filter = modules.ObservationFilter()
        super().__init__(observation_space, action_space)

    def build(self):
        """Build."""
        n_in = self.obs_filter.get_policy_ob_shape(self.observation_space)
        self.embedding = modules.ObservationEmbedding(n_in, self.params.embedding_size)
        layers = [
            nn.Linear(2 * self.params.embedding_size, self.params.size),
            nn.LayerNorm(self.params.size),
            nn.ReLU()
        ]
        for _ in range(self.params.pi_layers - 1):
            layers.append(nn.Linear(self.params.size, self.params.size))
            layers.append(nn.LayerNorm(self.params.size))
            layers.append(nn.ReLU())
        layers.append(TanhDiagGaussian(self.params.size, self.action_space.shape[0],
                                       constant_log_std=False))
        # initialize mean of the action distribution to zero.
        for p in layers[-1].fc_mean.parameters():
            nn.init.constant_(p, 0.)
        # scale init of action std_dev.
        nn.init.constant_(layers[-1].fc_logstd.bias.data,self.params.init_log_std)
        self.net = nn.Sequential(*layers)

    def forward(self, ob):
        """Forward."""
        ob = self.obs_filter.get_policy_observation(ob)
        ob = nest.map_structure(lambda z: z.float(), ob)
        return self.net(self.embedding(ob))


class QNet(ContinuousQFunctionBase):
    """Q Function."""

    def __init__(self, observation_space, action_space, mean, std):
        super().__init__(observation_space, action_space)
        self.ac_mean = mean['action']
        self.ac_std = std['action']
        self.device = None

    def _to_torch(self, x):
        if x is None:
            return x
        return torch.from_numpy(x).float().to(self.device)

    def _unnorm_action(self, item):
        ac, mean, std = item
        if mean is None:
            return ac
        return std * ac + mean

    def _norm_action(self, item):
        ac, mean, std = item
        if mean is None:
            return ac
        return (ac - mean) / std

    def build(self):
        """Build."""
        self.params = NetworkParams()
        self.obs_filter = modules.ObservationFilter()
        n_in = self.obs_filter.get_value_fn_ob_shape(self.observation_space)
        self.embedding = modules.ObservationEmbedding(n_in, self.params.embedding_size)
        layers = [
            nn.Linear(2 * self.params.embedding_size, self.params.size),
            nn.LayerNorm(self.params.size),
            nn.ReLU()
        ]
        for _ in range(self.params.vf_layers - 1):
            layers.append(nn.Linear(self.params.size, self.params.size))
            layers.append(nn.LayerNorm(self.params.size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.params.size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, ob, ac):
        """Forward."""
        ob = self.obs_filter.get_value_fn_observation(ob)
        ob = nest.map_structure(lambda z: z.float(), ob)
        if self.device is None:
            self.device = nest.flatten(ob)[0].device
            self.ac_mean = nest.map_structure(self._to_torch, self.ac_mean)
            self.ac_std = nest.map_structure(self._to_torch, self.ac_std)
        # combine actions, but action observations are normalized so we have to
        # unnormalize them first
        combined_ac = nest.map_structure(
            self._unnorm_action,
            nest.zip_structure(ob['action'], self.ac_mean, self.ac_std)
        )
        combined_ac['torque'] = (combined_ac['torque']
                                 + self.params.max_torque * ac)
        ob['action'] = nest.map_structure(
            self._norm_action,
            nest.zip_structure(combined_ac, self.ac_mean, self.ac_std)
        )
        return self.net(self.embedding(ob))


class QNetAppend(ContinuousQFunctionBase):
    """Q Function."""

    def build(self):
        """Build."""
        self.params = NetworkParams()
        self.obs_filter = modules.ObservationFilter()
        n_in = self.obs_filter.get_value_fn_ob_shape(self.observation_space)
        self.embedding = modules.ObservationEmbedding(n_in, self.params.embedding_size)
        self.action_embedding = nn.Sequential(
            nn.Linear(9, self.params.embedding_size),
            nn.ReLU()
        )
        layers = [
            nn.Linear(3 * self.params.embedding_size, self.params.size),
            nn.ReLU()
        ]
        for _ in range(self.params.vf_layers - 1):
            layers.append(nn.Linear(self.params.size, self.params.size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.params.size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, ob, ac):
        """Forward."""
        ob = self.obs_filter.get_value_fn_observation(ob)
        ob = nest.map_structure(lambda z: z.float(), ob)
        emb = torch.cat([self.embedding(ob), self.action_embedding(ac)], dim=1)
        return self.net(emb)


@gin.configurable
def policy_fn(env):
    """Create policy."""
    return Policy(PolicyNet(env.observation_space, env.action_space))


@gin.configurable
def qf_fn(env):
    """Create q function network."""
    return QFunction(QNet(env.observation_space, env.action_space,
                          env.venv.mean, env.venv.std))


@gin.configurable
def qf_append_fn(env):
    """Create q function network."""
    return QFunction(QNetAppend(env.observation_space, env.action_space))
