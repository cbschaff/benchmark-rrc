import gin
import numpy as np
import gym
from mp.const import CUBOID_MASS


DYNAMICS_PARAMS = {
    'mass': [0.26, 0.25, 0.021],
    'restitution': 0.8,
    'jointDamping': 0.0,
    'lateralFriction': 0.1,
    'spinningFriction': 0.1,
    'rollingFriction': 0.1,
    'linearDamping': 0.5,
    'angularDamping': 0.5,
    'contactStiffness': 0.1,
    'contactDamping': 0.05
}


class Parameter(object):
    def __init__(self, name, low, high):
        self.name = name
        self.low = np.array(low)
        self.high = np.array(high)
        self.shape = self.low.shape

    def sample(self):
        return np.random.rand(*self.shape) * (self.high - self.low) + self.low


class ParameterDict(object):
    def __init__(self, *params):
        self.names = [p.name for p in params]
        self.params = params

    def sample(self):
        return {p.name: p.sample() for p in self.params}

    def test(self, name, low=False, high=False):
        params = self.sample()
        if name not in self.names:
            raise ValueError(f"Can't test parameter. Invalid name: {name}")
        if low and high:
            raise ValueError("Can't test parameter. You must set exactly one of 'low' or 'high' to True.")
        if low:
            params[name] = self.params[name].low
        elif high:
            params[name] = self.params[name].high
        else:
            raise ValueError("Can't test parameter. You must set 'low' or 'high' to True.")
        return params


@gin.configurable
class TriFingerRandomizer(object):
    def __init__(self,
                 dynamics_scale=0.1,
                 cube_mass_scale=1.25,
                 position_scale=0.01,
                 velocity_scale=0.005,
                 torque_scale=0.005,
                 tip_force_scale=0.015,
                 tip_force_offset=0.095,
                 cube_rot_var=0.1,
                 cube_pos_var=0.001,
                 action_position=0.001,
                 action_torque=0.001,
                 timestep_low=0.003,
                 timestep_high=0.005):
        dynamics_params = []
        for name, p in DYNAMICS_PARAMS.items():
            low = np.array(p) * (1.0 - dynamics_scale)
            high = np.array(p) * (1.0 + dynamics_scale)
            dynamics_params.append(Parameter(name, low, high))

        dynamics_params.append(
            Parameter('cube_mass', (2 - cube_mass_scale) * CUBOID_MASS, cube_mass_scale * CUBOID_MASS)
        )
        self.dynamics_params = ParameterDict(*dynamics_params)

        self.robot_params = ParameterDict(*[
            Parameter('robot_position', 9 * [-position_scale], 9 * [position_scale]),
            Parameter('robot_velocity', 9 * [-velocity_scale], 9 * [velocity_scale]),
            Parameter("robot_torque", 9 * [-torque_scale], 9 * [torque_scale]),
            Parameter(
                "tip_force",
                3 * [tip_force_offset - tip_force_scale],
                3 * [tip_force_offset + tip_force_scale],
            ),
        ])
        self.cube_params = ParameterDict(*[
            Parameter('cube_pos', 3 * [-cube_pos_var], 3 * [cube_pos_var]),
            Parameter('cube_ori', 3 * [-cube_rot_var], 3 * [cube_rot_var])
        ])

        self.action_params = ParameterDict(*[
            Parameter('action_position', 9 * [-action_position], 9 * [action_position]),
            Parameter('action_torque', 9 * [-action_torque], 9 * [action_torque])
        ])

        self.timestep = Parameter('timestep', timestep_low, timestep_high)

    def sample_timestep(self):
        return self.timestep.sample()

    def sample_dynamics(self):
        return self.dynamics_params.sample()

    def sample_robot_noise(self):
        return self.robot_params.sample()

    def sample_cube_noise(self):
        return self.cube_params.sample()

    def sample_action_noise(self):
        return self.action_params.sample()

    def get_parameter_space(self):
        p = self.sample_dynamics()
        n = int(sum([np.prod(v.shape) for v in p.values()]))
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n,))
