#!/usr/bin/env python3
import numpy as np
import pybullet as p
from trifinger_simulation import TriFingerPlatform


class ZeroTorquePolicy(object):
    def __call__(self, *args, **kwargs):
        return np.zeros(9)


class CancelGravityPolicy(object):
    def __init__(self, env):
        self.id = env.platform.simfinger.finger_id

    def __call__(self, obs, *args, **kwargs):
        torque = p.calculateInverseDynamics(self.id,
                                            list(obs['robot_position']),
                                            list(obs['robot_velocity']),
                                            [0. for _ in range(9)])
        tas = TriFingerPlatform.spaces.robot_torque.gym
        return np.clip(np.array(torque), tas.low, tas.high)
