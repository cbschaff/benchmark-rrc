#!/usr/bin/env python3
from mp.const import COLLISION_TOLERANCE
workspace_id = 0

class CollisionConfig:
    def __init__(self, env):
        self.env = env
        self.finger_id = env.platform.simfinger.finger_id
        self.tip_ids = env.platform.simfinger.pybullet_tip_link_indices
        self.link_ids = env.platform.simfinger.pybullet_link_indices
        self.cube_id = env.platform.cube._object_id
        self. disabled_collisions = [((self.finger_id, tip_id), (self.cube_id, -1))
                                     for tip_id in self.tip_ids]

    def get_collision_conf(self, config_type):
        if config_type == "mpfc":  # previously: slacky_collision == True
            return {
                'body': self.finger_id,
                'joints': self.link_ids,
                'obstacles': [self.cube_id, workspace_id],
                'self_collisions': True,
                'extra_disabled_collisions': self.disabled_collisions,
                'max_distance': -COLLISION_TOLERANCE
            }
        elif config_type == "yaw_flip_path":
            # NOTE: This is for collision between finger tips!
            return {
                'body': self.finger_id,
                'joints': self.link_ids,
                'obstacles': [workspace_id],  # ignore collisions with cube
                'self_collisions': True,
                'extra_disabled_collisions': self.disabled_collisions,
                'max_distance': -COLLISION_TOLERANCE * 3
            }
        elif config_type == "yaw_flip_grasp":
            return {
                'body': self.finger_id,
                'joints': self.link_ids,
                'obstacles': [self.cube_id, workspace_id],
                'self_collisions': True,
                'extra_disabled_collisions': self.disabled_collisions,
                'max_distance': -COLLISION_TOLERANCE * 3  # larger collision admittance
            }
        else:  # previsouly: slacky_collision == False
            return {
                'body': self.finger_id,
                'joints': self.link_ids,
                'obstacles': [self.cube_id, workspace_id],
                'self_collisions': False
            }

    def get_collision_fn(self, config_type, diagnosis=False):
        from pybullet_planning.interfaces.robots.collision import get_collision_fn
        import functools
        config = self.get_collision_conf(config_type)
        return functools.partial(get_collision_fn(**config), diagnosis=diagnosis)
