import numpy as np

from pybullet_planning.utils import CIRCULAR_LIMITS

#####################################
# Reachability

def sample_reachable_base(robot, point, reachable_range=(0.25, 1.0)):
    from pybullet_planning.interfaces.env_manager.pose_transformation import unit_from_theta, point_from_pose
    radius = np.random.uniform(*reachable_range)
    x, y = radius*unit_from_theta(np.random.uniform(-np.pi, np.pi)) + point[:2]
    yaw = np.random.uniform(*CIRCULAR_LIMITS)
    base_values = (x, y, yaw)
    #set_base_values(robot, base_values)
    return base_values

def uniform_pose_generator(robot, gripper_pose, **kwargs):
    from pybullet_planning.interfaces.env_manager.pose_transformation import point_from_pose
    point = point_from_pose(gripper_pose)
    while True:
        base_values = sample_reachable_base(robot, point, **kwargs)
        if base_values is None:
            break
        yield base_values
        #set_base_values(robot, base_values)
        #yield get_pose(robot)
