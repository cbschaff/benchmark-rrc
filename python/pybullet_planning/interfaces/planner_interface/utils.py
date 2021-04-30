#!/usr/bin/env python3
import functools
from pybullet_planning.interfaces.robots.joint import get_joint_positions, get_joint_velocities, set_joint_positions_and_velocities
from pybullet_planning.interfaces.robots.body import get_pose, get_velocity, set_pose, set_velocity


def preserve_pos_and_vel(func):
    @functools.wraps(func)
    def wrapper_preserve_pos_and_vel(body, joints, *args, **kwargs):
        # save initial pose and velocities
        joint_pos = get_joint_positions(body, joints)
        joint_vel = get_joint_velocities(body, joints)

        ret = func(body, joints, *args, **kwargs)

        # put positions and velocities back!
        set_joint_positions_and_velocities(body, joints, joint_pos, joint_vel)
        return ret
    return wrapper_preserve_pos_and_vel


def preserve_pos_and_vel_wholebody(func):
    @functools.wraps(func)
    def wrapper_preserve_pos_and_vel_wholebody(cube_body, joints, finger_body, finger_joints, *args, **kwargs):
        # save initial pose and velocities
        cube_pose = get_pose(cube_body)
        cube_vel = get_velocity(cube_body)
        joint_pos = get_joint_positions(finger_body, finger_joints)
        joint_vel = get_joint_velocities(finger_body, finger_joints)

        ret = func(cube_body, joints, finger_body, finger_joints, *args, **kwargs)

        # put positions and velocities back!
        set_pose(cube_body, cube_pose)
        set_velocity(cube_body, linear=cube_vel[0], angular=cube_vel[1])
        set_joint_positions_and_velocities(finger_body, finger_joints, joint_pos, joint_vel)
        return ret
    return wrapper_preserve_pos_and_vel_wholebody

