import numpy as np
import copy
import pybullet

def create_robot_state(obs,kinematics):
    robot_state = [obs["robot_position"], obs["robot_velocity"], kinematics.get_tip_pos_pinnochio(obs["robot_position"])]
    return copy.deepcopy(robot_state)

def create_object_state(obs,kinematics):
    object_state = [obs["object_position"], obs["object_orientation"], np.asarray(pybullet.getEulerFromQuaternion(obs["object_orientation"])),
                    obs["goal_object_position"], obs["goal_object_orientation"],
                    np.asarray(pybullet.getEulerFromQuaternion(obs["goal_object_orientation"]))]
    return copy.deepcopy(object_state)

def get_robot_and_obj_state(obs,kinematics, last_states=None):
    if (last_states is None):
        return create_robot_state(obs, kinematics), create_object_state(obs, kinematics)
    else:
        robot_states = create_robot_state(obs, kinematics)
        obj_state = create_object_state(obs, kinematics)
        # now additionally propagate the states:
        last_states[:, 1:] = last_states[:, :-1]
        last_states[:, 0] = copy.deepcopy(obj_state[0])
        return robot_states, obj_state