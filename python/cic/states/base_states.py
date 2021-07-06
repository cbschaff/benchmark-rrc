from env.cube_env import ActionType
from mp.states import State

from cic.math_tools import rpy2Mat
from cic.utils import get_robot_and_obj_state
import numpy as np
import copy
import pinocchio as pin

import pybullet
from cic import rotation_primitives as rotation_primitives_cic

from scipy.spatial.transform import Rotation as R

'''
This file contains the basic movement primitives, i.e. small action sequences which can be combined together to obtain
more complex behavior.
'''


class IdlePrimitive(State):
    '''
    This primitive simply keeps the robot in its current position, for a specified amount of time.
    '''
    def __init__(self, env, kinematics=None, params=None):
        self.env = env
        self.goal_target = None

        self.duration = 100*4 # duration of the primitive in timesteps
        self.DEBUG = False
        self.iter_primitive_started = -1
        self.reference = None
        self.iteration = -1
        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def __call__(self, obs, info={}):
        # this is only called on first time:
        if (self.iteration==-1):
            self.iteration = 0
            self.iter_primitive_started = self.iteration
            self.reference = copy.deepcopy(obs['robot_position'])
            self.action_space_limits = self.env.action_space['position']

        action = self.get_action(position=np.clip(self.reference,self.action_space_limits.low,self.action_space_limits.high),frameskip=self._frameskip)

        self.iteration += 1
        state_finished = self.is_finished(self.iteration)

        if (state_finished):
            return action, self.next_state, info
        else:
            return action, self, info

    def reset(self):
        self.iteration = -1

    def is_finished(self, iteration):
        if (iteration>self.iter_primitive_started+self.duration):
            self.reset()
            return True
        else:
            return False

class IdlePrimitiveChangeGoal(IdlePrimitive):
    '''
    Copies the IdlePrimitive but has less duration and changes the goal location of the environment. This
    functionality is exploited inside the code for bayesian optimization.
    '''

    def __init__(self, env, kinematics=None, params=None):
        super(IdlePrimitiveChangeGoal, self).__init__(env, kinematics, params)
        self.duration = 1

    def is_finished(self, iteration):
        if (iteration>self.iter_primitive_started+self.duration):
            self.env.set_goal(orientation=self.env.goal_list[-1][0], pos=self.env.goal_list[-1][1])
            self.env.set_reach_start()  # log timeindex and cube pose at the start
            self.reset()
            return True
        else:
            return False

class IdlePrimitiveReachFinished(IdlePrimitive):
    '''
    Copies the idle primitive and does nothing exept for changing the parameter of the env,...
    '''

    def __init__(self, env, kinematics=None, params=None):
        super(IdlePrimitiveReachFinished, self).__init__(env, kinematics, params)
        self.duration = 1

    def is_finished(self, iteration):
        if (iteration>self.iter_primitive_started+self.duration):
            self.env.set_reach_finish()
            self.reset()
            return True
        else:
            return False

class IdlePrimitiveLong(IdlePrimitive):
    '''
    Copies the IdlePrimitive and siply has a longer duration.
    '''

    def __init__(self, env, kinematics=None, params=None):
        super(IdlePrimitiveLong, self).__init__(env, kinematics, params)
        self.duration = 10000

class InitResetPrimitive(State):
    '''
    This primitive is resetting the platform to pre-specified positions. It is used to enforce that the suceeding
    primitives start from the same initial conditions.
    '''
    def __init__(self, env, kinematics, params):
        self.env = env
        self.goal_target = None

        self.duration = 150*4 # duration of the primitive in timesteps
        self.DEBUG = False
        self.iter_primitive_started = -1
        self.iteration = -1
        self.reference = None

        self.kinematics = kinematics
        self.params = params

        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def __call__(self, obs, info={}):
        if (self.iteration==-1):
            self.iteration = 0
            self.iter_primitive_started = self.iteration

            self.action_space_limits = self.env.action_space['position']

            # first iteration -> compute cartesian reference position:
            unit_vec = np.asarray([0.0, 0.2, 0.2])
            offset = 15
            self.unit_vec_x = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(0+offset))).reshape(3, 3), unit_vec)
            self.unit_vec_y = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(240+offset))).reshape(3, 3), unit_vec)
            self.unit_vec_z = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(120+offset))).reshape(3, 3), unit_vec)

            self.reference = self.kinematics.inverse_kinematics_3_fingers(obs['robot_position'],self.unit_vec_x, self.unit_vec_y, self.unit_vec_z)
            self.reference = np.clip(self.reference, self.action_space_limits.low, self.action_space_limits.high)


        if (self.iteration==self.iter_primitive_started+100):
            unit_vec = np.asarray([0.0, 0.1, 0.1])
            offset = 15
            self.unit_vec_x = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(0+offset))).reshape(3, 3), unit_vec)
            self.unit_vec_y = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(240+offset))).reshape(3, 3), unit_vec)
            self.unit_vec_z = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(120+offset))).reshape(3, 3), unit_vec)
            self.reference = self.kinematics.inverse_kinematics_3_fingers(obs['robot_position'],self.unit_vec_x, self.unit_vec_y, self.unit_vec_z)
            self.reference = np.clip(self.reference, self.action_space_limits.low, self.action_space_limits.high)


        action = self.get_action(position=self.reference,frameskip=self._frameskip)
        self.iteration += 1
        state_finished = self.is_finished(self.iteration)

        if (state_finished):
            return action, self.next_state, info
        else:
            return action, self, info

    def reset(self):
        self.iteration = -1

    def is_finished(self, iteration):
        if (iteration>self.iter_primitive_started+self.duration):
            self.reset()
            return True
        else:
            return False

class ApproachObjectViapoints(State):
    '''
    This primitive approaches the fingers to desired grasping locations. This procedure is performed using several
    viapoints, i.e. the final position to be reached is approached step-by-step.
    This primitive can also be used to drive the fingers away safely from the cube by slowly increasing the distances
    to the desired grasping points.
    '''
    def __init__(self, env, kinematics, params, object, grasp, approach_grasp_xy, approach_grasp_h, duration, clip_pos, stop_early=False, assign_fingers=False, use_unrestricted_grasp=False):
        self.env = env
        self.goal_target = None
        self.stop_early = stop_early

        self.kinematics = kinematics
        self.params = params

        self.object = object
        self.grasp = grasp

        self.approach_grasp_xy = approach_grasp_xy
        self.approach_grasp_h = approach_grasp_h
        self.duration = duration
        self.counter = 0
        self.counter_end = len(duration)

        self.clip_pos = clip_pos

        self.assign_fingers = assign_fingers
        self.use_unrestricted_grasp = use_unrestricted_grasp

        self.DEBUG = False
        self.iteration = -1
        self.iter_primitive_started = -1
        self.reference = None

        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def __call__(self, obs, info={}):
        if (self.iteration==-1):
            self.iteration = 0
            self.action_space_limits = self.env.action_space['position']

            robot_state, state = get_robot_and_obj_state(obs, self.kinematics)
            self.iter_primitive_started = self.iteration
            self.counter = 0

            if(self.assign_fingers):
                if not(self.use_unrestricted_grasp):
                    self.grasp.assign_fingers(robot_state, state)
                else:
                    self.grasp.assign_fingers_unrestricted(robot_state, state)

            pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state, off_x=self.approach_grasp_xy[self.counter], off_y =self.approach_grasp_xy[self.counter], off_z=self.approach_grasp_h[self.counter], use_z = True)
            self.reference = self.kinematics.inverse_kinematics_3_fingers(robot_state[0], pos1,
                                                                          pos2, pos3)



        robot_state, state = get_robot_and_obj_state(obs, self.kinematics)
        reference = robot_state[0] + np.clip(self.reference - robot_state[0], -self.clip_pos[self.counter], self.clip_pos[self.counter])

        action = self.get_action(position=np.clip(reference,self.action_space_limits.low,self.action_space_limits.high),frameskip=self._frameskip)
        self.iteration += 1
        state_finished = self.is_finished(robot_state, state, self.iteration)

        if (state_finished):
            return action, self.next_state, info
        else:
            return action, self, info

    def reset(self):
        self.iteration = -1


    def is_finished(self, robot_state, cube_state, iteration):
        if (iteration>self.iter_primitive_started+self.duration[self.counter]):
            if (self.counter<self.counter_end-1):
                # not finished yet -> update position and return False
                self.counter += 1
                pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, cube_state,
                                                                  off_x=self.approach_grasp_xy[self.counter],
                                                                  off_y=self.approach_grasp_xy[self.counter],
                                                                  off_z=self.approach_grasp_h[self.counter])
                self.reference = self.kinematics.inverse_kinematics_3_fingers(robot_state[0], pos1,
                                                                              pos2, pos3)
                return False
            else:
                self.reset()
                return True
        else:
            return False


class MoveCubeFloorPrimitive(State):
    '''
    This primitive implements Cartesian Impedance Control in order to move the object in the ground plane.
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0]):
        self.env = env
        self.goal_target = None
        self.stop_early = stop_early
        self.use_fixed_reference = use_fixed_reference
        self.fixed_reference = fixed_reference

        self.object = object
        self.grasp = grasp

        self.kinematics = kinematics
        self.params = params

        self.grasp_xy = self.params.grasp_xy_floor
        self.grasp_h = self.params.grasp_h_floor
        self.gain_xy = 0.0
        self.gain_z = 0.0
        self.gain_xy_pre = self.params.gain_xy_pre_floor
        self.gain_z_pre = self.params.gain_z_pre_floor
        self.init_dur = self.params.init_dur_floor
        self.gain_xy_final = self.params.gain_xy_final_floor
        self.gain_z_final = self.params.gain_z_final_floor
        self.pos_gain_impedance = self.params.pos_gain_impedance_floor
        self.pos_gain = 0.0
        self.force_factor = self.params.force_factor_floor

        self.duration = 5000*4

        self.DEBUG = False
        self.iteration = -1
        self.iter_primitive_started = -1
        self.reference = None

        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def __call__(self, obs, info={}):
        if (self.iteration==-1):
            self.iteration = 0
            self.last_states = np.zeros((3, 100))
            self.action_space_limits = self.env.action_space['torque']

            # on first iteration:
            self.iter_primitive_started = self.iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.pos_gain = 0.0


        robot_state, state = get_robot_and_obj_state(obs, self.kinematics, last_states=self.last_states)
        #print (self.last_states[:,0])

        if (self.use_fixed_reference):
            state[3] = self.fixed_reference

        if (self.iteration == self.iter_primitive_started + self.init_dur):
            self.gain_xy = self.gain_xy_final
            self.gain_z = self.gain_z_final
            self.pos_gain = self.pos_gain_impedance

        # Calculate direction of the error:
        gen_dir = (state[3] - state[0])
        #print ("error: ", gen_dir)
        factor_dir = np.sqrt(np.sum((gen_dir) ** 2)) * 100
        gen_dir = (gen_dir) / np.sqrt(np.sum((gen_dir) ** 2))
        FACTOR = - self.pos_gain * np.clip(factor_dir, -2.0, 2.0)

        # directions from current end effector to center (unit vectors)
        dir1 = (state[0] - robot_state[2][0])
        dir1 = (dir1) / np.sqrt(np.sum((dir1) ** 2))
        dir2 = (state[0] - robot_state[2][1])
        dir2 = (dir2) / np.sqrt(np.sum((dir2) ** 2))
        dir3 = (state[0] - robot_state[2][2])
        dir3 = (dir3) / np.sqrt(np.sum((dir3) ** 2))

        pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state, off_x=self.grasp_xy,
                                                          off_y=self.grasp_xy, off_z=self.grasp_h)

        pos_signal_1 = robot_state[2][0] - pos1 + FACTOR * gen_dir
        vel_signal_1 = np.zeros((3))

        pos_signal_2 = robot_state[2][1] - pos2 + FACTOR * gen_dir
        vel_signal_2 = np.zeros((3))

        pos_signal_3 = robot_state[2][2] - pos3 + FACTOR * gen_dir
        vel_signal_3 = np.zeros((3))

        # this time we use all 3 fingers
        torque = self.kinematics.imp_ctrl_3_fingers([robot_state[0], robot_state[0], robot_state[0]],
                                                    [0.0 * dir1, 0.0 * dir2, 0.0 * dir3], [4, 8, 12], \
                                                    [vel_signal_1, vel_signal_2, vel_signal_3],
                                                    [pos_signal_1, pos_signal_2, pos_signal_3], \
                                                    [robot_state[1], robot_state[1], robot_state[1]],
                                                    [self.gain_xy, self.gain_xy, self.gain_xy], \
                                                    [self.gain_z, self.gain_z, self.gain_z],
                                                    [0,0,0])

        # potentially add additional component
        # TODO: to find out: on the floor might be even more stable without this additional component,...
        add_torque = self.kinematics.add_additional_force_3_fingers(robot_state[0], self.force_factor,
                                                                    self.grasp.get_edge_directions(robot_state,
                                                                                                   state),
                                                                    self.grasp.get_center_array(),
                                                                    correct_torque=True)

        torque = np.clip(torque, -0.36, 0.36)
        torque = self.kinematics.compute_gravity_compensation(robot_state,torque)

        action = self.get_action(torque=np.clip(torque,self.action_space_limits.low,self.action_space_limits.high),frameskip=self._frameskip)
        self.iteration += 1
        state_finished = self.is_finished(robot_state, state, self.iteration)

        if (state_finished):
            return action, self.next_state, info
        else:
            return action, self, info

    def reset(self):
        self.iteration = -1

    def is_finished(self, robot_state, cube_state, iteration):
        if (iteration > self.iter_primitive_started + self.duration):
            self.reset()
            return True
        else:
            return False

class MoveCubeFloorPrimitiveLvl1(MoveCubeFloorPrimitive):
    '''
    This primitive copies the "MoveCubeFloorPrimitive" and only changes the criterion when the primitive is finished.
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early):
        super(MoveCubeFloorPrimitiveLvl1, self).__init__(env, kinematics, params, object, grasp, stop_early)
        self.duration = 300*4

    def is_finished(self, robot_state, cube_state, iteration):
        if (self.use_fixed_reference):
            cube_state[3] = self.fixed_reference

        if (iteration > self.iter_primitive_started + self.duration):
            curr_dist = (cube_state[3] - self.last_states[:, 0]) * 100
            dir_progress = (self.last_states[:, 0] - self.last_states[:, 50]) * 100
            dist_law = (np.sum(dir_progress) >= 0.0 and np.sqrt(np.sum(curr_dist ** 2)) > 2.5)
            if (dist_law):
                self.reset()
                return True
            else:
                return False
        else:
            return False

class MoveCubeFloorPrimitiveLvl4(MoveCubeFloorPrimitive):
    '''
    This primitive copies the "MoveCubeFloorPrimitive" and only changes the criterion when the primitive is finished.
    In particular, the primitive is finished once the cube is close enough to the center.
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0]):
        super(MoveCubeFloorPrimitiveLvl4, self).__init__(env, kinematics, params, object, grasp, stop_early, use_fixed_reference=use_fixed_reference, fixed_reference=fixed_reference)
        self.duration = 300*4

    def is_finished(self, robot_state, cube_state, iteration):
        if (self.use_fixed_reference):
            cube_state[3] = self.fixed_reference

        curr_dist = (cube_state[3] - self.last_states[:, 0]) * 100
        # Add quick reset functionality when wanting to move on the floor
        if (np.sqrt(np.sum(curr_dist ** 2)) < 2.5):
            self.reset()
            return True

        if (iteration > self.iter_primitive_started + self.duration):
            dir_progress = (self.last_states[:, 0] - self.last_states[:, 50]) * 100
            dist_law = (np.sum(dir_progress) >= 0.0 and np.sqrt(np.sum(curr_dist ** 2)) > 2.5)
            if (dist_law):
                self.reset()
                return True
            else:
                return False
        else:
            return False


class MoveLiftCubePrimitive(State):
    '''
    This primitive implements Cartesian Impedance Control in order to move and lift the object. This functionality is
    required to solve stage 2 and 3 of the challenge.
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early):
        self.env = env
        self.goal_target = None
        self.stop_early = stop_early

        self.object = object
        self.grasp = grasp

        self.kinematics = kinematics
        self.params = params

        self.grasp_xy = self.params.grasp_xy_lift
        self.grasp_h = self.params.grasp_h_lift
        self.gain_xy = 0.0
        self.gain_z = 0.0
        self.gain_xy_pre = self.params.gain_xy_pre_lift
        self.gain_z_pre = self.params.gain_z_pre_lift
        self.init_dur = self.params.init_dur_lift
        self.gain_xy_ground = self.params.gain_xy_ground_lift
        self.gain_z_ground = self.params.gain_z_ground_lift
        self.pos_gain_impedance_ground = self.params.pos_gain_impedance_ground_lift
        self.gain_xy_lift = self.params.gain_xy_lift_lift
        self.gain_z_lift = self.params.gain_z_lift_lift
        self.pos_gain_impedance_lift = self.params.pos_gain_impedance_lift_lift
        self.pos_gain = 0.0
        self.force_factor = self.params.force_factor_lift
        self.switch_mode = self.params.switch_mode_lift
        self.clip_height = self.params.clip_height_lift

        self.start_lift = False


        self.duration = 5000*4

        self.DEBUG = False
        self.iteration = -1
        self.iter_primitive_started = -1
        self.reference = None

        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

        self.gain_increase_factor = 1.04
        self.max_interval_ctr = 20
        self.interval = 1800
        self.interval_ctr = 0

        self.integral_value = 0.0

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def update_gain(self):
        # if self.env.simulation:
        #     return
        if self.iteration % self.interval == 0 and self.interval_ctr < self.max_interval_ctr:
            self.gain_xy *= self.gain_increase_factor
            self.gain_z *= self.gain_increase_factor
            self.interval_ctr += 1

    def __call__(self, obs, info={}):
        if (self.iteration==-1):
            self.integral_value = 0.0
            self.iteration = 0
            self.last_states = np.zeros((3, 100))
            self.action_space_limits = self.env.action_space['torque']

            self.iter_primitive_started = self.iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.start_lift = False
            self.pos_gain = 0.0

        robot_state, state = get_robot_and_obj_state(obs, self.kinematics, last_states=self.last_states)

        if (self.iteration == self.iter_primitive_started + self.init_dur):
            self.gain_xy = self.gain_xy_ground
            self.gain_z = self.gain_z_ground
            self.pos_gain = self.pos_gain_impedance_ground

        xy_dist = np.sqrt(np.sum((state[3][:2] - state[0][:2]) ** 2))
        if (not(self.start_lift) and (self.iteration > self.iter_primitive_started + self.init_dur) and (xy_dist < self.switch_mode)):
            self.start_lift = True
            self.gain_xy = self.gain_xy_lift
            self.gain_z = self.gain_z_lift
            self.pos_gain = self.pos_gain_impedance_lift

        target = copy.deepcopy(state[3])
        if (self.start_lift):
            target[2] = state[0][2] + np.clip(target[2]-state[0][2],-self.clip_height,self.clip_height)
        else:
            target[2] = 0.0325

        # Calculate direction of the error:
        gen_dir = (target - state[0])
        #print ("error: ", gen_dir)
        factor_dir = np.sqrt(np.sum((gen_dir) ** 2)) * 100
        gen_dir = (gen_dir) / np.sqrt(np.sum((gen_dir) ** 2))
        FACTOR = - self.pos_gain * np.clip(factor_dir, -2.0, 2.0)

        if (self.start_lift):
            #UPDATE INTEGRAL GAIN:
            self.integral_value += self.params.grasp_normal_int_gain*FACTOR * gen_dir

        # directions from current end effector to center (unit vectors)
        dir1 = (state[0] - robot_state[2][0])
        dir1 = (dir1) / np.sqrt(np.sum((dir1) ** 2))
        dir2 = (state[0] - robot_state[2][1])
        dir2 = (dir2) / np.sqrt(np.sum((dir2) ** 2))
        dir3 = (state[0] - robot_state[2][2])
        dir3 = (dir3) / np.sqrt(np.sum((dir3) ** 2))

        pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state, off_x=self.grasp_xy,
                                                          off_y=self.grasp_xy, off_z=self.grasp_h)

        pos_signal_1 = robot_state[2][0] - pos1 + FACTOR * gen_dir + self.integral_value
        vel_signal_1 = np.zeros((3))

        pos_signal_2 = robot_state[2][1] - pos2 + FACTOR * gen_dir + self.integral_value
        vel_signal_2 = np.zeros((3))

        pos_signal_3 = robot_state[2][2] - pos3 + FACTOR * gen_dir + self.integral_value
        vel_signal_3 = np.zeros((3))

        # this time we use all 3 fingers
        torque = self.kinematics.imp_ctrl_3_fingers([robot_state[0], robot_state[0], robot_state[0]],
                                                    [0.0 * dir1, 0.0 * dir2, 0.0 * dir3], [4, 8, 12], \
                                                    [vel_signal_1, vel_signal_2, vel_signal_3],
                                                    [pos_signal_1, pos_signal_2, pos_signal_3], \
                                                    [robot_state[1], robot_state[1], robot_state[1]],
                                                    [self.gain_xy, self.gain_xy, self.gain_xy], \
                                                    [self.gain_z, self.gain_z, self.gain_z],
                                                    [0,0,0])

        # potentially add additional component
        add_torque = np.zeros((9))
        if (self.start_lift):
            add_torque = self.kinematics.add_additional_force_3_fingers(robot_state[0], self.force_factor,
                                                                        self.grasp.get_edge_directions(robot_state,
                                                                                                       state),
                                                                        [0,0,0],
                                                                        correct_torque=True)

        torque = np.clip(torque+add_torque, -0.36, 0.36)
        torque = self.kinematics.compute_gravity_compensation(robot_state, torque)

        action = self.get_action(torque=np.clip(torque, self.action_space_limits.low, self.action_space_limits.high),frameskip=self._frameskip)
        self.iteration += 1
        state_finished = self.is_finished(robot_state, state, self.iteration)

        if (state_finished):
            return action, self.next_state, info
        else:
            return action, self, info

    def reset(self):
        self.iteration = -1

    def is_finished(self, robot_state, cube_state, iteration):
        if (iteration > self.iter_primitive_started + self.duration):
            self.reset()
            return True
        else:
            return False

class MoveLiftCubePrimitiveLvl2(MoveLiftCubePrimitive):
    '''
    Incorporates excatly the same functionality as the MoveLiftCubePrimitive
    -> only has a different duration and a different reset function
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early):
        super(MoveLiftCubePrimitiveLvl2, self).__init__(env, kinematics, params, object, grasp, stop_early)
        self.duration = 1000*4

    def is_finished(self, robot_state, cube_state, iteration):
        target = copy.deepcopy(cube_state[3])
        if (self.start_lift):
            target[2] = cube_state[0][2] + np.clip(target[2]-cube_state[0][2],-self.clip_height,self.clip_height)
        else:
            target[2] = 0.0325

        if (iteration > self.iter_primitive_started + self.duration):
            curr_dist = (target - self.last_states[:, 0]) * 100
            dir_progress = (self.last_states[:, 0] - self.last_states[:, 50]) * 100
            dist_law = (np.sum(dir_progress) >= 0.0 and np.sqrt(np.sum(curr_dist ** 2)) > 2.5)
            if (dist_law):
                self.reset()
                return True
            else:
                return False
        else:
            return False

class MoveLiftCubePrimitiveLvl4(MoveLiftCubePrimitive):
    '''
    Incorporates excatly the same functionality as the MoveLiftCubePrimitive
    -> only has a different duration and a different reset function
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early):
        super(MoveLiftCubePrimitiveLvl4, self).__init__(env, kinematics, params, object, grasp, stop_early)
        self.switch_mode = 0.035
        self.duration = 1000*4

    def is_finished(self, robot_state, cube_state, iteration):
        target = copy.deepcopy(cube_state[3])
        if (self.start_lift):
            target[2] = cube_state[0][2] + np.clip(target[2]-cube_state[0][2],-self.clip_height,self.clip_height)
        else:
            target[2] = 0.0325

        if (iteration > self.iter_primitive_started + self.duration):
            curr_dist = (target - self.last_states[:, 0]) * 100
            dir_progress = (self.last_states[:, 0] - self.last_states[:, 50]) * 100
            dist_law = (np.sum(dir_progress) >= 0.0 and np.sqrt(np.sum(curr_dist ** 2)) > 4.5)
            if (dist_law):
                self.reset()
                return True
            else:
                return False
        else:
            return False

class MoveLiftCubeOrientPrimitive(State):
    '''
    This primitive implements Cartesian Impedance Control in order to move and lift the object. In addition, it contains
    components that should allow to also achieve a desired orientation of the object during the maneuver.
    '''
    def __init__(self, env, kinematics, params, object, grasp, stop_early):
        self.env = env
        self.goal_target = None
        self.stop_early = stop_early

        self.object = object
        self.grasp = grasp

        self.kinematics = kinematics
        self.params = params

        self.grasp_xy = self.params.orient_grasp_xy_lift
        self.grasp_h = self.params.orient_grasp_h_lift
        self.gain_xy = 0.0
        self.gain_z = 0.0
        self.gain_xy_pre = self.params.orient_gain_xy_pre_lift
        self.gain_z_pre = self.params.orient_gain_z_pre_lift
        self.init_dur = self.params.orient_init_dur_lift
        self.gain_xy_ground = self.params.orient_gain_xy_ground_lift
        self.gain_z_ground = self.params.orient_gain_z_ground_lift
        self.pos_gain_impedance_ground = self.params.orient_pos_gain_impedance_ground_lift
        self.gain_xy_lift = self.params.orient_gain_xy_lift_lift
        self.gain_z_lift = self.params.orient_gain_z_lift_lift
        self.pos_gain_impedance_lift = self.params.orient_pos_gain_impedance_lift_lift
        self.pos_gain = 0.0
        self.force_factor = self.params.orient_force_factor_lift
        self.switch_mode = self.params.orient_switch_mode_lift
        self.clip_height = self.params.orient_clip_height_lift

        self.force_factor_rot_lift = self.params.orient_force_factor_rot_lift
        self.force_factor_rot_ground = self.params.orient_force_factor_rot_ground

        self.start_lift = False
        self.start_moving = False


        self.duration = 5000*4

        self.DEBUG = False
        self.iteration = -1
        self.iter_primitive_started = -1
        self.reference = None

        self.planner = rotation_primitives_cic.RotationPrimitives(rotation_primitives_cic.to_H(np.eye(3)),
                                                                  rotation_primitives_cic.to_H(np.eye(3)))

        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

        self.integral_value = 0.0
        self.integral_value_orient = 0.0

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def __call__(self, obs, info={}):
        if (self.iteration==-1):
            self.integral_value = 0.0
            self.integral_value_orient = 0.0
            self.iteration = 0
            self.last_states = np.zeros((3, 100))
            self.action_space_limits = self.env.action_space['torque']

            self.iter_primitive_started = self.iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.start_lift = False
            self.pos_gain = 0.0
            self.start_moving = False

        robot_state, state = get_robot_and_obj_state(obs, self.kinematics, last_states=self.last_states)

        if (self.iteration == self.iter_primitive_started + self.init_dur):
            self.gain_xy = self.gain_xy_ground
            self.gain_z = self.gain_z_ground
            self.pos_gain = self.pos_gain_impedance_ground
            self.start_moving = True

        xy_dist = np.sqrt(np.sum((state[3][:2] - state[0][:2]) ** 2))
        if (not(self.start_lift) and (self.iteration > self.iter_primitive_started + self.init_dur) and (xy_dist < self.switch_mode)):
            self.start_lift = True
            self.gain_xy = self.gain_xy_lift
            self.gain_z = self.gain_z_lift
            self.pos_gain = self.pos_gain_impedance_lift

        target = copy.deepcopy(state[3])
        target[0] = np.clip(target[0],-0.08,0.08)
        target[1] = np.clip(target[1],-0.08,0.08)
        if (self.start_lift):
            target[2] = state[0][2] + np.clip(target[2]-state[0][2],-self.clip_height,self.clip_height)
        else:
            target[2] = 0.0325

        # Calculate direction of the error:
        gen_dir = (target - state[0])
        #print ("error: ", gen_dir)
        factor_dir = np.sqrt(np.sum((gen_dir) ** 2)) * 100
        gen_dir = (gen_dir) / np.sqrt(np.sum((gen_dir) ** 2))
        FACTOR = - self.pos_gain * np.clip(factor_dir, -2.0, 2.0)

        if (self.start_lift):
            #UPDATE INTEGRAL GAIN:
            self.integral_value += self.params.orient_int_pos_gain*FACTOR * gen_dir

        # directions from current end effector to center (unit vectors)
        dir1 = (state[0] - robot_state[2][0])
        dir1 = (dir1) / np.sqrt(np.sum((dir1) ** 2))
        dir2 = (state[0] - robot_state[2][1])
        dir2 = (dir2) / np.sqrt(np.sum((dir2) ** 2))
        dir3 = (state[0] - robot_state[2][2])
        dir3 = (dir3) / np.sqrt(np.sum((dir3) ** 2))

        pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state, off_x=self.grasp_xy,
                                                          off_y=self.grasp_xy, off_z=self.grasp_h)

        pos_signal_1 = robot_state[2][0] - pos1 + FACTOR * gen_dir + self.integral_value
        vel_signal_1 = np.zeros((3))

        pos_signal_2 = robot_state[2][1] - pos2 + FACTOR * gen_dir + self.integral_value
        vel_signal_2 = np.zeros((3))

        pos_signal_3 = robot_state[2][2] - pos3 + FACTOR * gen_dir + self.integral_value
        vel_signal_3 = np.zeros((3))

        # this time we use all 3 fingers
        torque = self.kinematics.imp_ctrl_3_fingers([robot_state[0], robot_state[0], robot_state[0]],
                                                    [0.0 * dir1, 0.0 * dir2, 0.0 * dir3], [4, 8, 12], \
                                                    [vel_signal_1, vel_signal_2, vel_signal_3],
                                                    [pos_signal_1, pos_signal_2, pos_signal_3], \
                                                    [robot_state[1], robot_state[1], robot_state[1]],
                                                    [self.gain_xy, self.gain_xy, self.gain_xy], \
                                                    [self.gain_z, self.gain_z, self.gain_z],
                                                    [0,0,0])

        # potentially add additional component when starting the lifting
        add_torque = np.zeros((9))
        if (self.start_lift):
            add_torque = self.kinematics.add_additional_force_3_fingers(robot_state[0], self.force_factor,
                                                                        self.grasp.get_edge_directions(robot_state,
                                                                                                       state),
                                                                        [0,0,0],
                                                                        correct_torque=True)

        torque = np.clip(torque+add_torque, -0.36, 0.36)

        # potentially add additional component concerning the rotation,... -> always activate this feature on ground and with lift,...
        if (self.start_moving):
            self.planner.set_goal(rotation_primitives_cic.to_H(
                np.asarray(pybullet.getMatrixFromQuaternion(obs["goal_object_orientation"])).reshape(3, 3)))
            self.planner.set_current_pose(
                rotation_primitives_cic.to_H(np.asarray(pybullet.getMatrixFromQuaternion(state[1])).reshape(3, 3)))
            sequence, sequence1g, sequence_direct = self.planner.get_control_seq()

            if not(self.start_lift):
                if (self.force_factor_rot_ground!=0.0):
                    # only care about the rotation in the ground plane
                    dir_rot = sequence_direct[0][0]
                    magnitude = sequence_direct[0][1]
                    dir_rot[np.argmax(np.abs(dir_rot))] = np.sign(magnitude) * np.sign(dir_rot[np.argmax(np.abs(dir_rot))]) * 1.0
                    dir_rot[np.abs(dir_rot) < 0.9] = 0.0

                    # rot_mat = R.from_rotvec(dir_rot * magnitude)
                    # reference_vector, _, _ = self.grasp.get_edge_directions(robot_state, state)
                    # reference_vector = np.matmul(rot_mat.as_matrix(), reference_vector)
                    # curr_reference_vector, _, _ = self.grasp.get_edge_directions(robot_state, state)
                    # gain = np.dot((curr_reference_vector / np.linalg.norm(curr_reference_vector)),
                    #               (reference_vector / np.linalg.norm(reference_vector)))
                    gain = np.cos(magnitude) # this should be easier here
                    rotation_value = np.pi / 2 * np.clip((1 - gain) * 2.5, 0, 1)
                    rotation_dir = np.matmul(np.asarray(pybullet.getMatrixFromQuaternion(state[1])).reshape(3, 3), dir_rot)
                    dir_rotatione = rotation_value*rotation_dir*self.force_factor_rot_ground

                    add_torque_force, add_torque_torque = self.kinematics.comp_combined_torque(copy.deepcopy(robot_state[0]),
                                                                                    [0, 0, 0],
                                                                                    dir_rotatione,
                                                                                    [1,1,1], state[0],
                                                                                    robot_state[2][0],
                                                                                    robot_state[2][1],
                                                                                    robot_state[2][2])
                else:
                    add_torque_torque = np.zeros((9))

            else:
                if (self.force_factor_rot_lift != 0.0):

                    magnitude = sequence_direct[1][1]
                    dir_rot = sequence_direct[1][0]*np.sign(magnitude)

                    # rot_mat = R.from_rotvec(dir_rot * magnitude)
                    # reference_vector, _, _ = self.grasp.get_edge_directions(robot_state, state)
                    # reference_vector = np.matmul(rot_mat.as_matrix(), reference_vector)
                    # curr_reference_vector, _, _ = self.grasp.get_edge_directions(robot_state, state)
                    # gain = np.dot((curr_reference_vector / np.linalg.norm(curr_reference_vector)),
                    #               (reference_vector / np.linalg.norm(reference_vector)))
                    gain = np.cos(magnitude)  # this should be easier here
                    self.integral_value_orient += self.params.orient_int_orient_gain * np.clip(gain, 0, 1) * np.sign(
                        magnitude)
                    self.integral_value_orient = np.clip(self.integral_value_orient,-10,10)
                    rotation_value = np.pi / 2 * (np.clip((1 - gain) * 2.5, 0, 1) + np.abs(self.integral_value_orient))
                    rotation_dir = np.matmul(np.asarray(pybullet.getMatrixFromQuaternion(state[1])).reshape(3, 3), dir_rot)
                    dir_rotatione = rotation_value*rotation_dir*self.force_factor_rot_lift

                    add_torque_force, add_torque_torque = self.kinematics.comp_combined_torque(copy.deepcopy(robot_state[0]),
                                                                                    [0, 0, 0],
                                                                                    dir_rotatione,
                                                                                    [1,1,1], state[0],
                                                                                    robot_state[2][0],
                                                                                    robot_state[2][1],
                                                                                    robot_state[2][2])
                else:
                    add_torque_torque = np.zeros((9))

            torque = torque + add_torque_torque

        torque = np.clip(torque, -0.36, 0.36)
        torque = self.kinematics.compute_gravity_compensation(robot_state, torque)

        action = self.get_action(torque=np.clip(torque, self.action_space_limits.low, self.action_space_limits.high),frameskip=self._frameskip)
        self.iteration += 1
        state_finished = self.is_finished(robot_state, state, self.iteration)

        if (state_finished):
            return action, self.next_state, info
        else:
            return action, self, info

    def reset(self):
        self.iteration = -1

    def set_dir_rot(self,dir_rot, magnitude):
        self.dir_rot = dir_rot
        self.dir_rot_magnitude = magnitude

    def is_finished(self, robot_state, cube_state, iteration):
        if (iteration > self.iter_primitive_started + self.duration):
            self.reset()
            return True
        else:
            return False

class MoveLiftCubeOrientPrimitiveLvl4(MoveLiftCubeOrientPrimitive):
    '''
    Incorporates excatly the same functionality as the MoveLiftCubeOrientPrimitive
    -> only has a different duration and different reset function
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early):
        super(MoveLiftCubeOrientPrimitiveLvl4, self).__init__(env, kinematics, params, object, grasp, stop_early)
        self.duration = 1000*4

    def is_finished(self, robot_state, cube_state, iteration):
        target = copy.deepcopy(cube_state[3])
        target[0] = np.clip(target[0],-0.08,0.08)
        target[1] = np.clip(target[1],-0.08,0.08)
        target[2] = np.clip(target[2], -0.08, 0.08)
        if (self.start_lift):
            target[2] = cube_state[0][2] + np.clip(target[2]-cube_state[0][2],-self.clip_height,self.clip_height)
        else:
            target[2] = 0.0325

        if (iteration > self.iter_primitive_started + self.duration):
            curr_dist = (target - self.last_states[:, 0]) * 100
            dir_progress = (self.last_states[:, 0] - self.last_states[:, 50]) * 100
            dist_law = (np.sum(dir_progress) >= 0.0 and np.sqrt(np.sum(curr_dist ** 2)) > 6.0)
            if (dist_law):
                self.reset()
                return True
            else:
                return False
        else:
            return False

class RotateCubeFloorPrimitive(State):
    '''
    This primitive implements Cartesian Impedance Control in order to rotate the object in the ground plane.
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0], dir_rot=[0,0,0]):
        self.env = env
        self.goal_target = None
        self.stop_early = stop_early
        self.use_fixed_reference = use_fixed_reference
        self.fixed_reference = fixed_reference
        self.dir_rot = dir_rot
        self.dir_rot_magnitude = 0

        self.object = object
        self.grasp = grasp

        self.kinematics = kinematics
        self.params = params

        self.grasp_xy = self.params.grasp_xy_rotate_ground
        self.grasp_h = self.params.grasp_h_rotate_ground
        self.gain_xy = 0.0
        self.gain_z = 0.0
        self.gain_xy_pre = self.params.gain_xy_pre_rotate_ground
        self.gain_z_pre = self.params.gain_z_pre_rotate_ground
        self.init_dur = self.params.init_dur_rotate_ground
        self.gain_xy_final = self.params.gain_xy_final_rotate_ground
        self.gain_z_final = self.params.gain_z_final_rotate_ground
        self.pos_gain_impedance = self.params.pos_gain_impedance_rotate_ground
        self.pos_gain = 0.0
        self.force_factor = self.params.force_factor_rotate_ground
        self.force_factor_ground_rot = self.params.force_factor_ground_rot_rotate_ground

        self.duration = 5000*4

        self.DEBUG = False
        self.iteration = -1
        self.iter_primitive_started = -1
        self.reference = None
        self.additional_force = False

        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def __call__(self, obs, info={}):
        if (self.iteration==-1):
            self.iteration = 0
            self.last_states = np.zeros((3, 100))
            self.action_space_limits = self.env.action_space['torque']

            self.iter_primitive_started = self.iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.pos_gain = 0.0
            self.additional_force = False

        robot_state, state = get_robot_and_obj_state(obs, self.kinematics, last_states=self.last_states)

        if (self.iteration==0):
            #additional computations that need the robot state and state,..
            rot_mat = R.from_rotvec(self.dir_rot_magnitude*self.dir_rot)
            self.reference_vector, _, _ = self.grasp.get_edge_directions(robot_state,state)
            # print (self.reference_vector)
            self.reference_vector = np.matmul(rot_mat.as_matrix(),self.reference_vector)
            # print (self.reference_vector)
            # input ("WAIT")


        curr_reference_vector, _, _ = self.grasp.get_edge_directions(robot_state, state)


        if (self.use_fixed_reference):
            state[3] = self.fixed_reference

        if (self.iteration == self.iter_primitive_started + self.init_dur):
            self.gain_xy = self.gain_xy_final
            self.gain_z = self.gain_z_final
            self.pos_gain = self.pos_gain_impedance
            self.additional_force = True

        # Calculate direction of the error:
        gen_dir = (state[3] - state[0])
        # print ("error: ", gen_dir)
        factor_dir = np.sqrt(np.sum((gen_dir) ** 2)) * 100
        gen_dir = (gen_dir) / np.sqrt(np.sum((gen_dir) ** 2))
        FACTOR = - self.pos_gain * np.clip(factor_dir, -2.0, 2.0)

        # directions from current end effector to center (unit vectors)
        dir1 = (state[0] - robot_state[2][0])
        dir1 = (dir1) / np.sqrt(np.sum((dir1) ** 2))
        dir2 = (state[0] - robot_state[2][1])
        dir2 = (dir2) / np.sqrt(np.sum((dir2) ** 2))
        dir3 = (state[0] - robot_state[2][2])
        dir3 = (dir3) / np.sqrt(np.sum((dir3) ** 2))

        pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state, off_x=self.grasp_xy,
                                                          off_y=self.grasp_xy, off_z=self.grasp_h)

        pos_signal_1 = robot_state[2][0] - pos1 + FACTOR * gen_dir
        vel_signal_1 = np.zeros((3))

        pos_signal_2 = robot_state[2][1] - pos2 + FACTOR * gen_dir
        vel_signal_2 = np.zeros((3))

        pos_signal_3 = robot_state[2][2] - pos3 + FACTOR * gen_dir
        vel_signal_3 = np.zeros((3))

        # this time we use all 3 fingers
        torque = self.kinematics.imp_ctrl_3_fingers([robot_state[0], robot_state[0], robot_state[0]],
                                                    [0.0 * dir1, 0.0 * dir2, 0.0 * dir3], [4, 8, 12], \
                                                    [vel_signal_1, vel_signal_2, vel_signal_3],
                                                    [pos_signal_1, pos_signal_2, pos_signal_3], \
                                                    [robot_state[1], robot_state[1], robot_state[1]],
                                                    [self.gain_xy, self.gain_xy, self.gain_xy], \
                                                    [self.gain_z, self.gain_z, self.gain_z],
                                                    [0,0,0])

        # potentially add additional component
        # TODO: to find out: on the floor might be even more stable without this additional component,...
        add_torque = self.kinematics.add_additional_force_3_fingers(robot_state[0], self.force_factor,
                                                                    self.grasp.get_edge_directions(robot_state,
                                                                                                   state),
                                                                    self.grasp.get_center_array(),
                                                                    correct_torque=True)
        torque = torque + add_torque

        if (self.additional_force):
            #Do potential gain scheduling:
            gain = np.dot((curr_reference_vector/np.linalg.norm(curr_reference_vector)),(self.reference_vector/np.linalg.norm(self.reference_vector)))
            #TODO: check this gain scheduling
            rotation_value = np.pi / 2 * np.clip((1-gain)*self.params.gain_scheduling,0,1)
            dir_rotatione = rotation_value * np.sign(self.dir_rot_magnitude) * self.dir_rot / np.sqrt(np.sum((self.dir_rot) ** 2))
            # vertical_idx, curr_axis_dir = self.grasp.get_axis(state)
            # idx_rotation_axes = np.argmax(np.abs(self.dir_rot))
            # dir_rotatione = curr_axis_dir[:, idx_rotation_axes]
            # # print (self.dir_rot[idx_rotation_axes] * dir_rotatione/ np.sqrt(np.sum((dir_rotatione) ** 2)))
            # # print (np.matmul(np.asarray(pybullet.getMatrixFromQuaternion(state[1])).reshape(3,3),self.dir_rot))
            # # input ("WAIT")
            # dir_rotatione = rotation_value * self.dir_rot[idx_rotation_axes] * dir_rotatione / np.sqrt(np.sum((dir_rotatione) ** 2))

            add_torque_force, add_torque_torque = self.kinematics.comp_combined_torque(copy.deepcopy(robot_state[0]),
                                                                            [0, 0, 0],
                                                                            self.force_factor_ground_rot * dir_rotatione,
                                                                            [1, 1, 1], state[0],
                                                                            robot_state[2][0],
                                                                            robot_state[2][1],
                                                                            robot_state[2][2])

            torque = torque + add_torque_torque

        torque = np.clip(torque, -0.36, 0.36)
        torque = self.kinematics.compute_gravity_compensation(robot_state, torque)

        action = self.get_action(torque=np.clip(torque, self.action_space_limits.low, self.action_space_limits.high),
                                 frameskip=self._frameskip)

        self.iteration += 1
        state_finished = self.is_finished(robot_state, state, self.iteration)

        if (state_finished):
            return action, self.next_state, info
        else:
            return action, self, info

    def reset(self):
        self.iteration = -1

    def set_dir_rot(self,dir_rot, magnitude):
        self.dir_rot = dir_rot
        self.dir_rot_magnitude = magnitude

    def set_fixed_ref(self,fixed_ref):
        self.fixed_reference = fixed_ref

    def is_finished(self, robot_state, cube_state, iteration):
        if (iteration > self.iter_primitive_started + self.duration):
            self.reset()
            return True
        else:
            return False

class RotateCubeFloorPrimitiveLvl4(RotateCubeFloorPrimitive):
    '''
    Incorporates excatly the same functionality as the RotateCubeFloorPrimitive
    -> only has a different duration and different reset function
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0], dir_rot=[0,0,0]):
        super(RotateCubeFloorPrimitiveLvl4, self).__init__(env, kinematics, params, object, grasp, stop_early, use_fixed_reference=use_fixed_reference, fixed_reference=fixed_reference, dir_rot=dir_rot)
        self.duration = 300*4

    def is_finished(self, robot_state, cube_state, iteration):

        if (iteration > self.iter_primitive_started + self.duration):
            self.reset()
            return True
        else:
            return False

class RotateCubeLiftPrimitive(State):
    '''
    This primitive implements Cartesian Impedance Control in order to rotate the object. The desired axis of rotation
    is now parallel to the ground plane, therefore the object has to be lifted to achieve the desired rotation.
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0], dir_rot=[0,0,0]):
        self.env = env
        self.goal_target = None
        self.stop_early = stop_early
        self.use_fixed_reference = use_fixed_reference
        self.fixed_reference = fixed_reference
        self.dir_rot = dir_rot

        self.object = object
        self.grasp = grasp

        self.kinematics = kinematics
        self.params = params

        self.grasp_xy = self.params.grasp_xy_rotate_lift
        self.grasp_h = self.params.grasp_h_rotate_lift
        self.gain_xy = 0.0
        self.gain_z = 0.0
        self.gain_xy_pre = self.params.gain_xy_pre_rotate_lift
        self.gain_z_pre = self.params.gain_z_pre_rotate_lift
        self.init_dur = self.params.init_dur_rotate_lift
        self.gain_xy_final = self.params.gain_xy_final_rotate_lift
        self.gain_z_final = self.params.gain_z_final_rotate_lift
        self.pos_gain_impedance = self.params.pos_gain_impedance_rotate_lift
        self.pos_gain = 0.0
        self.force_factor = self.params.force_factor_rotate_lift
        self.force_factor_center = self.params.force_factor_center_rotate_lift
        self.force_factor_rot = self.params.force_factor_rot_rotate_lift
        self.target_height_rot_lift = self.params.target_height_rot_lift_rotate_lift
        self.clip_height = self.params.clip_height_rotate_lift

        self.duration = 5000*4

        self.DEBUG = False
        self.iteration = -1
        self.iter_primitive_started = -1
        self.reference = None
        self.additional_force = False

        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def __call__(self, obs, info={}):
        if (self.iteration==-1):
            self.iteration = 0
            self.last_states = np.zeros((3, 100))
            self.action_space_limits = self.env.action_space['torque']

            self.iter_primitive_started = self.iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.pos_gain = 0.0
            self.additional_force = False



        robot_state, state = get_robot_and_obj_state(obs, self.kinematics, last_states=self.last_states)


        if (self.use_fixed_reference):
            state[3] = self.fixed_reference


        if (self.iteration == self.iter_primitive_started + self.init_dur):
            self.gain_xy = self.gain_xy_final
            self.gain_z = self.gain_z_final
            self.pos_gain = self.pos_gain_impedance
            self.additional_force = True

        target = copy.deepcopy(state[3])
        if (self.additional_force):
            target[2] = self.target_height_rot_lift
            target[2] = state[0][2] + np.clip(target[2]-state[0][2],-self.clip_height,self.clip_height)

        # Calculate direction of the error:
        gen_dir = (target - state[0])
        # print ("error: ", gen_dir)
        factor_dir = np.sqrt(np.sum((gen_dir) ** 2)) * 100
        gen_dir = (gen_dir) / np.sqrt(np.sum((gen_dir) ** 2))
        FACTOR = - self.pos_gain * np.clip(factor_dir, -2.0, 2.0)

        # directions from current end effector to center (unit vectors)
        dir1 = (state[0] - robot_state[2][0])
        dir1 = (dir1) / np.sqrt(np.sum((dir1) ** 2))
        dir2 = (state[0] - robot_state[2][1])
        dir2 = (dir2) / np.sqrt(np.sum((dir2) ** 2))
        dir3 = (state[0] - robot_state[2][2])
        dir3 = (dir3) / np.sqrt(np.sum((dir3) ** 2))

        pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state, off_x=self.grasp_xy,
                                                          off_y=self.grasp_xy, off_z=self.grasp_h)

        pos_signal_1 = robot_state[2][0] - pos1 + FACTOR * gen_dir
        vel_signal_1 = np.zeros((3))

        pos_signal_2 = robot_state[2][1] - pos2 + FACTOR * gen_dir
        vel_signal_2 = np.zeros((3))

        pos_signal_3 = robot_state[2][2] - pos3 + FACTOR * gen_dir
        vel_signal_3 = np.zeros((3))

        # this time we use all 3 fingers
        torque = self.kinematics.imp_ctrl_3_fingers([robot_state[0], robot_state[0], robot_state[0]],
                                                    [0.0 * dir1, 0.0 * dir2, 0.0 * dir3], [4, 8, 12], \
                                                    [vel_signal_1, vel_signal_2, vel_signal_3],
                                                    [pos_signal_1, pos_signal_2, pos_signal_3], \
                                                    [robot_state[1], robot_state[1], robot_state[1]],
                                                    [self.gain_xy, self.gain_xy, self.gain_xy], \
                                                    [self.gain_z, self.gain_z, self.gain_z],
                                                    [0,0,0])

        # potentially add additional component
        # TODO: to find out: on the floor might be even more stable without this additional component,...
        add_torque = self.kinematics.add_additional_force_3_fingers_center_gain(robot_state[0], self.force_factor,
                                                                    self.force_factor,
                                                                    self.grasp.get_edge_directions(robot_state,
                                                                                                   state),
                                                                    self.grasp.get_center_array(),
                                                                    correct_torque=True)
        torque = torque + add_torque

        if not(self.additional_force):
            vertical_idx, self.curr_axis_dir = self.grasp.get_axis(state)

        if (self.additional_force):
            rotation_value = np.pi / 2

            idx_rotation_axes = np.argmax(np.abs(self.dir_rot))
            dir_rotatione = self.curr_axis_dir[:, idx_rotation_axes]
            dir_rotatione = 1* self.dir_rot[idx_rotation_axes] * dir_rotatione / np.sqrt(np.sum((dir_rotatione) ** 2))
            # dir_rotatione = np.matmul(np.asarray(pybullet.getMatrixFromQuaternion(state[1])).reshape(3,3),self.dir_rot)

            add_torque_force, add_torque_torque = self.kinematics.comp_combined_torque(copy.deepcopy(robot_state[0]),
                                                                            [0, 0, 0],
                                                                            self.force_factor_rot * dir_rotatione,
                                                                            [1,1,1], state[0],
                                                                            robot_state[2][0],
                                                                            robot_state[2][1],
                                                                            robot_state[2][2])

            torque = torque + add_torque_torque

        torque = np.clip(torque, -0.36, 0.36)
        torque = self.kinematics.compute_gravity_compensation(robot_state, torque)

        action = self.get_action(torque=np.clip(torque, self.action_space_limits.low, self.action_space_limits.high),
                                 frameskip=self._frameskip)

        self.iteration += 1
        state_finished = self.is_finished(robot_state, state, self.iteration)

        if (state_finished):
            return action, self.next_state, info
        else:
            return action, self, info

    def reset(self):
        self.iteration = -1


    def set_dir_rot(self,dir_rot):
        self.dir_rot = dir_rot

    def set_fixed_ref(self,fixed_ref):
        self.fixed_reference = fixed_ref

    def is_finished(self, robot_state, cube_state, iteration):
        if (iteration > self.iter_primitive_started + self.duration):
            self.reset()
            return True
        else:
            return False

class RotateCubeLiftPrimitiveLvl4(RotateCubeLiftPrimitive):
    '''
    Incorporates excatly the same functionality as RotateCubeLiftPrimitive
    -> only has a different duration and different reset function
    '''

    def __init__(self, env, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0], dir_rot=[0,0,0]):
        super(RotateCubeLiftPrimitiveLvl4, self).__init__(env, kinematics, params, object, grasp, stop_early, use_fixed_reference=use_fixed_reference, fixed_reference=fixed_reference, dir_rot=dir_rot)
        if (self.env.simulation):
            self.duration = 250*4
        else:
            self.duration = 500*4

    def is_finished(self, robot_state, cube_state, iteration):

        if (iteration > self.iter_primitive_started + self.duration):
            self.reset()
            return True
        else:
            return False


class ApproachObjectViapointsSpecialGrasp(ApproachObjectViapoints):
    '''
    This primitive implements the same functionality as the ApproachObjectViapoints. However, now a special grasp is
    chosen. This special grasp is exploited when the cube has to be lifted to achieve the desired rotation.
    '''

    def __init__(self, env, kinematics, params, object, grasp, approach_grasp_xy, approach_grasp_h, duration, clip_pos, stop_early=False, assign_fingers=False, dir=[0,0,0]):
        super(ApproachObjectViapointsSpecialGrasp, self).__init__(env, kinematics, params, object, grasp, approach_grasp_xy, approach_grasp_h, duration, clip_pos,stop_early=stop_early, assign_fingers=assign_fingers)
        self.dir = dir

    def __call__(self, obs, info={}):
        if (self.iteration==-1):
            self.iteration = 0
            self.action_space_limits = self.env.action_space['position']

            robot_state, state = get_robot_and_obj_state(obs, self.kinematics)
            self.iter_primitive_started = self.iteration
            self.counter = 0

            if (self.assign_fingers):
                idx_rotation_axes = np.argmax(np.abs(self.dir))
                indicator = idx_rotation_axes
                # this call is only needed to compute the vertical axes,...
                self.grasp.assign_fingers(robot_state, state)
                vertical_idx, axes_dir = self.grasp.get_axis(state)
                if (vertical_idx < idx_rotation_axes):
                    indicator -= 1
                if (indicator == 0):
                    self.grasp.assign_fingers_y(robot_state, state, direction=np.sign(np.sum(self.dir)))
                else:
                    self.grasp.assign_fingers_x(robot_state, state, direction=np.sign(np.sum(self.dir)))

            pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state,
                                                              off_x=self.approach_grasp_xy[self.counter],
                                                              off_y=self.approach_grasp_xy[self.counter],
                                                              off_z=self.approach_grasp_h[self.counter])
            self.reference = self.kinematics.inverse_kinematics_3_fingers(robot_state[0], pos1,
                                                                          pos2, pos3)


        robot_state, state = get_robot_and_obj_state(obs, self.kinematics)
        reference = robot_state[0] + np.clip(self.reference - robot_state[0], -self.clip_pos[self.counter],
                                             self.clip_pos[self.counter])

        action = self.get_action(
            position=np.clip(reference, self.action_space_limits.low, self.action_space_limits.high), frameskip=self._frameskip)
        self.iteration += 1
        state_finished = self.is_finished(robot_state, state, self.iteration)

        if (state_finished):
            return action, self.next_state, info
        else:
            return action, self, info


    def set_axis(self,dir):
        self.dir = dir

from mp.grasping import get_planned_grasp
from mp.grasping.grasp_sampling import GraspSampler

class SetGraspPrimitive(State):
    '''
    This primitive sets the state such that it can be used by planned grasp code
    '''
    def __init__(self, env, kinematics, params):
        self.env = env
        self.goal_target = None

        self.kinematics = kinematics
        self.params = params

        self.duration = 1 # duration of the primitive in timesteps
        self.DEBUG = False
        self.iter_primitive_started = -1
        self.reference = None
        self.iteration = -1
        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def __call__(self, obs, info={}):
        # this is only called on first time:
        if (self.iteration==-1):
            self.iteration = 0
            self.iter_primitive_started = self.iteration
            self.reference = copy.deepcopy(obs['robot_position'])
            self.action_space_limits = self.env.action_space['position']

        action = self.get_action(position=np.clip(self.reference,self.action_space_limits.low,self.action_space_limits.high),frameskip=self._frameskip)

        self.iteration += 1
        state_finished = self.is_finished(self.iteration)

        if (state_finished):
            grasp_sampler = GraspSampler(
                self.env, obs['object_position'], obs['object_orientation'])
            custom_grasp = [grasp_sampler.get_custom_grasp(
                obs['robot_tip_positions'])]
            try:
                grasp, path = get_planned_grasp(self.env, obs['object_position'], obs['object_orientation'],
                                                obs['goal_object_position'], obs['goal_object_orientation'],
                                                tight=True, heuristic_grasps=custom_grasp)
            except Exception:
                grasp, path = custom_grasp[0], None
            info['grasp'] = grasp
            info['path'] = path
            return action, self.next_state, info

        else:
            return action, self, info

    def reset(self):
        self.iteration = -1

    def is_finished(self, iteration):
        if (iteration>self.iter_primitive_started+self.duration):
            self.reset()
            return True
        else:
            return False


class SetGraspFromOutsidePrimitive(State):
    '''
    This primitive sets a grasp compatible to CIC code based on the current position of the fingers,...
    '''
    def __init__(self, env, kinematics, params, object, grasp):
        self.env = env
        self.goal_target = None

        self.kinematics = kinematics
        self.params = params
        self.grasp = grasp

        self.duration = 1 # duration of the primitive in timesteps
        self.DEBUG = False
        self.iter_primitive_started = -1
        self.reference = None
        self.iteration = -1
        if (self.env.simulation):
            self._frameskip = 1
        else:
            self._frameskip = 5

    def connect(self, next_state, done_state):
        self.next_state = next_state
        self.done_state = done_state

    def __call__(self, obs, info={}):
        # this is only called on first time:
        if (self.iteration==-1):
            self.iteration = 0
            self.iter_primitive_started = self.iteration
            self.reference = copy.deepcopy(obs['robot_position'])
            self.action_space_limits = self.env.action_space['position']

        action = self.get_action(position=np.clip(self.reference,self.action_space_limits.low,self.action_space_limits.high),frameskip=self._frameskip)

        self.iteration += 1
        state_finished = self.is_finished(self.iteration)

        if (state_finished):
            robot_state, state = get_robot_and_obj_state(obs, self.kinematics)
            self.grasp.set_directions(robot_state,state)

            return action, self.next_state, info

        else:
            return action, self, info

    def reset(self):
        self.iteration = -1

    def is_finished(self, iteration):
        if (iteration>self.iter_primitive_started+self.duration):
            self.reset()
            return True
        else:
            return False
