import numpy as np
import copy
from .math_tools import rpy2Mat
import time

from itertools import permutations



class PrimitiveObject:
    def __init__(self):
        pass

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        pass


    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        pass

class IdlePrimitive(PrimitiveObject):
    def __init__(self, kinematics, params):
        super(IdlePrimitive, self).__init__()
        self.goal_target = None

        self.kinematics = kinematics
        self.params = params

        self.duration = 100*4 # duration of the primitive in timesteps
        self.DEBUG = False
        self.iter_primitive_started = -1
        self.reference = None

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        if (initial):
            self.iter_primitive_started = iteration
            self.reference = copy.deepcopy(robot_state[0])
        thisdict = {
            "torque": np.zeros((9)),
            "position": self.reference
        }
        return thisdict


    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        if (iteration>self.iter_primitive_started+self.duration):
            return True
        else:
            return False

class InitResetPrimitive(PrimitiveObject):
    def __init__(self, kinematics, params):
        super(InitResetPrimitive, self).__init__()
        self.goal_target = None

        self.duration = 400*4 # duration of the primitive in timesteps
        self.DEBUG = False
        self.iter_primitive_started = -1
        self.reference = None

        self.kinematics = kinematics
        self.params = params

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        if (initial):
            self.iter_primitive_started = iteration
            # first iteration -> compute cartesian reference position:
            unit_vec = np.asarray([0.0, 0.2, 0.2])
            offset = 15
            self.unit_vec_x = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(0+offset))).reshape(3, 3), unit_vec)
            self.unit_vec_y = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(240+offset))).reshape(3, 3), unit_vec)
            self.unit_vec_z = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(120+offset))).reshape(3, 3), unit_vec)

            self.reference = self.kinematics.inverse_kinematics_3_fingers(robot_state[0],self.unit_vec_x, self.unit_vec_y, self.unit_vec_z)

        if (iteration==self.iter_primitive_started+200):
            unit_vec = np.asarray([0.0, 0.1, 0.1])
            offset = 15
            self.unit_vec_x = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(0+offset))).reshape(3, 3), unit_vec)
            self.unit_vec_y = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(240+offset))).reshape(3, 3), unit_vec)
            self.unit_vec_z = np.matmul(np.asarray(rpy2Mat(0, 0, np.deg2rad(120+offset))).reshape(3, 3), unit_vec)
            self.reference = self.kinematics.inverse_kinematics_3_fingers(robot_state[0],self.unit_vec_x, self.unit_vec_y, self.unit_vec_z)

        thisdict = {
            "torque": np.zeros((9)),
            "position": self.reference
        }
        return thisdict


    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        if (iteration>self.iter_primitive_started+self.duration):
            return True
        else:
            return False

class MoveObjectFloorPrimitive(PrimitiveObject):
    '''
    Simplistic Movement Primitive for Moving the object
    -> also a grasp has to be chosen,...
    -> will consist of other primitives that also take care of how the cube is approached,...
    '''
    def __init__(self, kinematics, params, object, grasp, stop_early):
        super(MoveObjectFloorPrimitive, self).__init__()
        self.goal_target = None
        self.stop_early = stop_early

        self.object = object
        self.grasp = grasp

        self.kinematics = kinematics
        self.params = params

        self.grasp_xy = self.params.grasp_xy
        self.grasp_h = self.params.grasp_h
        self.gain_xy = 0.0
        self.gain_z = 0.0
        self.gain_xy_pre = self.params.gain_xy_pre
        self.gain_z_pre = self.params.gain_z_pre
        self.init_dur = self.params.init_dur
        self.gain_xy_final = self.params.gain_xy_final
        self.gain_z_final = self.params.gain_z_final
        self.pos_gain_impedance =  self.params.pos_gain_impedance
        self.pos_gain = 0.0
        self.force_factor = self.params.force_factor

        self.duration = 5000*4

        self.DEBUG = False
        self.iter_primitive_started = -1
        self.reference = None

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        '''
        Will contain multiple steps: 1-> approach the fingers to make contact
        2 -> manipulate the object
        '''
        if (initial):
            self.iter_primitive_started = iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.pos_gain = 0.0

        if (iteration==self.iter_primitive_started+self.init_dur):
            self.gain_xy = self.gain_xy_final
            self.gain_z = self.gain_z_final
            self.pos_gain = self.pos_gain_impedance

        #Calculate direction of the error:
        gen_dir = (state[3] - state[0])
        factor_dir = np.sqrt(np.sum((gen_dir) ** 2)) * 100
        gen_dir = (gen_dir) / np.sqrt(np.sum((gen_dir) ** 2))
        FACTOR = -0.25*self.pos_gain*np.clip(factor_dir, -2.0, 2.0)

        # directions from current end effector to center (unit vectors)
        dir1 = (state[0] - robot_state[2][0])
        dir1 = (dir1) / np.sqrt(np.sum((dir1) ** 2))
        dir2 = (state[0] - robot_state[2][1])
        dir2 = (dir2) / np.sqrt(np.sum((dir2) ** 2))
        dir3 = (state[0] - robot_state[2][2])
        dir3 = (dir3) / np.sqrt(np.sum((dir3) ** 2))

        pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state, off_x=self.grasp_xy, off_y =self.grasp_xy, off_z=self.grasp_h)

        pos_signal_1 = robot_state[2][0] - pos1 + FACTOR*gen_dir
        vel_signal_1 = np.zeros((3))

        pos_signal_2 = robot_state[2][1] - pos2 + FACTOR*gen_dir
        vel_signal_2 = np.zeros((3))

        pos_signal_3 = robot_state[2][2] - pos3 + FACTOR*gen_dir
        vel_signal_3 = np.zeros((3))

        torque = self.kinematics.imp_ctrl_3_fingers([robot_state[0],robot_state[0],robot_state[0]],[0.0*dir1,0.0*dir2,0.0*dir3],[4,8,12],\
                                         [vel_signal_1,vel_signal_2,vel_signal_3], [pos_signal_1,pos_signal_2,pos_signal_3],\
                                         [robot_state[1],robot_state[1],robot_state[1]], [self.gain_xy, self.gain_xy,self.gain_xy],\
                                         [self.gain_z, self.gain_z,self.gain_z], self.grasp.get_center_array())

        # potentially add additional component
        # TODO: to find out: on the floor might be even more stable without this additional component,...
        add_torque = self.kinematics.add_additional_force_3_fingers(robot_state[0],self.force_factor,self.grasp.get_edge_directions(robot_state,state),self.grasp.get_center_array(),correct_torque=True)

        torque = np.clip(torque+add_torque,-0.36,0.36)

        thisdict = {
            "torque": torque,
            "position": np.zeros((9))
        }
        return thisdict


    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        if (iteration>self.iter_primitive_started+self.duration):
            return True
        else:
            return False

class MoveCubeFloorPrimitive(PrimitiveObject):
    '''
    Simplistic Movement Primitive for Moving the Cube with a 3 finger grasp (this change is needed,...)
    '''

    def __init__(self, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0]):
        super(MoveCubeFloorPrimitive, self).__init__()
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
        self.iter_primitive_started = -1
        self.reference = None

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        '''
        Will contain multiple steps: 1-> approach the fingers to make contact
        2 -> manipulate the object
        '''
        if (self.use_fixed_reference):
            state[3] = self.fixed_reference

        if (initial):
            self.iter_primitive_started = iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.pos_gain = 0.0

        if (iteration == self.iter_primitive_started + self.init_dur):
            self.gain_xy = self.gain_xy_final
            self.gain_z = self.gain_z_final
            self.pos_gain = self.pos_gain_impedance

        # Calculate direction of the error:
        gen_dir = (state[3] - state[0])
        print ("error: ", gen_dir)
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

        thisdict = {
            "torque": torque,
            "position": np.zeros((9))
        }
        return thisdict

    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        if (iteration > self.iter_primitive_started + self.duration):
            return True
        else:
            return False

class MoveCubeFloorPrimitiveLvl1(MoveCubeFloorPrimitive):
    '''
    Incorporates excatly the same functionality as the movecubefloorprimitive
    -> only has a different duration
    -> only has different reset function,...
    '''

    def __init__(self, kinematics, params, object, grasp, stop_early):
        super(MoveCubeFloorPrimitiveLvl1, self).__init__(kinematics, params, object, grasp, stop_early)
        self.duration = 300*4

    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        if (self.use_fixed_reference):
            cube_state[3] = self.fixed_reference

        if (iteration > self.iter_primitive_started + self.duration):
            curr_dist = (cube_state[3] - cube_state[6][:, 0]) * 100
            dir_progress = (cube_state[6][:, 0] - cube_state[6][:, 50]) * 100
            dist_law = (np.sum(dir_progress) >= 0.0 and np.sqrt(np.sum(curr_dist ** 2)) > 2.5)
            if (dist_law):
                return True
            else:
                return False
        else:
            return False

class MoveCubeFloorPrimitiveLvl4(MoveCubeFloorPrimitive):
    '''
    Incorporates excatly the same functionality as the movecubefloorprimitive
    -> only has a different duration
    -> only has different reset function,...
    '''

    def __init__(self, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0]):
        super(MoveCubeFloorPrimitiveLvl4, self).__init__(kinematics, params, object, grasp, stop_early, use_fixed_reference=use_fixed_reference, fixed_reference=fixed_reference)
        self.duration = 300*4

    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        if (self.use_fixed_reference):
            cube_state[3] = self.fixed_reference

        curr_dist = (cube_state[3] - cube_state[6][:, 0]) * 100
        # Add quick reset functionality when wanting to move on the floor
        if (np.sqrt(np.sum(curr_dist ** 2)) < 2.5):
            return True

        if (iteration > self.iter_primitive_started + self.duration):
            dir_progress = (cube_state[6][:, 0] - cube_state[6][:, 50]) * 100
            dist_law = (np.sum(dir_progress) >= 0.0 and np.sqrt(np.sum(curr_dist ** 2)) > 2.5)
            if (dist_law):
                return True
            else:
                return False
        else:
            return False

class MoveLiftCubePrimitive(PrimitiveObject):
    '''
    Simplistic Movement Primitive for Moving the Cube with a 3 finger grasp (this change is needed,...)
    '''

    def __init__(self, kinematics, params, object, grasp, stop_early):
        super(MoveLiftCubePrimitive, self).__init__()
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
        self.iter_primitive_started = -1
        self.reference = None

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        '''
        Will contain multiple steps: 1-> approach the fingers to make contact
        2 -> manipulate the object
        '''
        if (initial):
            self.iter_primitive_started = iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.start_lift = False
            self.pos_gain = 0.0

        if (iteration == self.iter_primitive_started + self.init_dur):
            self.gain_xy = self.gain_xy_ground
            self.gain_z = self.gain_z_ground
            self.pos_gain = self.pos_gain_impedance_ground

        xy_dist = np.sqrt(np.sum((state[3][:2] - state[0][:2]) ** 2))
        if (not(self.start_lift) and (iteration > self.iter_primitive_started + self.init_dur) and (xy_dist < self.switch_mode)):
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
        print ("error: ", gen_dir)
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
        add_torque = np.zeros((9))
        if (self.start_lift):
            add_torque = self.kinematics.add_additional_force_3_fingers(robot_state[0], self.force_factor,
                                                                        self.grasp.get_edge_directions(robot_state,
                                                                                                       state),
                                                                        [0,0,0],
                                                                        correct_torque=True)

        torque = np.clip(torque+add_torque, -0.36, 0.36)

        thisdict = {
            "torque": torque,
            "position": np.zeros((9))
        }
        return thisdict

    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        if (iteration > self.iter_primitive_started + self.duration):
            return True
        else:
            return False

class MoveLiftCubePrimitiveLvl2(MoveLiftCubePrimitive):
    '''
    Incorporates excatly the same functionality as the movecubefloorprimitive
    -> only has a different duration
    -> only has different reset function,...
    '''

    def __init__(self, kinematics, params, object, grasp, stop_early):
        super(MoveLiftCubePrimitiveLvl2, self).__init__(kinematics, params, object, grasp, stop_early)
        self.duration = 300*4

    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        target = copy.deepcopy(cube_state[3])
        if (self.start_lift):
            target[2] = cube_state[0][2] + np.clip(target[2]-cube_state[0][2],-self.clip_height,self.clip_height)
        else:
            target[2] = 0.0325

        if (iteration > self.iter_primitive_started + self.duration):
            curr_dist = (target - cube_state[6][:, 0]) * 100
            dir_progress = (cube_state[6][:, 0] - cube_state[6][:, 50]) * 100
            dist_law = (np.sum(dir_progress) >= 0.0 and np.sqrt(np.sum(curr_dist ** 2)) > 2.5)
            if (dist_law):
                return True
            else:
                return False
        else:
            return False


class ApproachObjectViapoints(PrimitiveObject):

    def __init__(self, kinematics, params, object, grasp, approach_grasp_xy, approach_grasp_h, duration, clip_pos, stop_early=False, assign_fingers=False):
        super(ApproachObjectViapoints, self).__init__()
        self.goal_target = None
        self.stop_early = stop_early

        self.kinematics = kinematics
        self.params = params

        self.object = object
        self.grasp = grasp

        self.approach_grasp_xy = approach_grasp_xy
        self.approach_grasp_h = approach_grasp_h
        self.duration = duration*4
        self.counter = 0
        self.counter_end = len(duration)

        self.clip_pos = clip_pos

        self.assign_fingers = assign_fingers

        self.DEBUG = False
        self.iter_primitive_started = -1
        self.reference = None

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        '''
        Will contain multiple steps: 1-> approach the fingers
        2 -> manipulate the object,...
        -> very well "imagineable that the individual components are again primitives,..."
        '''
        if (initial):
            self.iter_primitive_started = iteration
            self.counter = 0
            if(self.assign_fingers):
                self.grasp.assign_fingers(robot_state, state)

            pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state, off_x=self.approach_grasp_xy[self.counter], off_y =self.approach_grasp_xy[self.counter], off_z=self.approach_grasp_h[self.counter])
            self.reference = self.kinematics.inverse_kinematics_3_fingers(robot_state[0], pos1,
                                                                          pos2, pos3)


        thisdict = {
            "torque": np.zeros((9)),
            "position": robot_state[0] + np.clip(self.reference - robot_state[0], -self.clip_pos[self.counter], self.clip_pos[self.counter])
        }
        return thisdict




    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
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
                return True
        else:
            return False

class ApproachObjectViapointsSpecialGrasp(ApproachObjectViapoints):

    def __init__(self, kinematics, params, object, grasp, approach_grasp_xy, approach_grasp_h, duration, clip_pos, stop_early=False, assign_fingers=False, dir=[0,0,0]):
        super(ApproachObjectViapointsSpecialGrasp, self).__init__(kinematics, params, object, grasp, approach_grasp_xy, approach_grasp_h, duration, clip_pos,stop_early=stop_early, assign_fingers=assign_fingers)
        self.dir = dir

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        '''
        Will contain multiple steps: 1-> approach the fingers
        2 -> manipulate the object,...
        -> very well "imagineable that the individual components are again primitives,..."
        '''
        if (initial):
            self.iter_primitive_started = iteration
            self.counter = 0
            if(self.assign_fingers):
                idx_rotation_axes = np.argmax(np.abs(self.dir))
                indicator = idx_rotation_axes
                # this call is only needed to compute the vertical axes,...
                self.grasp.assign_fingers(robot_state,state)
                vertical_idx, axes_dir = self.grasp.get_axis(state)
                if (vertical_idx < idx_rotation_axes):
                    indicator -= 1
                if (indicator == 0):
                    self.grasp.assign_fingers_y(robot_state, state, direction=np.sign(np.sum(self.dir)))
                else:
                    self.grasp.assign_fingers_x(robot_state, state, direction=np.sign(np.sum(self.dir)))

            pos1, pos2, pos3 = self.grasp.get_finger_position(robot_state, state, off_x=self.approach_grasp_xy[self.counter], off_y =self.approach_grasp_xy[self.counter], off_z=self.approach_grasp_h[self.counter])
            self.reference = self.kinematics.inverse_kinematics_3_fingers(robot_state[0], pos1,
                                                                          pos2, pos3)


        thisdict = {
            "torque": np.zeros((9)),
            "position": robot_state[0] + np.clip(self.reference - robot_state[0], -self.clip_pos[self.counter], self.clip_pos[self.counter])
        }
        return thisdict

    def set_axis(self,dir):
        self.dir = dir




class RotateCubeFloorPrimitive(PrimitiveObject):
    '''
    Simplistic Movement Primitive for Moving the Cube with a 3 finger grasp (this change is needed,...)
    '''

    def __init__(self, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0], dir_rot=[0,0,0]):
        super(RotateCubeFloorPrimitive, self).__init__()
        self.goal_target = None
        self.stop_early = stop_early
        self.use_fixed_reference = use_fixed_reference
        self.fixed_reference = fixed_reference
        self.dir_rot = dir_rot

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
        self.iter_primitive_started = -1
        self.reference = None
        self.additional_force = False

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        '''
        Will contain multiple steps: 1-> approach the fingers to make contact
        2 -> manipulate the object
        '''
        if (self.use_fixed_reference):
            state[3] = self.fixed_reference

        if (initial):
            self.iter_primitive_started = iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.pos_gain = 0.0
            self.additional_force = False

        if (iteration == self.iter_primitive_started + self.init_dur):
            self.gain_xy = self.gain_xy_final
            self.gain_z = self.gain_z_final
            self.pos_gain = self.pos_gain_impedance
            self.additional_force = True

        # Calculate direction of the error:
        gen_dir = (state[3] - state[0])
        print ("error: ", gen_dir)
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
            rotation_value = np.pi / 2
            vertical_idx, curr_axis_dir = self.grasp.get_axis(state)
            idx_rotation_axes = np.argmax(np.abs(self.dir_rot))
            dir_rotatione = curr_axis_dir[:, idx_rotation_axes]
            dir_rotatione = rotation_value * self.dir_rot[idx_rotation_axes] * dir_rotatione / np.sqrt(np.sum((dir_rotatione) ** 2))

            add_torque_force, add_torque_torque = self.kinematics.comp_combined_torque(copy.deepcopy(robot_state[0]),
                                                                            [0, 0, 0],
                                                                            self.force_factor_ground_rot * dir_rotatione,
                                                                            [1, 1, 1], state[0],
                                                                            robot_state[2][0],
                                                                            robot_state[2][1],
                                                                            robot_state[2][2])

            torque = torque + add_torque_torque

        torque = np.clip(torque, -0.36, 0.36)

        thisdict = {
            "torque": torque,
            "position": np.zeros((9))
        }
        return thisdict

    def set_dir_rot(self,dir_rot):
        self.dir_rot = dir_rot

    def set_fixed_ref(self,fixed_ref):
        self.fixed_reference = fixed_ref

    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        if (iteration > self.iter_primitive_started + self.duration):
            return True
        else:
            return False

class RotateCubeFloorPrimitiveLvl4(RotateCubeFloorPrimitive):
    '''
    Incorporates excatly the same functionality as the movecubefloorprimitive
    -> only has a different duration
    -> only has different reset function,...
    '''

    def __init__(self, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0], dir_rot=[0,0,0]):
        super(RotateCubeFloorPrimitiveLvl4, self).__init__(kinematics, params, object, grasp, stop_early, use_fixed_reference=use_fixed_reference, fixed_reference=fixed_reference, dir_rot=dir_rot)
        self.duration = 300*4

    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):

        if (iteration > self.iter_primitive_started + self.duration):
            return True
        else:
            return False


class RotateCubeLiftPrimitive(PrimitiveObject):
    '''
    Simplistic Movement Primitive for Moving the Cube with a 3 finger grasp (this change is needed,...)
    '''

    def __init__(self, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0], dir_rot=[0,0,0]):
        super(RotateCubeLiftPrimitive, self).__init__()
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
        self.iter_primitive_started = -1
        self.reference = None
        self.additional_force = False

    def step(self, robot_state, state, cube_state_filtered, goal, iteration, initial):
        '''
        Will contain multiple steps: 1-> approach the fingers to make contact
        2 -> manipulate the object
        '''
        if (self.use_fixed_reference):
            state[3] = self.fixed_reference

        if (initial):
            self.iter_primitive_started = iteration
            # TODO ALSO: RESET THE GAINS,...
            self.gain_xy = self.gain_xy_pre
            self.gain_z = self.gain_z_pre
            self.pos_gain = 0.0
            self.additional_force = False

        if (iteration == self.iter_primitive_started + self.init_dur):
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
        print ("error: ", gen_dir)
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
            dir_rotatione = -1* self.dir_rot[idx_rotation_axes] * dir_rotatione / np.sqrt(np.sum((dir_rotatione) ** 2))

            add_torque_force, add_torque_torque = self.kinematics.comp_combined_torque(copy.deepcopy(robot_state[0]),
                                                                            [0, 0, 0],
                                                                            self.force_factor_rot * dir_rotatione,
                                                                            [1,1,1], state[0],
                                                                            robot_state[2][0],
                                                                            robot_state[2][1],
                                                                            robot_state[2][2])

            torque = torque + add_torque_torque

        torque = np.clip(torque, -0.36, 0.36)

        thisdict = {
            "torque": torque,
            "position": np.zeros((9))
        }
        return thisdict

    def set_dir_rot(self,dir_rot):
        self.dir_rot = dir_rot

    def set_fixed_ref(self,fixed_ref):
        self.fixed_reference = fixed_ref

    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):
        if (iteration > self.iter_primitive_started + self.duration):
            return True
        else:
            return False

class RotateCubeLiftPrimitiveLvl4(RotateCubeLiftPrimitive):
    '''
    Incorporates excatly the same functionality as the movecubefloorprimitive
    -> only has a different duration
    -> only has different reset function,...
    '''

    def __init__(self, kinematics, params, object, grasp, stop_early, use_fixed_reference=False, fixed_reference=[0,0,0], dir_rot=[0,0,0]):
        super(RotateCubeLiftPrimitiveLvl4, self).__init__(kinematics, params, object, grasp, stop_early, use_fixed_reference=use_fixed_reference, fixed_reference=fixed_reference, dir_rot=dir_rot)
        self.duration = 500*4

    def is_finished(self, robot_state, cube_state, cube_state_filtered, goal, iteration):

        if (iteration > self.iter_primitive_started + self.duration):
            return True
        else:
            return False