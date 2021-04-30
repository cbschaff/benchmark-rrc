from mp.align_rotation import roll_and_pitch_aligned, get_yaw_diff
from mp.states import State
from .base_states import *

from cic import states as states_cic
from cic import rotation_primitives as rotation_primitives_cic
from cic.utils import get_robot_and_obj_state

import numpy as np
import pybullet

'''
This file contains the sequences from the Code of TU Darmstadt. Basically, a sequence is a concatenation of Primitives
to realize a more complex behavior.
'''


class BringCenterPrimitive(State):
    '''
    Primitive contains concatenation of base primitives which results in moving the cube to the center
    '''
    def __init__(self, env, kinematics, parameters, object, grasp):
        self.goinitpose = states_cic.InitResetPrimitive(env, kinematics=kinematics, params=parameters)
        self.approach = states_cic.ApproachObjectViapoints(env, kinematics, parameters, object, grasp,
                                                      parameters.approach_grasp_xy, parameters.approach_grasp_h,
                                                      parameters.approach_duration, parameters.approach_clip_pos,
                                                      stop_early=True, assign_fingers=True)
        self.bring_center = states_cic.MoveCubeFloorPrimitiveLvl4(env, kinematics, parameters, object, grasp, False,
                                                       use_fixed_reference=True, fixed_reference=[0, 0, 0.0325])
        self.reset_state = states_cic.ApproachObjectViapoints(env, kinematics, parameters, object, grasp,
                                                   parameters.reset_grasp_xy, parameters.reset_grasp_h,
                                                   parameters.reset_duration, parameters.reset_clip_pos,
                                                   stop_early=True, assign_fingers=False)

    def connect(self, next_state, done_state):
        self.goinitpose.connect(
            next_state=self.approach,
            done_state=done_state
        )
        self.approach.connect(
            next_state=self.bring_center,
            done_state=done_state
        )
        self.bring_center.connect(
            next_state=self.reset_state,
            done_state=done_state
        )
        self.reset_state.connect(
            next_state=next_state,
            done_state=done_state
        )

    def reset(self):
        self.goinitpose.reset()
        self.approach.reset()
        self.bring_center.reset()
        self.reset_state.reset()

    def __call__(self, obs, info={}):
        substate = self.goinitpose
        return substate(obs, info)

class BringFinalPoseOrientationPrimitive(State):
    '''
    Primitive contains concatenation of base primitives which results in moving the cube to the final location including lifting
    Note: Now in this implementation, the orientation of the cube is corrected
    '''
    def __init__(self, env, kinematics, parameters, object, grasp, grasp1=None):
        self.goinitpose = states_cic.InitResetPrimitive(env, kinematics=kinematics, params=parameters)
        self.approach = states_cic.ApproachObjectViapoints(env, kinematics, parameters, object, grasp,
                                                      parameters.approach_grasp_xy, parameters.approach_grasp_h,
                                                      parameters.approach_duration, parameters.approach_clip_pos,
                                                      stop_early=True, assign_fingers=True)

        self.grasp1 = grasp1
        if (grasp1 is None):
            grasp_to_be_used = grasp
        else:
            self.set_cic_grasp = states_cic.SetGraspFromOutsidePrimitive(env, kinematics,
                                                                         parameters, object, grasp1)
            grasp_to_be_used = grasp1

        self.move_cube_lift = states_cic.MoveLiftCubeOrientPrimitiveLvl4(env, kinematics, parameters, object, grasp_to_be_used,
                                                                False)
        self.reset_state = states_cic.ApproachObjectViapoints(env, kinematics, parameters, object, grasp_to_be_used,
                                                   parameters.reset_grasp_xy, parameters.reset_grasp_h,
                                                   parameters.reset_duration, parameters.reset_clip_pos,
                                                   stop_early=True, assign_fingers=False)

    def reset(self):
        self.goinitpose.reset()
        self.approach.reset()
        if not(self.grasp1 is None):
            self.set_cic_grasp.reset()
        self.move_cube_lift.reset()
        self.reset_state.reset()

    def connect(self, next_state, done_state):
        self.goinitpose.connect(
            next_state=self.approach,
            done_state=done_state
        )
        if (self.grasp1 is None):
            self.approach.connect(
                next_state=self.move_cube_lift,
                done_state=done_state
            )
        else:
            self.approach.connect(
                next_state=self.set_cic_grasp,
                done_state=done_state
            )
            self.set_cic_grasp.connect(
                next_state=self.move_cube_lift,
                done_state=done_state
            )

        self.move_cube_lift.connect(
            next_state=self.reset_state,
            done_state=done_state
        )
        self.reset_state.connect(
            next_state=next_state,
            done_state=done_state
        )

    def __call__(self, obs, info={}):
        substate = self.goinitpose
        return substate(obs, info)


class BringFinalPosePrimitive(State):
    '''
    Primitive contains concatenation of base primitives which results in moving the cube to the final location including lifting
    Note: at the moment in this final action, the orientation of the cube is not corrected / taken into account
    '''
    def __init__(self, env, kinematics, parameters, object, grasp):
        self.goinitpose = states_cic.InitResetPrimitive(env, kinematics=kinematics, params=parameters)
        self.approach = states_cic.ApproachObjectViapoints(env, kinematics, parameters, object, grasp,
                                                      parameters.approach_grasp_xy, parameters.approach_grasp_h,
                                                      parameters.approach_duration, parameters.approach_clip_pos,
                                                      stop_early=True, assign_fingers=True)
        self.move_cube_lift = states_cic.MoveLiftCubePrimitiveLvl2(env, kinematics, parameters, object, grasp,
                                                                False)
        self.reset_state = states_cic.ApproachObjectViapoints(env, kinematics, parameters, object, grasp,
                                                   parameters.reset_grasp_xy, parameters.reset_grasp_h,
                                                   parameters.reset_duration, parameters.reset_clip_pos,
                                                   stop_early=True, assign_fingers=False)

    def reset(self):
        self.goinitpose.reset()
        self.approach.reset()
        self.move_cube_lift.reset()
        self.reset_state.reset()

    def connect(self, next_state, done_state):
        self.goinitpose.connect(
            next_state=self.approach,
            done_state=done_state
        )
        self.approach.connect(
            next_state=self.move_cube_lift,
            done_state=done_state
        )
        self.move_cube_lift.connect(
            next_state=self.reset_state,
            done_state=done_state
        )
        self.reset_state.connect(
            next_state=next_state,
            done_state=done_state
        )

    def __call__(self, obs, info={}):
        substate = self.goinitpose
        return substate(obs, info)

class BringFinalPosePrimitiveLvl4(BringFinalPosePrimitive):
    '''
    Primitive contains concatenation of base primitives which results in moving the cube to the final location including lifting
    Note: at the moment in this final action, the orientation of the cube is not corrected / taken into account
    '''

    def __init__(self, env, kinematics, parameters, object, grasp):
        super(BringFinalPosePrimitiveLvl4, self).__init__(env, kinematics, parameters, object, grasp)
        self.move_cube_lift = states_cic.MoveLiftCubePrimitiveLvl4(env, kinematics, parameters, object, grasp,
                                                                   False)


class CorrectOrientationGroundPrimitive(State):
    '''
    Primitive contains concatenation of base primitives which results in rotating the cube into the right orientation when
    wanting to rotate around an axis perpendicular to the ground plane
    '''
    def __init__(self, env, kinematics, parameters, object, grasp,grasp1=None):
        self.goinitpose = states_cic.InitResetPrimitive(env, kinematics=kinematics, params=parameters)
        self.approach = states_cic.ApproachObjectViapoints(env, kinematics, parameters, object, grasp,
                                                      parameters.approach_grasp_xy, parameters.approach_grasp_h,
                                                      parameters.approach_duration, parameters.approach_clip_pos,
                                                      stop_early=True, assign_fingers=True, use_unrestricted_grasp=True)


        self.grasp1 = grasp1
        if (grasp1 is None):
            grasp_to_be_used = grasp
        else:
            self.set_cic_grasp = states_cic.SetGraspFromOutsidePrimitive(env, kinematics,
                                                                         parameters, object, grasp1)
            grasp_to_be_used = grasp1


        self.rotate = states_cic.RotateCubeFloorPrimitiveLvl4(env, kinematics, parameters, object, grasp_to_be_used, False, use_fixed_reference=True, fixed_reference=[0,0,0.0325],dir_rot=[0,0,1])
        self.reset_state = states_cic.ApproachObjectViapoints(env, kinematics, parameters, object, grasp_to_be_used,
                                                   parameters.reset_grasp_xy, parameters.reset_grasp_h,
                                                   parameters.reset_duration, parameters.reset_clip_pos,
                                                   stop_early=True, assign_fingers=False)

    def reset(self):
        self.goinitpose.reset()
        self.approach.reset()
        if not(self.grasp1 is None):
            self.set_cic_grasp.reset()
        self.rotate.reset()
        self.reset_state.reset()

    def set_rotation_directions(self, direction, position, magnitude):
        self.rotate.set_dir_rot(direction, magnitude)
        self.rotate.set_fixed_ref(position)

    def connect(self, next_state, done_state):
        self.goinitpose.connect(
            next_state=self.approach,
            done_state=done_state
        )
        if (self.grasp1 is None):
            self.approach.connect(
                next_state=self.rotate,
                done_state=done_state
            )
        else:
            self.approach.connect(
                next_state=self.set_cic_grasp,
                done_state=done_state
            )
            self.set_cic_grasp.connect(
                next_state=self.rotate,
                done_state=done_state
            )

        self.rotate.connect(
            next_state=self.reset_state,
            done_state=done_state
        )
        self.reset_state.connect(
            next_state=next_state,
            done_state=done_state
        )

    def __call__(self, obs, info={}):
        substate = self.goinitpose
        return substate(obs, info)


class CorrectOrientationAirPrimitive(State):
    '''
    Primitive contains concatenation of base primitives which results in rotating the cube into the right orientation when
    wanting to rotate around an axis parallel to the ground plane
    '''
    def __init__(self, env, kinematics, parameters, object, grasp, grasp1=None):
        # if (grasp1 is None):
        #     grasp1 = grasp
        self.goinitpose = states_cic.InitResetPrimitive(env, kinematics=kinematics, params=parameters)
        self.approachrotlift = states_cic.ApproachObjectViapointsSpecialGrasp(env, kinematics, parameters, object, grasp,
                                                 parameters.approach_grasp_xy, parameters.approach_grasp_h,
                                                 parameters.approach_duration, parameters.approach_clip_pos,
                                                 stop_early=True, assign_fingers=True, dir=[1,0,0])

        self.grasp1 = grasp1
        if (grasp1 is None):
            grasp_to_be_used = grasp
        else:
            self.set_cic_grasp = states_cic.SetGraspFromOutsidePrimitive(env, kinematics,
                                                                         parameters, object, grasp1)
            grasp_to_be_used = grasp1

        self.rotate_lift = states_cic.RotateCubeLiftPrimitiveLvl4(env, kinematics, parameters, object, grasp_to_be_used, False, use_fixed_reference=True, fixed_reference=[0,0,0.0325],dir_rot=[1,0,0])
        self.reset_state = states_cic.ApproachObjectViapoints(env, kinematics, parameters, object, grasp_to_be_used,
                                                   parameters.reset_grasp_xy, parameters.reset_grasp_h,
                                                   parameters.reset_duration, parameters.reset_clip_pos,
                                                   stop_early=True, assign_fingers=False)

    def reset(self):
        self.goinitpose.reset()
        self.approachrotlift.reset()
        if not(self.grasp1 is None):
            self.set_cic_grasp.reset()
        self.rotate_lift.reset()
        self.reset_state.reset()

    def set_rotation_directions(self, direction, position):
        self.approachrotlift.set_axis(direction)
        self.rotate_lift.set_dir_rot(direction)
        self.rotate_lift.set_fixed_ref(position)

    def connect(self, next_state, done_state):
        self.goinitpose.connect(
            next_state=self.approachrotlift,
            done_state=done_state
        )
        if (self.grasp1 is None):
            self.approachrotlift.connect(
                next_state=self.rotate_lift,
                done_state=done_state
            )
        else:
            self.approachrotlift.connect(
                next_state=self.set_cic_grasp,
                done_state=done_state
            )
            self.set_cic_grasp.connect(
                next_state=self.rotate_lift,
                done_state=done_state
            )

        self.rotate_lift.connect(
            next_state=self.reset_state,
            done_state=done_state
        )
        self.reset_state.connect(
            next_state=next_state,
            done_state=done_state
        )

    def __call__(self, obs, info={}):
        substate = self.goinitpose
        return substate(obs, info)


class AlignAxesPrimitive(State):
    '''
    Primitive contains concatenation of base primitives which results in rotating the cube into the right orientation.
    It includes a process of decision making which decides which rotations have to be executed after each other.
    '''
    def __init__(self, env, kinematics, parameters, object, grasp, grasp1=None):
        self.kinematics = kinematics
        self.grasp = grasp
        # print (env.goal['orientation'])
        self.planner = rotation_primitives_cic.RotationPrimitives(rotation_primitives_cic.to_H(np.eye(3)),
                                                              rotation_primitives_cic.to_H(np.eye(3)))

        self.bring_center = BringCenterPrimitive(env, kinematics, parameters, object, grasp)
        self.rotate_ground = CorrectOrientationGroundPrimitive(env, kinematics, parameters, object, grasp, grasp1=grasp1)
        self.rotate_air = CorrectOrientationAirPrimitive(env, kinematics, parameters, object, grasp, grasp1=grasp1)

        self.next_state = None

        # self.end_recursion = EndRecursion(env)
        # self.start_recursion = StartRecursion(env)

    def reset(self):
        self.bring_center.reset()
        self.rotate_ground.reset()
        self.rotate_air.reset()

    def connect(self, next_state, done_state):
        self.bring_center.connect(
            next_state=self,
            done_state=done_state
        )
        self.rotate_ground.connect(
            next_state=self,
            done_state=done_state
        )
        self.rotate_air.connect(
            next_state=self,
            done_state=done_state
        )

        #TODO: this is not yet finally cleanly implemented,...
        self.next_state = next_state

        # self.start_recursion.connect(
        #     next_state=next_state
        # )




    def __call__(self, obs, info={}):
        # Maybe the selection here slighttly breaks the recursion,...
        robot_state, state = get_robot_and_obj_state(obs, self.kinematics)
        self.planner.set_goal(rotation_primitives_cic.to_H(np.asarray(pybullet.getMatrixFromQuaternion(obs["goal_object_orientation"])).reshape(3, 3)))
        self.planner.set_current_pose(
            rotation_primitives_cic.to_H(np.asarray(pybullet.getMatrixFromQuaternion(state[1])).reshape(3, 3)))
        sequence, sequence1g, sequence_direct = self.planner.get_control_seq()

        print("SEQUENCE")
        print(sequence)
        print(sequence1g)
        print(sequence_direct)
        # input ("WAIT")
        # sequence_direct[0][1] = np.deg2rad(90)

        if (np.abs(np.rad2deg(sequence[0][1])) > 10):
            print("rotate crucial")
            desired = [0, 0, 0.0325]
            xy_dist = np.sqrt(np.sum((desired[:2] - state[0][:2]) ** 2))
            if (xy_dist > 0.05):
                self.high_lvl_primitive = 1  # move center sequence
            else:
                # the correction with the sign is needed since this is not adaptive
                dir_rot = copy.deepcopy(sequence[0][0] * np.sign(sequence[0][1]))
                dir_rot[np.abs(dir_rot) < 0.95] = 0.0
                dir_rot[(dir_rot) > 0.95] = 1.0
                dir_rot[(dir_rot) < -0.95] = -1.0
                # this call is only needed to compute the vertical axes,...
                self.grasp.assign_fingers(robot_state, state)
                vertical_idx, axes_dir = self.grasp.get_axis(state)
                if not (vertical_idx == np.argmax(np.abs(dir_rot))):
                    self.rotate_air.set_rotation_directions(dir_rot,[state[0][0], state[0][1], 0.0325])
                    self.high_lvl_primitive = 4  # rotate in air
                else:
                    self.high_lvl_primitive = 3  # rotate on ground
                    dir_rot = np.matmul(np.asarray(pybullet.getMatrixFromQuaternion(state[1])).reshape(3, 3), dir_rot)
                    self.rotate_ground.set_rotation_directions(dir_rot, [state[0][0], state[0][1], 0.0325])

        elif (np.abs(np.rad2deg(sequence_direct[0][1]) > 45)):
            sequence_direct[0][1] = np.clip(sequence_direct[0][1],-np.deg2rad(60),np.deg2rad(60))
            print("rotate minor")
            desired = [0, 0, 0.0325]
            xy_dist = np.sqrt(np.sum((desired[:2] - state[0][:2]) ** 2))
            if (xy_dist > 0.05):
                self.high_lvl_primitive = 1  # move center sequence
            else:
                dir_rot = sequence_direct[0][0]
                dir_rot[np.argmax(np.abs(dir_rot))] = np.sign(dir_rot[np.argmax(np.abs(dir_rot))])
                dir_rot[np.abs(dir_rot) < 0.9] = 0.0
                state[3] = np.asarray(desired)
                self.grasp.assign_fingers(robot_state, state)
                vertical_idx, axes_dir = self.grasp.get_axis(state)
                if not (vertical_idx == np.argmax(np.abs(dir_rot))):
                    self.rotate_air.set_rotation_directions(dir_rot,[state[0][0], state[0][1], 0.0325])
                    self.high_lvl_primitive = 4  # rotate in air
                else:
                    self.high_lvl_primitive = 3  # rotate on ground
                    dir_rot = np.matmul(np.asarray(pybullet.getMatrixFromQuaternion(state[1])).reshape(3, 3), dir_rot)
                    self.rotate_ground.set_rotation_directions(dir_rot, [state[0][0], state[0][1], 0.0325], sequence_direct[0][1])
        else:
            print("move direct")
            self.high_lvl_primitive = 2

        if (self.high_lvl_primitive==1):
            substate = self.bring_center
            return substate(obs, info)
        elif (self.high_lvl_primitive==3):
            substate = self.rotate_ground
            return substate(obs, info)
        elif (self.high_lvl_primitive==4):
            substate = self.rotate_air
            return substate(obs, info)
        else:
            substate = self.next_state
            return substate(obs, info)
