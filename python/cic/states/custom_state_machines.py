import gin
from mp import states

from cic import states as states_cic
from cic import grasping_class as cic_grasping_class

'''
This file contains the StateMachines of the Code from TU Darmstadt.
'''

@gin.configurable
class CICStateMachineLvl1(states_cic.StateMachineCIC):
    def build(self):
        self.init = states_cic.IdlePrimitive(self.env, kinematics=self.main_ctrl.kinematics, params=self.parameters)
        self.done = states.FailureState(self.env)
        self.goinitpose = states_cic.InitResetPrimitive(self.env, kinematics=self.main_ctrl.kinematics,
                                                        params=self.parameters)
        self.grasp = cic_grasping_class.ThreeFingerGrasp(self.main_ctrl.kinematics, self.parameters, self.object)
        self.approach = states_cic.ApproachObjectViapoints(self.env, self.main_ctrl.kinematics, self.parameters,
                                                           self.object, self.grasp,
                                                           self.parameters.approach_grasp_xy,
                                                           self.parameters.approach_grasp_h,
                                                           self.parameters.approach_duration,
                                                           self.parameters.approach_clip_pos,
                                                           stop_early=True, assign_fingers=True)
        self.move_cube_floor = states_cic.MoveCubeFloorPrimitiveLvl1(self.env, self.main_ctrl.kinematics,
                                                                     self.parameters, self.object,
                                                                     self.grasp, False)
        self.reset_state = states_cic.ApproachObjectViapoints(self.env, self.main_ctrl.kinematics, self.parameters,
                                                              self.object, self.grasp,
                                                              self.parameters.reset_grasp_xy,
                                                              self.parameters.reset_grasp_h,
                                                              self.parameters.reset_duration,
                                                              self.parameters.reset_clip_pos,
                                                              stop_early=True, assign_fingers=False)
        #
        self.init.connect(next_state=self.goinitpose, done_state=self.done)
        # self.init.connect(next_state=self.init, done_state=self.done)
        self.goinitpose.connect(next_state=self.approach, done_state=self.done)
        self.approach.connect(next_state=self.move_cube_floor, done_state=self.done)
        self.move_cube_floor.connect(next_state=self.reset_state, done_state=self.done)
        self.reset_state.connect(next_state=self.init, done_state=self.done)
        self.goto_init_pose = states.GoToInitPoseState(self.env)

        # return initial state
        return self.init

@gin.configurable
class CICStateMachineLvl2(states_cic.StateMachineCIC):
    def build(self):
        self.init = states_cic.IdlePrimitive(self.env, kinematics=self.main_ctrl.kinematics, params=self.parameters)
        self.done = states.FailureState(self.env)
        self.goinitpose = states_cic.InitResetPrimitive(self.env, kinematics=self.main_ctrl.kinematics,
                                                        params=self.parameters)
        self.grasp = cic_grasping_class.ThreeFingerGrasp(self.main_ctrl.kinematics, self.parameters, self.object)
        self.approach = states_cic.ApproachObjectViapoints(self.env, self.main_ctrl.kinematics, self.parameters,
                                                           self.object, self.grasp,
                                                           self.parameters.approach_grasp_xy,
                                                           self.parameters.approach_grasp_h,
                                                           self.parameters.approach_duration,
                                                           self.parameters.approach_clip_pos,
                                                           stop_early=True, assign_fingers=True)

        self.grasp1 = cic_grasping_class.ThreeFingerGraspExternal(self.main_ctrl.kinematics, self.parameters, self.object)

        self.set_cic_grasp1 = states_cic.SetGraspFromOutsidePrimitive(self.env, self.main_ctrl.kinematics, self.parameters, self.object, self.grasp1)

        if (self.new_grasp):
            grasp_to_be_used = self.grasp1
        else:
            grasp_to_be_used = self.grasp

        self.move_cube_lift = states_cic.MoveLiftCubePrimitiveLvl2(self.env, self.main_ctrl.kinematics, self.parameters,
                                                                   self.object, grasp_to_be_used,
                                                                   False)
        self.reset_state = states_cic.ApproachObjectViapoints(self.env, self.main_ctrl.kinematics, self.parameters,
                                                              self.object, grasp_to_be_used,
                                                              self.parameters.reset_grasp_xy,
                                                              self.parameters.reset_grasp_h,
                                                              self.parameters.reset_duration,
                                                              self.parameters.reset_clip_pos,
                                                              stop_early=True, assign_fingers=False)

        self.init.connect(next_state=self.goinitpose, done_state=self.done)
        self.goinitpose.connect(next_state=self.approach, done_state=self.done)
        if (self.new_grasp):
            self.approach.connect(next_state=self.set_cic_grasp1, done_state=self.done)
        else:
            self.approach.connect(next_state=self.move_cube_lift, done_state=self.done)

        self.set_cic_grasp1.connect(next_state=self.move_cube_lift, done_state=self.done)

        self.move_cube_lift.connect(next_state=self.reset_state, done_state=self.done)
        # TODO: check where we want to reset to (-> is init the best choice?)
        self.reset_state.connect(next_state=self.init, done_state=self.done)

        return self.init

@gin.configurable
class Lvl3Joint(states_cic.StateMachineCIC):
    def build(self):
        self.init = states_cic.IdlePrimitive(self.env, kinematics=self.main_ctrl.kinematics, params=self.parameters)
        self.done = states.FailureState(self.env)
        self.goinitpose = states_cic.InitResetPrimitive(self.env, kinematics=self.main_ctrl.kinematics,
                                                        params=self.parameters)
        self.grasp = cic_grasping_class.ThreeFingerGrasp(self.main_ctrl.kinematics, self.parameters, self.object)
        self.approach = states_cic.ApproachObjectViapoints(self.env, self.main_ctrl.kinematics, self.parameters,
                                                           self.object, self.grasp,
                                                           self.parameters.approach_grasp_xy,
                                                           self.parameters.approach_grasp_h,
                                                           self.parameters.approach_duration,
                                                           self.parameters.approach_clip_pos,
                                                           stop_early=True, assign_fingers=True)

        self.set_grasp = states_cic.SetGraspPrimitive(self.env, kinematics=self.main_ctrl.kinematics, params=self.parameters)

        self.move_to_goal = states.MoveToGoalState(self.env)

        self.move_cube_lift = states_cic.MoveLiftCubePrimitiveLvl2(self.env, self.main_ctrl.kinematics, self.parameters,
                                                                   self.object, self.grasp,
                                                                   False)
        self.reset_state = states_cic.ApproachObjectViapoints(self.env, self.main_ctrl.kinematics, self.parameters,
                                                              self.object, self.grasp,
                                                              self.parameters.reset_grasp_xy,
                                                              self.parameters.reset_grasp_h,
                                                              self.parameters.reset_duration,
                                                              self.parameters.reset_clip_pos,
                                                              stop_early=True, assign_fingers=False)
        self.init.connect(next_state=self.goinitpose, done_state=self.done)
        self.goinitpose.connect(next_state=self.approach, done_state=self.done)
        self.approach.connect(next_state=self.set_grasp, done_state=self.done)
        self.set_grasp.connect(next_state=self.move_to_goal, done_state=self.done)
        self.move_to_goal.connect(next_state=self.approach, failure_state=self.approach)

        # TODO: check where we want to reset to (-> is init the best choice?)
        self.reset_state.connect(next_state=self.init, done_state=self.done)

        return self.init



@gin.configurable
class CICStateMachineLvl4(states_cic.StateMachineCIC):
    def build(self):
        self.init = states_cic.IdlePrimitive(self.env, kinematics=self.main_ctrl.kinematics, params=self.parameters)
        self.done = states.FailureState(self.env)
        self.grasp = cic_grasping_class.ThreeFingerGrasp(self.main_ctrl.kinematics, self.parameters, self.object)
        self.grasp1 = cic_grasping_class.ThreeFingerGraspExternal(self.main_ctrl.kinematics, self.parameters,
                                                                  self.object)
        if not(self.new_grasp):
            # do not use new grasp:
            self.grasp1 = None

        self.align_axes = states_cic.AlignAxesPrimitive(self.env, self.main_ctrl.kinematics, self.parameters,
                                                        self.object, self.grasp, grasp1=self.grasp1)

        self.bring_final = states_cic.BringFinalPoseOrientationPrimitive(self.env, self.main_ctrl.kinematics, self.parameters,
                                                              self.object, self.grasp, grasp1=self.grasp1)

        self.init.connect(next_state=self.align_axes, done_state=self.done)

        self.align_axes.connect(next_state=self.bring_final, done_state=self.done)

        self.bring_final.connect(next_state=self.align_axes, done_state=self.done)

        return self.init

@gin.configurable
class CICStateMachineBO(states_cic.StateMachineCIC):
    def build(self):
        self.init = states_cic.IdlePrimitive(self.env, kinematics=self.main_ctrl.kinematics, params=self.parameters)
        self.done = states.FailureState(self.env)
        self.grasp = cic_grasping_class.ThreeFingerGrasp(self.main_ctrl.kinematics, self.parameters, self.object)
        self.grasp1 = cic_grasping_class.ThreeFingerGraspExternal(self.main_ctrl.kinematics, self.parameters,
                                                                  self.object)

        self.change_goal = states_cic.IdlePrimitiveChangeGoal(self.env, self.main_ctrl.kinematics, self.parameters)

        if not(self.new_grasp):
            # do not use new grasp:
            self.grasp1 = None

        self.align_axes = states_cic.AlignAxesPrimitive(self.env, self.main_ctrl.kinematics, self.parameters,
                                                        self.object, self.grasp, grasp1=self.grasp1)

        self.bring_final = states_cic.BringFinalPoseOrientationPrimitive(self.env, self.main_ctrl.kinematics, self.parameters,
                                                              self.object, self.grasp, grasp1=self.grasp1)

        self.do_nothing = states_cic.IdlePrimitiveLong(self.env, kinematics=self.main_ctrl.kinematics, params=self.parameters)


        # self.init.connect(next_state=self.bring_center, done_state=self.done)
        self.init.connect(next_state=self.change_goal, done_state=self.done)
        self.change_goal.connect(next_state=self.align_axes, done_state=self.done)
        # self.bring_final.connect(next_state=self.bring_final, done_state=self.done)

        self.align_axes.connect(next_state=self.do_nothing, done_state=self.done)
        self.do_nothing.connect(next_state=self.do_nothing, done_state=self.done)
        # self.bring_final.connect(next_state=self.bring_center, done_state=self.done)
        # self.bring_final.connect(next_state=self.align_axes, done_state=self.done)

        return self.init

class CICwithPG(states_cic.StateMachineCIC):
    def build(self):
        self.goto_init_pose = states.GoToInitPoseState(self.env)
        self.align_object = states.AlignObjectSequenceState(self.env)
        self.planned_grasp = states.PlannedGraspState(self.env)

        self.grasp = cic_grasping_class.ThreeFingerGraspExternal(self.main_ctrl.kinematics, self.parameters, self.object)

        self.set_cic_grasp = states_cic.SetGraspFromOutsidePrimitive(self.env, self.main_ctrl.kinematics, self.parameters, self.object, self.grasp)

        self.move_cube_lift = states_cic.MoveLiftCubePrimitiveLvl4(self.env, self.main_ctrl.kinematics, self.parameters,
                                                                   self.object, self.grasp,
                                                                   False)
        self.wait = states.WaitState(
            self.env, 30 if self.env.simulation else 1000)
        self.failure = states.FailureState(self.env)

        # define transitions between states
        self.goto_init_pose.connect(next_state=self.align_object,
                                    failure_state=self.failure)
        self.align_object.connect(next_state=self.planned_grasp,
                                  failure_state=self.failure)
        self.planned_grasp.connect(next_state=self.set_cic_grasp,
                                   failure_state=self.align_object)

        self.set_cic_grasp.connect(next_state=self.move_cube_lift, done_state=self.failure)
        self.move_cube_lift.connect(next_state=self.wait, done_state=self.failure)
        self.wait.connect(next_state=self.goto_init_pose, failure_state=self.failure)

        # return initial state
        return self.goto_init_pose
