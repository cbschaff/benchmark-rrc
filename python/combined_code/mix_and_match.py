import gin
from mp import states

import cpc.states as cpc_states
from cic import states as states_cic
from cic import grasping_class as cic_grasping_class
from mp.set_hyperparams import set_hyperparams
from cic import object_class as cic_object_class
from cic import control_main_class as cic_control_main_class
from cic import parameters_new_grasp as cic_parameters_new_grasp

from cpc import parameters as cpc_params


class TG(states.State):
    def __init__(self, env):
        super().__init__(env)
        self.goto_init_pose = cpc_states.SimpleGoToInitState(self.env)
        self.triangulate_fingers = cpc_states.AlignState(self.env)
        self.lower = cpc_states.LowerState(self.env)
        self.grasp = cpc_states.IntoState(self.env)

    def connect(self, next_state, failure_state):
        self.goto_init_pose.connect(
            next_state=self.triangulate_fingers, failure_state=self.goto_init_pose
        )
        self.triangulate_fingers.connect(next_state=self.lower, failure_state=self.goto_init_pose)
        self.lower.connect(next_state=self.grasp, failure_state=self.goto_init_pose)
        self.grasp.connect(next_state=next_state, failure_state=self.goto_init_pose)

    def reset(self):
        self.goto_init_pose.reset()
        self.lower.reset()
        self.grasp.reset()
        self.triangulate_fingers.reset()

    def __call__(self, obs, info={}):
        return self.goto_init_pose(obs, info)


class PG(states.State):
    def __init__(self, env):
        super().__init__(env)
        self.goto_init_pose = states.GoToInitPoseState(self.env)
        self.planned_grasp = states.PlannedGraspState(self.env)

    def connect(self, next_state, failure_state):
        self.goto_init_pose.connect(
            next_state=self.planned_grasp, failure_state=failure_state
        )
        self.planned_grasp.connect(
            next_state=next_state, failure_state=failure_state
        )

    def reset(self):
        self.goto_init_pose.reset()
        self.planned_grasp.reset()

    def __call__(self, obs, info={}):
        return self.goto_init_pose(obs, info)


class CG(states.State):
    def __init__(self, env, main_ctrl, parameters, object, parameters_cic, skip_grasp_init=False):
        super().__init__(env)
        self.main_ctrl = main_ctrl
        self.parameters = parameters
        self.object = object
        self.parameters_cic = parameters_cic
        self.skip_grasp_init = skip_grasp_init

        # Params required for some States
        _grasp = cic_grasping_class.ThreeFingerGrasp(
            self.main_ctrl.kinematics, self.parameters_cic, self.object
        )

        # Instantiate States
        self.goinitpose = states_cic.InitResetPrimitive(
            self.env, self.main_ctrl.kinematics, self.parameters_cic
        )
        self.approach = states_cic.ApproachObjectViapoints(
            self.env,
            self.main_ctrl.kinematics,
            self.parameters_cic,
            self.object,
            _grasp,
            self.parameters_cic.approach_grasp_xy,
            self.parameters_cic.approach_grasp_h,
            self.parameters_cic.approach_duration,
            self.parameters_cic.approach_clip_pos,
            stop_early=True,
            assign_fingers=True,
        )

        # NOTE: when called, this state put the grasp information in info['grasp']
        self.set_grasp = states_cic.SetGraspPrimitive(
            self.env, kinematics=self.main_ctrl.kinematics, params=self.parameters_cic
        )

    def connect(self, next_state, failure_state):
        self.goinitpose.connect(next_state=self.approach, done_state=failure_state)
        if (self.skip_grasp_init):
            self.approach.connect(next_state=next_state, done_state=failure_state)
        else:
            self.approach.connect(next_state=self.set_grasp, done_state=failure_state)
            self.set_grasp.connect(next_state=next_state, done_state=failure_state)

    def reset(self):
        self.goinitpose.reset()
        self.approach.reset()
        self.set_grasp.reset()

    def __call__(self, obs, info={}):
        return self.goinitpose(obs, info)


class CICApproachGoal(states.State):
    def __init__(self, env, main_ctrl, parameters, object, parameters_cic):
        super().__init__(env)
        self.main_ctrl = main_ctrl
        self.parameters = parameters
        self.object = object
        self.parameters_cic = parameters_cic

        _grasp = cic_grasping_class.ThreeFingerGraspExternal(
            self.main_ctrl.kinematics, self.parameters_cic, self.object
        )

        # Instantiate States
        self.set_cic_grasp = states_cic.SetGraspFromOutsidePrimitive(
            self.env, self.main_ctrl.kinematics, self.parameters_cic, self.object, _grasp
        )
        self.move_cube_lift = states_cic.MoveLiftCubeOrientPrimitiveLvl4(
            self.env,
            self.main_ctrl.kinematics,
            self.parameters_cic,
            self.object,
            _grasp,
            False,
        )

    def connect(self, next_state, failure_state):
        # define transitions between states
        self.set_cic_grasp.connect(
            next_state=self.move_cube_lift, done_state=failure_state
        )
        self.move_cube_lift.connect(
            next_state=next_state, done_state=failure_state
        )

    def reset(self):
        self.set_cic_grasp.reset()
        self.move_cube_lift.reset()

    def __call__(self, obs, info={}):
        return self.set_cic_grasp(obs, info)


#########################################
# Mixed State Machines
#########################################


class StateMachineCombined(states.StateMachine):
    def __init__(self, env):
        # needed for cic states:
        self.main_ctrl = cic_control_main_class.TriFingerController(env)
        self.object = cic_object_class.Cube()
        self.new_grasp = True
        self.env = env
        difficulty = env.difficulty
        if (difficulty == 1):
            self.parameters_cic = cic_parameters_new_grasp.CubeLvl1Params(env)
        elif (difficulty == 2 or difficulty == 3):
            self.parameters_cic = cic_parameters_new_grasp.CubeLvl2Params(env)
        elif (difficulty == 4):
            self.parameters_cic = cic_parameters_new_grasp.CubeLvl4Params(env)

        # needed for cpc:
        if difficulty == 1:
            self.parameters = cpc_params.CubeLvl1Params(env)
        elif (difficulty == 4):
            self.parameters = cpc_params.CubeLvl4Params(env)
        else:
            self.parameters = cpc_params.CubeParams(env)

        # Set hyperparams for MP-PG
        set_hyperparams(env.simulation)

        self.init_state = self.build()

    def build(self):
        # THOSE 5 States are the skeleton
        self.init = states_cic.IdlePrimitive(self.env)
        self.align_obj = states.AlignObjectSequenceState(self.env)
        self.goto_init_pos = states.GoToInitPoseState(self.env)
        self.failure = states.FailureState(self.env)
        self.do_nothing = states_cic.IdlePrimitiveLong(self.env)

        self.build_grasp_and_approach()

        # START CONNECTING THE STATE MACHINES
        self.init.connect(next_state=self.align_obj, done_state=self.failure)
        self.align_obj.connect(self.goto_init_pos, self.failure)
        self.goto_init_pos.connect(self.grasp, self.failure)
        self.grasp.connect(self.approach, self.failure)
        self.approach.connect(self.do_nothing, self.do_nothing)
        self.do_nothing.connect(next_state=self.do_nothing, done_state=self.failure)
        return self.init

    def build_grasp_and_approach(self):
        """Returns the start state.

        The following states must be defined in this method:
        - self.grasp
        - self.approach
        """
        raise NotImplementedError


@gin.configurable
class CICwithCG(StateMachineCombined):
    def build_grasp_and_approach(self):
        self.grasp = CG(
            self.env, self.main_ctrl, self.parameters, self.object, self.parameters_cic, skip_grasp_init=True
        )
        self.approach = CICApproachGoal(
            self.env, self.main_ctrl, self.parameters, self.object, self.parameters_cic
        )


@gin.configurable
class CICwithPG(StateMachineCombined):
    def build_grasp_and_approach(self):
        self.grasp = PG(self.env)
        self.approach = CICApproachGoal(
            self.env, self.main_ctrl, self.parameters, self.object, self.parameters_cic
        )
        return self.grasp


@gin.configurable
class CICwithTG(StateMachineCombined):
    def build_grasp_and_approach(self):
        self.grasp = TG(self.env)
        self.approach = CICApproachGoal(
            self.env, self.main_ctrl, self.parameters, self.object, self.parameters_cic
        )
        return self.grasp


@gin.configurable
class MPwithCG(StateMachineCombined):
    def build_grasp_and_approach(self):
        self.grasp = CG(
            self.env, self.main_ctrl, self.parameters, self.object, self.parameters_cic
        )
        self.approach = states.MoveToGoalState(self.env)
        return self.grasp


@gin.configurable
class MPwithPG(StateMachineCombined):
    def build_grasp_and_approach(self):
        self.grasp = PG(self.env)
        self.approach = states.MoveToGoalState(self.env)
        return self.grasp


@gin.configurable
class MPwithTG(StateMachineCombined):
    def build_grasp_and_approach(self):
        self.grasp = TG(self.env)
        self.approach = states.MoveToGoalState(self.env)
        return self.grasp


@gin.configurable
class CPCwithCG(StateMachineCombined):
    def build_grasp_and_approach(self):
        self.grasp = CG(
            self.env, self.main_ctrl, self.parameters, self.object, self.parameters_cic
        )
        self.approach = cpc_states.GoalWithOrientState(
            self.env, self.parameters.k_p_goal, self.parameters.k_p_into,
            self.parameters.k_i_goal, self.parameters.k_p_ang, self.parameters.k_i_ang,
            self.parameters.gain_increase_factor, self.parameters.interval,
            self.parameters.max_interval_ctr)
        return self.grasp


@gin.configurable
class CPCwithPG(StateMachineCombined):
    def build_grasp_and_approach(self):
        self.grasp = PG(self.env)
        self.approach = cpc_states.GoalWithOrientState(
            self.env, self.parameters.k_p_goal, self.parameters.k_p_into,
            self.parameters.k_i_goal, self.parameters.k_p_ang, self.parameters.k_i_ang,
            self.parameters.gain_increase_factor, self.parameters.interval,
            self.parameters.max_interval_ctr)
        return self.grasp


@gin.configurable
class CPCwithTG(StateMachineCombined):
    def build_grasp_and_approach(self):
        self.grasp = TG(self.env)
        self.approach = cpc_states.GoalWithOrientState(
            self.env, self.parameters.k_p_goal, self.parameters.k_p_into,
            self.parameters.k_i_goal, self.parameters.k_p_ang, self.parameters.k_i_ang,
            self.parameters.gain_increase_factor, self.parameters.interval,
            self.parameters.max_interval_ctr)
        return self.grasp
