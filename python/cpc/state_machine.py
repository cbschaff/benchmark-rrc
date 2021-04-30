import gin
from mp import states as mp_states

from cpc import states as cpc_states
from cpc import parameters as cpc_params


class StateMachine(object):
    def __init__(self, env, parameters=None):
        self.env = env
        difficulty = env.difficulty
        if parameters is not None:
            self.parameters = parameters
        else:
            if difficulty == 1:
                self.parameters = cpc_params.CubeLvl1Params(env)
            elif (difficulty == 4):
                self.parameters = cpc_params.CubeLvl4Params(env)
            else:
                self.parameters = cpc_params.CubeParams(env)
        self.init_state = self.build()

    def build(self):
        """Instantiate states and connect them.

        make sure to make all of the states instance variables so that
        they are reset when StateMachine.reset is called!

        Returns: base_states.State (initial state)


        Ex:

        self.state1 = State1(env, args)
        self.state2 = State2(env, args)

        self.state1.connect(...)
        self.state2.connect(...)

        return self.state1
        """
        raise NotImplementedError

    def reset(self):
        self.state = self.init_state
        self.info = {}
        print("==========================================")
        print("Resetting State Machine...")
        print(f"Entering State: {self.state.__class__.__name__}")
        print("==========================================")
        for attr in vars(self).values():
            if isinstance(attr, mp_states.State):
                attr.reset()

    def __call__(self, obs):
        prev_state = self.state
        action, self.state, self.info = self.state(obs, self.info)
        if prev_state != self.state:
            print("==========================================")
            print(f"Entering State: {self.state.__class__.__name__}")
            print("==========================================")
            prev_state.reset()
        if action['frameskip'] == 0:
            return self.__call__(obs)
        else:
            return action


@gin.configurable
class CPCStateMachine(StateMachine):
    def build(self):
        self.goto_init_pose = cpc_states.GoToInitState(self.env)
        self.align_to_object = cpc_states.AlignState(self.env)
        self.lower = cpc_states.LowerState(self.env)
        self.grasp = cpc_states.IntoState(self.env)
        self.move_to_goal = cpc_states.MoveToGoalState(
            self.env, self.parameters.k_p_goal, self.parameters.k_p_into, self.parameters.k_i_goal, self.parameters.gain_increase_factor, self.parameters.interval, self.parameters.max_interval_ctr)
        self.failure = mp_states.FailureState(self.env)

        # define transitions between states
        self.goto_init_pose.connect(next_state=self.align_to_object,
                                    failure_state=self.failure)
        self.align_to_object.connect(next_state=self.lower,
                                     failure_state=self.goto_init_pose)
        self.lower.connect(next_state=self.grasp,
                           failure_state=self.goto_init_pose)
        self.grasp.connect(next_state=self.move_to_goal,
                           failure_state=self.goto_init_pose)
        self.move_to_goal.connect(
            next_state=self.move_to_goal, failure_state=self.goto_init_pose)
        return self.goto_init_pose


@gin.configurable
class CPCStateMachineL4(StateMachine):
    def build(self):
        self.goto_init_pose = cpc_states.GoToInitState(self.env)
        self.preorient_object = cpc_states.AlignObjectSequenceState(self.env, self.parameters)
        self.align_to_object = cpc_states.AlignState(self.env)
        self.lower = cpc_states.LowerState(self.env)
        self.grasp = cpc_states.IntoState(self.env)
        self.move_to_goal = cpc_states.GoalWithOrientState(
            self.env, self.parameters.k_p_goal, self.parameters.k_p_into, self.parameters.k_i_goal, self.parameters.k_p_ang, self.parameters.k_i_ang, self.parameters.gain_increase_factor, self.parameters.interval, self.parameters.max_interval_ctr)
        self.failure = mp_states.FailureState(self.env)

        # define transitions between states
        self.goto_init_pose.connect(next_state=self.align_to_object,
                                    failure_state=self.preorient_object)
        self.preorient_object.connect(next_state=self.goto_init_pose,
                                      failure_state=self.goto_init_pose)
        self.align_to_object.connect(next_state=self.lower,
                                     failure_state=self.goto_init_pose)
        self.lower.connect(next_state=self.grasp,
                           failure_state=self.goto_init_pose)
        self.grasp.connect(next_state=self.move_to_goal,
                           failure_state=self.goto_init_pose)
        self.move_to_goal.connect(
            next_state=self.move_to_goal, failure_state=self.goto_init_pose)
        return self.goto_init_pose

class MixedStateMachine1(StateMachine):
    """
    Use Planned Grasp with CPC
    """

    def build(self):
        self.goto_init_pose = cpc_states.GoToInitState(self.env)
        self.init_after_preorient = cpc_states.GoToInitState(self.env)
        self.preorient_object = mp_states.AlignObjectSequenceState(self.env)
        self.planned_grasp = mp_states.PlannedGraspState(self.env)
        self.move_to_goal = cpc_states.MoveToGoalState(self.env)
        self.failure = mp_states.FailureState(self.env)

        # define transitions between states
        self.goto_init_pose.connect(next_state=self.preorient_object,
                                    failure_state=self.failure)
        self.preorient_object.connect(next_state=self.init_after_preorient,
                                      failure_state=self.goto_init_pose)
        self.init_after_preorient.connect(next_state=self.planned_grasp,
                                          failure_state=self.failure)
        self.planned_grasp.connect(next_state=self.move_to_goal,
                                   failure_state=self.goto_init_pose)
        self.move_to_goal.connect(
            next_state=self.move_to_goal, failure_state=self.goto_init_pose)
        return self.goto_init_pose


class MixedStateMachine1_L4(StateMachine):
    """
    Use Planned Grasp with CPC for difficulty = 4.
    """

    def build(self):
        self.goto_init_pose = cpc_states.GoToInitState(self.env)
        self.init_after_preorient = cpc_states.GoToInitState(self.env)
        self.preorient_object = mp_states.AlignObjectSequenceState(self.env)
        self.planned_grasp = mp_states.PlannedGraspState(self.env)
        self.move_to_goal = cpc_states.GoalWithOrientState(self.env)
        self.failure = mp_states.FailureState(self.env)

        # define transitions between states
        self.goto_init_pose.connect(next_state=self.preorient_object,
                                    failure_state=self.failure)
        self.preorient_object.connect(next_state=self.init_after_preorient,
                                      failure_state=self.goto_init_pose)
        self.init_after_preorient.connect(next_state=self.planned_grasp,
                                          failure_state=self.failure)
        self.planned_grasp.connect(next_state=self.move_to_goal,
                                   failure_state=self.goto_init_pose)
        self.move_to_goal.connect(
            next_state=self.move_to_goal, failure_state=self.goto_init_pose)
        return self.goto_init_pose


class MixedStateMachine2(StateMachine):
    """
    Use Triangle Grasp and Motion Planning
    """

    def build(self):
        self.goto_init_pose = cpc_states.GoToInitState(self.env)
        self.init_after_preorient = cpc_states.GoToInitState(self.env)
        self.preorient_object = mp_states.AlignObjectSequenceState(self.env)
        self.align_to_object = cpc_states.AlignState(self.env)
        self.lower = cpc_states.LowerState(self.env)
        self.grasp = cpc_states.IntoState(self.env)
        self.move_to_goal = mp_states.MoveToGoalState(self.env)
        self.failure = mp_states.FailureState(self.env)

        # define transitions between states
        self.goto_init_pose.connect(next_state=self.preorient_object,
                                    failure_state=self.failure)
        self.preorient_object.connect(next_state=self.init_after_preorient,
                                      failure_state=self.failure)
        self.init_after_preorient.connect(next_state=self.align_to_object,
                                          failure_state=self.failure)
        self.align_to_object.connect(next_state=self.lower,
                                     failure_state=self.goto_init_pose)
        self.lower.connect(next_state=self.grasp,
                           failure_state=self.goto_init_pose)
        self.grasp.connect(next_state=self.move_to_goal,
                           failure_state=self.goto_init_pose)
        self.move_to_goal.connect(
            next_state=self.move_to_goal, failure_state=self.goto_init_pose)
        return self.goto_init_pose
