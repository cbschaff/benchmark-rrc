from .base_states import *
from mp.states import WaitState, AlignYawReleaseState


class AlignObjectSequenceState(SimpleState):
    def __init__(self, env, parameters):
        super().__init__(env)
        self.premanip_info = GetPreManipInfoState(env)
        self.prealign = PreAlignState(env)
        self.prelower = PreLowerState(env)
        self.preinto = PreIntoState(env)
        self.pregoal = PreGoalState(env, parameters.pitch_k_p_goal, parameters.pitch_k_p_into, parameters.pitch_k_i_goal, parameters.pitch_k_p_ang,
                                    parameters.pitch_k_i_ang, parameters.gain_increase_factor, parameters.interval, parameters.max_interval_ctr, parameters.pitch_rot_factor, parameters.pitch_lift_factor)
        self.yaw_orient = PreGoalState2(env, parameters.yaw_k_p_goal, parameters.yaw_k_p_into, parameters.yaw_k_i_goal, parameters.yaw_k_p_ang,
                                        parameters.yaw_k_i_ang, parameters.gain_increase_factor, parameters.interval, parameters.max_interval_ctr)
        self.preorient = PreOrientState(env)
        self.release = AlignYawReleaseState(self.env)
        self.wait = WaitState(env, 10 if env.simulation else 300)

    def reset(self):
        self.premanip_info.reset()
        self.prealign.reset()
        self.prelower.reset()
        self.preinto.reset()
        self.pregoal.reset()
        self.preorient.reset()
        self.yaw_orient.reset()
        self.release.reset()
        self.wait.reset()

    def connect(self, next_state, failure_state):
        self.premanip_info.connect(
            next_state=self.prealign,
            failure_state=next_state,
        )
        self.prealign.connect(
            next_state=self.prelower,
            failure_state=failure_state,
        )
        self.prelower.connect(
            next_state=self.preinto,
            failure_state=failure_state,
        )
        self.preinto.connect(
            next_state=self.pregoal,
            failure_state=failure_state
        )
        self.pregoal.connect(
            next_state=self.preorient,
            failure_state=failure_state
        )
        self.preorient.connect(
            next_state=self.yaw_orient,
            failure_state=self.yaw_orient
        )
        self.yaw_orient.connect(
            next_state=self.release,
            failure_state=self.release)
        self.release.connect(
            next_state=self.premanip_info,
            failure_state=failure_state
        )

    def __call__(self, obs, info=None):
        info = dict() if info is None else info
        return self.premanip_info(obs, info)


class AlignObjectSequenceStateEval(SimpleState):
    def __init__(self, env, parameters):
        super().__init__(env)
        self.init_pose = SimpleGoToInitState(env)
        self.premanip_info = GetPreManipInfoState(env)
        self.prealign = PreAlignState(env)
        self.prelower = PreLowerState(env)
        self.preinto = PreIntoState(env)
        self.pregoal = PreGoalState(env, parameters.pitch_k_p_goal, parameters.pitch_k_p_into, parameters.pitch_k_i_goal, parameters.pitch_k_p_ang,
                                    parameters.pitch_k_i_ang, parameters.gain_increase_factor, parameters.interval, parameters.max_interval_ctr, parameters.pitch_rot_factor, parameters.pitch_lift_factor)
        self.yaw_orient = PreGoalState2(env, parameters.yaw_k_p_goal, parameters.yaw_k_p_into, parameters.yaw_k_i_goal, parameters.yaw_k_p_ang,
                                        parameters.yaw_k_i_ang, parameters.gain_increase_factor, parameters.interval, parameters.max_interval_ctr)
        self.preorient = PreOrientState(env)
        self.release = AlignYawReleaseState(self.env)
        self.wait = WaitState(env, 10 if env.simulation else 300)

    def reset(self):
        self.init_pose.reset()
        self.premanip_info.reset()
        self.prealign.reset()
        self.prelower.reset()
        self.preinto.reset()
        self.pregoal.reset()
        self.preorient.reset()
        self.yaw_orient.reset()
        self.release.reset()
        self.wait.reset()

    def connect(self, next_state, failure_state):
        self.init_pose.connect(
            next_state=self.premanip_info,
            failure_state=next_state
        )
        self.premanip_info.connect(
            next_state=self.prealign,
            failure_state=next_state,
        )
        self.prealign.connect(
            next_state=self.prelower,
            failure_state=failure_state,
        )
        self.prelower.connect(
            next_state=self.preinto,
            failure_state=failure_state,
        )
        self.preinto.connect(
            next_state=self.pregoal,
            failure_state=failure_state
        )
        self.pregoal.connect(
            next_state=self.preorient,
            failure_state=failure_state
        )
        self.preorient.connect(
            next_state=self.yaw_orient,
            failure_state=self.yaw_orient
        )
        self.yaw_orient.connect(
            next_state=self.release,
            failure_state=self.release)
        self.release.connect(
            next_state=self.init_pose,
            failure_state=self.init_pose
        )

    def __call__(self, obs, info=None):
        info = dict() if info is None else info
        return self.init_pose(obs, info)
