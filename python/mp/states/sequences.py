from mp.align_rotation import roll_and_pitch_aligned, get_yaw_diff
from .base_states import *
from .state_machine import State
import numpy as np


class CenteringPrimitiveState(State):
    def __init__(self, env):
        super().__init__(env)
        self.heuristic_grasp = HeuristicGraspState(env)
        self.random_grasp = RandomGraspState(env)
        self.partial_grasp = PartialGraspState(env)
        self.move_to_center = MoveToCenterState(env)
        self.wait = WaitState(env, 10 if env.simulation else 300)

    def reset(self):
        self.heuristic_grasp.reset()
        self.random_grasp.reset()
        self.partial_grasp.reset()
        self.move_to_center.reset()
        self.wait.reset()

    def connect(self, next_state, failure_state):
        self.heuristic_grasp.connect(
            next_state=self.move_to_center,
            failure_state=self.random_grasp,
        )
        self.random_grasp.connect(
            next_state=self.move_to_center,
            failure_state=self.partial_grasp,
        )
        self.partial_grasp.connect(
            next_state=self.move_to_center,
            failure_state=failure_state,
        )
        self.move_to_center.connect(
            next_state=self.wait,
            failure_state=failure_state
        )
        self.wait.connect(
            next_state=next_state,
            failure_state=failure_state
        )

    def __call__(self, obs, info=None):
        info = dict() if info is None else info
        return self.heuristic_grasp(obs, info)


class PitchingPrimitiveState(State):
    def __init__(self, env):
        super().__init__(env)
        self.pitching_grasp = PitchingGraspState(env)
        self.align_cube = AlignRollAndPitchState(env)
        self.wait = WaitState(env, 10 if env.simulation else 300)

    def reset(self):
        self.pitching_grasp.reset()
        self.align_cube.reset()
        self.wait.reset()

    def connect(self, next_state, failure_state):
        self.pitching_grasp.connect(
            next_state=self.align_cube,
            failure_state=failure_state,
        )
        self.align_cube.connect(
            next_state=self.wait,
            failure_state=failure_state
        )
        self.wait.connect(
            next_state=next_state,
            failure_state=failure_state
        )

    def __call__(self, obs, info=None):
        info = dict() if info is None else info
        return self.pitching_grasp(obs, info)


class YawingPrimitiveState(State):
    def __init__(self, env):
        super().__init__(env)
        self.yawing_grasp = YawingGraspState(env)
        self.align_cube = AlignYawState(env)
        self.wait = WaitState(env, 10 if env.simulation else 300)

    def reset(self):
        self.yawing_grasp.reset()
        self.align_cube.reset()
        self.wait.reset()

    def connect(self, next_state, failure_state):
        self.yawing_grasp.connect(
            next_state=self.align_cube,
            failure_state=failure_state,
        )
        self.align_cube.connect(
            next_state=self.wait,
            failure_state=failure_state
        )
        self.wait.connect(
            next_state=next_state,
            failure_state=failure_state
        )

    def __call__(self, obs, info=None):
        info = dict() if info is None else info
        return self.yawing_grasp(obs, info)


class AlignObjectSequenceState(State):
    def __init__(self, env):
        super().__init__(env)
        self.center_object = CenteringPrimitiveState(env)
        self.align_pitch = PitchingPrimitiveState(env)
        self.align_yaw = YawingPrimitiveState(env)

    def reset(self):
        self.center_object.reset()
        self.align_pitch.reset()
        self.align_yaw.reset()

    def object_yaw_oriented(self, obs):
        return np.abs(get_yaw_diff(
            obs['object_orientation'],
            obs['goal_object_orientation'])
        ) < np.pi / 4

    def object_roll_and_pitch_oriented(self, obs):
        return roll_and_pitch_aligned(
            obs['object_orientation'],
            obs['goal_object_orientation']
        )

    def object_centered(self, obs):
        dist_from_center = np.linalg.norm(
            obs['object_position'][:2]
        )
        return dist_from_center < 0.07

    def connect(self, next_state, failure_state):
        self.center_object.connect(
            next_state=self,
            failure_state=failure_state,
        )
        self.align_pitch.connect(
            next_state=self,
            failure_state=self.center_object,  # force a recentering to jiggle the cube
        )
        self.align_yaw.connect(
            next_state=self,
            failure_state=self.center_object,  # force a recentering to jiggle the cube
        )
        self.next_state = next_state

    def __call__(self, obs, info=None):
        info = dict() if info is None else info

        # Choose which substate to transition to
        if self.env.info['difficulty'] == 4:
            if not self.object_centered(obs):
                sub_state = self.center_object
            elif not self.object_roll_and_pitch_oriented(obs):
                sub_state = self.align_pitch
            elif not self.object_yaw_oriented(obs):
                sub_state = self.align_yaw
            else:
                sub_state = self.next_state
        else:
            if not self.object_centered(obs):
                sub_state = self.center_object
            else:
                sub_state = self.next_state
        return sub_state(obs, info)

class AlignObjectSequenceState20deg(AlignObjectSequenceState):
    '''
    Copies the IdlePrimitive but has less duration and simply changes the goal
    '''

    def __init__(self, env):
        super(AlignObjectSequenceState20deg, self).__init__(env)

    def object_yaw_oriented(self, obs):
        return np.abs(get_yaw_diff(
            obs['object_orientation'],
            obs['goal_object_orientation'])
        ) < np.deg2rad(20)
