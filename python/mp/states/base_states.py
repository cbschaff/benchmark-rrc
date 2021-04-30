from collections import deque
import numpy as np
import pybullet as p

from .state_machine import State
from mp import base_policies, grasping
from mp.action_sequences import ScriptedActions
from mp.const import CONTRACTED_JOINT_CONF, INIT_JOINT_CONF
from mp.utils import Transform


class FailureState(State):
    def __init__(self, env):
        self.env = env

    def reset(self):
        pass

    def __call__(self, obs, info={}):
        # hold current position forever
        return self.get_action(position=obs['robot_position']), self, info


class OpenLoopState(State):
    """Base class for open-loop control states."""
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.reset()

    def reset(self):
        self.actions = None

    def connect(self, next_state, failure_state):
        self.next_state = next_state
        self.failure_state = failure_state

    def get_action_generator(self, obs, info):
        """Yields (action, info) tuples."""
        raise NotImplementedError

    def __call__(self, obs, info=None):
        info = dict() if info is None else info
        try:
            if self.actions is None:
                self.actions = self.get_action_generator(obs, info)
            action, info = self.actions.__next__()
            return action, self, info
        except Exception as e:
            self.reset()
            if isinstance(e, StopIteration):
                # query the next state
                return self.next_state(obs, info)
            else:
                print(f"Caught error: {e}")
                return self.get_action(frameskip=0), self.failure_state, info


class WaitState(OpenLoopState):
    """Pause the robot at the current position for certain steps."""

    def __init__(self, env, steps):
        super().__init__(env)
        self.steps = steps

    def get_action_generator(self, obs, info=None):
        yield self.get_action(position=obs['robot_position'],
                              frameskip=self.steps), info


class GoToInitPoseState(OpenLoopState):
    """Move the fingers to INIT_JOINT_CONF.

    The policy contracts the fingers first (`CONTRACTED_JOINT_CONF`), and then
    moves them to the initial position (`INIT_JOINT_CONF`).
    This sequence of motions prevents the fingers from touching the object during the motion.
    """

    def __init__(self, env, steps=300):
        super().__init__(env)
        self.steps = steps

    def get_action_generator(self, obs, info=None):
        for pos in [CONTRACTED_JOINT_CONF, INIT_JOINT_CONF]:
            yield self.get_action(position=pos, frameskip=self.steps // 2), info


class RandomGraspState(OpenLoopState):
    """Sample random grasp and move fingers to the grasp positions.

    Sample three random points on the object that meets conditions,
    and move the fingers to the points.

    Added to info:
        + grasp
    """

    def get_action_generator(self, obs, info=None):
        grasp = grasping.sample_grasp(self.env, obs['object_position'],
                                      obs['object_orientation'])
        info['grasp'] = grasp
        actions = grasping.get_grasp_approach_actions(
            self.env, obs, grasp
        )
        for pos in actions:
            yield self.get_action(position=pos, frameskip=1), info


class HeuristicGraspState(OpenLoopState):
    """Use pre-defined heuristic grasp and move fingers to the grasp positions.

    Check pre-defined heuristic grasps one by one if it will work,
    And use the one that works to generate approach actions.

    Added to info:
        + grasp
    """

    def get_action_generator(self, obs, info):
        grasps = grasping.get_all_heuristic_grasps(
            self.env, obs['object_position'], obs['object_orientation'],
            avoid_edge_faces=False
        )
        actions = None
        for grasp in grasps:
            try:
                actions = grasping.get_grasp_approach_actions(
                    self.env, obs, grasp
                )
                break
            except RuntimeError:
                pass

        if actions is None:
            raise RuntimeError("All heuristic grasps failed...")
        else:
            info['grasp'] = grasp
            for pos in actions:
                yield self.get_action(position=pos, frameskip=1), info


class PlannedGraspState(OpenLoopState):
    """Use planned grasp and move fingers to the grasp positions.

    Sample a grasp and check if the grasp is feasible with wholebody planning.
    Use the obtained 'planned grasp' to generate approach actions.

    Added to info:
        + grasp
        + path
    """

    def get_action_generator(self, obs, info):
        grasp, path = grasping.get_planned_grasp(
            self.env,
            obs['object_position'],
            obs['object_orientation'],
            obs['goal_object_position'],
            obs['goal_object_orientation'],
            tight=True,
            use_rrt=True,
            use_ori=self.env.info['difficulty'] == 4
        )
        info['grasp'] = grasp
        info['path'] = path
        actions = grasping.get_grasp_approach_actions(self.env, obs, grasp)
        for pos in actions:
            yield self.get_action(position=pos, frameskip=1), info


class PartialGraspState(OpenLoopState):
    """Performs 'partial grasp' of which only one or two fingers are used
    to reach the object.

    This samples a 'partial' grasp that uses only one or two fingers to reach the object.
    During the motion, the rest of the fingers are fixed at the `INIT_JOINT_CONF` position.

    Added to info:
        + grasp
    """

    def get_action_generator(self, obs, info):
        done = False
        obj_pos = obs['object_position']
        obj_ori = obs['object_orientation']
        while not done:
            grasp = grasping.sample_partial_grasp(self.env, obj_pos, obj_ori)
            if len(grasp.valid_tips) == 1:
                # NOTE: if only one of the tips is valid, check if
                # a. the tip is far from origin than object
                # b. angle between tip-to-origin vector and object's y-axis > 30 degree
                dist = np.linalg.norm(grasp.base_tip_pos[:, :2], axis=1)
                tip_id = np.argmax(dist)
                origin_to_tip = grasp.base_tip_pos[tip_id, :2]
                origin_to_tip /= np.linalg.norm(origin_to_tip)
                cube_yaxis = Transform(np.array([0, 0, 0]), obj_ori)(np.array([0, 1, 0]))[:2]
                cube_yaxis /= np.linalg.norm(cube_yaxis)
                # cube_yaxis = Rotation.from_quat(obj_ori).apply(np.array([0, 1, 0]))  # Is this same?
                if dist[tip_id] > np.linalg.norm(obj_pos) and np.dot(origin_to_tip, cube_yaxis) < 1 / 2:
                    print('grasp.valid_tips', grasp.valid_tips)
                    done = True
            else:
                done = True

        if grasp is None:
            raise RuntimeError("grasp is not found")

        actions = grasping.get_grasp_approach_actions(self.env, obs, grasp)
        info['grasp'] = grasp
        for pos in actions:
            yield self.get_action(position=pos, frameskip=1), info


class PitchingGraspState(OpenLoopState):
    """Find a grasp for aligning pitch orientation, and move fingers to the grasp positions.

    This state expects the object is a cube.

    Added to info:
        + grasp
        + pitch_axis
        + pitch_angle
    """

    def get_action_generator(self, obs, info):
        obj_pos = obs['object_position']
        obj_ori = obs['object_orientation']

        grasp, pitch_axis, pitch_angle = grasping.get_pitching_grasp(
            self.env, obj_pos, obj_ori, obs['goal_object_orientation']
        )

        if grasp is None:
            raise RuntimeError("Pitching grasp failed...")

        info['grasp'] = grasp
        info['pitch_axis'] = pitch_axis
        info['pitch_angle'] = pitch_angle
        actions = grasping.get_grasp_approach_actions(self.env, obs, grasp)
        for pos in actions:
            yield self.get_action(position=pos, frameskip=1), info


class YawingGraspState(OpenLoopState):
    """Find a grasp for aligning yaw orientation, and move fingers to the grasp positions.

    Sample a grasp and run wholebody planning (:class:`grasping.wholebody_planning.WholeBodyPlanner`)
    to check if the grasp is feasible.
    Once a valid grasp is found, use it to generate action sequence to move
    fingers to the grasp positions.

    Added to info:
        + grasp
        + path
    """

    def get_action_generator(self, obs, info):
        obj_pos = obs['object_position']
        obj_ori = obs['object_orientation']

        candidate_step_angle = [np.pi * 2 / 3, np.pi / 2, np.pi / 3]
        for step_angle in candidate_step_angle:
            grasp, path = grasping.get_yawing_grasp(
                self.env, obj_pos, obj_ori,
                obs['goal_object_orientation'],
                step_angle=step_angle
            )
            if grasp is None:
                continue

            try:
                actions = grasping.get_grasp_approach_actions(
                    self.env, obs, grasp
                )
                break
            except RuntimeError:
                continue

        info['grasp'] = grasp
        info['path'] = path
        actions = grasping.get_grasp_approach_actions(self.env, obs, grasp)
        for pos in actions:
            yield self.get_action(position=pos, frameskip=1), info


class MoveToCenterState(OpenLoopState):
    """Move the object to center.

    This state moves the fingers to the given grasp positions, moves
    the objcet to center, and then release fingers from it.
    This expects that the fingers are grasping the cube when this state is reached.

    Expect for info to have:
        * grasp
    Remove from info:
        - grasp
    """

    def get_action_generator(self, obs, info):
        if 'grasp' not in info:
            raise RuntimeError("info does not contain grasp information!")

        action_sequence = ScriptedActions(
            self.env, obs['robot_tip_positions'], info['grasp']
        )
        action_sequence.add_grasp(coef=0.6)
        action_sequence.add_move_to_center(coef=0.6)
        action_sequence.add_release(2.0)
        action_sequence.add_raise_tips(height=0.08)
        actions = action_sequence.get_action_sequence(
            frameskip=1,
            action_repeat=5,
            action_repeat_end=10
        )
        del info['grasp']
        for pos in actions:
            yield self.get_action(position=pos, frameskip=2), info


class AlignRollAndPitchState(OpenLoopState):
    """Align roll and pitch.

    This expects that the fingers are grasping the cube when this state is reached.

    Expect for info to have:
        * grasp
        * pitch_axis
        * pitch_angle

    Remove from info:
        - grasp
        - pitch_axis
        - pitch_angle
    """

    def get_action_generator(self, obs, info):
        if 'grasp' not in info or 'pitch_axis' not in info:
            raise RuntimeError("info does not contain grasp information!")

        action_sequence = ScriptedActions(
            self.env, obs['robot_tip_positions'], info['grasp']
        )
        action_sequence.add_grasp(coef=0.6)
        pitch_angle = np.sign(info['pitch_angle']) * np.pi / 2
        action_sequence.add_pitch_rotation(
            height=0.045,
            rotate_axis=info['pitch_axis'],
            rotate_angle=pitch_angle,
            coef=0.6)
        action_sequence.add_release()
        action_sequence.add_raise_tips()
        actions = action_sequence.get_action_sequence(
            frameskip=1,
            action_repeat=6,
            action_repeat_end=10
        )
        del info['grasp']
        del info['pitch_axis']
        del info['pitch_angle']
        for pos in actions:
            yield self.get_action(position=pos, frameskip=2), info


class AlignYawReleaseState(OpenLoopState):
    """Simply release the object.

    Moves the finger away from the object.
    This state is only used within :class:`AlignYawState`.
    This expects that the fingers are grasping the cube when this state is reached.

    Expect for info to have:
        * grasp

    Remove from info:
        - grasp
        - path
    """

    def get_action_generator(self, obs, info):
        if 'grasp' not in info:
            raise RuntimeError("info does not contain grasp information!")

        info['grasp'].update(obs['object_position'], obs['object_orientation'])
        action_sequence = ScriptedActions(
            self.env, obs['robot_tip_positions'], info['grasp']
        )
        action_sequence.add_release2(1.5)
        action_sequence.add_raise_tips(height=0.08)
        actions = action_sequence.get_action_sequence(
            frameskip=1,
            action_repeat=6,
            action_repeat_end=10
        )
        del info['grasp']
        if 'path' in info.keys():
            del info['path']
        for pos in actions:
            yield self.get_action(position=pos, frameskip=2), info


class AlignYawState(State):
    """Align yaw of the object

    This use the given planned path together with force control to rotate the object.
    This expects that the fingers are grasping the cube when this state is reached.

    Expect for info to have:
        * path
    """

    def __init__(self, env):
        super().__init__(env)
        self.release = AlignYawReleaseState(env)
        self.reset()

    def reset(self):
        self.pi = None
        self.release.reset()

    def connect(self, next_state, failure_state):
        self.release.connect(next_state, failure_state)

    def __call__(self, obs, info=None):
        assert "path" in info

        if self.pi is None:
            self.pi = base_policies.PlanningAndForceControlPolicy(
                self.env, obs, base_policies.CancelGravityPolicy(self.env),
                info['path'],
                adjust_tip=False,
                adjust_tip_ori=False,
                action_repeat=4
            )

        if not self.pi.at_end_of_sequence(self.pi._step):
            action = self.pi(obs)
            return self.get_action(
                position=action['position'], torque=action['torque']
            ), self, info
        else:
            self.reset()
            return self.get_action(frameskip=0), self.release, info


class MoveToGoalState(State):
    """The main state for carrying the object to the goal.

    Uses the planned path together with force control to carry
    the ojbect to goal.
    This expects that the fingers are grasping the cube when this state is reached.

    Expect for info to have:
        * path
    """

    BO_action_repeat = 12  # BO

    def __init__(self, env):
        super().__init__(env)
        self.grasp_check_failed_count = 0
        self.pi = None

    def reset(self):
        self.pi = None

    def object_grasped(self, obs, grasp):
        T_cube_to_base = Transform(obs['object_position'],
                                   obs['object_orientation'])
        target_tip_pos = T_cube_to_base(grasp.cube_tip_pos)
        center_of_tips = np.mean(target_tip_pos, axis=0)
        dist = np.linalg.norm(target_tip_pos - obs['robot_tip_positions'])
        center_dist = np.linalg.norm(center_of_tips - np.mean(obs['robot_tip_positions'], axis=0))
        object_is_grasped = center_dist < 0.07 and dist < 0.10
        if object_is_grasped:
            self.grasp_check_failed_count = 0
        else:
            self.grasp_check_failed_count += 1
            print('incremented grasp_check_failed_count')
            print(f'center_dist: {center_dist:.4f}\tdist: {dist:.4f}')

        return self.grasp_check_failed_count < 5

    def connect(self, next_state, failure_state):
        self.next_state = next_state
        self.failure_state = failure_state

    def __call__(self, obs, info=None):
        info = dict() if info is None else info
        assert "path" in info

        if self.pi is None:
            self.pi = base_policies.PlanningAndForceControlPolicy(
                self.env, obs, base_policies.CancelGravityPolicy(self.env),
                info['path'],
                adjust_tip=True,
                adjust_tip_ori=False,
                action_repeat=self.BO_action_repeat
            )

        if self.object_grasped(obs, info['grasp']):
            action = self.pi(obs)
            return self.get_action(
                position=action['position'], torque=action['torque']
            ), self, info

        print("Object has been dropped!")
        self.reset()  # cleanup
        return self.get_action(frameskip=0), self.failure_state, info
