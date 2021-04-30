"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import pybullet as p
import numpy as np
import gym
import time
import cv2
from env.cube_env import ActionType
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation import camera


EXCEP_MSSG = "================= captured exception =================\n" + \
    "{message}\n" + "{error}\n" + '=================================='


class frameskip_to:
    '''
    A Context Manager that sets action type and action space temporally
    This applies to all wrappers and the origianl environment recursively
    '''
    def __init__(self, frameskip, env):
        self.frameskip = frameskip
        self.env = env
        self.org_frameskip = env.unwrapped.frameskip

    def __enter__(self):
        self.env.unwrapped.frameskip = self.frameskip

    def __exit__(self, type, value, traceback):
        self.env.unwrapped.frameskip = self.org_frameskip


class action_type_to:
    '''
    A Context Manager that sets action type and action space temporally
    This applies to all wrappers and the origianl environment recursively
    '''

    def __init__(self, action_type, env):
        self.action_type = action_type
        self.action_space = self._get_action_space(action_type)
        self.get_config(env)
        self.env = env

    def get_config(self, env):
        self.orig_action_spaces = [env.action_type]
        self.orig_action_types = [env.action_space]
        while hasattr(env, 'env'):
            env = env.env
            self.orig_action_types.append(env.action_type)
            self.orig_action_spaces.append(env.action_space)

    def __enter__(self):
        env = self.env
        env.action_space = self.action_space
        env.action_type = self.action_type
        while hasattr(env, 'env'):
            env = env.env
            env.action_space = self.action_space
            env.action_type = self.action_type

    def __exit__(self, type, value, traceback):
        ind = 0
        env = self.env
        env.action_space = self.orig_action_spaces[ind]
        env.action_type = self.orig_action_types[ind]
        while hasattr(env, 'env'):
            ind += 1
            env = env.env
            env.action_space = self.orig_action_spaces[ind]
            env.action_type = self.orig_action_types[ind]

    def _get_action_space(self, action_type):
        import gym
        from trifinger_simulation import TriFingerPlatform
        from env.cube_env import ActionType
        spaces = TriFingerPlatform.spaces
        if action_type == ActionType.TORQUE:
            action_space = spaces.robot_torque.gym
        elif action_type == ActionType.POSITION:
            action_space = spaces.robot_position.gym
        elif action_type == ActionType.TORQUE_AND_POSITION:
            action_space = gym.spaces.Dict(
                {
                    "torque": spaces.robot_torque.gym,
                    "position": spaces.robot_position.gym,
                }
            )
        else:
            raise ValueError('unknown action type')
        return action_space




class NewToOldObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
            "goal_object_orientation",
            "tip_force",
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": env.observation_space['robot']['position'],
                "robot_velocity": env.observation_space['robot']['velocity'],
                "robot_torque": env.observation_space['robot']['torque'],
                "robot_tip_positions": env.observation_space['robot']['tip_positions'],
                "object_position": env.observation_space["achieved_goal"]["position"],
                "object_orientation": env.observation_space["achieved_goal"]["orientation"],
                "goal_object_position": env.observation_space["desired_goal"]["position"],
                "goal_object_orientation": env.observation_space["desired_goal"]["orientation"],
                "tip_force": env.observation_space["robot"]["tip_force"],
                "action_torque": env.observation_space['robot']['torque'],
                "action_position": env.observation_space['robot']['position'],
            }
        )

    def observation(self, obs):
        old_obs = {
            "robot_position": obs['robot']['position'],
            "robot_velocity": obs['robot']['velocity'],
            "robot_torque": obs['robot']['torque'],
            "robot_tip_positions": obs['robot']['tip_positions'],
            "tip_force": obs['robot']['tip_force'],
            "object_position": obs['achieved_goal']['position'],
            "object_orientation": obs['achieved_goal']['orientation'],
            "goal_object_position": obs['desired_goal']['position'],
            "goal_object_orientation": obs['desired_goal']['orientation'],
        }
        if self.action_space == self.observation_space['robot_position']:
            old_obs['action_torque'] = np.zeros_like(obs['action'])
            old_obs['action_position'] = obs['action']
        elif self.action_space == self.observation_space['robot_torque']:
            old_obs['action_torque'] = obs['action']
            old_obs['action_position'] = np.zeros_like(obs['action'])
        else:
            old_obs['action_torque'] = obs['action']['torque']
            old_obs['action_position'] = obs['action']['position']
        return old_obs


class AdaptiveActionSpaceWrapper(gym.Wrapper):
    """Create a unified action space for torque and position control."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Dict({
            'position': gym.spaces.Box(
                low=trifingerpro_limits.robot_position.low,
                high=trifingerpro_limits.robot_position.high,
            ),
            'torque': gym.spaces.Box(
                low=trifingerpro_limits.robot_torque.low,
                high=trifingerpro_limits.robot_torque.high,
            ),
            'frameskip': gym.spaces.Box(low=np.zeros(1),
                                        high=np.inf * np.ones(1))
        })

    def _clip_action(self, action):
        clipped_action = {'torque': None, 'position': None,
                          'frameskip': action['frameskip']}
        if action['torque'] is not None:
            clipped_action['torque'] = np.clip(
                action['torque'],
                self.action_space['torque'].low,
                self.action_space['torque'].high
            )
        if action['position'] is not None:
            clipped_action['position'] = np.clip(
                action['position'],
                self.action_space['position'].low,
                self.action_space['position'].high
            )
        return clipped_action

    def step(self, action):
        action = self._clip_action(action)
        with frameskip_to(action['frameskip'], self.env):
            if action['torque'] is None:
                with action_type_to(ActionType.POSITION, self.env):
                    return self.env.step(action['position'])
            elif action['position'] is None:
                with action_type_to(ActionType.TORQUE, self.env):
                    return self.env.step(action['torque'])
            else:
                with action_type_to(ActionType.TORQUE_AND_POSITION, self.env):
                    return self.env.step({
                        'position': action['position'],
                        'torque': action['torque']
                    })


class TimingWrapper(gym.Wrapper):
    """Set timing constraints for realtime control based on the frameskip and
    action frequency."""

    def __init__(self, env, dt):
        super().__init__(env)
        self.dt = dt

    def reset(self):
        self.t = None
        self.frameskip = None
        return self.env.reset()

    def step(self, action):
        if self.t is not None:
            elapsed_time = time.time() - self.t
            min_elapsed_time = self.frameskip * self.dt
            if elapsed_time < min_elapsed_time:
                time.sleep(min_elapsed_time - elapsed_time)

        self.t = time.time()
        self.frameskip = action['frameskip']
        return self.env.step(action)


class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cameras = camera.TriFingerCameras(image_size=(360, 270))
        self.metadata = {"render.modes": ["rgb_array"]}
        self._initial_reset = True
        self._accum_reward = 0
        self._reward_at_step = 0

    def reset(self):
        import pybullet as p
        obs = self.env.reset()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])
        self._accum_reward = 0
        self._reward_at_step = 0
        if self._initial_reset:
            self._episode_idx = 0
            self._initial_reset = False
        else:
            self._episode_idx += 1
        return obs

    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)
        self._accum_reward += reward
        self._reward_at_step = reward
        return observation, reward, is_done, info

    def render(self, mode='rgb_array', **kwargs):
        assert mode == 'rgb_array', 'RenderWrapper Only supports rgb_array mode'
        images = self.cameras.cameras[0].get_image(), self.cameras.cameras[1].get_image()
        height = images[0].shape[1]
        two_views = np.concatenate((images[0], images[1]), axis=1)
        two_views = cv2.putText(two_views, 'step_count: {:06d}'.format(self.env.unwrapped.step_count), (10, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 0),
                    thickness=1, lineType=cv2.LINE_AA)

        two_views = cv2.putText(two_views, 'episode: {}'.format(self._episode_idx), (10, 70),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 0),
                    thickness=1, lineType=cv2.LINE_AA)

        two_views = cv2.putText(two_views, 'reward: {:.2f}'.format(self._reward_at_step), (10, height - 130),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        two_views = cv2.putText(two_views, 'acc_reward: {:.2f}'.format(self._accum_reward), (10, height - 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        return two_views


class PyBulletClearGUIWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])
        return obs
