from .cube_env import RealRobotCubeEnv, ActionType
import env.wrappers as wrappers


def get_initializer(name):
    from env import initializers
    if name is None:
        return None
    if hasattr(initializers, name):
        return getattr(initializers, name)
    else:
        raise ValueError(f"Can't find initializer: {name}")


def get_reward_fn(name):
    from env import reward_fns
    if name is None:
        return reward_fns.competition_reward
    if hasattr(reward_fns, name):
        return getattr(reward_fns, name)
    else:
        raise ValueError(f"Can't find reward function: {name}")


def get_termination_fn(name):
    from env import termination_fns
    if name is None:
        return None
    if hasattr(termination_fns, name):
        return getattr(termination_fns, name)
    elif hasattr(termination_fns, "generate_" + name):
        return getattr(termination_fns, "generate_" + name)()
    else:
        raise ValueError(f"Can't find termination function: {name}")


def make_env(cube_goal_pose, goal_difficulty, action_space, frameskip=1,
             sim=False, visualization=False, reward_fn=None,
             termination_fn=None, initializer=None, episode_length=119000,
             rank=0, monitor=False, path=None):
    reward_fn = get_reward_fn(reward_fn)
    initializer = get_initializer(initializer)(goal_difficulty)
    termination_fn = get_termination_fn(termination_fn)
    if action_space not in ['torque', 'position', 'torque_and_position', 'position_and_torque']:
        raise ValueError(f"Unknown action space: {action_space}.")
    if action_space == 'torque':
        action_type = ActionType.TORQUE
    elif action_space in ['torque_and_position', 'position_and_torque']:
        action_type = ActionType.TORQUE_AND_POSITION
    else:
        action_type = ActionType.POSITION
    env = RealRobotCubeEnv(cube_goal_pose,
                           goal_difficulty,
                           action_type=action_type,
                           frameskip=frameskip,
                           sim=sim,
                           visualization=visualization,
                           reward_fn=reward_fn,
                           termination_fn=termination_fn,
                           initializer=initializer,
                           episode_length=episode_length,
                           path=path)
    env.seed(seed=rank)
    env.action_space.seed(seed=rank)
    env = wrappers.NewToOldObsWrapper(env)
    env = wrappers.AdaptiveActionSpaceWrapper(env)
    if not sim:
        env = wrappers.TimingWrapper(env, 0.001)
    if visualization:
        env = wrappers.PyBulletClearGUIWrapper(env)
    if monitor:
        env = wrappers.RenderWrapper(env)
    return env
