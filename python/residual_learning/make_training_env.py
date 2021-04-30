"""Define training env."""
import torch
import gin
import os
from dl.rl.envs import SubprocVecEnv, DummyVecEnv, EpisodeInfo, VecObsNormWrapper
from env.make_env import make_env
from trifinger_simulation.tasks import move_cube
from .residual_wrappers import ResidualWrapper, RandomizedEnvWrapper


@gin.configurable
def make_training_env(nenv, state_machine, goal_difficulty, action_space,
                      residual_state, frameskip=1, sim=True, visualization=False,
                      reward_fn=None, termination_fn=None, initializer=None,
                      episode_length=100000, monitor=False, seed=0,
                      domain_randomization=False, norm_observations=False,
                      max_torque=0.1):

    # dummy goal dict
    goal = move_cube.sample_goal(goal_difficulty)

    goal_dict = {
        'position': goal.position,
        'orientation': goal.orientation
    }

    def _env(rank):
        def _thunk():
            env = make_env(
                cube_goal_pose=goal_dict,
                goal_difficulty=goal_difficulty,
                action_space=action_space,
                frameskip=frameskip,
                sim=sim,
                visualization=visualization,
                reward_fn=reward_fn,
                termination_fn=termination_fn,
                initializer=initializer,
                episode_length=10*episode_length,  # make this long enough to ensure that we have "episode_length" steps in the residual_state.
                rank=rank,
                monitor=monitor,
            )
            if domain_randomization:
                env = RandomizedEnvWrapper(env)
            env = ResidualWrapper(env, state_machine, frameskip,
                                  max_torque, residual_state,
                                  max_length=episode_length)
            env = EpisodeInfo(env)
            env.seed(seed + rank)
            return env
        return _thunk

    if nenv > 1:
        env = SubprocVecEnv([_env(i) for i in range(nenv)], context='fork')
    else:
        env = DummyVecEnv([_env(0)])
        env.reward_range = env.envs[0].reward_range

    if norm_observations:
        env = VecObsNormWrapper(env)
        norm_params = torch.load(
            os.path.join(os.path.dirname(__file__), 'obs_norm.pt')
        )
        env.load_state_dict(norm_params)
        assert env.mean is not None
        assert env.std is not None
    return env
