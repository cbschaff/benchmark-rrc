"""Place initializers here.

These will be passed as an arguement to the training env, allowing us to
easily try out different cube initializations (i.e. for cirriculum learning).
"""

import os
from collections import namedtuple
from trifinger_simulation.tasks import move_cube
from trifinger_simulation.tasks.move_cube import Pose
from scipy.spatial.transform import Rotation
from trifinger_simulation.tasks.move_cube import _ARENA_RADIUS, _max_height
from mp.align_rotation import project_cube_xy_plane
from mp.utils import sample_uniform_from_circle
import numpy as np
import random
import pickle as pkl


class RandomInitializer:
    """Initializer that samples random initial states and goals."""

    def __init__(self, difficulty):
        """Initialize.
        Args:
            difficulty (int):  Difficulty level for sampling goals.
        """
        self.difficulty = difficulty

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        return move_cube.sample_goal(difficulty=-1)

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return move_cube.sample_goal(difficulty=self.difficulty)


class EvalEpisodesInitializer:
    '''Initialize episodes according to json files saved in eval_episodes'''

    def __init__(self, difficulty):
        self._counter = 0
        self.difficulty = difficulty
        self.eval_dir = 'eval_episodes/level{}'.format(difficulty)
        self.episodes = []
        # self._load_episodes()
        self._init_flag = [False, False]  # Flag to maintain the initialization counter without assuming if get_initial_state is called before get_goal

    def get_initial_state(self):
        if not self.episodes:
            self._load_episodes()
        ret = self.episodes[self._counter].initial_state
        self._update_counter()
        self._init_flag[0] = True
        return ret

    def get_goal(self):
        ret = self.episodes[self._counter].goal
        self._update_counter()
        self._init_flag[1] = True
        return ret

    def _update_counter(self):
        '''update the counter which is maintained to avoid accessing non-existing evaluation episode'''
        assert self._counter < len(self.episodes), 'Only {} eval episodes found, however, the function is called {} times'.format(len(self.episodes), self._counter)
        if all(self._init_flag):
            self._counter += 1
            self._init_flag = [False, False]

    def _load_episodes(self):
        assert os.path.isdir(self.eval_dir), 'Make sure that you have generated evaluation episodes'
        EvalEpisode = namedtuple('EvalEpisode', ['initial_state', 'goal'])
        files = os.listdir(self.eval_dir)
        assert len(files) % 2 == 0, 'Even number of files are expected in {}'.format(self.eval_dir)
        num_episodes = len(files) // 2
        for i in range(num_episodes):
            with open(os.path.join(self.eval_dir, '{:05d}-init.json'.format(i)), 'r') as f:
                init = Pose.from_json(f.read())
            with open(os.path.join(self.eval_dir, '{:05d}-goal.json'.format(i)), 'r') as f:
                goal = Pose.from_json(f.read())
            self.episodes.append(EvalEpisode(init, goal))


class Task4SmallRotation:
    def __init__(self, difficulty, orientation_error_threshold=np.pi/2 * 0.5):
        if difficulty != 4:
            raise ValueError("Task4SmallRotation initializer should only be "
                             f"used with goal difficulty 4. difficulty: {difficulty}")
        self.difficulty = 4
        self.init = None
        self.orientation_error_threshold = orientation_error_threshold

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        self.init = move_cube.sample_goal(difficulty=-1)
        return self.init

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        ori_error = 100000  # some large value
        while ori_error < np.pi/2 * 0.1 or \
              ori_error > self.orientation_error_threshold:
            goal = move_cube.sample_goal(difficulty=4)
            # goal.position[:2] = self.init.position[:2]  # TEMP: align x and y
            ori_error = self._weighted_orientation_error(goal)
            # pos_error = self._weighted_position_error(goal)
        return goal

    def _weighted_orientation_error(self, goal):
        goal_rot = Rotation.from_quat(goal.orientation)
        init_rot = Rotation.from_quat(self.init.orientation)
        error_rot = goal_rot.inv() * init_rot
        orientation_error = error_rot.magnitude()
        return orientation_error

    def _weighted_position_error(self, goal):
        range_xy_dist = _ARENA_RADIUS * 2
        range_z_dist = _max_height

        xy_dist = np.linalg.norm(
            goal.position[:2] - self.init.position[:2]
        )
        z_dist = abs(goal.position[2] - self.init.position[2])
        # weight xy- and z-parts by their expected range
        return (xy_dist / range_xy_dist + z_dist / range_z_dist) / 2


class TrainingInitializer:
    """Init in a tighter radius."""

    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.goal = None

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        init = move_cube.sample_goal(difficulty=-1)
        init.position[:2] = sample_uniform_from_circle(0.07)
        if self.difficulty == 4:
            self.goal = move_cube.sample_goal(self.difficulty)

            # sample orientation
            projected_goal_ori = project_cube_xy_plane(self.goal.orientation)
            z_rot_noise = Rotation.from_euler(
                'z', (np.pi / 2 * 0.70) * random.random()
            )
            init.orientation = (
                z_rot_noise * Rotation.from_quat(projected_goal_ori)
            ).as_quat()
        return init

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        if self.difficulty == 4:
            if self.goal is None:
                raise ValueError("Goal is unset. Call get_initial_state before "
                                 "get_goal.")
            else:
                goal = self.goal
                self.goal = None
        else:
            goal = move_cube.sample_goal(difficulty=self.difficulty)
        return goal


class CenteredInitializer:
    """Init in a tighter radius."""

    def __init__(self, difficulty):
        self.difficulty = difficulty

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        init = move_cube.sample_goal(difficulty=-1)
        init.position[:2] = sample_uniform_from_circle(0.10)
        return init

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return move_cube.sample_goal(difficulty=self.difficulty)


class BOInitializer:
    """Initializer that is based on some generated file that needs to be loaded,..."""

    def __init__(self, difficulty):
        """Initialize.
        Args:
            difficulty (int):  Difficulty level for sampling goals.
        """
        self.difficulty = difficulty
        with open("/ws/src/usercode/python/cic/bayesian_opt/content/pos.pkl", 'rb') as f:
            init_arr_params = pkl.load(f)
        with open("/ws/src/usercode/python/cic/bayesian_opt/content/iter_idx.txt", 'r') as f:
            curr_idx = int(f.readline())
        self.init_information = np.asarray(init_arr_params[:, curr_idx], dtype=float)

    def get_initial_state(self):
        """Get the initial position based on the information from the file. Do not read z coordinate to ensure
        that this position is on the ground plane"""
        ex_state = move_cube.sample_goal(difficulty=-1)
        ex_state.position = self.init_information[0:2] # read only two values, z-coordinate is unchanged
        ex_state.orientation = self.init_information[3:7]
        return ex_state

    def get_goal(self):
        """Get the goal position based on the information from the file."""
        ex_state = move_cube.sample_goal(difficulty=self.difficulty)
        ex_state.position = self.init_information[7:10]
        ex_state.orientation = self.init_information[10:14]
        return ex_state


random_init = RandomInitializer
eval_init = EvalEpisodesInitializer
small_rot_init = Task4SmallRotation
training_init = TrainingInitializer
centered_init = CenteredInitializer
bo_init = BOInitializer
