#!/usr/bin/env python3
import numpy as np
from trifinger_simulation.tasks import move_cube
from trifinger_simulation.trifinger_platform import TriFingerPlatform

COLLISION_TOLERANCE = 3.5 * 1e-03
MU = 0.5
CUBOID_SIZE = np.array((0.065, 0.065, 0.065))
CUBOID_HALF_SIZE = CUBOID_SIZE / 2
CUBOID_MASS = 0.094
VIRTUAL_CUBOID_HALF_SIZE = CUBOID_HALF_SIZE + 0.007
MIN_HEIGHT = min(CUBOID_HALF_SIZE)
MAX_HEIGHT = move_cube._max_height
ARENA_RADIUS = move_cube._ARENA_RADIUS
AVG_POSE_STEPS = 200
# INIT_JOINT_CONF = TriFingerPlatform.spaces.robot_position.default
INIT_JOINT_CONF = np.array([0.0, 0.9, -2.0, 0.0, 0.9, -2.0, 0.0, 0.9, -2.0], dtype=np.float32)
CONTRACTED_JOINT_CONF = np.array([0.0, 1.4, -2.4, 0.0, 1.4, -2.4, 0.0, 1.4, -2.4], dtype=np.float32)
CLEARED_JOINT_CONF = np.array([0.0, 1.4, -0.6, 0.0, 1.4, -0.6, 0.0, 1.4, -0.6], dtype=np.float32)

TMP_VIDEO_DIR = '/tmp/rrc_videos'
CUSTOM_LOGDIR = '/output'  # only applicable when it is running on Singularity

EXCEP_MSSG = "================= captured exception =================\n" + \
    "{message}\n" + "{error}\n" + '=================================='

# colors
TRANSLU_CYAN = (0, 1, 1, 0.4)
TRANSLU_YELLOW = (1, 1, 0, 0.4)
TRANSLU_BLUE = (0, 0, 1, 0.4)
TRANSLU_RED = (1, 0, 0, 0.4)
