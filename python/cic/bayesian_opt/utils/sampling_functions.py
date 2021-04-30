# this file incorporates all the sampling functions which are needed to generate the data for the GP,...
import sys
sys.path.append('../../')
import numpy as np
import pickle as pkl
import pybullet as p
from cic import math_tools
from cic.bayesian_opt.const import SIMULATION
from scipy.spatial.transform import Rotation
import copy

random = np.random.RandomState()
_CUBE_WIDTH = 0.065
_ARENA_RADIUS = 0.195

_cube_3d_radius = _CUBE_WIDTH * np.sqrt(3) / 2
_max_cube_com_distance_to_center = _ARENA_RADIUS - _cube_3d_radius

_min_height = _CUBE_WIDTH / 2
_max_height = 0.1

def define_sample_fct(name,num_samples, path, path2=None):
    if (name=='sample_basic_stage_3'):
        return sample_basic_stage_3(num_samples, path, path2=path2)
    elif (name=='sample_basic'):
        return sample_basic(num_samples, path, path2=path2)
    elif (name=='sample_basic_stage_2'):
        return sample_basic_stage_2(num_samples, path, path2=path2)
    elif (name=='sample_rot_lift_directions'):
        return sample_rot_lift_directions(num_samples, path, path2=path2)
    elif (name=='sample_rot_ground_directions'):
        return sample_rot_ground_directions(num_samples, path, path2=path2)
    elif (name=='sample_lift_w_orientations_directions'):
        return sample_lift_w_orientations_directions(num_samples, path, path2=path2)
    elif (name=='sample_for_align_exp'):
        return sample_for_align_exp(num_samples, path, path2=path2)
    else:
        print ('NO SAMPLING DISTRIBUTION CHOSEN')
        return 0


def random_xy():
    # sample uniform position in circle (https://stackoverflow.com/a/50746409)
    radius = _max_cube_com_distance_to_center * np.sqrt(random.random())
    theta = random.uniform(0, 2 * np.pi)

    # x,y-position of the cube
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    return x, y

def random_rotation(magnitude):
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(magnitude * axis)

def sample_random_num(low,high):
    return (2 * np.random.rand(1) - 1)*((high-low)/2) + (high+low)/2

def sample_basic_stage_3(num_samples, path, path2=None):
    init_x = [-0.025, 0.025]
    init_y = [-0.025, 0.025]
    init_z = [0.0325, 0.0325]

    init_r = [0.0, 0.0]
    init_p = [0.0, 0.0]
    init_yaw = [np.deg2rad(0.0), np.deg2rad(360.0)]

    target_r = [0.0, 0.0]
    target_p = [0.0, 0.0]
    target_yaw = [0.0, 0.0]

    rand_arr = np.random.rand(12, num_samples)
    rand_arr = (rand_arr - 0.5)
    rand_arr[0, :] = rand_arr[0, :] * (init_x[1] - init_x[0]) + 0.5 * (init_x[1] + init_x[0])
    rand_arr[1, :] = rand_arr[1, :] * (init_y[1] - init_y[0]) + 0.5 * (init_y[1] + init_y[0])
    rand_arr[2, :] = rand_arr[2, :] * (init_z[1] - init_z[0]) + 0.5 * (init_z[1] + init_z[0])
    rand_arr[3, :] = rand_arr[3, :] * (init_r[1] - init_r[0]) + 0.5 * (init_r[1] + init_r[0])
    rand_arr[4, :] = rand_arr[4, :] * (init_p[1] - init_p[0]) + 0.5 * (init_p[1] + init_p[0])
    rand_arr[5, :] = rand_arr[5, :] * (init_yaw[1] - init_yaw[0]) + 0.5 * (init_yaw[1] + init_yaw[0])

    rand_arr[9, :] = rand_arr[9, :] * (target_r[1] - target_r[0]) + 0.5 * (target_r[1] + target_r[0])
    rand_arr[10, :] = rand_arr[10, :] * (target_p[1] - target_p[0]) + 0.5 * (target_p[1] + target_p[0])
    rand_arr[11, :] = rand_arr[11, :] * (target_yaw[1] - target_yaw[0]) + 0.5 * (target_yaw[1] + target_yaw[0])

    for i in range(num_samples):
        x, y = random_xy()
        z = random.uniform(_min_height, _max_height)
        rand_arr[6, i] = x
        rand_arr[7, i] = y
        rand_arr[8, i] = z

    target_arr = np.zeros((14, num_samples))
    for i in range(num_samples):
        target_arr[:3, i] = rand_arr[:3, i]
        target_arr[3:7, i] = np.asarray(p.getQuaternionFromEuler(rand_arr[3:6, i]))
        target_arr[7:10, i] = rand_arr[6:9, i]
        target_arr[10:14, i] = np.asarray(p.getQuaternionFromEuler(rand_arr[9:12, i]))

    if not(SIMULATION):
        target_arr[0, :8] = 1.0
        target_arr[0, 8:] = 1.0
        target_arr[1, :8] = 0.0
        target_arr[1, 8:] = 1.0
        target_arr[2, :] = 0.0325
        target_arr[6,:] = 1.0

    if (path is None):
        return target_arr
    initial_pos_path = str(path + 'pos.pkl')
    with open(initial_pos_path, 'wb') as f:
        pkl.dump(target_arr, f)

    if not (path2 is None):
        initial_pos_path = str(path2 + 'pos.pkl')
        with open(initial_pos_path, 'wb') as f:
            pkl.dump(target_arr, f)


def sample_basic(num_samples,path, path2=None):
    init_x = [-0.025, 0.025]
    init_y = [-0.025, 0.025]
    init_z = [0.0325, 0.0325]

    init_r = [0.0, 0.0]
    init_p = [0.0, 0.0]
    init_yaw = [np.deg2rad(0.0), np.deg2rad(360.0)]

    target_x = [-0.11, 0.11]
    target_y = [-0.11, 0.11]
    target_z = [0.0325, 0.0325]

    target_r = [0.0, 0.0]
    target_p = [0.0, 0.0]
    target_yaw = [0.0, 0.0]

    rand_arr = np.random.rand(12, num_samples)
    rand_arr = (rand_arr - 0.5)
    rand_arr[0, :] = rand_arr[0, :] * (init_x[1] - init_x[0]) + 0.5 * (init_x[1] + init_x[0])
    rand_arr[1, :] = rand_arr[1, :] * (init_y[1] - init_y[0]) + 0.5 * (init_y[1] + init_y[0])
    rand_arr[2, :] = rand_arr[2, :] * (init_z[1] - init_z[0]) + 0.5 * (init_z[1] + init_z[0])
    rand_arr[3, :] = rand_arr[3, :] * (init_r[1] - init_r[0]) + 0.5 * (init_r[1] + init_r[0])
    rand_arr[4, :] = rand_arr[4, :] * (init_p[1] - init_p[0]) + 0.5 * (init_p[1] + init_p[0])
    rand_arr[5, :] = rand_arr[5, :] * (init_yaw[1] - init_yaw[0]) + 0.5 * (init_yaw[1] + init_yaw[0])
    rand_arr[6, :] = np.clip(rand_arr[6, :] * (target_x[1] - target_x[0]) + 0.5 * (target_x[1] + target_x[0]),-0.1,0.1)
    rand_arr[7, :] = np.clip(rand_arr[7, :] * (target_y[1] - target_y[0]) + 0.5 * (target_y[1] + target_y[0]),-0.1,0.1)
    rand_arr[8, :] = rand_arr[8, :] * (target_z[1] - target_z[0]) + 0.5 * (target_z[1] + target_z[0])
    rand_arr[9, :] = rand_arr[9, :] * (target_r[1] - target_r[0]) + 0.5 * (target_r[1] + target_r[0])
    rand_arr[10, :] = rand_arr[10, :] * (target_p[1] - target_p[0]) + 0.5 * (target_p[1] + target_p[0])
    rand_arr[11, :] = rand_arr[11, :] * (target_yaw[1] - target_yaw[0]) + 0.5 * (target_yaw[1] + target_yaw[0])

    target_arr = np.zeros((14, num_samples))
    for i in range(num_samples):
        target_arr[:3, i] = rand_arr[:3, i]
        target_arr[3:7, i] = np.asarray(p.getQuaternionFromEuler(rand_arr[3:6, i]))
        target_arr[7:10, i] = rand_arr[6:9, i]
        target_arr[10:14, i] = np.asarray(p.getQuaternionFromEuler(rand_arr[9:12, i]))

    if not(SIMULATION):
        target_arr[0, :8] = 1.0
        target_arr[0, 8:] = 1.0
        target_arr[1, :8] = 0.0
        target_arr[1, 8:] = 1.0
        target_arr[2, :] = 0.0325
        target_arr[6,:] = 1.0

    if (path is None):
        return target_arr

    initial_pos_path = str(path + 'pos.pkl')
    with open(initial_pos_path, 'wb') as f:
        pkl.dump(target_arr, f)

    if not(path2 is None):
        initial_pos_path = str(path2 + 'pos.pkl')
        with open(initial_pos_path, 'wb') as f:
            pkl.dump(target_arr, f)


def sample_basic_stage_2(num_samples,path, path2=None):
    init_x = [-0.025, 0.025]
    init_y = [-0.025, 0.025]
    init_z = [0.0325, 0.0325]

    init_r = [0.0, 0.0]
    init_p = [0.0, 0.0]
    init_yaw = [np.deg2rad(0.0), np.deg2rad(360.0)]

    target_x = [0.0, 0.0]
    target_y = [0.0, 0.0]
    target_z = [0.0825-0.015, 0.0825+0.01]

    target_r = [0.0, 0.0]
    target_p = [0.0, 0.0]
    target_yaw = [0.0, 0.0]

    rand_arr = np.random.rand(12, num_samples)
    rand_arr = (rand_arr - 0.5)
    rand_arr[0, :] = rand_arr[0, :] * (init_x[1] - init_x[0]) + 0.5 * (init_x[1] + init_x[0])
    rand_arr[1, :] = rand_arr[1, :] * (init_y[1] - init_y[0]) + 0.5 * (init_y[1] + init_y[0])
    rand_arr[2, :] = rand_arr[2, :] * (init_z[1] - init_z[0]) + 0.5 * (init_z[1] + init_z[0])
    rand_arr[3, :] = rand_arr[3, :] * (init_r[1] - init_r[0]) + 0.5 * (init_r[1] + init_r[0])
    rand_arr[4, :] = rand_arr[4, :] * (init_p[1] - init_p[0]) + 0.5 * (init_p[1] + init_p[0])
    rand_arr[5, :] = rand_arr[5, :] * (init_yaw[1] - init_yaw[0]) + 0.5 * (init_yaw[1] + init_yaw[0])
    rand_arr[6, :] = rand_arr[6, :] * (target_x[1] - target_x[0]) + 0.5 * (target_x[1] + target_x[0])
    rand_arr[7, :] = rand_arr[7, :] * (target_y[1] - target_y[0]) + 0.5 * (target_y[1] + target_y[0])
    rand_arr[8, :] = rand_arr[8, :] * (target_z[1] - target_z[0]) + 0.5 * (target_z[1] + target_z[0])
    rand_arr[9, :] = rand_arr[9, :] * (target_r[1] - target_r[0]) + 0.5 * (target_r[1] + target_r[0])
    rand_arr[10, :] = rand_arr[10, :] * (target_p[1] - target_p[0]) + 0.5 * (target_p[1] + target_p[0])
    rand_arr[11, :] = rand_arr[11, :] * (target_yaw[1] - target_yaw[0]) + 0.5 * (target_yaw[1] + target_yaw[0])



    target_arr = np.zeros((14, num_samples))
    for i in range(num_samples):
        target_arr[:3, i] = rand_arr[:3, i]
        target_arr[3:7, i] = np.asarray(p.getQuaternionFromEuler(rand_arr[3:6, i]))
        target_arr[7:10, i] = rand_arr[6:9, i]
        target_arr[10:14, i] = np.asarray(p.getQuaternionFromEuler(rand_arr[9:12, i]))

    if not(SIMULATION):
        target_arr[0, :8] = 1.0
        target_arr[0, 8:] = 1.0
        target_arr[1, :8] = 0.0
        target_arr[1, 8:] = 1.0
        target_arr[2, :] = 0.0325
        target_arr[6,:] = 1.0

    if (path is None):
        return target_arr
    initial_pos_path = str(path + 'pos.pkl')
    with open(initial_pos_path, 'wb') as f:
        pkl.dump(target_arr, f)

    if not(path2 is None):
        initial_pos_path = str(path2 + 'pos.pkl')
        with open(initial_pos_path, 'wb') as f:
            pkl.dump(target_arr, f)

def sample_rot_lift_directions(num_samples,path, path2=None):
    import pybullet as p
    if not((num_samples==8) or (num_samples==4)):
        print ("NOT OPTIMAL SAMPLING")


    target_arr = np.zeros((14, num_samples))

    # this is if we first want to rotate into special position on the real system:
    if not(SIMULATION):
        target_arr[0, :8] = 1.0
        target_arr[0, 8:] = 1.0
        target_arr[1, :8] = 0.0
        target_arr[1, 8:] = 1.0
        target_arr[2, :] = 0.0325
        target_arr[6,:] = 1.0


    else:
        # THIS BLOCK IS FOR SETTING THE INITIAL CONDITION (IN SIMULATION,...)
        init_x = [-0.025, 0.025]
        init_y = [-0.025, 0.025]
        init_z = [0.0325, 0.0325]

        init_r = [0.0, 0.0]
        init_p = [0.0, 0.0]
        init_yaw = [np.deg2rad(0.0), np.deg2rad(360.0)]

        rand_arr = np.random.rand(6, num_samples)
        rand_arr = (rand_arr - 0.5)
        rand_arr[0, :] = rand_arr[0, :] * (init_x[1] - init_x[0]) + 0.5 * (init_x[1] + init_x[0])
        rand_arr[1, :] = rand_arr[1, :] * (init_y[1] - init_y[0]) + 0.5 * (init_y[1] + init_y[0])
        rand_arr[2, :] = rand_arr[2, :] * (init_z[1] - init_z[0]) + 0.5 * (init_z[1] + init_z[0])
        rand_arr[3, :] = rand_arr[3, :] * (init_r[1] - init_r[0]) + 0.5 * (init_r[1] + init_r[0])
        rand_arr[4, :] = rand_arr[4, :] * (init_p[1] - init_p[0]) + 0.5 * (init_p[1] + init_p[0])
        rand_arr[5, :] = rand_arr[5, :] * (init_yaw[1] - init_yaw[0]) + 0.5 * (init_yaw[1] + init_yaw[0])

        for i in range(num_samples):
            target_arr[:3, i] = copy.deepcopy(rand_arr[:3, i])
            target_arr[3:7, i] = copy.deepcopy(np.asarray(p.getQuaternionFromEuler(rand_arr[3:6, i])))


    i=1
    for j in range(num_samples):
        if (i%4==0):
            r,p,y = 0.0, -np.pi/2, 0.0
        elif (i%3==0):
            r, p, y = 0.0, np.pi / 2, 0.0
        elif (i%2==0):
            r, p, y = -np.pi / 2, 0.0, 0.0
        else:
            r, p, y = np.pi / 2, 0.0, 0.0
        i += 1
        rotMat = math_tools.rpy2Mat(r,p,y)
        rotMat = Rotation.from_matrix(rotMat)
        target_arr[10:14,j] = rotMat.as_quat()

        x, y = random_xy()
        z = 0.0325
        target_arr[7, j] = 0.0 #np.clip(x,-0.08,0.08)
        target_arr[8, j] = 0.0 #np.clip(y,-0.08,0.08)
        target_arr[9, j] = z

    if (path is None):
        return target_arr
    initial_pos_path = str(path + 'pos.pkl')
    with open(initial_pos_path, 'wb') as f:
        pkl.dump(target_arr, f)

    if not(path2 is None):
        initial_pos_path = str(path2 + 'pos.pkl')
        with open(initial_pos_path, 'wb') as f:
            pkl.dump(target_arr, f)


def sample_rot_ground_directions(num_samples,path, path2=None):
    import pybullet as p
    if not((num_samples==8) or (num_samples==4)):
        print ("NOT OPTIMAL SAMPLING")


    target_arr = np.zeros((14, num_samples))

    # this is if we first want to rotate into special position on the real system:
    if not(SIMULATION):
        target_arr[0, :8] = 1.0
        target_arr[0, 8:] = 1.0
        target_arr[1, :8] = 0.0
        target_arr[1, 8:] = 1.0
        target_arr[2, :] = 0.0325
        target_arr[6,:] = 1.0


    else:
        # THIS BLOCK IS FOR SETTING THE INITIAL CONDITION (IN SIMULATION,...)
        init_x = [-0.025, 0.025]
        init_y = [-0.025, 0.025]
        init_z = [0.0325, 0.0325]

        init_r = [0.0, 0.0]
        init_p = [0.0, 0.0]
        init_yaw = [np.deg2rad(0.0), np.deg2rad(360.0)]

        rand_arr = np.random.rand(6, num_samples)
        rand_arr = (rand_arr - 0.5)
        rand_arr[0, :] = rand_arr[0, :] * (init_x[1] - init_x[0]) + 0.5 * (init_x[1] + init_x[0])
        rand_arr[1, :] = rand_arr[1, :] * (init_y[1] - init_y[0]) + 0.5 * (init_y[1] + init_y[0])
        rand_arr[2, :] = rand_arr[2, :] * (init_z[1] - init_z[0]) + 0.5 * (init_z[1] + init_z[0])
        rand_arr[3, :] = rand_arr[3, :] * (init_r[1] - init_r[0]) + 0.5 * (init_r[1] + init_r[0])
        rand_arr[4, :] = rand_arr[4, :] * (init_p[1] - init_p[0]) + 0.5 * (init_p[1] + init_p[0])
        rand_arr[5, :] = rand_arr[5, :] * (init_yaw[1] - init_yaw[0]) + 0.5 * (init_yaw[1] + init_yaw[0])

        for i in range(num_samples):
            target_arr[:3, i] = copy.deepcopy(rand_arr[:3, i])
            target_arr[3:7, i] = copy.deepcopy(np.asarray(p.getQuaternionFromEuler(rand_arr[3:6, i])))

    rand_arr = np.random.rand(1, num_samples) # for random value
    rand_arr2 = np.random.rand(1, num_samples) # for sign
    rand_arr2[rand_arr2<0.5] = -1
    rand_arr2[rand_arr2>=0.5] = 1

    # only set the yaw direction here. Currently setup as: Rotation by at least 25 deg,....
    for j in range(num_samples):
        r, p, y = 0.0, 0.0, rand_arr2[0,j]*np.deg2rad(55*rand_arr[0,j]+35)

        rotMat = math_tools.rpy2Mat(r,p,y)
        rotMat = Rotation.from_matrix(rotMat)
        target_arr[10:14,j] = rotMat.as_quat()

        x, y = random_xy()
        z = 0.0325
        target_arr[7, j] = 0.0 #np.clip(x,-0.08,0.08)
        target_arr[8, j] = 0.0 #np.clip(y,-0.08,0.08)
        target_arr[9, j] = z

    if (path is None):
        return target_arr
    initial_pos_path = str(path + 'pos.pkl')
    with open(initial_pos_path, 'wb') as f:
        pkl.dump(target_arr, f)

    if not(path2 is None):
        initial_pos_path = str(path2 + 'pos.pkl')
        with open(initial_pos_path, 'wb') as f:
            pkl.dump(target_arr, f)


def sample_lift_w_orientations_directions(num_samples,path, path2=None):
    import pybullet as p
    # this is for trying to keep the rotation and ALSO lifting the cube,....
    if not((num_samples==8) or (num_samples==4)):
        print ("NOT OPTIMAL SAMPLING")


    target_arr = np.zeros((14, num_samples))

    # this is if we first want to rotate into special position on the real system:
    if not(SIMULATION):
        target_arr[0, :8] = 1.0
        target_arr[0, 8:] = 1.0
        target_arr[1, :8] = 0.0
        target_arr[1, 8:] = 1.0
        target_arr[2, :] = 0.0325
        target_arr[6,:] = 1.0


    else:
        # THIS BLOCK IS FOR SETTING THE INITIAL CONDITION (IN SIMULATION,...)
        init_x = [-0.025, 0.025]
        init_y = [-0.025, 0.025]
        init_z = [0.0325, 0.0325]

        init_r = [0.0, 0.0]
        init_p = [0.0, 0.0]
        init_yaw = [np.deg2rad(0.0), np.deg2rad(360.0)]

        rand_arr = np.random.rand(6, num_samples)
        rand_arr = (rand_arr - 0.5)
        rand_arr[0, :] = rand_arr[0, :] * (init_x[1] - init_x[0]) + 0.5 * (init_x[1] + init_x[0])
        rand_arr[1, :] = rand_arr[1, :] * (init_y[1] - init_y[0]) + 0.5 * (init_y[1] + init_y[0])
        rand_arr[2, :] = rand_arr[2, :] * (init_z[1] - init_z[0]) + 0.5 * (init_z[1] + init_z[0])
        rand_arr[3, :] = rand_arr[3, :] * (init_r[1] - init_r[0]) + 0.5 * (init_r[1] + init_r[0])
        rand_arr[4, :] = rand_arr[4, :] * (init_p[1] - init_p[0]) + 0.5 * (init_p[1] + init_p[0])
        rand_arr[5, :] = rand_arr[5, :] * (init_yaw[1] - init_yaw[0]) + 0.5 * (init_yaw[1] + init_yaw[0])

        for i in range(num_samples):
            target_arr[:3, i] = copy.deepcopy(rand_arr[:3, i])
            target_arr[3:7, i] = copy.deepcopy(np.asarray(p.getQuaternionFromEuler(rand_arr[3:6, i])))


    rand_arr = np.random.rand(3, num_samples) # for random value
    rand_arr2 = np.random.rand(3, num_samples) # for sign
    rand_arr2[rand_arr2<0.5] = -1
    rand_arr2[rand_arr2>=0.5] = 1


    for j in range(num_samples):
        # r, p, y = rand_arr2[0,j]*np.deg2rad(30*rand_arr[0,j]), rand_arr2[1,j]*np.deg2rad(30*rand_arr[1,j]), rand_arr2[2,j]*np.deg2rad(30*rand_arr[2,j])
        #
        # rotMat = math_tools.rpy2Mat(r,p,y)
        # rotMat = Rotation.from_matrix(rotMat)
        # target_arr[10:14,j] = rotMat.as_quat()

        # new sampling function:
        rotation_obj = random_rotation(np.deg2rad(sample_random_num(-45,45)))
        rot_in_euler = rotation_obj.as_euler('xyz', degrees=True)
        while True:
            if (np.max(np.abs(rot_in_euler))>=45):
                rotation_obj = random_rotation(np.deg2rad(sample_random_num(-45, 45)))
                rot_in_euler = rotation_obj.as_euler('xyz', degrees=True)
            else:
                break
        target_arr[10:14, j] = rotation_obj.as_quat()

        x, y = random_xy()
        z = random.uniform(_min_height, _max_height)
        target_arr[7, j] = np.clip(x,-0.08,0.08)
        target_arr[8, j] = np.clip(y,-0.08,0.08)
        target_arr[9, j] = z

    if (path is None):
        return target_arr
    initial_pos_path = str(path + 'pos.pkl')
    with open(initial_pos_path, 'wb') as f:
        pkl.dump(target_arr, f)

    if not(path2 is None):
        initial_pos_path = str(path2 + 'pos.pkl')
        with open(initial_pos_path, 'wb') as f:
            pkl.dump(target_arr, f)


def sample_for_align_exp(num_samples, path, path2=None):
    # this is a specialized sampling function that assures that at least a rotation around one axis is needed
    init_x = [-0.025, 0.025]
    init_y = [-0.025, 0.025]
    init_z = [0.0325, 0.0325]

    init_r = [0.0, 0.0]
    init_p = [0.0, 0.0]
    init_yaw = [np.deg2rad(0.0), np.deg2rad(360.0)]

    target_r = [np.deg2rad(-180.0), np.deg2rad(180.0)]
    target_p = [np.deg2rad(-180.0), np.deg2rad(180.0)]
    target_yaw = [np.deg2rad(-180.0), np.deg2rad(180.0)]

    rand_arr = np.random.rand(12, num_samples)
    rand_arr = (rand_arr - 0.5)
    rand_arr[0, :] = rand_arr[0, :] * (init_x[1] - init_x[0]) + 0.5 * (init_x[1] + init_x[0])
    rand_arr[1, :] = rand_arr[1, :] * (init_y[1] - init_y[0]) + 0.5 * (init_y[1] + init_y[0])
    rand_arr[2, :] = rand_arr[2, :] * (init_z[1] - init_z[0]) + 0.5 * (init_z[1] + init_z[0])
    rand_arr[3, :] = rand_arr[3, :] * (init_r[1] - init_r[0]) + 0.5 * (init_r[1] + init_r[0])
    rand_arr[4, :] = rand_arr[4, :] * (init_p[1] - init_p[0]) + 0.5 * (init_p[1] + init_p[0])
    rand_arr[5, :] = rand_arr[5, :] * (init_yaw[1] - init_yaw[0]) + 0.5 * (init_yaw[1] + init_yaw[0])

    rand_arr[9, :] = rand_arr[9, :] * (target_r[1] - target_r[0]) + 0.5 * (target_r[1] + target_r[0])
    rand_arr[10, :] = rand_arr[10, :] * (target_p[1] - target_p[0]) + 0.5 * (target_p[1] + target_p[0])
    rand_arr[11, :] = rand_arr[11, :] * (target_yaw[1] - target_yaw[0]) + 0.5 * (target_yaw[1] + target_yaw[0])

    for i in range(num_samples):
        x, y = random_xy()
        z = random.uniform(_min_height, _max_height)
        rand_arr[6, i] = x
        rand_arr[7, i] = y
        rand_arr[8, i] = z

    target_arr = np.zeros((14, num_samples))
    for i in range(num_samples):
        target_arr[:3, i] = rand_arr[:3, i]
        target_arr[3:7, i] = np.asarray(p.getQuaternionFromEuler(rand_arr[3:6, i]))
        target_arr[7:10, i] = rand_arr[6:9, i]
        # roll,pitch,yaw = rand_arr[9, i], rand_arr[10,i], rand_arr[11,i]
        # while True:
        #     if (np.max(np.abs([roll,pitch,yaw]))<np.deg2rad(45)):
        #         roll,pitch,yaw = np.deg2rad(np.random.randint(-180,180)), np.deg2rad(np.random.randint(-180,180)), np.deg2rad(np.random.randint(-180,180))
        #     else:
        #         break
        # target_arr[10:14, i] = np.asarray(p.getQuaternionFromEuler([roll,pitch,yaw]))

        # new sampling function:
        rotation_obj = random_rotation(np.deg2rad(sample_random_num(-360,360)))
        rot_in_euler = rotation_obj.as_euler('xyz', degrees=True)
        while True:
            if (np.max(np.abs(rot_in_euler))<45):
                rotation_obj = random_rotation(np.deg2rad(sample_random_num(-360, 360)))
                rot_in_euler = rotation_obj.as_euler('xyz', degrees=True)
            else:
                break
        target_arr[10:14, i] = rotation_obj.as_quat()

    if not(SIMULATION):
        target_arr[0, :8] = 1.0
        target_arr[0, 8:] = 1.0
        target_arr[1, :8] = 0.0
        target_arr[1, 8:] = 1.0
        target_arr[2, :] = 0.0325
        target_arr[6,:] = 1.0

    if (path is None):
        return target_arr
    initial_pos_path = str(path + 'pos.pkl')
    with open(initial_pos_path, 'wb') as f:
        pkl.dump(target_arr, f)

    if not (path2 is None):
        initial_pos_path = str(path2 + 'pos.pkl')
        with open(initial_pos_path, 'wb') as f:
            pkl.dump(target_arr, f)
