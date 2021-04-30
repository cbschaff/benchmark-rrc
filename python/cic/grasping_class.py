import numpy as np
import copy
from .math_tools import rpy2Mat
import time

from itertools import permutations

class BasicGraspingPrimitive():
    def __init__(self, kinematics, params, object):

        self.kinematics = kinematics
        self.params = params
        self.object = object

        ## viapoints
        self.viapoints = None
        self.goal_target = None

        # Reference variable
        ## Variable to measure primitive progress and satisfy
        self.error = 9999.
        self.error_matrix = np.ones((3,3))*999.
        self.d_error = 9999.

        self.error_thrs = 0.003
        self.d_error_thrs = 0.0001
        self.v_thrs = 0.0001

        self.DEBUG = True

    def start(self, cube_state, des_cube_state, robot_state):
        self.viapoints = self.get_grasping_viapoints(cube_state , robot_state)

    def combinatorial_matching(self, fingers_poses, cube_targets):
        'Cube targets is provided as a list of candidates. We wanna find optimal candidate + optimal matching by intensive Search'
        best_desired_poses = cube_targets[0]
        best_perm = [0,1,2]
        d_error = 1000000000
        perm = permutations([0,1,2])
        for cube_target in cube_targets:
            for idx in perm:
                er = np.mean((fingers_poses - cube_target[:,idx])**2)
                if er< d_error:
                    d_error = er
                    best_desired_poses = cube_target
                    best_perm = idx
                    print(best_perm)
        return best_desired_poses[:,best_perm], best_perm, best_desired_poses


    def get_grasping_viapoints(self, cube_state , robot_state, center=True):
        ## Set State ##
        cube_pose = cube_state[0]
        robot_xyz = np.stack(robot_state[2])

        x_off = 1.4
        pos_f_c = np.array([[x_off,0.,0.],[-x_off,0.5,0.],[-x_off,-0.5,0.],[-x_off,0.,0.],[x_off,-0.5,0.],[x_off,0.5,0.],
                            [0.,0.,x_off],[0.,0.5,-x_off],[0.,-0.5,-x_off],[0.,0.,-x_off],[0.,-0.5,x_off],[0,0.5,x_off]]).T
        pos_f_w = self.object.in_world(cube_state=cube_state, pos_c = pos_f_c, offset_prop = True)
        pos_f_candidates = [pos_f_w[:,:3], pos_f_w[:,3:6],pos_f_w[:,6:9], pos_f_w[:,9:]]
        candidates_clean = []
        for candidate in pos_f_candidates:
            check = np.sum(candidate[2,:]<0.)
            if check<1:
                candidates_clean.append(candidate)



        if len(candidates_clean)>0:
            self.goal_target, _, __ = self.combinatorial_matching(fingers_poses=robot_xyz, cube_targets=pos_f_candidates)

        #print('fingers poses in world frame: ', goal_target)
        #time.sleep(60)


        return [self.goal_target]

    def is_satisfied(self, cube_state, des_cube_state, robot_state):
        robot_state_np = np.stack((robot_state[2][0], robot_state[2][1], robot_state[2][2]), 1)
        current_error_matrix = (self.goal_target - robot_state_np) ** 2
        current_error = np.mean(current_error_matrix)

        self.d_error = np.abs(self.error - current_error)
        self.error_matrix = current_error_matrix
        self.error = current_error

        if self.DEBUG:
            print('Current Goal is:', self.goal_target)
            print('Current Robot State in XYZ is:', robot_state[2])
            print('current_error is ', current_error)
            print('cube state is:', cube_state[2])

        return False

    def compute_goal_pose(self, cube_state, des_cube_state, robot_state):
        self.viapoints = self.get_grasping_viapoints(cube_state , robot_state)
        target = self.viapoints[0]
        #target = np.stack(robot_state[2])
        #print('Viapoint is:', target)
        self.goal_target = target
        #print(target)
        return target

    def grasping_gains(self):
        pass

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def skew(self, vector):
        """
        this function returns a numpy array with the skew symmetric cross product matrix for vector.
        the skew symmetric cross product matrix is defined such that
        np.cross(a, b) = np.dot(skew(a), b)
        :param vector: An array like vector to create the skew symmetric cross product matrix for
        :return: A numpy array of the skew symmetric cross product vector
        """

        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])



    def assign_y(self, edge_arr, p0, p1, p2, direction):
        asign1, cent1, dist1 = self.assign_y_internal(edge_arr, p0, p1, p2, direction)
        asign2, cent2, dist2 = self.assign_y_internal(edge_arr, p0, p2, p1, direction)
        if (dist1 <= dist2):
            return asign1, cent1
        else:
            # print ("before" + str(asign2))
            asign2[1:] = np.flipud(asign2[1:])
            # print("after" + str(asign2))
            cent2[1:] = np.flipud(cent2[1:])
            return asign2, cent2

    def assign_y_internal(self, edge_arr, p0, p1, p2, direction):

        max_dist = 1
        add_grasp1 = max_dist - 1
        add_grasp2 = max_dist + 1

        if (add_grasp1 > 3):
            add_grasp1 = 0
        if (add_grasp1 < 0):
            add_grasp1 = 3
        if (add_grasp2 > 3):
            add_grasp2 = 0
        if (add_grasp2 < 0):
            add_grasp2 = 3

        pot_ret = [[max_dist, add_grasp1, add_grasp2], [add_grasp2, max_dist, add_grasp1],
                   [add_grasp1, add_grasp2, max_dist]]
        cent = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        config1 = np.sum((edge_arr[:, max_dist] - p0) ** 2 + (edge_arr[:, add_grasp1] - p1) ** 2 + (
                    edge_arr[:, add_grasp2] - p2) ** 2)
        config2 = np.sum((edge_arr[:, max_dist] - p1) ** 2 + (edge_arr[:, add_grasp1] - p2) ** 2 + (
                    edge_arr[:, add_grasp2] - p0) ** 2)
        config3 = np.sum((edge_arr[:, max_dist] - p2) ** 2 + (edge_arr[:, add_grasp1] - p0) ** 2 + (
                    edge_arr[:, add_grasp2] - p1) ** 2)
        dist_arr = [config1, config2, config3]
        dec = np.argmin([config1, config2, config3])

        dist_1 = copy.deepcopy(dist_arr[dec])
        pot_ret_1 = copy.deepcopy(pot_ret[dec])
        cent_1 = copy.deepcopy(cent[dec])

        max_dist = 3
        add_grasp1 = max_dist - 1
        add_grasp2 = max_dist + 1

        if (add_grasp1 > 3):
            add_grasp1 = 0
        if (add_grasp1 < 0):
            add_grasp1 = 3
        if (add_grasp2 > 3):
            add_grasp2 = 0
        if (add_grasp2 < 0):
            add_grasp2 = 3

        pot_ret = [[max_dist, add_grasp1, add_grasp2], [add_grasp2, max_dist, add_grasp1],
                   [add_grasp1, add_grasp2, max_dist]]
        cent = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        config1 = np.sum((edge_arr[:, max_dist] - p0) ** 2 + (edge_arr[:, add_grasp1] - p1) ** 2 + (
                    edge_arr[:, add_grasp2] - p2) ** 2)
        config2 = np.sum((edge_arr[:, max_dist] - p1) ** 2 + (edge_arr[:, add_grasp1] - p2) ** 2 + (
                    edge_arr[:, add_grasp2] - p0) ** 2)
        config3 = np.sum((edge_arr[:, max_dist] - p2) ** 2 + (edge_arr[:, add_grasp1] - p0) ** 2 + (
                    edge_arr[:, add_grasp2] - p1) ** 2)
        dist_arr = [config1, config2, config3]
        dec = np.argmin([config1, config2, config3])

        dist_2 = copy.deepcopy(dist_arr[dec])
        pot_ret_2 = copy.deepcopy(pot_ret[dec])
        cent_2 = copy.deepcopy(cent[dec])

        moment_dir = np.dot(self.skew(self.unit_vector(edge_arr[:, 0]-edge_arr[:, 2])),self.unit_vector(edge_arr[:, 1]-edge_arr[:, 3]))
        if (direction==-1 or direction==1):
            if (np.sign(moment_dir[2])*np.sign(direction)==1):
                return pot_ret_1, cent_1, dist_1
            else:
                return pot_ret_2, cent_2, dist_2

        # if (direction==-1):
        #     return pot_ret_1, cent_1, dist_1
        # elif (direction==1):
        #     return pot_ret_2, cent_2, dist_2

        if (dist_1 < dist_2):
            return pot_ret_1, cent_1, dist_1
        else:
            return pot_ret_2, cent_2, dist_2

    def assign_x_internal(self, edge_arr, p0, p1, p2, direction):

        max_dist = 0
        add_grasp1 = max_dist-1
        add_grasp2 = max_dist+1

        if (add_grasp1>3):
            add_grasp1 = 0
        if (add_grasp1<0):
            add_grasp1 = 3
        if (add_grasp2>3):
            add_grasp2 = 0
        if (add_grasp2<0):
            add_grasp2 = 3

        pot_ret = [[max_dist,add_grasp1,add_grasp2],[add_grasp2,max_dist,add_grasp1],[add_grasp1,add_grasp2,max_dist]]
        cent = [[1,0,0],[0,1,0],[0,0,1]]
        config1 = np.sum((edge_arr[:,max_dist]-p0)**2+(edge_arr[:,add_grasp1]-p1)**2+(edge_arr[:,add_grasp2]-p2)**2)
        config2 = np.sum((edge_arr[:,max_dist]-p1)**2+(edge_arr[:,add_grasp1]-p2)**2+(edge_arr[:,add_grasp2]-p0)**2)
        config3 = np.sum((edge_arr[:,max_dist]-p2)**2+(edge_arr[:,add_grasp1]-p0)**2+(edge_arr[:,add_grasp2]-p1)**2)
        dist_arr = [config1, config2, config3]
        dec = np.argmin([config1, config2, config3])

        dist_1 = copy.deepcopy(dist_arr[dec])
        pot_ret_1 = copy.deepcopy(pot_ret[dec])
        cent_1 = copy.deepcopy(cent[dec])

        max_dist = 2
        add_grasp1 = max_dist-1
        add_grasp2 = max_dist+1

        if (add_grasp1>3):
            add_grasp1 = 0
        if (add_grasp1<0):
            add_grasp1 = 3
        if (add_grasp2>3):
            add_grasp2 = 0
        if (add_grasp2<0):
            add_grasp2 = 3

        pot_ret = [[max_dist,add_grasp1,add_grasp2],[add_grasp2,max_dist,add_grasp1],[add_grasp1,add_grasp2,max_dist]]
        cent = [[1,0,0],[0,1,0],[0,0,1]]
        config1 = np.sum((edge_arr[:,max_dist]-p0)**2+(edge_arr[:,add_grasp1]-p1)**2+(edge_arr[:,add_grasp2]-p2)**2)
        config2 = np.sum((edge_arr[:,max_dist]-p1)**2+(edge_arr[:,add_grasp1]-p2)**2+(edge_arr[:,add_grasp2]-p0)**2)
        config3 = np.sum((edge_arr[:,max_dist]-p2)**2+(edge_arr[:,add_grasp1]-p0)**2+(edge_arr[:,add_grasp2]-p1)**2)
        dist_arr = [config1, config2, config3]
        dec = np.argmin([config1, config2, config3])

        dist_2 = copy.deepcopy(dist_arr[dec])
        pot_ret_2 = copy.deepcopy(pot_ret[dec])
        cent_2 = copy.deepcopy(cent[dec])

        moment_dir = np.dot(self.skew(self.unit_vector(edge_arr[:, 1]-edge_arr[:, 3])),self.unit_vector(edge_arr[:, 0]-edge_arr[:, 2]))
        if (direction==-1 or direction==1):
            if (np.sign(moment_dir[2])*np.sign(direction)==1):
                return pot_ret_1, cent_1, dist_1
            else:
                return pot_ret_2, cent_2, dist_2

        # if (direction==1):
        #     return pot_ret_1, cent_1, dist_1
        # elif (direction==-1):
        #     return pot_ret_2, cent_2, dist_2

        if (dist_1<dist_2):
            return pot_ret_1, cent_1, dist_1
        else:
            return pot_ret_2, cent_2, dist_2

    def assign_x(self, edge_arr, p0, p1, p2, direction):
        asign1, cent1, dist1 = self.assign_x_internal(edge_arr, p0, p1, p2, direction)
        asign2, cent2, dist2 = self.assign_x_internal(edge_arr, p0, p2, p1, direction)
        if (dist1<=dist2):
            return asign1, cent1
        else:
            asign2[1:] = np.flipud(asign2[1:])
            cent2[1:] = np.flipud(cent2[1:])
            return asign2, cent2

    def assign_target_old(self, edge_arr, target, p0, p1, p2):

        dist_from_target = (np.sum((edge_arr-target.reshape(3,1))**2,axis=0))
        max_dist = np.argmax(dist_from_target)
        add_grasp1 = max_dist-1
        add_grasp2 = max_dist+1

        if (add_grasp1>3):
            add_grasp1 = 0
        if (add_grasp1<0):
            add_grasp1 = 3
        if (add_grasp2>3):
            add_grasp2 = 0
        if (add_grasp2<0):
            add_grasp2 = 3

        pot_ret = [[max_dist,add_grasp1,add_grasp2],[add_grasp2,max_dist,add_grasp1],[add_grasp1,add_grasp2,max_dist]]
        cent = [[1,0,0],[0,1,0],[0,0,1]]
        config1 = np.sum((edge_arr[:,max_dist]-p0)**2+(edge_arr[:,add_grasp1]-p1)**2+(edge_arr[:,add_grasp2]-p2)**2)
        config2 = np.sum((edge_arr[:,max_dist]-p1)**2+(edge_arr[:,add_grasp1]-p2)**2+(edge_arr[:,add_grasp2]-p0)**2)
        config3 = np.sum((edge_arr[:,max_dist]-p2)**2+(edge_arr[:,add_grasp1]-p0)**2+(edge_arr[:,add_grasp2]-p1)**2)
        dec = np.argmin([config1,config2,config3])

        return pot_ret[dec], cent[dec]

    def assign_internal(self, edge_arr, target, p0, p1, p2):

        dist_from_target = (np.sum((edge_arr-target.reshape(3,1))**2,axis=0))
        max_dist = np.argmax(dist_from_target)
        add_grasp1 = max_dist-1
        add_grasp2 = max_dist+1

        if (add_grasp1>3):
            add_grasp1 = 0
        if (add_grasp1<0):
            add_grasp1 = 3
        if (add_grasp2>3):
            add_grasp2 = 0
        if (add_grasp2<0):
            add_grasp2 = 3

        pot_ret = [[max_dist,add_grasp1,add_grasp2],[add_grasp2,max_dist,add_grasp1],[add_grasp1,add_grasp2,max_dist]]
        cent = [[1,0,0],[0,1,0],[0,0,1]]
        config1 = np.sum((edge_arr[:,max_dist]-p0)**2+(edge_arr[:,add_grasp1]-p1)**2+(edge_arr[:,add_grasp2]-p2)**2)
        config2 = np.sum((edge_arr[:,max_dist]-p1)**2+(edge_arr[:,add_grasp1]-p2)**2+(edge_arr[:,add_grasp2]-p0)**2)
        config3 = np.sum((edge_arr[:,max_dist]-p2)**2+(edge_arr[:,add_grasp1]-p0)**2+(edge_arr[:,add_grasp2]-p1)**2)
        dist_arr = [config1,config2,config3]
        dec = np.argmin([config1,config2,config3])

        return pot_ret[dec], cent[dec], dist_arr[dec]

    def assign_target(self, edge_arr, target, p0, p1, p2):
        asign1, cent1, dist1 = self.assign_internal(edge_arr, target, p0, p1, p2)
        asign2, cent2, dist2 = self.assign_internal(edge_arr, target, p0, p2, p1)
        if (dist1<=dist2):
            return asign1, cent1
        else:
            asign2[1:] = np.flipud(asign2[1:])
            cent2[1:] = np.flipud(cent2[1:])
            return asign2, cent2

    def assign_target_unrestricted(self, edge_arr, target, p0, p1, p2):
        # just tries to find the best grasp without caring about where the object has to be placed
        asign1, cent1, dist1 = self.assign_x_internal(edge_arr, p0, p1, p2, 0)
        asign2, cent2, dist2 = self.assign_x_internal(edge_arr, p0, p2, p1, 0)
        asign3, cent3, dist3 = self.assign_y_internal(edge_arr, p0, p1, p2, 0)
        asign4, cent4, dist4 = self.assign_y_internal(edge_arr, p0, p2, p1, 0)
        asign2[1:] = np.flipud(asign2[1:])
        cent2[1:] = np.flipud(cent2[1:])
        asign4[1:] = np.flipud(asign4[1:])
        cent4[1:] = np.flipud(cent4[1:])
        dist = np.asarray([dist1,dist2,dist3,dist4])
        asign = [asign1, asign2, asign3, asign4]
        cent = [cent1, cent2, cent3, cent4]

        best_grasp = np.argmin(dist)
        return asign[best_grasp], cent[best_grasp]


class TwoFingerGrasp(BasicGraspingPrimitive):
    def __init__(self, kinematics, params, object):
        super(TwoFingerGrasp, self).__init__(kinematics, params, object)
        # This additional offset should ensure to keep the third finger away,..
        self.excl_add_offset = 0.25


    def calc_grasp_position(self):
        '''
        This function might be used to calculate all the possible grasp locations if there are multiple to consider
        '''
        pass

    def assign_fingers(self,robot_state, cube_state, off_x=0.0, off_y =0.0, off_z=0.0):
        off_y +=  self.excl_add_offset
        '''
        This function should assign the fingers to where you want to grasp potnetially,...
        '''
        locations, vertical_axes_idx, axes_dir = self.object.compute_edge_loc_entire_cube(cube_state, off_x, off_y, off_z)
        self.asign_arr, self.cent_arr = self.assign_y(locations, robot_state[2][0], robot_state[2][1], robot_state[2][2])

    def get_finger_position(self,robot_state, cube_state, off_x=0.0, off_y =0.0, off_z=0.0, use_z = True):
        # Note: quick fix, use z has no effect here,...
        '''
        This function should return the desired positions of the fingers,...
        '''
        off_y +=  self.excl_add_offset

        #Step 1: calculate the locations:
        locations, vertical_axes_idx, axes_dir = self.object.compute_edge_loc_entire_cube(cube_state, off_x, off_y,
                                                                                          off_z)
        #Step 2: use the assignment and return the finger positions
        return locations[:,self.asign_arr[0]], locations[:,self.asign_arr[1]], locations[:,self.asign_arr[2]]

    def get_edge_directions(self,robot_state,cube_state):
        edge_dir = self.object.compute_edge_dir_entire_cube(cube_state)
        return edge_dir[:, self.asign_arr[0]], edge_dir[:, self.asign_arr[1]], edge_dir[:, self.asign_arr[2]]


    def get_center_array(self):
        return (np.asarray(self.cent_arr)>0.99)

class ThreeFingerGrasp(BasicGraspingPrimitive):
    def __init__(self, kinematics, params, object):
        super(ThreeFingerGrasp, self).__init__(kinematics, params, object)
        # This additional offset should potentially ensure to keep the third finger away,..
        self.excl_add_offset = 0.0


    def calc_grasp_position(self):
        '''
        This function might be used to calculate all the possible grasp locations if there are multiple to consider
        '''
        pass

    def assign_fingers(self,robot_state, cube_state, off_x=0.0, off_y =0.0, off_z=0.0):
        off_y +=  self.excl_add_offset
        '''
        This function should assign the fingers to where you want to grasp potnetially,...
        '''
        locations, vertical_axes_idx, axes_dir = self.object.compute_edge_loc_entire_cube(cube_state, off_x, off_y, off_z, compute_vertical=True)
        self.asign_arr, self.cent_arr = self.assign_target(locations, cube_state[3], robot_state[2][0], robot_state[2][1], robot_state[2][2])

    def assign_fingers_unrestricted(self,robot_state, cube_state, off_x=0.0, off_y =0.0, off_z=0.0):
        off_y +=  self.excl_add_offset
        '''
        This function should assign the fingers to where you want to grasp potnetially,...
        '''
        locations, vertical_axes_idx, axes_dir = self.object.compute_edge_loc_entire_cube(cube_state, off_x, off_y, off_z, compute_vertical=True)
        self.asign_arr, self.cent_arr = self.assign_target_unrestricted(locations, cube_state[3], robot_state[2][0], robot_state[2][1], robot_state[2][2])

    def assign_fingers_x(self,robot_state, cube_state, direction=0, off_x=0.0, off_y =0.0, off_z=0.0):
        off_y +=  self.excl_add_offset
        '''
        This function defines grasp around current x-axis
        '''
        locations, vertical_axes_idx, axes_dir = self.object.compute_edge_loc_entire_cube(cube_state, off_x, off_y, off_z, compute_vertical=True)
        self.asign_arr, self.cent_arr = self.assign_x(locations, robot_state[2][0], robot_state[2][1], robot_state[2][2], direction)

    def assign_fingers_y(self,robot_state, cube_state, direction=0, off_x=0.0, off_y =0.0, off_z=0.0):
        off_y +=  self.excl_add_offset
        '''
        This function defines grasp around current y-axis
        '''
        locations, vertical_axes_idx, axes_dir = self.object.compute_edge_loc_entire_cube(cube_state, off_x, off_y, off_z, compute_vertical=True)
        self.asign_arr, self.cent_arr = self.assign_y(locations, robot_state[2][0], robot_state[2][1], robot_state[2][2], direction)



    def get_finger_position(self,robot_state, cube_state, off_x=0.0, off_y =0.0, off_z=0.0, use_z=True):
        # use z variable has no effect here,...
        '''
        This function should return the desired positions of the fingers,...
        '''
        off_y +=  self.excl_add_offset

        #Step 1: calculate the locations:
        locations, vertical_axes_idx, axes_dir = self.object.compute_edge_loc_entire_cube(cube_state, off_x, off_y,
                                                                                          off_z)
        #Step 2: use the assignment and return the finger positions
        return locations[:,self.asign_arr[0]], locations[:,self.asign_arr[1]], locations[:,self.asign_arr[2]]

    def get_edge_directions(self,robot_state,cube_state):
        edge_dir = self.object.compute_edge_dir_entire_cube(cube_state)
        return edge_dir[:, self.asign_arr[0]], edge_dir[:, self.asign_arr[1]], edge_dir[:, self.asign_arr[2]]


    def get_center_array(self):
        return (np.asarray(self.cent_arr)>0.99)

    def get_axis(self,cube_state):
        '''
        should only return vertical axis
        '''

        #Step 1: calculate the locations:
        locations, vertical_axes_idx, axes_dir = self.object.compute_edge_loc_entire_cube(cube_state)
        return vertical_axes_idx, axes_dir


class ThreeFingerGraspExternal(ThreeFingerGrasp):
    def __init__(self, kinematics, params, object):
        super(ThreeFingerGraspExternal, self).__init__(kinematics, params, object)
        # This additional offset should potentially ensure to keep the third finger away,..
        self.excl_add_offset = 0.0

    def get_center_array(self):
        # in case of this grasp, do not treat any finger as the central one,...
        return [0,0,0]

    def get_axis(self,cube_state):
        '''
        should only return vertical axis
        '''

        # print ("THE get_axis FUNCTION IS NOT CORRECTLY IMPLEMENTED WITH THIS GRASP")
        # return 0

        # USE OLD FUNCIONALIYT:
        locations, vertical_axes_idx, axes_dir = self.object.compute_edge_loc_entire_cube(cube_state)
        return vertical_axes_idx, axes_dir

    def set_directions(self, robot_state, state):
        self.object.set_directions(robot_state, state)


    def get_finger_position(self,robot_state, cube_state, off_x=0.0, off_y =0.0, off_z=0.0, use_z=False):
        '''
        This function should return the desired positions of the fingers,...
        '''
        if (not use_z):
            off_z = 0.0
        off_y +=  self.excl_add_offset

        #Step 1: calculate the locations:
        pos1, pos2, pos3 = self.object.compute_edge_loc_special(cube_state, off_x, off_y,
                                                                                          off_z)
        #Step 2: use the assignment and return the finger positions
        return pos1, pos2, pos3

    def get_edge_directions(self,robot_state,cube_state):
        pos1, pos2, pos3 = self.object.compute_edge_dir_special(cube_state)
        return pos1, pos2, pos3


