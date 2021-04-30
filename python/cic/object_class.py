import numpy as np
from .math_tools import rpy2Mat
import pybullet


class Object:
    def __init__(self):
        self._CUBE_WIDTH_x = None
        self._CUBE_WIDTH_y = None
        self._CUBE_WIDTH_z = None

    def faces_xyz(self, cube_state):
        ## Check Axis pointing Up and get the angle ##
        rot = np.asarray(rpy2Mat(cube_state[2][0], cube_state[2][1], cube_state[2][2])).reshape(3, 3)

        face = np.eye(3)*np.array([self._CUBE_WIDTH_x, self._CUBE_WIDTH_y, self._CUBE_WIDTH_z])
        faces_in_cube = np.concatenate((face, -face),1)
        faces_in_world = np.matmul(rot, faces_in_cube) + np.tile(cube_state[0][:,None],[1,6])

        return faces_in_world

    def offseted_faces_xyz(self, cube_state, offset):
        ## Check Axis pointing Up and get the angle ##
        rot = np.asarray(rpy2Mat(cube_state[2][0], cube_state[2][1], cube_state[2][2])).reshape(3, 3)

        face = np.eye(3)*np.array([self._CUBE_WIDTH_x + offset, self._CUBE_WIDTH_y+offset, self._CUBE_WIDTH_z+offset])
        faces_in_cube = np.concatenate((face, -face),1)
        faces_in_world = np.matmul(faces_in_cube.T, rot).T + np.tile(cube_state[0][:,None],[1,6])

        return faces_in_world

    def in_cube(self, cube_state, pos_w):
        rot_c2w = np.asarray(rpy2Mat(cube_state[2][0], cube_state[2][1], cube_state[2][2])).reshape(3, 3)

        pos_c = np.matmul(pos_w.T - cube_state[0], rot_c2w.T).T
        return pos_c

    def in_world(self, cube_state, pos_c, offset_prop = False ):
        rot_c2w = np.asarray(rpy2Mat(cube_state[2][0], cube_state[2][1], cube_state[2][2])).reshape(3, 3)

        if offset_prop:
            pos_c = pos_c * np.array([[self._CUBE_WIDTH_x , self._CUBE_WIDTH_y, self._CUBE_WIDTH_z]]).T

        #print('position C:', pos_c)
        pos_w = np.matmul(pos_c.T, rot_c2w).T + np.tile(cube_state[0][:,None],[1,pos_c.shape[1]])
        return pos_w




class Cuboid(Object):
    def __init__(self):
        super(Cube, self).__init__()
        self._CUBE_WIDTH_x = 0.01
        self._CUBE_WIDTH_y = 0.07
        self._CUBE_WIDTH_z = 0.01

    def compute_edge_loc_entire_cube(self, cube_state, off_x=0.0, off_y=0.0, off_z=0.0, compute_vertical=False):

        # add potential offset to true cube locations:
        cube_width_x = self._CUBE_WIDTH_x+off_x
        cube_width_y = self._CUBE_WIDTH_y+off_y
        cube_width_z = self._CUBE_WIDTH_z   # z-offset is added later


        rot_p0 = np.asarray(rpy2Mat(0.0,0.0,cube_state[2][2])).reshape(3,3)

        side_points = np.zeros((3, 6))
        side_points[:, 0] = cube_state[0]
        side_points[:, 1] = cube_state[0]
        side_points[:, 2] = cube_state[0]
        side_points[:, 3] = cube_state[0]
        side_points[:, 4] = cube_state[0]
        side_points[:, 5] = cube_state[0]

        # manipulation matrix calculates the sides,..
        manip_mat = np.zeros((3, 6))
        # careful this here changes up things:
        # first x direction:
        manip_mat[0, 0] += cube_width_x / 2.0
        manip_mat[0, 1] -= cube_width_x / 2.0
        manip_mat[1, 2] += cube_width_y / 2.0
        manip_mat[1, 3] -= cube_width_y / 2.0
        manip_mat[2, 4] += cube_width_z / 2.0
        manip_mat[2, 5] -= cube_width_z / 2.0
        manip_mat = np.matmul(rot_p0, manip_mat)

        edge_rot = side_points + manip_mat
        edge_rot[2, :] = edge_rot[2, :] + off_z

        if (compute_vertical):
            x_axis = np.abs(manip_mat[2,0]-manip_mat[2,1])
            y_axis = np.abs(manip_mat[2,2]-manip_mat[2,3])
            z_axis = np.abs(manip_mat[2,4]-manip_mat[2,5])
            axes_info = np.asarray([x_axis,y_axis,z_axis])
            self.vertical_axes_idx = np.argmax(axes_info)
        self.vertical_axes_idx = 2

        x_axis_dir = manip_mat[:,0]-manip_mat[:,1]
        y_axis_dir = manip_mat[:,2]-manip_mat[:,3]
        z_axis_dir = manip_mat[:,4]-manip_mat[:,5]

        axes_dir = np.zeros((3, 3))
        axes_dir[0,:] = x_axis_dir[:]
        axes_dir[1,:] = y_axis_dir[:]
        axes_dir[2,:] = z_axis_dir[:]

        count = 0
        edge_rot2 = np.zeros((3, 4))
        for i in range(3):
            if not(i==self.vertical_axes_idx):
                edge_rot2[:,count] = edge_rot[:,2*i]
                edge_rot2[:,count+2] = edge_rot[:,2*i+1]
                count += 1

        return edge_rot2, self.vertical_axes_idx, axes_dir

    def compute_edge_dir_entire_cube(self, cube_state):
        # manipulation matrix calculates the sides,..
        rot_p0 = np.asarray(rpy2Mat(0.0,0.0,cube_state[2][2]).reshape(3,3))
        manip_mat = np.zeros((3, 6))
        # careful this here changes up things:
        # first x direction:
        manip_mat[0, 0] += 1.0
        manip_mat[0, 1] -= 1.0
        manip_mat[1, 2] += 1.0
        manip_mat[1, 3] -= 1.0
        manip_mat[2, 4] += 1.0
        manip_mat[2, 5] -= 1.0
        manip_mat = np.matmul(rot_p0, manip_mat)

        count = 0
        manip_mat2 = np.zeros((3, 4))
        self.vertical_axes_idx = 2
        for i in range(3):
            if not(i==self.vertical_axes_idx):
                manip_mat2[:,count] = manip_mat[:,2*i]
                manip_mat2[:,count+2] = manip_mat[:,2*i+1]
                count += 1

        return manip_mat2

class Cube(Object):
    def __init__(self):
        super(Cube, self).__init__()
        # TODO: find out what is appropriate here (maybe even use pybullet,...)
        self._CUBE_WIDTH_x = 0.065#0.025
        self._CUBE_WIDTH_y = 0.065#0.025
        self._CUBE_WIDTH_z = 0.065#0.025

    def compute_edge_loc_entire_cube(self, cube_state, off_x=0.0, off_y=0.0, off_z=0.0, compute_vertical=False):

        # add potential offset to true cube locations:
        cube_width_x = self._CUBE_WIDTH_x+off_x
        cube_width_y = self._CUBE_WIDTH_y+off_y
        # TODO: implement this nicer (this is now added cause maybe also the cube is flipped,...) and the z offset is applied globally,...
        cube_width_z = self._CUBE_WIDTH_z+off_x   # z-offset is added later


        #rot_p0 = np.asarray(rpy2Mat(0.0,0.0,cube_state[2][2])).reshape(3,3)
        #rot_p0 = np.asarray(rpy2Mat(cube_state[2][0],cube_state[2][1],cube_state[2][2])).reshape(3,3)
        rot_p0 = np.asarray(pybullet.getMatrixFromQuaternion(cube_state[1])).reshape(3, 3)

        side_points = np.zeros((3, 6))
        side_points[:, 0] = cube_state[0]
        side_points[:, 1] = cube_state[0]
        side_points[:, 2] = cube_state[0]
        side_points[:, 3] = cube_state[0]
        side_points[:, 4] = cube_state[0]
        side_points[:, 5] = cube_state[0]

        # manipulation matrix calculates the sides,..
        manip_mat = np.zeros((3, 6))
        # careful this here changes up things:
        # first x direction:
        manip_mat[0, 0] += cube_width_x / 2.0
        manip_mat[0, 1] -= cube_width_x / 2.0
        manip_mat[1, 2] += cube_width_y / 2.0
        manip_mat[1, 3] -= cube_width_y / 2.0
        manip_mat[2, 4] += cube_width_z / 2.0
        manip_mat[2, 5] -= cube_width_z / 2.0
        manip_mat = np.matmul(rot_p0, manip_mat)

        edge_rot = side_points + manip_mat
        edge_rot[2, :] = edge_rot[2, :] + off_z

        if (compute_vertical):
            x_axis = np.abs(manip_mat[2,0]-manip_mat[2,1])
            y_axis = np.abs(manip_mat[2,2]-manip_mat[2,3])
            z_axis = np.abs(manip_mat[2,4]-manip_mat[2,5])
            axes_info = np.asarray([x_axis,y_axis,z_axis])
            self.vertical_axes_idx = np.argmax(axes_info)
        #self.vertical_axes_idx = 2

        x_axis_dir = manip_mat[:,0]-manip_mat[:,1]
        y_axis_dir = manip_mat[:,2]-manip_mat[:,3]
        z_axis_dir = manip_mat[:,4]-manip_mat[:,5]

        axes_dir = np.zeros((3, 3))
        axes_dir[0,:] = x_axis_dir[:]
        axes_dir[1,:] = y_axis_dir[:]
        axes_dir[2,:] = z_axis_dir[:]

        count = 0
        edge_rot2 = np.zeros((3, 4))
        for i in range(3):
            if not(i==self.vertical_axes_idx):
                edge_rot2[:,count] = edge_rot[:,2*i]
                edge_rot2[:,count+2] = edge_rot[:,2*i+1]
                count += 1

        return edge_rot2, self.vertical_axes_idx, axes_dir

    def compute_edge_dir_entire_cube(self, cube_state):
        # manipulation matrix calculates the sides,..
        #rot_p0 = np.asarray(rpy2Mat(0.0,0.0,cube_state[2][2]).reshape(3,3))
        #rot_p0 = np.asarray(rpy2Mat(cube_state[2][0],cube_state[2][1],cube_state[2][2])).reshape(3,3)
        rot_p0 = np.asarray(pybullet.getMatrixFromQuaternion(cube_state[1])).reshape(3, 3)
        manip_mat = np.zeros((3, 6))
        # careful this here changes up things:
        # first x direction:
        manip_mat[0, 0] += 1.0
        manip_mat[0, 1] -= 1.0
        manip_mat[1, 2] += 1.0
        manip_mat[1, 3] -= 1.0
        manip_mat[2, 4] += 1.0
        manip_mat[2, 5] -= 1.0
        manip_mat = np.matmul(rot_p0, manip_mat)

        count = 0
        manip_mat2 = np.zeros((3, 4))
        #self.vertical_axes_idx = 2
        for i in range(3):
            if not(i==self.vertical_axes_idx):
                manip_mat2[:,count] = manip_mat[:,2*i]
                manip_mat2[:,count+2] = manip_mat[:,2*i+1]
                count += 1

        return manip_mat2

    def set_directions(self, robot_state, cube_state):
        tip_1 = robot_state[2][0]
        tip_2 = robot_state[2][1]
        tip_3 = robot_state[2][2]

        rot_p0 = np.asarray(pybullet.getMatrixFromQuaternion(cube_state[1])).reshape(3, 3)
        cube_center = cube_state[0]

        dir_1 = tip_1 - cube_center
        dir_2 = tip_2 - cube_center
        dir_3 = tip_3 - cube_center

        self.dir1_unnormalized = np.matmul(rot_p0.T,dir_1)
        factor = np.max(np.abs(self.dir1_unnormalized))/ (0.065/2.0)
        self.dir1_unnormalized = self.dir1_unnormalized / factor

        self.dir2_unnormalized = np.matmul(rot_p0.T,dir_2)
        factor = np.max(np.abs(self.dir2_unnormalized))/ (0.065/2.0)
        self.dir2_unnormalized = self.dir2_unnormalized / factor

        self.dir3_unnormalized = np.matmul(rot_p0.T,dir_3)
        factor = np.max(np.abs(self.dir3_unnormalized))/ (0.065/2.0)
        self.dir3_unnormalized = self.dir3_unnormalized / factor

        import copy
        self.edge_dir1 = copy.deepcopy(self.dir1_unnormalized)
        self.edge_dir1 = self.edge_dir1 / np.linalg.norm(self.edge_dir1)
        self.edge_dir1[np.argmax(np.abs(self.edge_dir1))] = np.sign(self.edge_dir1[np.argmax(np.abs(self.edge_dir1))])
        self.edge_dir1[np.abs(self.edge_dir1)<0.9] = 0.0

        self.edge_dir2 = copy.deepcopy(self.dir2_unnormalized)
        self.edge_dir2 = self.edge_dir2 / np.linalg.norm(self.edge_dir2)
        self.edge_dir2[np.argmax(np.abs(self.edge_dir2))] = np.sign(self.edge_dir2[np.argmax(np.abs(self.edge_dir2))])
        self.edge_dir2[np.abs(self.edge_dir2)<0.9] = 0.0

        self.edge_dir3 = copy.deepcopy(self.dir3_unnormalized)
        self.edge_dir3 = self.edge_dir3 / np.linalg.norm(self.edge_dir3)
        self.edge_dir3[np.argmax(np.abs(self.edge_dir3))] = np.sign(self.edge_dir3[np.argmax(np.abs(self.edge_dir3))])
        self.edge_dir3[np.abs(self.edge_dir3)<0.9] = 0.0

        dir_1 = dir_1 / np.linalg.norm(dir_1)
        dir_2 = dir_2 / np.linalg.norm(dir_2)
        dir_3 = dir_3 / np.linalg.norm(dir_3)

        self.dir1 = np.matmul(rot_p0.T,dir_1)
        self.dir2 = np.matmul(rot_p0.T,dir_2)
        self.dir3 = np.matmul(rot_p0.T,dir_3)

    def compute_edge_dir_special(self, cube_state):
        # manipulation matrix calculates the sides,..
        #rot_p0 = np.asarray(rpy2Mat(0.0,0.0,cube_state[2][2]).reshape(3,3))
        #rot_p0 = np.asarray(rpy2Mat(cube_state[2][0],cube_state[2][1],cube_state[2][2])).reshape(3,3)
        rot_p0 = np.asarray(pybullet.getMatrixFromQuaternion(cube_state[1])).reshape(3, 3)

        # dir1 = np.matmul(rot_p0, self.dir1)
        # dir2 = np.matmul(rot_p0, self.dir2)
        # dir3 = np.matmul(rot_p0, self.dir3)
        dir1 = np.matmul(rot_p0, self.edge_dir1)
        dir2 = np.matmul(rot_p0, self.edge_dir2)
        dir3 = np.matmul(rot_p0, self.edge_dir3)

        return dir1, dir2, dir3


    def compute_edge_loc_special(self, cube_state, off_x=0.0, off_y=0.0, off_z=0.0, compute_vertical=False):

        # add potential offset to true cube locations:
        cube_offset_x = off_x
        cube_offset_y = off_y
        # # TODO: implement this nicer (this is now added cause maybe also the cube is flipped,...) and the z offset is applied globally,...
        # cube_width_z = self._CUBE_WIDTH_z+off_x   # z-offset is added later

        pos1 = self.dir1_unnormalized + self.dir1*cube_offset_x
        pos2 = self.dir2_unnormalized + self.dir2*cube_offset_x
        pos3 = self.dir3_unnormalized + self.dir3*cube_offset_x

        rot_p0 = np.asarray(pybullet.getMatrixFromQuaternion(cube_state[1])).reshape(3, 3)

        pos1 = np.matmul(rot_p0,pos1) + cube_state[0] + np.asarray([0,0,off_z])
        pos2 = np.matmul(rot_p0,pos2) + cube_state[0] + np.asarray([0,0,off_z])
        pos3 = np.matmul(rot_p0,pos3) + cube_state[0] + np.asarray([0,0,off_z])

        return pos1, pos2, pos3









