import numpy as np
from scipy.spatial.transform import Rotation
import numpy as np
import pybullet as p


def todegree(w):
    return w*180/np.pi


def torad(w):
    return w*np.pi/180


def angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def add_one(index):
    if index+1 == 3:
        index_out = 0
    else:
        index_out = index+1
    return index_out


def to_H(R, T=np.zeros(3)):
    H = np.eye(4)
    H[:-1,:-1] = R
    H[:-1,-1] = T
    return H

def closest_axis_2_userdefined(H, vec):
    #print (H)
    #print (np.linalg.inv(H[:-1,:-1]))
    min_angle = 190
    x_des = np.array(vec)
    index = 0
    sign = 0
    reverse = False
    for i in range(3):
        x = H[:-1, i]
        theta = todegree(angle(x, x_des))
        #print (theta)
        if theta > 90:
            theta = theta - 180
            if theta ==0:
                reverse = True
        if min_angle > np.abs(theta):
            min_angle = np.abs(theta)
            index = i
            if theta == 0.:
                if reverse:
                    sign = -1
                else:
                    sign = 1
            else:
                sign = np.sign(theta)
    return min_angle, index, sign

def R_2vect(vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]
    if x==0 and y==0 and z==0:
        z=1

    # The rotation angle.
    angle = np.arccos(np.clip(np.dot(vector_orig, vector_fin),-1,1))

    # Trig functions (only need to do this maths once!).
    ca = np.cos(angle)
    sa = np.sin(angle)

    R = np.eye(4)
    # Calculate the rotation matrix elements.
    R[0, 0] = 1.0 + (1.0 - ca) * (x ** 2 - 1.0)
    R[0, 1] = -z * sa + (1.0 - ca) * x * y
    R[0, 2] = y * sa + (1.0 - ca) * x * z
    R[1, 0] = z * sa + (1.0 - ca) * x * y
    R[1, 1] = 1.0 + (1.0 - ca) * (y ** 2 - 1.0)
    R[1, 2] = -x * sa + (1.0 - ca) * y * z
    R[2, 0] = -y * sa + (1.0 - ca) * x * z
    R[2, 1] = x * sa + (1.0 - ca) * y * z
    R[2, 2] = 1.0 + (1.0 - ca) * (z ** 2 - 1.0)
    return R, axis, angle



class RotationPrimitives():

    def __init__(self, H0, Hg):
        self.H0 = H0
        self.Hg = Hg

    def set_goal(self, Hg):
        self.Hg = Hg

    def set_current_pose(self,H0):
        self.H0 = H0

    def get_control_seq(self, ax=None):
        ## Control Sequence will provide rotation vector and desired rotation to achieve target ##

        ################## Goal to Viapoint 2 ###################################
        theta, index, sign = self.closest_axis_2_normal(self.Hg)
        des_vec = np.array([0, 0, sign * 1])
        R, r_vec_via2gw, ang_via = self.R_2vect(self.Hg[:-1, index], des_vec)
        H_via2 = np.matmul(R, self.Hg)
        H_via2[:-1,-1] = self.Hg[:-1,-1]
        H_via2[2, -1] = 0.
        r_vec_via2g = np.matmul(H_via2[:-1,:-1].T, r_vec_via2gw)
        c2g = [r_vec_via2g, -ang_via, r_vec_via2gw]

        #########################################################################

        ############# From Floor to Viapoint 1 ####################
        index_H0, sign_H0 = self.find_index_z(self.H0)
        #theta_0 ,index_H0, sign_H0 = self.closest_axis_2_normal(self.H0)
        # print (index_H0, sign_H0, index, sign)
        # input ("WAIT")
        rot_index, ang_floor = self.find_rot_z(index_H0, sign_H0, index, sign)
        if rot_index is not None:
            r_vec_floor = np.zeros(3)
            r_vec_floor[rot_index] = 1
            rotation_floor = Rotation.from_rotvec(ang_floor * r_vec_floor)
            R_floor_1 = rotation_floor.as_matrix()
            R_floor_1 = to_H(R=R_floor_1)
            H_via1 = np.matmul(self.H0, R_floor_1)
            #H_via1[1,-1] = 0.3
        else:
            r_vec_floor = np.zeros(3)
            r_vec_floor[index] = 1
            ang_floor = 0.
            H_via1 = self.H0
        r_vec_floor_w = np.matmul(self.H0[:-1,:-1], r_vec_floor)
        c01 = [r_vec_floor, ang_floor, r_vec_floor_w]
        ####################################################

        ############ From Viapoint 1 to Viapoint 2 ################
        if index == 0:
            vec_1 = H_via1[:-1, 1]
            vec_2 = H_via2[:-1, 1]
        else:
            vec_1 = H_via1[:-1, 0]
            vec_2 = H_via2[:-1, 0]
        R12, r_vec_via12_p, ang_via12 = self.R_2vect(vec_1, vec_2)
        r_vec_via12w = np.zeros(3)
        r_vec_via12w[2] = np.sign(r_vec_via12_p[2])
        r_vec_via12 = np.matmul(H_via1[:-1,:-1].T, r_vec_via12w)
        c12 = [r_vec_via12, ang_via12, r_vec_via12w]
        ###########################################################

        ##### COMPUTE SHORTCUT: ########
        rot_12 = to_H(Rotation.from_rotvec(c12[0]*c12[1]).as_matrix())
        rot2g = to_H(Rotation.from_rotvec(c2g[0]*c2g[1]).as_matrix())
        rot1g = np.matmul(rot_12,rot2g)
        if np.allclose(rot1g, np.eye(4)):
            c1g = [np.array([0,0,1]), 0.]
        else:
            rot1g = Rotation.from_matrix(rot1g[:-1,:-1]).as_rotvec()
            c1g = [rot1g / np.linalg.norm(rot1g,ord=2), np.linalg.norm(rot1g,ord=2)]

        ##### Compute rotation from start to Via-2 ##
        R_via2 = H_via2[:-1,:-1]
        R_init = self.H0[:-1,:-1]
        R_to_2 = np.matmul(R_init.T, R_via2)
        if np.allclose(R_to_2, np.eye(3)):
            c_to_2 = [np.array([0, 0, 1]), 0.]
        else:
            rot_to_2 = Rotation.from_matrix(R_to_2).as_rotvec()
            c_to_2 = [rot_to_2 / np.linalg.norm(rot_to_2, ord=2), np.linalg.norm(rot_to_2, ord=2)]

        ##### Compute rotation from start to Goal ###
        R_g = self.Hg[:-1, :-1]
        R_init = self.H0[:-1, :-1]
        R_to_g = np.matmul(R_init.T, R_g)
        if np.allclose(R_to_g, np.eye(3)):
            c_to_g = [np.array([0, 0, 1]), 0.]
        else:
            rot_to_g = Rotation.from_matrix(R_to_g).as_rotvec()
            c_to_g = [rot_to_g / np.linalg.norm(rot_to_g, ord=2), np.linalg.norm(rot_to_g, ord=2)]

        command_seq = [c01, c12,c2g]
        return command_seq, [c1g], [c_to_2, c_to_g]

    def find_index_z(self, H):
        big_Z = 0.0
        index = 0
        for i in range(3):
            z = H[2, i]
            # print(z)
            if np.abs(z) > big_Z:
                big_Z = np.abs(z)
                sign = np.sign(z)
                index = i
        return index, sign

    def find_rot_z(self, index_H0, sign_H0, index, sign):
        if index == index_H0:
            if sign == sign_H0:
                return None, None
            else:
                angle = np.pi
                if index == 0:
                    rot_over = 1
                else:
                    rot_over = 0
                return rot_over, angle
        else:
            rot_over = 0
            while (rot_over == index or rot_over == index_H0):
                rot_over += 1
            if sign == sign_H0:
                angle = -np.pi / 2
                if add_one(rot_over) != index_H0:
                    angle = -angle
            else:
                angle = np.pi / 2
                if add_one(rot_over) != index_H0:
                    angle = -angle
            return rot_over, angle

    def closest_axis_2_normal(self, H):
        # print (H)
        # print (np.linalg.inv(H[:-1,:-1]))
        min_angle = 190
        x_des = np.array([0, 0, 1])
        index = 0
        sign = 0
        reverse = False
        for i in range(3):
            x = H[:-1, i]
            theta = todegree(angle(x, x_des))
            # print (theta)
            if theta > 90:
                theta = theta - 180
                if theta ==0:
                    reverse = True
            if min_angle > np.abs(theta):
                min_angle = np.abs(theta)
                index = i
                if theta == 0.:
                    if reverse:
                        sign = -1
                    else:
                        sign = 1
                else:
                    sign = np.sign(theta)
        return min_angle, index, sign


    def R_2vect(self, vector_orig, vector_fin):
        """Calculate the rotation matrix required to rotate from one vector to another.
        For the rotation of one vector to another, there are an infinit series of rotation matrices
        possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
        plane between the two vectors.  Hence the axis-angle convention will be used to construct the
        matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
        angle is the arccosine of the dot product of the two unit vectors.
        Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
        the rotation matrix R is::
                  |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
            R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
                  | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
        @param R:           The 3x3 rotation matrix to update.
        @type R:            3x3 numpy array
        @param vector_orig: The unrotated vector defined in the reference frame.
        @type vector_orig:  numpy array, len 3
        @param vector_fin:  The rotated vector defined in the reference frame.
        @type vector_fin:   numpy array, len 3
        """

        # Convert the vectors to unit vectors.
        vector_orig = vector_orig / np.linalg.norm(vector_orig)
        vector_fin = vector_fin / np.linalg.norm(vector_fin)

        # The rotation axis (normalised).
        axis = np.cross(vector_orig, vector_fin)
        axis_len = np.linalg.norm(axis)
        if axis_len != 0.0:
            axis = axis / axis_len

        # Alias the axis coordinates.
        x = axis[0]
        y = axis[1]
        z = axis[2]
        if x==0 and y==0 and z==0:
            z=1

        # The rotation angle.
        angle = np.arccos(np.clip(np.dot(vector_orig, vector_fin),-1,1))

        # Trig functions (only need to do this maths once!).
        ca = np.cos(angle)
        sa = np.sin(angle)

        R = np.eye(4)
        # Calculate the rotation matrix elements.
        R[0, 0] = 1.0 + (1.0 - ca) * (x ** 2 - 1.0)
        R[0, 1] = -z * sa + (1.0 - ca) * x * y
        R[0, 2] = y * sa + (1.0 - ca) * x * z
        R[1, 0] = z * sa + (1.0 - ca) * x * y
        R[1, 1] = 1.0 + (1.0 - ca) * (y ** 2 - 1.0)
        R[1, 2] = -x * sa + (1.0 - ca) * y * z
        R[2, 0] = -y * sa + (1.0 - ca) * x * z
        R[2, 1] = x * sa + (1.0 - ca) * y * z
        R[2, 2] = 1.0 + (1.0 - ca) * (z ** 2 - 1.0)
        return R, axis, angle


def calculate_mutltiple_goals(init_information, obs):
    goal_orients = []

    # This first calcuation step allows to calculate a viapoint! I.e. it yields a goal orientation for some axis alignment
    init_orient = np.zeros(3)
    init_orient[:2] = np.asarray(init_information[:2])
    init_orient = init_orient / np.linalg.norm(init_orient)
    current_orient = np.asarray(p.getMatrixFromQuaternion(obs["object_orientation"])).reshape(3, 3)
    theta, index, sign = closest_axis_2_userdefined(to_H(current_orient), init_orient)
    des_vec = sign * np.array(init_orient)
    Rot1, r_vec_via2gw, ang_via = R_2vect(current_orient[:, index], des_vec)
    first_goal = np.matmul(Rot1[:-1, :-1], current_orient)
    first_goal = Rotation.from_matrix(first_goal)
    goal_orients.append([first_goal.as_quat()+[0.001,0.001,0.001,0.001],[0,0,0.0325]]) #addition of noise needed since otherwise problems code,...

    # this second calculation applies the relative transformation which is desired based on the current observation!!!
    # now take into account the desired rotation from the target information:
    des_rotation = np.asarray(p.getMatrixFromQuaternion(init_information[10:14])).reshape(3, 3)
    init_orient = np.asarray([1,0,0])
    theta, index, sign = closest_axis_2_userdefined(
        to_H(current_orient), init_orient)
    des_vec = sign * np.array(init_orient)
    Rot1, r_vec_via2gw, ang_via = R_2vect(current_orient[:, index], des_vec)
    second_goal = np.matmul(Rot1[:-1, :-1], current_orient)
    # now apply rotation:
    second_goal = np.matmul(des_rotation, second_goal)
    # now rotate back to orientation that we are now at:
    second_goal = np.matmul(Rot1[:-1, :-1].T, second_goal)
    second_goal = Rotation.from_matrix(second_goal)
    goal_orients.append([second_goal.as_quat()+[0.001,0.001,0.001,0.001],[init_information[7],init_information[8],init_information[9]]]) #addition of noise needed since otherwise problems code,...

    return goal_orients