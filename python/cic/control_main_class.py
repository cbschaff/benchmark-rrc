import numpy as np
import copy
import pinocchio as pin
import os

from numpy.linalg import norm, solve
import numpy.linalg as la

'''
This file contains all the relevant low-level control algorithms and functionality
-> mapping from desired positions in joint or cartesian space to control actions,...
'''

class TriFingerController:
    def __init__(self, env):
        self.env = env
        self.tri_finger_filename = "trifinger_mod.urdf"

        self.model = pin.buildModelFromUrdf(os.path.join(os.path.dirname(__file__), self.tri_finger_filename))
        self.data = self.model.createData()

        self.kinematics = TriFingerKin(self.model, self.data, self.env)
        self.params = ModelParams()

        self.robot_state = None

        self.end_effectors_id = [4,8,12]

    def pos_ctrl(self, q, v, q_des):
        # this function implements the conversion from a desired configuration to torques
        # position signal:
        print('position control', q)
        res_torque = np.array([6.0, 6.0, 6.0]) * (q_des - q)
        res_torque = res_torque - 1 * np.array([0.1, 0.1, 0.1]) * v
        return res_torque

    def position_control_fingers(self, model, data, q, v, location, threshold):

        # function drives fingers to desired location
        count = np.zeros((3,))
        ac = np.zeros((9,))

        JOINT_ID = [4, 8, 12]
        for i in range(3):
            dist = np.sqrt(np.sum(np.square(data.oMi[JOINT_ID[i]].translation - location[:, i])))

            q_des = self.kinematics.inverse_kinematics(model, data, copy.deepcopy(q), copy.deepcopy(location[:, i]), JOINT_ID[i])
            ac[i*3:(i+1)*3] = copy.deepcopy(self.pos_ctrl(q[i*3:(i+1)*3], v[i*3:(i+1)*3], q_des[i*3:(i+1)*3]))

        ac = np.clip(ac, -0.36, 0.36)
        return ac, count

    def cartesian_position_control(self, model, data, q, v, location, threshold):
        q = self.kinematics.trafo_q(q)
        v = self.kinematics.trafo_q(v)

        ## Compute Jacobian ##
        _JAC = pin.computeJointJacobians(self.model, self.data, q)
        jacobian_list = []
        for idx in self.end_effectors_id:
            #pin.forwardKinematics(self.model, self.data, self.trafo_q(q))
            jacobian = pin.getJointJacobian(self.model, self.data, idx, pin.ReferenceFrame.WORLD, )[:3, :]
            jacobian_list.append(jacobian)
            #print('index is ',idx,'with jacobian: ',jacobian)
        ######################

        ## Get the Cartesian fingers Pose ##
        # print('state robot is : ', self.robot_state[2])
        # print('stacked robot state:', np.stack(self.robot_state[2], axis=1))
        # print('goal state is:', self.goal_target)

        fingers_xyz = np.stack(self.robot_state[2], axis=1)
        xyz_error = self.goal_target - fingers_xyz
        #print('Error in XYZ is:', xyz_error)
        ####################################
        ac = np.zeros((9,))
        JOINTS = [[0,3], [4,7], [8,11]]
        for i in range(3):
            jac = jacobian_list[i]
            jac = np.linalg.pinv(jac)
            er  =  xyz_error[:, i]
            q_er = jac @ er
            ac[i*3:(i+1)*3] = q_er[JOINTS[i][0]: JOINTS[i][1]] -  np.array([0.1, 0.1, 0.1]) * v[self.end_effectors_id[i]-4:self.end_effectors_id[i]-1]

        ac = np.clip(ac, -0.36, 0.36)
        return ac






class TriFingerKin():
    def __init__(self, model, data, env):
        self.model = model
        self.data = data
        self.env = env

    def get_tip_pos_pinnochio(self,q):
        q = self.trafo_q(q)
        pin.forwardKinematics(self.model, self.data, q)

        return [self.data.oMi[4].translation,self.data.oMi[8].translation,self.data.oMi[12].translation]

    def compute_gravity_compensation(self,robot_state,torque):
        pin.forwardKinematics(self.model, self.data, self.trafo_q(robot_state[0]))
        b = pin.rnea(self.model, self.data, self.trafo_q(robot_state[0]),
                     self.trafo_q(robot_state[1]), np.zeros(12))
        add = self.inv_trafo_q(b)
        torque += add
        torque = np.clip(torque, -0.36, 0.36)
        return torque

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


    def inverse_kinematics(self, q, x_des, joint_id):

        if (joint_id==4):
            joint_id = 0
        elif (joint_id==8):
            joint_id = 1
        elif(joint_id==12):
            joint_id = 2

        q0 = self.env.pinocchio_utils.inverse_kinematics(joint_id, x_des, q)
        if (q0 is not None):
            return q0

        q = self.trafo_q(q)
        eps = 1e-3
        DT = 0.04
        damp = 1e-6
        JOINT_ID = joint_id  # 3
        i = 0
        pin.forwardKinematics(self.model, self.data, q)

        while True:
            pin.forwardKinematics(self.model, self.data, q)

            x = self.data.oMi[JOINT_ID].translation
            R = self.data.oMi[JOINT_ID].rotation

            a = (np.reshape((x - x_des), (3, 1)))
            err = np.matmul(R.T, a)

            if norm(err) < eps:
                success = True
                return self.inv_trafo_q(q)
                # return q
            J = pin.computeJointJacobian(self.model, self.data, q, JOINT_ID)[:3, :]
            v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(3), err))
            q = pin.integrate(self.model, q, v * DT)

            i += 1
            if (i > 1000):
                # return result latest after 1000 iterations:
                return self.inv_trafo_q(q)
                # return q
        return q

    def inverse_kinematics_3_fingers(self, q, fing1_des, fing2_des, fing3_des):

        des_joints = np.zeros((9))
        des_joints[:3] = copy.deepcopy(self.inverse_kinematics(copy.deepcopy(q), fing1_des, 4))[:3]
        des_joints[3:6] = copy.deepcopy(self.inverse_kinematics(copy.deepcopy(q), fing2_des, 8))[3:6]
        des_joints[6:9] = copy.deepcopy(self.inverse_kinematics(copy.deepcopy(q), fing3_des, 12))[6:9]

        return des_joints

    def forward_kinematics(self, q):
        pass

    def jacobian(self, q):
        pass

    def inverse_jacobian(self, q):
        pass

    def imp_ctrl(self, q, force_dir, joint_id, vel, pos, v, K_xy, K_z):

        q = self.trafo_q(q)
        v = self.trafo_q(v)

        m = 1.0

        d_xy = np.sqrt(4 * m * 500)
        d_z = np.sqrt(4 * m * 500)
        M = np.eye(3) * m
        M_inv = np.linalg.pinv(M)
        D = np.eye(3)
        D[0,0] = D[1,1] = d_xy
        D[2,2] = d_z
        K = np.eye(3)

        K[0,0] = K_xy
        K[1, 1] = K_xy
        K[2, 2] = K_z

        F = 0.01#0.05
        M_joint = pin.crba(self.model, self.data, q)

        pos_gain = np.matmul(K, pos)
        damp_gain = np.matmul(D, vel)
        # damp_gain = 0.0
        force_gain = F*force_dir

        acc = np.matmul(M_inv, (force_gain - pos_gain - damp_gain))

        # TODO: I think it is sufficient if this is calculated only once:
        full_J_var = pin.computeJointJacobiansTimeVariation(self.model,self.data,q,v)
        J_var_frame = pin.getJointJacobianTimeVariation(self.model, self.data, joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:]
        dyn_comp = np.matmul(J_var_frame,v)

        acc = acc - dyn_comp

        # TODO: I think it is sufficient if this is calculated only once:
        full_JAC = pin.computeJointJacobians(self.model,self.data,q)
        J_NEW = pin.getJointJacobian(self.model,self.data,joint_id,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        J = J_NEW

        J_inv = np.matmul(np.linalg.pinv(np.matmul(J.T,J)),J.T)
        # print (J_inv)
        torque = np.matmul(M_joint,np.matmul(J_inv,acc))

        return self.inv_trafo_q(torque)

    def imp_ctrl_3_fingers(self, q, force_dir, joint_id, vel, pos, v, K_xy, K_z,center_arr):
        ac = np.zeros(9)
        for i in range(3):
            if not(center_arr[i]):
                ac[i*3:(i+1)*3] = copy.deepcopy(self.imp_ctrl(q[i],force_dir[i],joint_id[i],vel[i],pos[i],v[i],K_xy[i],K_z[i]))[i*3:(i+1)*3]
        return ac

    def comp_combined_force(self, q, des_force, des_torque, in_touch):

        if (np.sum(in_touch)==0):
            return np.zeros((9)), np.zeros((9))

        q = self.trafo_q(q)

        # TODO: I think it is sufficient if this is calculated only once:
        full_JAC = pin.computeJointJacobians(self.model, self.data, q)
        J_trans_inv_mat = np.zeros((3,3*12))

        for i in range(3):
            if (in_touch[i]==1):
                idx = i * 4 + 4
                idx_low = idx-4
                J_NEW = pin.getJointJacobian(self.model, self.data, idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                J_trans = J_NEW.T
                J_trans_inv = np.matmul(np.linalg.pinv(np.matmul(J_trans.T, J_trans)), J_trans.T)
                J_trans_inv_mat[:,idx_low*3:idx*3] = J_trans_inv

        add_torque_force = np.matmul(np.matmul(np.linalg.pinv(np.matmul(J_trans_inv_mat.T, J_trans_inv_mat)), J_trans_inv_mat.T),des_force)
        add_torque_force1 = self.inv_trafo_q(add_torque_force[:12])
        add_torque_force2 = self.inv_trafo_q(add_torque_force[12:24])
        add_torque_force3 = self.inv_trafo_q(add_torque_force[24:36])

        return (add_torque_force1+add_torque_force2+add_torque_force3), np.zeros((9))

    def comp_combined_torque(self, q, des_force, des_torque, in_touch, center, p1, p2, p3):

        P = [p1,p2,p3]

        # only rotate when completely in touch!
        if (np.sum(in_touch)!=3):
            return np.zeros((9)), np.zeros((9))

        q = self.trafo_q(q)

        # TODO: I think it is sufficient if this is calculated only once:
        full_JAC = pin.computeJointJacobians(self.model, self.data, q)
        J_trans_inv_mat = np.zeros((3,3*12))

        for i in range(3):
            if (in_touch[i]==1):
                idx = i * 4 + 4
                idx_low = idx-4
                J_NEW = pin.getJointJacobian(self.model, self.data, idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                J_trans = J_NEW.T
                J_trans_inv = np.matmul(np.linalg.pinv(np.matmul(J_trans.T, J_trans)), J_trans.T)
                # print (np.shape(J_trans_inv))
                J_trans_inv = np.matmul(self.skew(P[i]-center),J_trans_inv)
                # print(np.shape(J_trans_inv))
                # input("STOOP")
                J_trans_inv_mat[:,idx_low*3:idx*3] = J_trans_inv


        add_torque_force = np.matmul(np.matmul(np.linalg.pinv(np.matmul(J_trans_inv_mat.T, J_trans_inv_mat)), J_trans_inv_mat.T),des_torque)
        add_torque_force1 = self.inv_trafo_q(add_torque_force[:12])
        add_torque_force2 = self.inv_trafo_q(add_torque_force[12:24])
        add_torque_force3 = self.inv_trafo_q(add_torque_force[24:36])
        # print (J_trans_inv_mat)
        # input ("WAIT....")
        # net_force = net_force + np.matmul(J_trans_inv, torque)

        # print (net_force)
        # input ("NET FORCE")
        return np.zeros((9)), (add_torque_force1+add_torque_force2+add_torque_force3)


    def comp_effective_force(self, q, torque, in_touch):

        q = self.trafo_q(q)
        torque = self.trafo_q(torque)

        net_force = np.zeros((3))

        full_JAC = pin.computeJointJacobians(self.model,self.data,q)
        for i in range(3):
            if (in_touch[i]==1):
                idx = i*4+4
                J_NEW = pin.getJointJacobian(self.model,self.data,idx,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                J_trans = J_NEW.T
                J_trans_inv = np.matmul(np.linalg.pinv(np.matmul(J_trans.T,J_trans)),J_trans.T)
                net_force = net_force + np.matmul(J_trans_inv,torque)

        # print (net_force)
        # input ("NET FORCE")
        return net_force

    def add_additional_force_3_fingers(self,q,force_factor,edge_dir,center,correct_torque=False):
        in_touch = [int(not (center[0])), int(not (center[1])), int(not (center[2]))]
        if not (center[0]):
            add_torque_force1, zolo = self.comp_combined_force(copy.deepcopy(q),
                                                               -force_factor * edge_dir[0],
                                                               [0, 0, 0],
                                                               [1, 0, 0])
        else:
            add_torque_force1 = np.zeros((9))
        if not (center[1]):
            add_torque_force2, zolo = self.comp_combined_force(copy.deepcopy(q),
                                                               -force_factor * edge_dir[1],
                                                               [0, 0, 0],
                                                               [0, 1, 0])
        else:
            add_torque_force2 = np.zeros((9))
        if not (center[2]):
            add_torque_force3, zolo = self.comp_combined_force(copy.deepcopy(q),
                                                               -force_factor * edge_dir[2],
                                                               [0, 0, 0],
                                                               [0, 0, 1])
        else:
            add_torque_force3 = np.zeros((9))

        add_torque_through_normal = add_torque_force1 + add_torque_force2 + add_torque_force3

        if not(correct_torque):
            return add_torque_through_normal

        net_force = self.comp_effective_force(copy.deepcopy(q),
                                              add_torque_through_normal,
                                              in_touch)
        add_torque_froce, add_torque_torque = self.comp_combined_force(copy.deepcopy(q),
                                                                       -net_force, [0, 0, 0], in_touch)
        return add_torque_through_normal + add_torque_froce

    def add_additional_force_3_fingers_center_gain(self,q,force_factor,force_factor_center,edge_dir,center,correct_torque=False):
        in_touch = [int(not (center[0])), int(not (center[1])), int(not (center[2]))]
        if not (center[0]):
            add_torque_force1, zolo = self.comp_combined_force(copy.deepcopy(q),
                                                               -force_factor * edge_dir[0],
                                                               [0, 0, 0],
                                                               [1, 0, 0])
        else:
            add_torque_force1, zolo = self.comp_combined_force(copy.deepcopy(q),
                                                               -force_factor_center * edge_dir[0],
                                                               [0, 0, 0],
                                                               [1, 0, 0])
        if not (center[1]):
            add_torque_force2, zolo = self.comp_combined_force(copy.deepcopy(q),
                                                               -force_factor * edge_dir[1],
                                                               [0, 0, 0],
                                                               [0, 1, 0])
        else:
            add_torque_force2, zolo = self.comp_combined_force(copy.deepcopy(q),
                                                               -force_factor_center * edge_dir[1],
                                                               [0, 0, 0],
                                                               [0, 1, 0])
        if not (center[2]):
            add_torque_force3, zolo = self.comp_combined_force(copy.deepcopy(q),
                                                               -force_factor * edge_dir[2],
                                                               [0, 0, 0],
                                                               [0, 0, 1])
        else:
            add_torque_force3, zolo = self.comp_combined_force(copy.deepcopy(q),
                                                               -force_factor_center * edge_dir[2],
                                                               [0, 0, 0],
                                                               [0, 0, 1])

        add_torque_through_normal = add_torque_force1 + add_torque_force2 + add_torque_force3

        if not(correct_torque):
            return add_torque_through_normal

        net_force = self.comp_effective_force(copy.deepcopy(q),
                                              add_torque_through_normal,
                                              in_touch)
        add_torque_froce, add_torque_torque = self.comp_combined_force(copy.deepcopy(q),
                                                                       -net_force, [0, 0, 0], in_touch)
        return add_torque_through_normal + add_torque_froce

    def trafo_q(self, q):
        # Function in order to account for the larger state space
        q_new = np.zeros(12)
        q_new[0:3] = q[0:3]
        q_new[3] = 0.0
        q_new[4:7] = q[3:6]
        q_new[7] = 0.0
        q_new[8:11] = q[6:9]
        q_new[11] = 0.0
        return q_new

    def inv_trafo_q(self, q):
        # function does the inverse of the one above,...
        q_new = np.zeros(9)
        q_new[0:3] = q[0:3]
        q_new[3:6] = q[4:7]
        q_new[6:9] = q[8:11]
        return q_new


class ModelParams():
    def __init__(self):
        # TODO: put all the relevant parameters at this single place here!

        # PARAMS FOR THE APROACH PRIMITIVE
        self.approach_grasp_xy = [0.1, 0.1, 0.09, 0.036195386201143265]
        self.approach_grasp_h = [0.1, 0.0, 0.0, 0.0]
        self.approach_duration = [500, 1000, 1500, 2000]
        self.approach_clip_pos = [0.03,0.03,0.03,0.015]

        # PARAMS FOR THE RESET PRIMITIVE
        self.reset_grasp_xy = [0.1,0.2]
        self.reset_grasp_h = [0.0,0.05]
        self.reset_duration = [200,400]
        self.reset_clip_pos = [0.03,0.015]

        # PARAMS FOR THE GRASP FLOOR PRIMITIVE:
        self.grasp_xy = -0.002748661208897829
        self.grasp_h = -0.004752709995955229
        self.gain_xy_pre = 431.3075256347656
        self.gain_z_pre = 350.3428955078125
        self.init_dur = 100
        self.gain_xy_final = 952.9378662109375
        self.gain_z_final = 1364.1202392578125
        self.pos_gain_impedance = 0.06652811169624329
        self.force_factor = 0.613079309463501










