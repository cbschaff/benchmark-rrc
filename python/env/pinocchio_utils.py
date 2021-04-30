import numpy as np

import pinocchio


class PinocchioUtils:
    """
    Consists of kinematic methods for the finger platform.
    """

    def __init__(self):
        """
        Initializes the finger model on which control's to be performed.
        """
        self.urdf_path = '/opt/blmc_ei/src/robot_properties_fingers/urdf/pro/trifingerpro.urdf'
        self.tip_link_names = [
            "finger_tip_link_0",
            "finger_tip_link_120",
            "finger_tip_link_240",
        ]
        self.robot_model = pinocchio.buildModelFromUrdf(self.urdf_path)
        self.data = self.robot_model.createData()
        self.tip_link_ids = [
            self.robot_model.getFrameId(link_name)
            for link_name in self.tip_link_names
        ]

    def forward_kinematics(self, joint_positions):
        """
        Compute end effector positions for the given joint configuration.

        Args:
            finger (SimFinger): a SimFinger object
            joint_positions (list): Flat list of angular joint positions.

        Returns:
            List of end-effector positions. Each position is given as an
            np.array with x,y,z positions.
        """
        pinocchio.framesForwardKinematics(
            self.robot_model, self.data, joint_positions,
        )

        return [
            np.asarray(self.data.oMf[link_id].translation).reshape(-1).tolist()
            for link_id in self.tip_link_ids
        ]

    def compute_jacobian(self, finger_id, q0):
        """
        Compute the jacobian of a finger at configuration q0.

        Args:
            finger_id (int): an index specifying the end effector. (i.e. 0 for
                             the first finger, 1 for the second, ...)
            q0 (np.array):   The current joint positions.

        Returns:
            An np.array containing the jacobian.
        """
        frame_id = self.tip_link_ids[finger_id]
        return pinocchio.computeFrameJacobian(
            self.robot_model,
            self.data,
            q0,
            frame_id,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

    def _project_onto_constraints(self, pos):
        from trifinger_simulation.trifinger_platform import TriFingerPlatform
        space = TriFingerPlatform.spaces.robot_position
        return np.clip(pos, space.low, space.high)

    def _line_search(self, finger_id, xdes, q0, dq, max_iter=10, dt=1.0):
        """
        Performs a line search, reducing the step size until the end effector
        moves closer to the desired position.

        Args:
            finger_id (int): an index specifying the end effector. (i.e. 0 for
                             the first finger, 1 for the second, ...)
            xdes (np.array): desired xyz position of the end effector
            q0 (np.array):   The current joint positions.
            dq (np.array):   The update direction.
            max_iter (int):  The maximum number of iterations.
            dt (float):      The initial step size.

        Returns:
            q (np.array):      The updated joint positions.
            error (float):     The error in end effector position.
            step size (float): The step size used.
        """
        xcurrent = self.forward_kinematics(q0)[finger_id]
        original_error = np.linalg.norm(xdes - xcurrent)
        error = np.inf
        q = q0
        iter = 0
        while error >= original_error:
            q = pinocchio.integrate(self.robot_model, q0, dt * dq)
            q = self._project_onto_constraints(q)
            xcurrent = self.forward_kinematics(q)[finger_id]
            error = np.linalg.norm(xdes - xcurrent)
            dt /= 2
            iter += 1
            if iter == max_iter:
                # Likely at a local minimum
                return q0, original_error, 0
        return q, error, 2 * dt

    def _compute_dq(self, finger_id, xdes, q0):
        """
        Computes the direction in which to update joint positions.

        Args:
            finger_id (int): an index specifying the end effector. (i.e. 0 for
                             the first finger, 1 for the second, ...)
            xdes (np.array): desired xyz position of the end effector
            q0 (np.array):   The current joint positions.

        Returns:
            An np.array containing the direction in joint space which moves
            the end effector closer to the desired position.
        """
        Ji = self.compute_jacobian(finger_id, q0)[:3, :]
        frame_id = self.tip_link_ids[finger_id]
        xcurrent = self.data.oMf[frame_id].translation
        Jinv = np.linalg.pinv(Ji)
        return Jinv.dot(xdes - xcurrent)

    def inverse_kinematics(self, finger_id, xdes, q0, tol=0.001, max_iter=20):
        """
        Compute the joint positions which approximately result in a given
        end effector position.

        Args:
            finger_id (int): an index specifying the end effector. (i.e. 0 for
                             the first finger, 1 for the second, ...)
            xdes (np.array): desired xyz position of the end effector
            q0 (np.array):   The current joint positions.
            tol (float):     The tolerated error in end effector position.
            max_iter:        The maximum number of iterations.

        Returns:
            An np.array of joint positions which achieve the desired end
            effector position.
        """
        iter = 0
        q = self._project_onto_constraints(q0)
        xcurrent = self.forward_kinematics(q)[finger_id]
        error = np.linalg.norm(xdes - xcurrent)
        dt = 1.0
        prev_error = np.inf
        while error > tol and (prev_error - error) > 1e-5 and iter < max_iter:
            dq = self._compute_dq(finger_id, xdes, q)
            # start the line search with a step size a bit larger than the
            # previous step size.
            dt = min(1.0, 2 * dt)
            prev_error = error
            q, error, dt = self._line_search(finger_id, xdes, q, dq, dt=dt)
            iter += 1
        if error > tol:
            return None
        return q
