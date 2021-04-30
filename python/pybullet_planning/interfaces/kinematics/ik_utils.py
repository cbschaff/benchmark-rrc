import math
import numpy as np
import pybullet as p

from pybullet_planning.utils import CLIENT

def inverse_kinematics_helper(robot, link, target_pose, null_space=None):
    (target_point, target_quat) = target_pose
    assert target_point is not None
    if null_space is not None:
        assert target_quat is not None
        lower, upper, ranges, rest = null_space

        kinematic_conf = p.calculateInverseKinematics(robot, link, target_point,
                                                      lowerLimits=lower, upperLimits=upper, jointRanges=ranges, restPoses=rest,
                                                      physicsClientId=CLIENT)
    elif target_quat is None:
        #ikSolver = p.IK_DLS or p.IK_SDLS
        kinematic_conf = p.calculateInverseKinematics(robot, link, target_point,
                                                      #lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp, jointDamping=jd,
                                                      # solver=ikSolver, maxNumIterations=-1, residualThreshold=-1,
                                                      physicsClientId=CLIENT)
    else:
        kinematic_conf = p.calculateInverseKinematics(robot, link, target_point, target_quat, physicsClientId=CLIENT)
    if (kinematic_conf is None) or any(map(math.isnan, kinematic_conf)):
        return None
    return kinematic_conf

def is_pose_close(pose, target_pose, pos_tolerance=1e-3, ori_tolerance=1e-3*np.pi):
    (point, quat) = pose
    (target_point, target_quat) = target_pose
    if (target_point is not None) and not np.allclose(point, target_point, atol=pos_tolerance, rtol=0):
        return False
    if (target_quat is not None) and not np.allclose(quat, target_quat, atol=ori_tolerance, rtol=0):
        # TODO: account for quaternion redundancy
        return False
    return True

def inverse_kinematics(robot, link, target_pose, max_iterations=200, custom_limits={}, **kwargs):
    from pybullet_planning.interfaces.env_manager.pose_transformation import all_between
    from pybullet_planning.interfaces.robots import get_movable_joints, set_joint_positions, get_link_pose, get_custom_limits

    movable_joints = get_movable_joints(robot)
    for iterations in range(max_iterations):
        # TODO: stop is no progress
        # TODO: stop if collision or invalid joint limits
        kinematic_conf = inverse_kinematics_helper(robot, link, target_pose)
        if kinematic_conf is None:
            return None
        set_joint_positions(robot, movable_joints, kinematic_conf)
        if is_pose_close(get_link_pose(robot, link), target_pose, **kwargs):
            break
    else:
        return None
    lower_limits, upper_limits = get_custom_limits(robot, movable_joints, custom_limits)
    if not all_between(lower_limits, kinematic_conf, upper_limits):
        return None
    return kinematic_conf


def snap_sols(sols, q_guess, joint_limits, weights=None, best_sol_only=False):
    """get the best solution based on closeness to the q_guess and weighted joint diff

    Parameters
    ----------
    sols : [type]
        [description]
    q_guess : [type]
        [description]
    joint_limits : [type]
        [description]
    weights : [type], optional
        [description], by default None
    best_sol_only : bool, optional
        [description], by default False

    Returns
    -------
    lists of joint conf (list)
    or
    joint conf (list)
    """
    import numpy as np
    valid_sols = []
    dof = len(q_guess)
    if not weights:
        weights = [1.0] * dof
    else:
        assert dof == len(weights)

    for sol in sols:
        test_sol = np.ones(dof)*9999.
        for i in range(dof):
            for add_ang in [-2.*np.pi, 0, 2.*np.pi]:
                test_ang = sol[i] + add_ang
                if (test_ang <= joint_limits[i][1] and test_ang >= joint_limits[i][0] and \
                    abs(test_ang - q_guess[i]) < abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol.tolist())

    if len(valid_sols) == 0:
        return []

    if best_sol_only:
        best_sol_ind = np.argmin(np.sum((weights*(valid_sols - np.array(q_guess)))**2,1))
        return valid_sols[best_sol_ind]
    else:
        return valid_sols

def sample_multiple_ik_with_collision(ik, joint_collision_fn, sample_fn, tip_positions, num_samples, max_trials=30, tolerance=1e-02):
    num_trial = 0
    ik_solutions = []
    done = False

    # test if any IK solution exists
    ik_sol = sample_ik_solution(ik, sample_fn, tip_positions)
    if ik_sol is None:
        return []

    while not done:
        # solve ik solutions for num_samples times and filter out those in collision
        current_ik_solutions = [sample_ik_solution(ik, sample_fn, tip_positions) for _ in range(num_samples)]
        current_ik_solutions = [sol for sol in current_ik_solutions if sol is not None]  # filter out None
        current_ik_solutions = [current_ik_sol for current_ik_sol in current_ik_solutions if not joint_collision_fn(current_ik_sol)]
        if len(current_ik_solutions) == 0:
            return []
        current_ik_solutions = np.asarray(current_ik_solutions)

        if len(ik_solutions) == 0:
            ik_solutions.append(current_ik_solutions[0])

        while True:
            #axis: ik_solusions, sample, dim
            errors = np.array([np.abs(current_ik_solutions - ik_solution) for ik_solution in ik_solutions])
            max_error_for_each_solutions = np.max(errors, axis=2)
            bool_array = multiply_bool_array(max_error_for_each_solutions > tolerance)
            if bool_array.any():
                idx = np.argmax(bool_array)
                ik_solutions.append(current_ik_solutions[idx])
                #throw away similar vectors
                current_ik_solutions = current_ik_solutions[np.where(bool_array==True)]
            else:
                break
        num_trial += num_samples
        if len(ik_solutions) >= num_samples or num_trial >= max_trials:
            done = True
    return np.array(ik_solutions[:num_samples])

def multiply_bool_array(bool_array):
    if len(bool_array) == 1:
        return bool_array[0]
    else:
        output = bool_array[0]
        for i in range(1, len(bool_array)):
            output = np.multiply(output, bool_array[i])
        return output


def sample_ik_solution(ik, sample_fn, target_tip_positions, max_trials=5):
    done = False
    count = 0
    while not done and count < max_trials:
        joint_conf = sample_fn()
        joint_conf = np.asarray(joint_conf, dtype=np.float64)
        ik_solution = [
            ik(i,
               np.asarray(target_tip_positions[i], dtype=np.float32),
               np.asarray(joint_conf, dtype=np.float64)
            )
            for i in range(3)
        ]
        # ik_solution = [ik(i, target_tip_positions[i], joint_conf) for i in range(3)]
        done = (ik_solution[0] is not None) and (ik_solution[1] is not None) and (ik_solution[2] is not None)
        count += 1
    if done:
        ik_solution = [ik_solution[i][3*i:3*(i+1)] for i in range(3)]
        return np.concatenate(ik_solution)
    else:
        return None


def sample_partial_ik_solution(ik, sample_fn, target_tip_positions, max_trials=5):
    """If IK solution exists in any of the tip positions, returns those solution"""
    done = False
    count = 0
    while not done and count < max_trials:
        joint_conf = sample_fn()
        joint_conf = np.asarray(joint_conf, dtype=np.float64)
        ik_solution = [
            ik(i,
               np.asarray(target_tip_positions[i], dtype=np.float32),
               np.asarray(joint_conf, dtype=np.float64)
            )
            for i in range(3)
        ]
        # ik_solution = [ik(i, target_tip_positions[i], joint_conf) for i in range(3)]
        done = (ik_solution[0] is not None) or (ik_solution[1] is not None) or (ik_solution[2] is not None)
        count += 1
    if done:
        for i in range(3):
            if ik_solution[i] is None:
                ik_solution[i] = np.array([None] * 3)
            else:
                ik_solution[i] = ik_solution[i][3*i:3*(i+1)]

        return np.concatenate(ik_solution)
    else:
        return None


def sample_no_collision_ik(ik, joint_collision_fn, sample_fn, target_tip_positions, n_samples=3):
    ik_solution = sample_ik_solution(ik, sample_fn, target_tip_positions)
    if ik_solution is None:
        return None
    ik_solutions = [ik_solution]
    ik_solutions += [sample_ik_solution(ik, sample_fn, target_tip_positions) for _ in range(n_samples)]
    ik_solutions = [ik_sol for ik_sol in ik_solutions if joint_collision_fn(ik_sol)]
    return ik_solutions
