from cpc import CPCStateMachine as CPCwithTG
from cpc import CPCStateMachineL4 as CPCwithTGL4
from cic.states import CICStateMachineLvl2 as CICwithCG
from cic.states import CICStateMachineLvl4 as CICwithCGL4
from cic.states import CICStateMachineLvl1 as CICwithCGL1
from mp.state_machines import MPStateMachine as MPwithPG
from residual_learning import residual_state_machines as rsm
from cic import parameters_new_grasp as cic_parameters_new_grasp
from mp import states, base_policies
from cpc import parameters as cpc_params
from combined_code import mix_and_match as mm


state_machines = {
    'mp-pg-l1': MPwithPG,
    'mp-pg-l2': MPwithPG,
    'mp-pg-l3': MPwithPG,
    'mp-pg-l4': MPwithPG,
    'cic-cg-l1': CICwithCGL1,
    'cic-cg-l2': CICwithCG,
    'cic-cg-l3': CICwithCG,
    'cic-cg-l4': CICwithCGL4,
    'cpc-tg-l1': CPCwithTG,
    'cpc-tg-l2': CPCwithTG,
    'cpc-tg-l3': CPCwithTG,
    'cpc-tg-l4': CPCwithTGL4,
    'residual-mp-pg-l3': rsm.ResidualMP_with_PG_LVL3,
    'residual-mp-pg-l4': rsm.ResidualMP_with_PG_LVL4,
    'residual-cic-cg-l3': rsm.ResidualCIC_with_CG_LVL3,
    'residual-cic-cg-l4': rsm.ResidualCIC_with_CG_LVL4,
    'residual-cpc-tg-l3': rsm.ResidualCPC_with_TG_LVL3,
    'residual-cpc-tg-l4': rsm.ResidualCPC_with_TG_LVL4,
    'mp-cg-l4': mm.MPwithCG,
    'mp-tg-l4': mm.MPwithTG,
    'cic-pg-l4': mm.CICwithPG,
    'cic-tg-l4': mm.CICwithTG,
    'cpc-pg-l4': mm.CPCwithPG,
    'cpc-cg-l4': mm.CPCwithCG,
}


def create_state_machine(difficulty, method, env, residual=False, bo=False):
    if residual:
        if method not in ['mp-pg', 'cic-cg', 'cpc-tg'] and difficulty in [1, 2]:
            raise ValueError("Residual policies are only available for methods "
                             "'mp-pg', 'cic-cg', 'cpc-tg' and difficulties 3 and 4."
                             f"Method: {method}, difficulty: {difficulty}.")
    if bo:
        if method not in ['mp-pg', 'cic-cg', 'cpc-tg'] and difficulty in [1, 2]:
            raise ValueError("BO optimized parameters are only available for methods "
                             "'mp-pg', 'cic-cg', 'cpc-tg' and difficulties 3 and 4."
                             f"Method: {method}, difficulty: {difficulty}.")
    if method not in ['mp-pg', 'cic-cg', 'cpc-tg'] and difficulty != 4:
        raise ValueError(f'{method} is only implemented for difficulty 4.')
    id = method + f'-l{difficulty}'
    if residual:
        id = 'residual-' + id
    if id not in state_machines:
        raise ValueError(
            f"Unknown method: {method}. Options are: "
            "mp-pg, cic-cg, cpc-tg, mp-cg, mp-tg, cic-pg, cic-tg, cpc-pg, cpc-cg."
        )
    if bo:
        return create_bo_state_machine(id, env, difficulty)
    else:
        return state_machines[id](env)


def create_bo_state_machine(id, env, difficulty):
    if 'mp-pg' in id:
        return mp_bo_wrapper(id, env, difficulty)
    elif 'cic-cg' in id:
        return cic_bo_wrapper(id, env, difficulty)
    else:
        return cpc_bo_wrapper(id, env, difficulty)


def mp_bo_wrapper(id, env, difficulty):
    if difficulty == 3:
        if (env.simulation):
            states.MoveToGoalState.BO_action_repeat = 10
            base_policies.PlanningAndForceControlPolicy.BO_num_tipadjust_steps = 184
        else:
            states.MoveToGoalState.BO_action_repeat = 26  # 12  # (int) [1, 100], default: 12
            base_policies.PlanningAndForceControlPolicy.BO_num_tipadjust_steps = 63  # 50  # (int) [10, 200], default: 50
    elif difficulty == 4:
        if (env.simulation):
            states.MoveToGoalState.BO_action_repeat = 13
            base_policies.PlanningAndForceControlPolicy.BO_num_tipadjust_steps = 161
        else:
            states.MoveToGoalState.BO_action_repeat = 29  # 12  # (int) [1, 100], default: 12
            base_policies.PlanningAndForceControlPolicy.BO_num_tipadjust_steps = 182  # 50  # (int) [10, 200], default: 50
    return state_machines[id](env)


def cic_bo_wrapper(id, env, difficulty):
    if difficulty == 3:
        parameters = cic_parameters_new_grasp.CubeLvl2Params(env)
    elif difficulty == 4:
        parameters = cic_parameters_new_grasp.CubeLvl4Params(env)
        if (env.simulation):
            parameters.orient_grasp_xy_lift = -0.01932485358
            parameters.orient_grasp_h_lift = 0.0167107629776001
            parameters.orient_gain_xy_lift_lift = 500.0
            parameters.orient_gain_z_lift_lift = 974.5037078857422
            parameters.orient_pos_gain_impedance_lift_lift = 0.015002169609069825
            parameters.orient_force_factor_lift = 0.6673897802829742
            parameters.orient_force_factor_rot_lift = 0.010000000000000002
            parameters.orient_int_orient_gain = 0.0003590885430574417
            parameters.orient_int_pos_gain = 0.008034629583358766
        else:
            parameters.orient_grasp_xy_lift = -0.03926035182
            parameters.orient_grasp_h_lift = -0.005355795621871948
            parameters.orient_gain_xy_lift_lift = 895.7465827465057
            parameters.orient_gain_z_lift_lift = 1500.0
            parameters.orient_pos_gain_impedance_lift_lift = 0.01427580736577511
            parameters.orient_force_factor_lift = 0.49047523438930507
            parameters.orient_force_factor_rot_lift = 0.0022044302672147753
            parameters.orient_int_orient_gain = 0.027903699278831486
            parameters.orient_int_pos_gain = 0.013680822849273681
    return state_machines[id](env, parameters=parameters)


def cpc_bo_wrapper(id, env, difficulty):
    if difficulty == 3:
        parameters = cpc_params.CubeParams(env)
        if (env.simulation):
            parameters.interval = 9
            parameters.gain_increase_factor = 1.1110639113783836
            parameters.k_p_goal = 0.5408251136541367
            parameters.k_p_into = 0.17404515892267228
            parameters.k_i_goal = 0.00801944613456726
        else:
            parameters.interval = 3000  # 1800  # Range: 500 - 3000 not super important
            parameters.gain_increase_factor = 1.7353031241893768  # 1.04  # Range: 1.01 - 2.0
            parameters.k_p_goal = 0.5804646849632262  # 0.75  # Range: 0.3 - 1.5, same for l4
            parameters.k_p_into = 0.1  # 0.2  # Range: 0.1 - 0.6, same for l4
            parameters.k_i_goal = 0.00801206259727478  # 0.005  # Range: 0.0008 - 0.1, same for l4
    if difficulty == 4:
        parameters = cpc_params.CubeLvl4Params(env)
        if (env.simulation):
            parameters.interval = 10
            parameters.gain_increase_factor = 1.2431243617534635
            parameters.k_p_goal = 0.4393719419836998
            parameters.k_p_into = 0.21185509711503983
            parameters.k_i_goal = 0.008012341380119324
            parameters.k_p_ang = 0.02238279849290848
            parameters.k_i_ang = 0.0019905194759368898
        else:
            parameters.interval = 579
            parameters.gain_increase_factor = 1.07002716961503
            parameters.k_p_goal = 0.6011996507644652
            parameters.k_p_into = 0.13088179603219033
            parameters.k_i_goal = 0.006161301851272583
            parameters.k_p_ang = 0.06160478860139847
            parameters.k_i_ang = 0.0007573306798934938
    return state_machines[id](env, parameters=parameters)
