#!/usr/bin/env python3
from mp.states import MoveToGoalState
from mp.base_policies import PlanningAndForceControlPolicy


def set_hyperparams(simulation):
    '''Set two hyperparameters that are Bayesian-Optimized'''
    # NOTE: These values will NOT be resetted when state.reset() is called.
    if simulation:
        MoveToGoalState.BO_action_repeat = 13  # (int) [1, 100], default: 12
        PlanningAndForceControlPolicy.BO_num_tipadjust_steps = 161  # (int) [10, 200], default: 50
    else:
        MoveToGoalState.BO_action_repeat = 29  # (int) [1, 100], default: 12
        PlanningAndForceControlPolicy.BO_num_tipadjust_steps = 182  # (int) [10, 200], default: 50
