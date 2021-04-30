from .residual_state import ResidualState
from mp.state_machines import MPStateMachine
from cic.states.custom_state_machines import CICStateMachineLvl2, CICStateMachineLvl4
from cpc.state_machine import CPCStateMachine, CPCStateMachineL4
import os

MP_LVL3_LOGDIR = os.path.join(os.path.dirname(__file__), 'models/mp_lvl3')
CIC_LVL3_LOGDIR = os.path.join(os.path.dirname(__file__), 'models/cic_lvl3')
CPC_LVL3_LOGDIR = os.path.join(os.path.dirname(__file__), 'models/cpc_lvl3')
MP_LVL4_LOGDIR = os.path.join(os.path.dirname(__file__), 'models/mp_lvl4')
CIC_LVL4_LOGDIR = os.path.join(os.path.dirname(__file__), 'models/cic_lvl4')
CPC_LVL4_LOGDIR = os.path.join(os.path.dirname(__file__), 'models/cpc_lvl4')


class ResidualMP_with_PG_LVL3(MPStateMachine):
    def build(self):
        init_state = super().build()
        self.move_to_goal = ResidualState(self.move_to_goal, MP_LVL3_LOGDIR)
        # rewire connections to use new state
        self.planned_grasp.connect(self.move_to_goal, self.align_object)
        return init_state


class ResidualMP_with_PG_LVL4(MPStateMachine):
    def build(self):
        init_state = super().build()
        self.move_to_goal = ResidualState(self.move_to_goal, MP_LVL4_LOGDIR)
        # rewire connections to use new state
        self.planned_grasp.connect(self.move_to_goal, self.align_object)
        return init_state


class ResidualCIC_with_CG_LVL3(CICStateMachineLvl2):
    def build(self):
        init_state = super().build()
        self.move_cube_lift = ResidualState(self.move_cube_lift, CIC_LVL3_LOGDIR)
        # rewire connections to use new state
        if not (self.new_grasp):
            self.approach.connect(next_state=self.move_cube_lift, done_state=self.done)
        self.set_cic_grasp1.connect(next_state=self.move_cube_lift, done_state=self.done)
        return init_state


class ResidualCIC_with_CG_LVL4(CICStateMachineLvl4):
    def build(self):
        init_state = super().build()
        self.bring_final.move_cube_lift = ResidualState(
                            self.bring_final.move_cube_lift, CIC_LVL4_LOGDIR)
        # rewire connections to use new state
        self.align_axes.connect(next_state=self.bring_final, done_state=self.done)
        self.bring_final.connect(next_state=self.align_axes, done_state=self.done)
        return init_state


class ResidualCPC_with_TG_LVL3(CPCStateMachine):
    def build(self):
        init_state = super().build()
        self.move_to_goal = ResidualState(self.move_to_goal, CPC_LVL3_LOGDIR)
        # rewire connections to use new state
        self.grasp.connect(next_state=self.move_to_goal,
                           failure_state=self.goto_init_pose)
        self.move_to_goal.connect(
            next_state=self.move_to_goal, failure_state=self.goto_init_pose)
        return init_state


class ResidualCPC_with_TG_LVL4(CPCStateMachineL4):
    def build(self):
        init_state = super().build()
        self.move_to_goal = ResidualState(self.move_to_goal, CPC_LVL4_LOGDIR)
        # rewire connections to use new state
        self.grasp.connect(next_state=self.move_to_goal,
                           failure_state=self.goto_init_pose)
        self.move_to_goal.connect(
            next_state=self.move_to_goal, failure_state=self.goto_init_pose)
        return init_state
