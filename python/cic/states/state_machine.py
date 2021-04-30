from mp import states
from cic import object_class as cic_object_class
from cic import control_main_class as cic_control_main_class
from cic import parameters as cic_parameters
from cic import parameters_new_grasp as cic_parameters_new_grasp


class StateMachineCIC(states.StateMachine):
    def __init__(self, env, main_ctrl=None, object=None, parameters=None, new_grasp=True):
        if (main_ctrl is None):
            main_ctrl = cic_control_main_class.TriFingerController(env)
        if (object is None):
            object = cic_object_class.Cube()

        if (parameters is None):
            if not(new_grasp):
                difficulty = env.difficulty
                if (difficulty == 1):
                    parameters = cic_parameters.CubeLvl1Params(env)
                elif (difficulty == 2 or difficulty == 3):
                    parameters = cic_parameters.CubeLvl2Params(env)
                elif (difficulty == 4):
                    parameters = cic_parameters.CubeLvl4Params(env)
            else:
                difficulty = env.difficulty
                if (difficulty == 1):
                    parameters = cic_parameters_new_grasp.CubeLvl1Params(env)
                elif (difficulty == 2 or difficulty == 3):
                    parameters = cic_parameters_new_grasp.CubeLvl2Params(env)
                elif (difficulty == 4):
                    parameters = cic_parameters_new_grasp.CubeLvl4Params(env)

        self.env = env
        self.main_ctrl = main_ctrl
        self.object = object
        self.parameters = parameters
        self.new_grasp = new_grasp
        self.init_state = self.build()
