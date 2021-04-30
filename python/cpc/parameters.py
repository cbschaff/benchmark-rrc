class CubeLvl1Params():
    def __init__(self, env):
        if env.simulation:
            self.k_p_goal = 0.75  # Range: 0.3 - 1.5, same for l4
            self.k_p_into = 0.18  # Range: 0.1 - 0.6, same for l4
            self.k_i_goal = 0.008  # Range: 0.0008 - 0.1, same for l4
            self.interval = 1  # Range: 500 - 3000 not super important
            self.gain_increase_factor = 1.004  # Range: 1.01 - 2.0
            self.max_interval_ctr = 2000
        else:
            self.k_p_goal = 0.65
            self.k_p_into = 0.2
            self.k_i_goal = 0.004
            self.interval = 1800  # Range: 500 - 3000 not super important
            self.gain_increase_factor = 1.04  # Range: 1.01 - 2.0
            self.max_interval_ctr = 30


class CubeParams():
    def __init__(self, env):
        if env.simulation:
            self.k_p_goal = 0.9  # Range: 0.3 - 1.5, same for l4
            self.k_p_into = 0.20  # Range: 0.1 - 0.6, same for l4
            self.k_i_goal = 0.007  # Range: 0.0008 - 0.1, same for l4
            self.interval = 1  # Range: 500 - 3000 not super important
            self.gain_increase_factor = 1.004  # Range: 1.01 - 2.0
            self.max_interval_ctr = 2000
        else:
            self.k_p_goal = 0.65
            self.k_p_into = 0.25
            self.k_i_goal = 0.004
            self.interval = 1800  # Range: 500 - 3000 not super important
            self.gain_increase_factor = 1.04  # Range: 1.01 - 2.0
            self.max_interval_ctr = 30


class CubeLvl4Params():
    def __init__(self,env):
        if env.simulation:
            self.gain_increase_factor = 1.002
            self.k_p_goal = 0.9
            self.k_p_into = 0.25
            self.k_i_goal = 0.006
            self.k_p_ang = 0.01
            self.k_i_ang = 0.0005
            self.pitch_k_p_goal = 0.45
            self.pitch_k_p_into = 0.22
            self.pitch_k_i_goal = 0.004
            self.pitch_k_p_ang = 0.25
            self.pitch_k_i_ang = 0.0005
            self.yaw_k_p_goal = 0.9
            self.yaw_k_p_into = 0.25
            self.yaw_k_i_goal = 0.006
            self.yaw_k_p_ang = 0.01
            self.yaw_k_i_ang = 0.0005
            self.interval = 3
            self.max_interval_ctr = 2000
            self.pitch_rot_factor = 1.5
            self.pitch_lift_factor = 2.46
        else:
            self.gain_increase_factor = 1.04
            self.k_p_goal = 0.65
            self.k_p_into = 0.28
            self.k_i_goal = 0.004
            self.k_p_ang = 0.05
            self.k_i_ang = 0.0005
            self.pitch_k_p_goal = 0.5
            self.pitch_k_p_into = 0.36
            self.pitch_k_i_goal = 0.005
            self.pitch_k_p_ang = 0.15
            self.pitch_k_i_ang = 0.002
            self.yaw_k_p_goal = 0.65
            self.yaw_k_p_into = 0.28
            self.yaw_k_i_goal = 0.004
            self.yaw_k_p_ang = 0.05
            self.yaw_k_i_ang = 0.001
            self.interval = 1800
            self.max_interval_ctr = 30
            self.pitch_rot_factor = 1.18
            self.pitch_lift_factor = 1.38
