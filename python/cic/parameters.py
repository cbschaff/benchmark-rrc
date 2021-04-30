class CuboidParams():
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

class CubeLvl1Params():
    def __init__(self,env):
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
        self.grasp_xy_floor = -0.002748661208897829-0.04
        self.grasp_h_floor = 0.0
        self.gain_xy_pre_floor = 294.8115234375
        self.gain_z_pre_floor = 249.4430389404297
        self.init_dur_floor = 100
        self.gain_xy_final_floor = 294.8115234375
        self.gain_z_final_floor = 249.4430389404297
        self.pos_gain_impedance_floor = 0.032040052115917206
        self.force_factor_floor = 0.613079309463501

class CubeLvl2Params():
    def __init__(self, env):
        # TODO: put all the relevant parameters at this single place here!

        # PARAMS FOR THE APROACH PRIMITIVE
        self.approach_grasp_xy = [0.15, 0.15, 0.15, 0.09, 0.05, 0.0] #0.036195386201143265
        self.approach_grasp_h = [0.15, 0.1 , 0.0, 0.0, 0.0, 0.0]
        self.approach_duration = [250, 500, 750, 850, 1100, 1250]#[500, 1000, 1500, 2000]
        self.approach_clip_pos = [0.03,0.03,0.03,0.03,0.015,0.015]

        # PARAMS FOR THE RESET PRIMITIVE
        self.reset_grasp_xy = [0.1,0.2]
        self.reset_grasp_h = [0.0,0.05]
        self.reset_duration = [200,400]
        self.reset_clip_pos = [0.03,0.015]

        # PARAMS FOR THE LIFT PRIMITIVE:
        self.grasp_normal_int_gain = 0.0
        if (env.simulation):
            self.grasp_xy_lift = -0.002748661208897829-0.04+0.0034238076210021985
            self.grasp_h_lift = 0.0+0.02
        else:
            self.grasp_xy_lift = -0.002748661208897829-0.04 -0.02
            self.grasp_h_lift = 0.0 + 0.02
        self.gain_xy_pre_lift = 294.8115234375
        self.gain_z_pre_lift = 249.4430389404297
        self.init_dur_lift = 100
        self.gain_xy_ground_lift = 294.8115234375
        self.gain_z_ground_lift = 249.4430389404297
        self.pos_gain_impedance_ground_lift = 0.032040052115917206

        self.switch_mode_lift = 0.02
        self.clip_height_lift = 0.0881337970495224
        if (env.simulation):
            self.gain_xy_lift_lift = 906.2994122505188#725.27587890625
            self.gain_z_lift_lift = 762.2084140777588#915.2654418945312
            self.pos_gain_impedance_lift_lift = 0.015682869404554364#0.024435650557279587
            self.force_factor_lift = 0.7260927557945251#0.613079309463501
        else:
            self.gain_xy_lift_lift = 842.6861107349396#725.27587890625
            self.gain_z_lift_lift = 1348.3159720897675#915.2654418945312
            self.pos_gain_impedance_lift_lift = 0.019911217913031576#0.024435650557279587
            self.force_factor_lift = 0.4958663940429688#0.613079309463501


class CubeLvl4Params():
    def __init__(self, env):
        # TODO: put all the relevant parameters at this single place here!

        # PARAMS FOR THE APROACH PRIMITIVE
        self.approach_grasp_xy = [0.15, 0.15, 0.15, 0.09, 0.05, 0.0] #0.036195386201143265
        self.approach_grasp_h = [0.15, 0.1 , 0.0, 0.0, 0.0, 0.0]
        self.approach_duration = [150, 300, 450, 600, 750, 900]#[250, 500, 750, 850, 1100, 1250]#[500, 1000, 1500, 2000]
        self.approach_clip_pos = [0.03,0.03,0.03,0.03,0.015,0.015]

        # PARAMS FOR THE RESET PRIMITIVE
        self.reset_grasp_xy = [0.1,0.2,0.2]
        self.reset_grasp_h = [0.0,0.05,0.2]
        self.reset_duration = [100,200,350]#[200,400,600]
        self.reset_clip_pos = [0.03,0.015,0.015]

        # PARAMS FOR THE GRASP FLOOR PRIMITIVE:
        self.grasp_xy_floor = -0.002748661208897829-0.04
        self.grasp_h_floor = 0.0
        self.gain_xy_pre_floor = 294.8115234375
        self.gain_z_pre_floor = 249.4430389404297
        self.init_dur_floor = 100
        self.gain_xy_final_floor = 294.8115234375
        self.gain_z_final_floor = 249.4430389404297
        self.pos_gain_impedance_floor = 0.032040052115917206
        self.force_factor_floor = 0.613079309463501

        # PARAMS FOR THE LIFT PRIMITIVE:
        self.grasp_xy_lift = -0.002748661208897829-0.04
        self.grasp_h_lift = 0.0
        self.gain_xy_pre_lift = 294.8115234375
        self.gain_z_pre_lift = 249.4430389404297
        self.init_dur_lift = 100
        self.gain_xy_ground_lift = 294.8115234375
        self.gain_z_ground_lift = 249.4430389404297
        self.pos_gain_impedance_ground_lift = 0.032040052115917206

        self.switch_mode_lift = 0.02
        self.gain_xy_lift_lift = 725.27587890625
        self.gain_z_lift_lift = 915.2654418945312
        self.pos_gain_impedance_lift_lift = 0.024435650557279587
        self.clip_height_lift = 0.0881337970495224
        self.force_factor_lift = 0.613079309463501

        #PARAMS ROTATE GROUND
        self.grasp_xy_rotate_ground = 0.0022661592811346054 - 0.04 + 0.015980666875839232
        self.grasp_h_rotate_ground = -0.0036394810304045677 -0.00455500602722168
        self.gain_xy_pre_rotate_ground = 294.8115234375
        self.gain_z_pre_rotate_ground = 249.4430389404297
        self.init_dur_rotate_ground = 100
        self.gain_xy_final_rotate_ground = 605.1593139767647#704.6562671661377#611.3788452148438
        self.gain_z_final_rotate_ground = 1243.352460861206#988.0030393600464#600.0
        self.pos_gain_impedance_rotate_ground = 0.02067116618156433#0.011825856864452361#0.0#0.024435650557279587
        self.force_factor_rotate_ground = 0.38862146139144893#0.3449445694684983#0.61
        self.force_factor_ground_rot_rotate_ground = 0.15#0.07469592690467834#0.05#0.02#0.075
        self.gain_scheduling = 20.0

        # PARAMS ROTATE LIFT
        if (env.simulation):
            self.grasp_xy_rotate_lift = 0.0022661592811346054 - 0.04 + 0.015220232009887699 + 0.02
            self.grasp_h_rotate_lift = 0.0 + 0.02 + 0.012779669761657713
        else:
            self.grasp_xy_rotate_lift = 0.0022661592811346054 - 0.04 -0.0030424451828002935
            self.grasp_h_rotate_lift = 0.0 + 0.008263015747070314

        self.gain_xy_pre_rotate_lift = 294.8115234375
        self.gain_z_pre_rotate_lift = 249.4430389404297
        self.init_dur_rotate_lift = 100

        if (env.simulation):
            self.gain_xy_final_rotate_lift = 756.1515808105469#792.4256086349487
            self.gain_z_final_rotate_lift = 1350.2868175506592#1391.9575095176697
            self.pos_gain_impedance_rotate_lift = 0.03663674935698509#0.03219028003513813
            self.force_factor_rotate_lift = 0.9886613845825196#0.8432791709899903
            self.force_factor_center_rotate_lift = 0.6068137288093567
            self.force_factor_rot_rotate_lift = 0.18974763333797454#0.33588245630264285#0.02#0.5218366980552673 * 0.15
            self.target_height_rot_lift_rotate_lift = 0.09
        else:
            self.gain_xy_final_rotate_lift = 671.0243940353394#725.27587890625
            self.gain_z_final_rotate_lift = 1473.1580078601837#915.2654418945312
            self.pos_gain_impedance_rotate_lift = 0.026690984442830086#0.024435650557279587
            self.force_factor_rotate_lift = 0.6484385967254639#0.45589959621429443
            self.force_factor_center_rotate_lift = 0.6068137288093567
            self.force_factor_rot_rotate_lift = 0.02#0.5218366980552673 * 0.15
            self.target_height_rot_lift_rotate_lift = 0.09

        self.clip_height_rotate_lift = 0.0881337970495224

        # PARAMS FOR THE LIFT PRIMITIVE UNDER KEEPING ORIENTATION
        self.orient_int_orient_gain = 0.0
        self.orient_int_pos_gain = 0.0

        self.orient_grasp_xy_lift = -0.002748661208897829-0.04 -0.02
        self.orient_grasp_h_lift = 0.0 -0.012131761014461517#+ 0.005928935948759317#+ 0.02
        self.orient_gain_xy_pre_lift = 294.8115234375
        self.orient_gain_z_pre_lift = 249.4430389404297
        self.orient_init_dur_lift = 100
        self.orient_gain_xy_ground_lift = 294.8115234375
        self.orient_gain_z_ground_lift = 249.4430389404297
        self.orient_pos_gain_impedance_ground_lift = 0.032040052115917206

        self.orient_switch_mode_lift = 0.035
        self.orient_gain_xy_lift_lift = 600.0#636.13525390625#842.6861107349396#725.27587890625
        self.orient_gain_z_lift_lift = 1221.789836883545#1300.06005859375#1348.3159720897675#915.2654418945312
        self.orient_pos_gain_impedance_lift_lift = 0.017376086339354516#0.025942541658878326#0.019911217913031576#0.024435650557279587
        self.orient_clip_height_lift = 0.0881337970495224
        self.orient_force_factor_lift = 0.5796286344528199#0.444672167301178#0.4958663940429688#0.613079309463501
        self.orient_force_factor_rot_lift = 0.0007480829954147339#0.0006446315092034638#0.001#0.5218366980552673 * 0.15
        self.orient_force_factor_rot_ground = 0.0