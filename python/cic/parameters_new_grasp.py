class CuboidParams():
    def __init__(self):
        # TODO: put all the relevant parameters at this single place here!

        # PARAMS FOR THE APROACH PRIMITIVE
        self.approach_grasp_xy = [0.1, 0.1, 0.09, 0.036195386201143265]
        self.approach_grasp_h = [0.1, 0.0, 0.0, 0.0]
        self.approach_duration = [500, 1000, 1500, 2000]
        approach_duration_new = []
        factor = 4
        approach_duration_new.append(self.approach_duration[0] * factor)
        for i in range(len(self.approach_duration) - 1):
            approach_duration_new.append(
                (self.approach_duration[i + 1] - self.approach_duration[i]) * factor + approach_duration_new[i])
        print(approach_duration_new)
        self.approach_duration = approach_duration_new
        self.approach_clip_pos = [0.03,0.03,0.03,0.015]

        # PARAMS FOR THE RESET PRIMITIVE
        self.reset_grasp_xy = [0.1,0.2]
        self.reset_grasp_h = [0.0,0.05]
        self.reset_duration = [200,400]
        self.reset_clip_pos = [0.03,0.015]
        reset_duration_new = []
        reset_duration_new.append(self.reset_duration[0] * factor)
        for i in range(len(self.reset_duration) - 1):
            reset_duration_new.append(
                (self.reset_duration[i + 1] - self.reset_duration[i]) * factor + reset_duration_new[i])
        print(reset_duration_new)
        self.reset_duration = reset_duration_new

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
        self.approach_duration = [250, 500, 750, 1000]
        self.approach_clip_pos = [0.03,0.03,0.03,0.015]
        approach_duration_new = []
        factor = 4
        approach_duration_new.append(self.approach_duration[0] * factor)
        for i in range(len(self.approach_duration) - 1):
            approach_duration_new.append(
                (self.approach_duration[i + 1] - self.approach_duration[i]) * factor + approach_duration_new[i])
        print(approach_duration_new)
        self.approach_duration = approach_duration_new

        # PARAMS FOR THE RESET PRIMITIVE
        self.reset_grasp_xy = [0.1,0.2]
        self.reset_grasp_h = [0.0,0.05]
        self.reset_duration = [200,400]
        self.reset_clip_pos = [0.03,0.015]
        reset_duration_new = []
        reset_duration_new.append(self.reset_duration[0] * factor)
        for i in range(len(self.reset_duration) - 1):
            reset_duration_new.append(
                (self.reset_duration[i + 1] - self.reset_duration[i]) * factor + reset_duration_new[i])
        print(reset_duration_new)
        self.reset_duration = reset_duration_new

        # PARAMS FOR THE GRASP FLOOR PRIMITIVE:
        self.grasp_xy_floor = -0.002748661208897829-0.04
        self.grasp_h_floor = 0.0
        self.gain_xy_pre_floor = 294.8115234375
        self.gain_z_pre_floor = 249.4430389404297
        self.init_dur_floor = 100
        if (env.simulation):
            self.gain_xy_final_floor = 693.0245697498322
            self.gain_z_final_floor = 744.9281394481659
            self.pos_gain_impedance_floor = 0.015
        else:
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
        approach_duration_new = []
        factor = 4
        approach_duration_new.append(self.approach_duration[0]*factor)
        for i in range(len(self.approach_duration)-1):
            approach_duration_new.append((self.approach_duration[i+1]-self.approach_duration[i])* factor+approach_duration_new[i])
        print (approach_duration_new)
        self.approach_duration = approach_duration_new
        self.approach_clip_pos = [0.03,0.03,0.03,0.03,0.015,0.015]

        # PARAMS FOR THE RESET PRIMITIVE
        self.reset_grasp_xy = [0.1,0.2]
        self.reset_grasp_h = [0.0,0.05]
        self.reset_duration = [200,400]
        self.reset_clip_pos = [0.03,0.015]
        reset_duration_new = []
        reset_duration_new.append(self.reset_duration[0] * factor)
        for i in range(len(self.reset_duration) - 1):
            reset_duration_new.append(
                (self.reset_duration[i + 1] - self.reset_duration[i]) * factor + reset_duration_new[i])
        print(reset_duration_new)
        self.reset_duration = reset_duration_new

        # PARAMS FOR THE LIFT PRIMITIVE:

        if (env.simulation):
            self.grasp_normal_int_gain = 0.007945765256881714
            self.grasp_xy_lift = -0.002748661208897829-0.04+0.0034238076210021985 + 0.019999999999999997#-0.02229524075984955
            self.grasp_h_lift = 0.0+0.02 -0.02838870149105787#-0.012001985907554625
        else:
            self.grasp_normal_int_gain = 0.007708187699317932
            self.grasp_xy_lift = -0.002748661208897829-0.04 + 0.019999999999999997
            self.grasp_h_lift = 0.0 -0.017401267290115353
        self.gain_xy_pre_lift = 294.8115234375
        self.gain_z_pre_lift = 249.4430389404297
        self.init_dur_lift = 100
        self.gain_xy_ground_lift = 294.8115234375
        self.gain_z_ground_lift = 249.4430389404297
        self.pos_gain_impedance_ground_lift = 0.032040052115917206

        # TODO: might want to consider changing this also on the real platform,....
        self.switch_mode_lift = 0.02
        self.clip_height_lift = 0.0881337970495224
        if (env.simulation):
            self.switch_mode_lift = 0.035
            self.gain_xy_ground_lift = 693.0245697498322
            self.gain_z_ground_lift = 744.9281394481659
            self.pos_gain_impedance_ground_lift = 0.032040052115917206
            self.gain_xy_lift_lift = 693.0245697498322#906.2994122505188 #400.0
            self.gain_z_lift_lift = 744.9281394481659#762.2084140777588 #722.6458370685577
            self.pos_gain_impedance_lift_lift = 0.01481801524758339#0.015682869404554364 #0.06
            self.force_factor_lift = 0.32288771867752075#0.7260927557945251 #0.7495793342590333
        else:
            self.gain_xy_lift_lift = 752.3177862167358
            self.gain_z_lift_lift = 1158.7008893489838
            self.pos_gain_impedance_lift_lift = 0.02216305151581764
            self.force_factor_lift = 0.6339841365814209


class CubeLvl4Params():
    def __init__(self, env):
        # TODO: put all the relevant parameters at this single place here!

        # PARAMS FOR THE APROACH PRIMITIVE
        self.approach_grasp_xy = [0.15, 0.15, 0.15, 0.09, 0.05, 0.0] #0.036195386201143265
        self.approach_grasp_h = [0.15, 0.1 , 0.0, 0.0, 0.0, 0.0]
        self.approach_duration = [150, 300, 450, 600, 750, 900]#[250, 500, 750, 850, 1100, 1250]#[500, 1000, 1500, 2000]
        self.approach_clip_pos = [0.03,0.03,0.03,0.03,0.015,0.015]
        approach_duration_new = []
        factor = 4
        approach_duration_new.append(self.approach_duration[0] * factor)
        for i in range(len(self.approach_duration) - 1):
            approach_duration_new.append(
                (self.approach_duration[i + 1] - self.approach_duration[i]) * factor + approach_duration_new[i])
        print(approach_duration_new)
        self.approach_duration = approach_duration_new

        # PARAMS FOR THE RESET PRIMITIVE
        self.reset_grasp_xy = [0.05,0.1, 0.1, 0.1, 0.1, 0.1, 0.15]
        self.reset_grasp_h = [0.0,  0.0, 0.05,0.1, 0.15,0.2, 0.15]
        self.reset_duration = [50,  100, 150, 200, 250, 300, 350]#[200,400,600]
        self.reset_clip_pos = [0.015,0.015,0.015, 0.015,0.015, 0.03, 0.03]
        reset_duration_new = []
        reset_duration_new.append(self.reset_duration[0] * factor)
        for i in range(len(self.reset_duration) - 1):
            reset_duration_new.append(
                (self.reset_duration[i + 1] - self.reset_duration[i]) * factor + reset_duration_new[i])
        print(reset_duration_new)
        self.reset_duration = reset_duration_new

        # # PARAMS FOR THE RESET PRIMITIVE
        # self.reset_grasp_xy = [0.1,0.2,0.2]
        # self.reset_grasp_h = [0.0,0.05,0.2]
        # self.reset_duration = [100,200,350]#[200,400,600]
        # self.reset_clip_pos = [0.03,0.015,0.015]

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

        if (env.simulation):
            self.grasp_xy_floor = -0.002748661208897829-0.04+0.0034238076210021985 + 0.019999999999999997
            self.grasp_h_floor = 0.0+0.02 -0.02838870149105787
            self.gain_xy_pre_floor = 294.8115234375
            self.gain_z_pre_floor = 249.4430389404297
            self.init_dur_floor = 100
            self.gain_xy_final_floor = 693.0245697498322
            self.gain_z_final_floor = 744.9281394481659
            self.pos_gain_impedance_floor = 0.015#0.032040052115917206
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
        # NOTE: THOSE REAL PARAMS ARE NOT FULLY SET YET,...
        if (env.simulation):
            self.grasp_xy_rotate_ground = 0.0022661592811346054 - 0.04 + 0.0012392520904541023#-0.007273907661437987#+ 0.0012835693359374983#+ 0.00017907857894897475
            self.grasp_h_rotate_ground = -0.0036394810304045677 + 0.006026389598846434 #0.009729766845703126#0.0013717865943908697#+ 0.006139404773712158
        else:
            self.grasp_xy_rotate_ground = 0.0022661592811346054 - 0.02 -0.004453001022338867#3+ 0.008769088983535766#+ 0.015980666875839232
            self.grasp_h_rotate_ground = 0.0#-0.0036394810304045677 + 0.011232354640960694#-0.00455500602722168

        self.gain_xy_pre_rotate_ground = 294.8115234375
        self.gain_z_pre_rotate_ground = 249.4430389404297
        self.init_dur_rotate_ground = 100
        if (env.simulation):
            self.gain_xy_final_rotate_ground = 556.7359089851379#803.606367111206#400.0#562.4382019042969
            self.gain_z_final_rotate_ground = 1211.8926048278809#773.1327414512634#1124.0761876106262#1127.987027168274
            self.pos_gain_impedance_rotate_ground = 0.0326704466342926#0.03204612851142883#0.03113249659538269#0.03760003924369812
            self.force_factor_rotate_ground = 0.9432543694972992#1.11380175948143#0.3808450520038605#0.21552527211606504 # perhaps use 0.6 instead of 1.11
            self.force_factor_ground_rot_rotate_ground = 0.2#0.14088733196258546#0.2#0.12187141180038452
            self.gain_scheduling = 20.0
        else:
            self.gain_xy_final_rotate_ground = 736.6782665252686#499.13966059684753#605.1593139767647#704.6562671661377#611.3788452148438
            self.gain_z_final_rotate_ground = 1000.9935677051544#1197.0452547073364#1243.352460861206#988.0030393600464#600.0
            self.pos_gain_impedance_rotate_ground = 0.026462331414222717#0.013593301996588706#0.02067116618156433#0.011825856864452361#0.0#0.024435650557279587
            self.force_factor_rotate_ground = 0.6#0.42808679342269895#0.38862146139144893#0.3449445694684983#0.61
            self.force_factor_ground_rot_rotate_ground = 0.16#0.13#0.2#0.2#0.163668292760849#0.15#0.07469592690467834#0.05#0.02#0.075
            self.gain_scheduling = 20.0

        # PARAMS ROTATE LIFT
        if (env.simulation):
            self.grasp_xy_rotate_lift = 0.0022661592811346054 - 0.04 + 0.00303934335708618#-0.02
            self.grasp_h_rotate_lift = 0.0 -0.017098468542099 #-0.00025067448616027804
        else:
            self.grasp_xy_rotate_lift = 0.0022661592811346054 - 0.04 -0.00622488021850586#+ 0.009622406959533692
            self.grasp_h_rotate_lift = 0.0 + 0.014247066974639896#+ 0.005761625766754149

        self.gain_xy_pre_rotate_lift = 294.8115234375
        self.gain_z_pre_rotate_lift = 249.4430389404297
        self.init_dur_rotate_lift = 100

        if (env.simulation):
            self.gain_xy_final_rotate_lift = 1007.8075528144836#1294.4734156131744
            self.gain_z_final_rotate_lift = 1220.6658482551575#1028.2374143600464
            self.pos_gain_impedance_rotate_lift = 0.02185524344444275#0.01539862297475338
            self.force_factor_rotate_lift = 0.6349194526672364#1.0
            self.force_factor_center_rotate_lift = 0.6068137288093567
            self.force_factor_rot_rotate_lift = 0.02#0.34193327307701116
            self.target_height_rot_lift_rotate_lift = 0.09
        else:
            self.gain_xy_final_rotate_lift = 940.2498722076416#820.5114454030991
            self.gain_z_final_rotate_lift = 1295.9117591381073#1133.7310016155243
            self.pos_gain_impedance_rotate_lift = 0.026721017956733706#0.02095555767416954
            self.force_factor_rotate_lift = 0.886374345421791#1.0
            self.force_factor_center_rotate_lift = 0.6068137288093567
            self.force_factor_rot_rotate_lift = 0.01#0.02
            self.target_height_rot_lift_rotate_lift = 0.09

        self.clip_height_rotate_lift = 0.0881337970495224

        # PARAMS FOR THE LIFT PRIMITIVE UNDER KEEPING ORIENTATION
        self.orient_int_orient_gain = 0.019696855545043947#0.0
        self.orient_int_pos_gain = 0.009067282676696778#0.0

        if (env.simulation):
            self.orient_int_orient_gain = 0.0005 #0.00127358540892601#
            self.orient_int_pos_gain = 0.0051201748847961425#0.005
            self.orient_grasp_xy_lift = -0.002748661208897829-0.04+0.0034238076210021985 + 0.019999999999999997#-0.02229524075984955
            self.orient_grasp_h_lift = 0.0+0.02 -0.02838870149105787#-0.012001985907554625
        else:
            self.orient_grasp_xy_lift = -0.002748661208897829-0.04 + 0.012545742988586423
            self.orient_grasp_h_lift = 0.0 + 0.002259004116058349
        self.orient_gain_xy_pre_lift = 294.8115234375
        self.orient_gain_z_pre_lift = 249.4430389404297
        self.orient_init_dur_lift = 100
        self.orient_gain_xy_ground_lift = 294.8115234375
        self.orient_gain_z_ground_lift = 249.4430389404297
        self.orient_pos_gain_impedance_ground_lift = 0.032040052115917206

        self.orient_switch_mode_lift = 0.035
        if (env.simulation):
            self.orient_pos_gain_impedance_ground_lift = 0.01#0.0125
            self.orient_gain_xy_ground_lift = 693.0245697498322
            self.orient_gain_z_ground_lift = 744.9281394481659
            self.orient_switch_mode_lift = 0.05
            self.orient_gain_xy_lift_lift = 700#842.1571612358093#672.3513275384903#829.543000459671
            self.orient_gain_z_lift_lift = 750#1378.0078768730164#1425.8808374404907#1101.177817583084
            self.orient_pos_gain_impedance_lift_lift = 0.0125#0.01641496576368809#0.015692157968878746#0.0197468575835228#0.015233442336320877
            self.orient_clip_height_lift = 0.0881337970495224
            self.orient_force_factor_lift = 0.59#0.35#0.5970059156417846#0.4772495344281197#0.5929181218147278
            self.orient_force_factor_rot_lift = 0.004292513728141785#0.0001#0.0014684607088565826#0.0005483480170369149#0.004665868282318115
        else:
            self.orient_gain_xy_lift_lift = 600.0
            self.orient_gain_z_lift_lift = 900.0
            self.orient_pos_gain_impedance_lift_lift = 0.020851443633437158
            self.orient_clip_height_lift = 0.0881337970495224
            self.orient_force_factor_lift = 0.35
            self.orient_force_factor_rot_lift = 0.0017775391042232514#0.0011627759784460067
        self.orient_force_factor_rot_ground = 0.0