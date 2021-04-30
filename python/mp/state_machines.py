from mp import states


class MPStateMachine(states.StateMachine):
    def build(self):
        self.goto_init_pose = states.GoToInitPoseState(self.env)
        self.align_object = states.AlignObjectSequenceState(self.env)
        self.planned_grasp = states.PlannedGraspState(self.env)
        self.move_to_goal = states.MoveToGoalState(self.env)
        self.wait = states.WaitState(
            self.env, 30 if self.env.simulation else 1000)
        self.failure = states.FailureState(self.env)

        # define transitions between states
        self.goto_init_pose.connect(next_state=self.align_object,
                                    failure_state=self.failure)
        self.align_object.connect(next_state=self.planned_grasp,
                                  failure_state=self.goto_init_pose)
        self.planned_grasp.connect(next_state=self.move_to_goal,
                                   failure_state=self.align_object)
        self.move_to_goal.connect(next_state=None, failure_state=self.wait)
        self.wait.connect(next_state=self.goto_init_pose,
                          failure_state=self.failure)

        # return initial state
        return self.goto_init_pose
