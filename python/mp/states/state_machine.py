class State(object):
    def __init__(self, env):
        self.env = env

    def connect(self, *states):
        raise NotImplementedError

    def __call__(self, obs, info=None):
        raise NotImplementedError

    def reset(self):
        """clear any internal variables this state may keep"""
        raise NotImplementedError

    def get_action(self, position=None, torque=None, frameskip=1):
        return {'position': position, 'torque': torque, 'frameskip': frameskip}


class StateMachine(object):
    def __init__(self, env):
        self.env = env
        self.init_state = self.build()

    def build(self):
        """Instantiate states and connect them.

        make sure to make all of the states instance variables so that
        they are reset when StateMachine.reset is called!

        Returns: base_states.State (initial state)


        Ex:

        self.state1 = State1(env, args)
        self.state2 = State2(env, args)

        self.state1.connect(...)
        self.state2.connect(...)

        return self.state1
        """
        raise NotImplementedError

    def reset(self):
        self.state = self.init_state
        self.info = {}
        print("==========================================")
        print("Resetting State Machine...")
        print(f"Entering State: {self.state.__class__.__name__}")
        print("==========================================")
        for attr in vars(self).values():
            if isinstance(attr, State):
                attr.reset()

    def __call__(self, obs):
        prev_state = self.state
        action, self.state, self.info = self.state(obs, self.info)
        if prev_state != self.state:
            print("==========================================")
            print(f"Entering State: {self.state.__class__.__name__}")
            print("==========================================")
        if action['frameskip'] == 0:
            return self.__call__(obs)
        else:
            return action
