
from mctspy.simluator import SimulatorInterface


class FrozenLakeSimulator(SimulatorInterface):

    def __init__(self, env):
        self.env = env

    def step(self, state, action):
        self.env.s = state
        next_state, reward, *_ = self.env.step(action)
        
        return next_state, reward, "agent_0"

    def state_is_terminal(self, state):
        return self.env.desc.flat[state] in (b"G", b"H")

    def enumerate_actions(self, state):
        return set(range(self.env.action_space.n))

    def get_initial_state(self):
        return self.env.reset(), "agent_0"

    def get_agent_num(self):
        return 1

    def get_current_agent(self, state):
        return "agent_0"

    def get_terminal_value(self, state):
        return {"agent_0": 0.0}