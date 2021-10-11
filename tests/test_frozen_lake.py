
import random
import pytest
gym_envs = pytest.importorskip("gym.envs.toy_text")

from functools import partial
from collections import defaultdict

from mctspy.tree import DecisionNode, MCTS
from mctspy.policies import uct_action
from mctspy.simulator import SimulatorInterface


class FrozenLakeMCTS(SimulatorInterface):

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


def random_rollout_value(state, env: FrozenLakeMCTS):
    """ Rollout the environment till terminal state with random actions.
    """
    cummulative_reward = defaultdict(int)

    while not env.state_is_terminal(state):
        agent_id = env.get_current_agent(state)
        state, reward, next_agent_id = env.step(
            state, 
            random.choice(tuple(env.enumerate_actions(state)))
        )
        cummulative_reward[agent_id] += reward
        agent_id = next_agent_id

    return cummulative_reward


def test_build_tree():
    random.seed(0)

    env = gym_envs.FrozenLakeEnv(is_slippery=False, map_name="4x4")
    env = FrozenLakeMCTS(env)
    
    # initial_state, agent_id = env.get_initial_state()
    initial_state, agent_id = 10, "agent_0"

    mcts = MCTS(env, uct_action, partial(random_rollout_value, env=env), 50)
    mcts_root = DecisionNode(initial_state, 0, {}, agent_id)
    
    mcts.build_tree(mcts_root)
    
    pass


def test_play():
    random.seed(0)

    env = gym_envs.FrozenLakeEnv(is_slippery=False, map_name="4x4")
    env = FrozenLakeMCTS(env)
    
    state = env.get_initial_state()
    trajectory = [state]

    mcts = MCTS(env, uct_action, partial(random_rollout_value, env=env), 50)
    mcts_root = DecisionNode(state, 0, 0, {})
    current = mcts_root

    while not env.state_is_terminal(state):
        mcts.build_tree(current)

        action = uct_action(current, 0)
        state, reward = env.step(state, action)
        
        current = current.children[action].children[state]
        
        trajectory.append(state)

    pass
