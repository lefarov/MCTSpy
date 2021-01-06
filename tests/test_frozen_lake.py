import random

from functools import partial
from gym.envs.toy_text import FrozenLakeEnv

from mctspy.tree import DecisionNode, MCTSSimulatorInterface, MCTS, ucb_action


class FrozenLakeMCTS(MCTSSimulatorInterface):

    def __init__(self, env):
        self.env = env

    def step(self, state, action):
        self.env.s = state
        next_state, reward, *_ = self.env.step(action)
        
        return next_state, reward
    
    def state_reward(self, state):
        return 1 if self.env.desc.flat[state] == b"G" else 0

    def state_is_terminal(self, state):
        return (
            self.env.desc.flat[state] == b"G" or 
            self.env.desc.flat[state] == b"H"
        )

    def enumerate_actions(self, state):
        return set(range(self.env.action_space.n))

    def get_initial_state(self):
        return self.env.reset()


def random_rollout_value(state, env: FrozenLakeMCTS):
    """ Rollout the environment till terminal state with random actions.
    """
    cummulative_reward = 0
    while not env.state_is_terminal(state):
        state, reward = env.step(
            state, 
            random.choice(tuple(env.enumerate_actions(state)))
        )
        cummulative_reward += reward

    return cummulative_reward

def test_build_tree():
    env = FrozenLakeEnv(is_slippery=False, map_name="4x4")
    env = FrozenLakeMCTS(env)

    mcts = MCTS(env, ucb_action, partial(random_rollout_value, env=env), 5000)
    mcts_root = DecisionNode(env.get_initial_state(), 0, 0, {})
    
    mcts.build_tree(mcts_root)
    assert ucb_action(mcts_root) == 1


def test_play():
    env = FrozenLakeEnv(is_slippery=False, map_name="4x4")
    env = FrozenLakeMCTS(env)
    
    state = env.get_initial_state()
    trajectory = [state]

    mcts = MCTS(env, ucb_action, partial(random_rollout_value, env=env), 1000)
    mcts_root = DecisionNode(state, 0, 0, {})
    current = mcts_root

    while not env.state_is_terminal(state):
        mcts.build_tree(current)

        action = max(
            (chance_node for chance_node in current.children.values()), 
            key=lambda chance_node: chance_node.value / chance_node.visits
        ).action

        state, reward = env.step(state, action)
        current = current.children[action].children[state]
        
        trajectory.append(state)

    pass
