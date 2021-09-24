import random
from functools import partial

from mctspy.policies import uct_action
from mctspy.tree import MCTS, POMCP, DecisionNode
from mctspy.utilities import random_rollout_value
from simulations.tic_tac import *


game = TicTac()
seed = 1337

for i in range(1):

    state, _ = game.get_initial_state()
    while not game.state_is_terminal(state):
        # actions = game.enumerate_actions(state)

        tree = POMCP(game, uct_action, partial(random_rollout_value, env=game, seed=1337), 50)
        root = DecisionNode(None, 0, {}, state.nextAgentId, [state], [])

        tree.build_tree(root)

        # tree.
        # nextAction = random.choice(tuple(actions))
        nextAction = uct_action(root, 0)
        print(f"Next action player #{state.nextAgentId}: {nextAction}")
        state, obs, reward, _ = game.step(state, nextAction)
        print(f"Reward: {reward} Obs: {obs}")
        print(state)

    print(f"Winner: {state.winnerId}")
