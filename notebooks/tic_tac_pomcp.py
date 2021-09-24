import random
from functools import partial

from mctspy.policies import uct_action
from mctspy.tree import POMCP, DecisionNode
from mctspy.utilities import random_rollout_value
from simulations.tic_tac import *

N = 100
wins = 0
verbal = False

game = TicTac()
seed = 1337
mcts = POMCP(
    game, uct_action, partial(random_rollout_value, env=game, seed=1337), 50
)

def report(message):
    if verbal:
        print(message)

for i in range(N):

    state, _ = game.get_initial_state()
    while not game.state_is_terminal(state):

        root = DecisionNode(None, 0, {}, state.nextAgentId, [state], [])
        mcts.build_tree(root)

        # tree.
        # nextAction = random.choice(tuple(actions))
        nextAction = uct_action(root, 0)
        report(f"Next action player #{state.nextAgentId}: {nextAction}")
        state, obs, reward, _ = game.step(state, nextAction)
        report(f"Reward: {reward} Obs: {obs}")
        report(state)

    report(f"Winner: {state.winnerId}")
    if state.winnerId == 0:
        wins += 1

print(f"Winrate: {wins / N}")
