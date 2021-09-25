import random
from functools import partial

from mctspy.policies import uct_action
from mctspy.tree import POMCP, DecisionNode
from mctspy.utilities import random_rollout_value
from simulations.tic_tac import TicTac

n_games = 5
wins = 0
verbose = False

game = TicTac()
n_iters = 10000
seed = 1337
my_agent_id = 0  # We play for X

state_value_estimator = partial(random_rollout_value, env=game, seed=seed)
mcts = POMCP(game, uct_action, state_value_estimator, n_iters)

def report(message):
    if verbose:
        print(message)

for i in range(n_games):

    state, _ = game.get_initial_state()
    mcts_root = DecisionNode(None, 0, {}, state.nextAgentId, [state], [None])

    node = mcts_root
    while not game.state_is_terminal(state):

        # Extend tree from current node. 
        # I.e. if node is not an initial state, some tree already exists
        mcts.build_tree(node)

        # Select the best "sense" action acording to UCB with 0 ecxploration
        action = uct_action(node, 0)
        state, observation, reward, agent_id = game.step(state, action)
        assert agent_id == my_agent_id

        # Move into Decision Node for "move" action
        node = node.children[action].children[observation]

        # Select the best "move" action
        action = uct_action(node, 0)
        state, observation, reward, agent_id = game.step(state, action)
        assert agent_id != my_agent_id

        # Move into Decision Node for oponnent's "sense"
        node = node.children[action].children[observation]

        # --- This part is hidden in the true simulator ---
        # Opponent's random "sense"
        action_sense = random.choice(tuple(game.enumerate_actions(state)))
        state, observation_sense, reward, agent_id = game.step(state, action_sense)
        assert agent_id != my_agent_id
        
        # Opponent's random "move"
        action_move = random.choice(tuple(game.enumerate_actions(state)))
        state, observation_move, reward, agent_id = game.step(state, action_move)
        assert agent_id == my_agent_id

        # "Promote" next node to the MCTS root
        # Problems TODO:
        # 1. In Blind Chess we don't get opponents "sense" action and observation
        # 2. Even if would get them, what if we didn't add the node in our tree
        #    which was actually selected by opponenet? What would happen to belief state?
        history = node.history
        node = node.children[action_sense].children[observation_sense]
        node = node.children[action_move].children[observation_move]
        node.history = history


    report(f"Winner: {state.winnerId}")
    if state.winnerId == 0:
        wins += 1

print(f"Winrate: {wins / n_games}")
