import random
from functools import partial

from mctspy.policies import uct_action
from mctspy.tree import POMCP, DecisionNode
from mctspy.utilities import random_rollout_value, pull_children_belief_state
from bots.tictac.simulator import TicTac


def report(message):
    if verbose:
        print(message)


n_games = 1
wins = 0
verbose = False
promote_opponent_child = True

game = TicTac()
# TODO: even 1000 iterations we don't try every opponents move,
#       that results in key error! WTF? 1000 iterations, Carl!
n_iters = 1000
seed = 1337
my_agent_id = 0  # We play for X

state_value_estimator = partial(random_rollout_value, env=game, seed=seed)
mcts = POMCP(game, uct_action, state_value_estimator, n_iters)

for i in range(n_games):

    state, _ = game.get_initial_state()
    mcts_root = DecisionNode(None, 0, {}, state.nextAgentId, [state], {state.nextAgentId: [None]})

    node = mcts_root
    while not game.state_is_terminal(state):

        # TODO: make TicTacState hashable and assert that state is in node's belief state

        # Extend tree from current node. 
        # I.e. if node is not an initial state, some tree already exists
        mcts.build_tree(node)

        # Select the best "sense" action according to UCB with 0 exploration
        action = uct_action(node, 0)
        state, observation, reward, agent_id = game.step(state, action)
        assert agent_id == my_agent_id

        # Move into Decision Node for "move" action
        node = node.children[action].children[observation]

        # Select the best "move" action
        action = uct_action(node, 0)
        state, observation, reward, agent_id = game.step(state, action)
        assert agent_id != my_agent_id

        # Move into Decision Node for opponent's "sense"
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
        # --- End of the hidden part ---

        # Check if the game was ended by me or opponent
        # (we need this additional check since the children in MCTS will be empty in this case)
        if game.state_is_terminal(state):
            break

        if promote_opponent_child:
            # "Promote" next node to the MCTS root
            history = node.history
            node = node.children[action_sense].children[observation_sense]
            node = node.children[action_move].children[observation_move]
            # We discard the opponent history since we don't know it for the Blind Chess simulator
            node.history = history
            # At this point, our node contains the correct belief state (with opponent's move)
            # and correct history, i.e. where the last observation is actually one recorded by our agent

        else:
            updated_belief_state = pull_children_belief_state(node, my_agent_id, last_observation=observation_move)
            # We still don't know which opponent's node to select,
            # so we probably need to discard the built sub-tree altogether.
            # TODO: in Blind Chess we can see move action and observation, 
            #       so maybe we can sample the branch that has the same move action and observation
            #       important! (we don't have these opponent moves, if opponent didn't take our figure)
            node = DecisionNode(None, 0, {}, my_agent_id, updated_belief_state, node.history)

        # Problems TODO:
        # 1. In Blind Chess we don't get opponents "sense" action and observation
        # 2. Even if would get them, what if we didn't add the node in our tree
        #    which was actually selected by opponent? What would happen to belief state?
        # Possible solutions:
        # 2. Should we then construct belief state that include all possible opponents moves?
        #    But all possible moves from which state? If in our current belief state more than
        #    one particle, it will exlode!

    report(f"Winner: {state.winnerId}")
    if state.winnerId == 0:
        wins += 1

print(f"Winrate: {wins / n_games}")
