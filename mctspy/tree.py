
import random

from collections import deque
from typing import Hashable, Callable


@dataclass
class DecisionNode:
    state: Hashable  # State of simulation
    value: float  # Estimate of a state value, i.e V(s) in RL notation
    children: dict  # Dict of decision nodes


@dataclass
class ChanceNode:
    action: Hashable  # Taken action
    visits: int  # Number of visits
    value: float  # Estiamte of a state-action pair value, i.e. Q(s, a) in RL notation
    children: dict  # Dict of chanve nodes



def ucb_action(decision_node):
    """ Select the action in decision node according to UCB formula
    """
    pass



class MCST:

    def __init__(self, num_iterations, state_value_estimator):
        self.num_iterations = num_iterations
        self.state_value_estimator = state_value_estimator

        self.root = None
        self.stack = None

    def build_tree(self, state: Hashable):
        # Reset tree
        self.root = DecisionNode(state, 0, {})
        self.stack = deque()

        for i in range(self.num_iterations):
            value = self.expand(self.root, self.stack, self.state_value_estimator)
            self.backup(value, self.stack)

    @staticmethod
    def expand(node: DecisionNode, stack: deque, state_value_estimator: Callable) -> float: 
        # Start at provided node (root)
        current_node = node
        
        while not current_node.state.is_end_of_game():
            # Get available actions
            available_actions = set(current_node.state.enumerate_moves())
            
            # If not all actions were tried at least once
            if not current_node.children.keys() == available_actions:
                # Sample random from untried actions
                action = random.choice(tuple(available_actions - current_node.children.keys()))
                # Advance simulation
                next_state = current_node.state.apply_move(action)

                # Append nodes to the tree
                decision_node = DecisionNode(next_state, 0, {})
                chance_node = ChanceNode(action, 1, 0, {next_state: decision_node})
                current_node.children[action] = chance_node

                # Record chane nodes for backpropagation
                stack.append(chance_node)

                # Rollout simulation till the end using default policy
                # TODO: should we implement off tree insertion?
                return state_value_estimator(next_state)
        
            # Select the best action according to UCB
            action = ucb(current_node)
            # Advance simulation
            next_state = current_node.state.apply_move(action)
            
            # Increment visites of Chance Node
            chance_node = current_node.children[action]
            chance_node.visits += 1
            # Append nodes to the tree
            if next_state not in chance_node.children:
                decision_node = DecisionNode(next_state, 0, {})
                chance_node.children[next_state] = decision_node
            
            # Record nodes for backpropagation
            stack.append(chance_node)

            # Go to the next state
            current_node = chance_node.children[next_state]

        return current_node.state.score_game()

    @staticmethod
    def backup(value: float, stack: deque):
        # Go in reverse over self.stack and update nodes
        pass
