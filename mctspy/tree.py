
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



def ucb_action(decision_node: DecisionNode):
    """ Select the action in decision node according to UCB formula.
    """
    pass


def random_action(decision_node: DecisionNode):
    """ Choose actoins randomly at every decision node.
    """
    return random.choice(tuple(decision_node.children.keys()))



class MCST:

    def __init__(
        self, num_iterations: int, action_selector: Callable, state_value_estimator: Callable
    ) -> None:
        
        self.num_iterations = num_iterations
        self.action_selector = action_selector
        self.state_value_estimator = state_value_estimator

        self.root = None
        self.stack = None

    def build_tree(self, state: Hashable):
        self.root = DecisionNode(state, 0, {})
        self.stack = deque()

        for i in range(self.num_iterations):
            # Assert that every iteration is started with clear stack
            assert not self.stack

            # Rollout siulation and expand tree
            value = MCST.expand(self.root, self.stack, self.state_value_estimator)

            # Backup the resulting value
            MCST.backup(value, self.stack)

    @staticmethod
    def expand(
        node: DecisionNode, stack: deque, action_selector: Callable, state_value_estimator: Callable
    ) -> float: 
        
        # Start at provided node (root)
        current_node = node
        # Go untill the terminal state
        while not current_node.state.is_end_of_game():
            # Get available actions
            available_actions = set(current_node.state.enumerate_moves())
            
            # If not all actions were tried at least once
            if not current_node.children.keys() == available_actions:
                # Sample random from untried actions
                action = random.choice(tuple(available_actions - current_node.children.keys()))
                # Advance simulation
                # TODO: implement intermediate rewards for common RL interface
                next_state = current_node.state.apply_move(action)

                # Append nodes to the tree
                decision_node = DecisionNode(next_state, 0, {})
                chance_node = ChanceNode(action, 1, 0, {next_state: decision_node})
                current_node.children[action] = chance_node

                # Record chane nodes for backpropagation
                stack.append(chance_node)

                # Rollout simulation till the end using default policy
                # TODO: implement off tree insertion
                return state_value_estimator(next_state)
        
            # Select the best action according to provided policy
            action = action_selector(current_node)
            # Advance simulation
            next_state = current_node.state.apply_move(action)
            
            # Increment visites of Chance Node
            chance_node = current_node.children[action]
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
        # Go in reverse over stack and update nodes
        while stack:
            chance_node = deque.pop()

            # Accumulate value and counts
            chance_node.visits += 1
            chance_node.value += value
