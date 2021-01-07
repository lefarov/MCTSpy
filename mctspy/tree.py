
import abc
import math
import random

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Callable, Tuple, Set, Deque


@dataclass
class DecisionNode:
    state: Hashable  # State of simulation
    visits: int  # Total number of visits (i.e, sum of visits of children nodes)
    reward: float  # Last observed reward in this state
    children: dict  # Dict of decision nodes


@dataclass
class ChanceNode:
    action: Hashable  # Taken action
    visits: int  # Number of visits
    value: float  # Estiamte of a state-action pair value, i.e. Q(s, a) in RL notation
    children: dict  # Dict of chanve nodes


def ucb(
    value: float, visits: int, total_visits: int, exploration_constant: float
) -> float:
    return value / visits + exploration_constant * math.sqrt(math.log(total_visits) / visits)


def ucb_action(
    decision_node: DecisionNode, exploration_constant: float = 1 / math.sqrt(2)
):
    """ Select the action in decision node according to UCB formula.
    """
    # Construct the list of (score, action) tuples
    ucb_scores = [
        (
            ucb(chance_node.value, chance_node.visits, decision_node.visits, exploration_constant),
            action,
        ) 
        for action, chance_node in decision_node.children.items()
    ]
    # Sort tuples according to score
    ucb_scores.sort(reverse=True)
    
    # Go over sorted UCB scores and select all actions with maximu score
    best_actions = []
    for score, action in ucb_scores:
        if score < ucb_scores[0][0]:
            break

        best_actions.append(action)
    
    return random.choice(best_actions)


def random_action(decision_node: DecisionNode):
    """ Choose actoins randomly at every decision node.
    """
    return random.choice(tuple(decision_node.children.keys()))


class MCTSSimulatorInterface(abc.ABC):

    @abc.abstractmethod
    def step(self, state: Hashable, action: Hashable) -> Tuple[Hashable, float]:
        """ Step through simulation.
        """
        pass

    @abc.abstractmethod
    def state_is_terminal(self, state: Hashable) -> bool:
        """ Check if state is terminal.
        """
        pass

    @abc.abstractmethod
    def enumerate_actions(self, state: Hashable) -> Set:
        """ Enumerate all possivle actions for given state.
        """
        pass
    
    @abc.abstractmethod
    def get_initial_state(self) -> Hashable:
        """ Get initial state.
        """
        pass


class MCTS:
    # TODO: can we implement heap in MCTS to select best actions in O(1)?

    def __init__(
        self,
        simulator: MCTSSimulatorInterface,  
        action_selection_policy: Callable[[DecisionNode], Hashable], 
        state_value_estimator: Callable[[Hashable], float],
        num_iterations: int,
    ) -> None:
        
        self.simulator = simulator
        self.action_selection_policy = action_selection_policy
        # TODO: implement off tree insertion
        self.state_value_estimator = state_value_estimator
        self.num_iterations = num_iterations

        self.stack = deque()

    def build_tree(self, node: DecisionNode):
        for i in range(self.num_iterations):
            # Assert that every iteration is started with clear stack
            assert not self.stack

            # Rollout siulation and expand tree
            value = MCTS.expand(
                node, 
                self.simulator, 
                self.stack, 
                self.action_selection_policy, 
                self.state_value_estimator
            )

            # Backup the resulting value
            MCTS.backup(self.stack)

    @staticmethod
    def expand(
        node: DecisionNode,
        simulator: MCTSSimulatorInterface,
        stack: Deque, 
        action_selection_policy: Callable[[DecisionNode], Hashable], 
        state_value_estimator: Callable[[Hashable], float],
    ) -> float: 
        
        # Start at provided node (root)
        current_node = node
        stack.append(current_node)

        # Go untill the terminal state
        while not simulator.state_is_terminal(current_node.state):
            # Get available actions
            available_actions = simulator.enumerate_actions(current_node.state)
            
            # If not all actions were tried at least once
            if not current_node.children.keys() == available_actions:
                # Sample random from untried actions
                action = random.choice(tuple(available_actions - current_node.children.keys()))
                # Advance simulation
                next_state, reward = simulator.step(current_node.state, action)

                # Estimate next state value using estimator and append nodes to the tree
                decision_node = DecisionNode(
                    next_state, 
                    0,
                    reward + state_value_estimator(next_state), 
                    {}
                )
                chance_node = ChanceNode(action, 0, 0.0, {next_state: decision_node})
                current_node.children[action] = chance_node

                # Record nodes for backpropagation
                stack.append(chance_node)
                stack.append(decision_node)

                break
        
            # Select the best action according to provided policy
            action = action_selection_policy(current_node)
            # Advance simulation
            next_state, reward = simulator.step(current_node.state, action)
            
            chance_node = current_node.children[action]
            # Append nodes to the tree if needed
            if next_state not in chance_node.children:
                chance_node.children[next_state] = DecisionNode(next_state, 0, 0.0, {})

            chance_node.children[next_state].reward = reward
            # Record nodes for backpropagation
            stack.append(chance_node)
            stack.append(chance_node.children[next_state])

            # Go to the next state
            current_node = chance_node.children[next_state]

    @staticmethod
    def backup(stack: Deque):
        # Total Rollout Return (discounted)
        cummulative_reward = stack.pop().reward
        
        # Go in reverse over stack and update nodes
        while stack:
            current_node = stack.pop()
            current_node.visits += 1

            if isinstance(current_node, DecisionNode):
                cummulative_reward += current_node.reward

            else:
                current_node.value += cummulative_reward
