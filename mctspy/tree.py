
import abc
import random

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Callable, Tuple, Set, Deque


@dataclass
class DecisionNode:
    state: Hashable  # State of simulation
    reward: float  # Last observed reward in this state
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



class MCTSSimulatorInterface(abc.ABC):

    @abs.abstractmethod
    def step(self, state: Hashable, action: Hashable) -> Tuple[Hashable, float]:
        """ Step through simulation.
        """
        pass

    @abs.abstractmethod
    def state_is_terminal(self, state: Hashable) -> bool:
        """ Check if state is terminal.
        """
        pass

    @abs.abstractmethod
    def enumerate_actions(self, state: Hashable) -> Set:
        """ Enumerate all possivle actions for given state
        """
        pass



class MCST:

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

        self.root = None
        self.stack = None

    def build_tree(self, state: Hashable):
        self.root = DecisionNode(state, 0, {})
        self.stack = deque()

        for i in range(self.num_iterations):
            # Assert that every iteration is started with clear stack
            assert not self.stack

            # Rollout siulation and expand tree
            value = MCST.expand(
                self.root, 
                self.simulator, 
                self.stack, 
                self.action_selection_policy, 
                self.state_value_estimator
            )

            # Backup the resulting value
            MCST.backup(value, self.stack)

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
                    reward + state_value_estimator(next_state), 
                    {}
                )
                chance_node = ChanceNode(action, 0, 0, {next_state: decision_node})
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
                chance_node.children[next_state] = DecisionNode(next_state, reward, {})
            
            # Record nodes for backpropagation
            stack.append(chance_node)
            stack.append(chance_node.children[next_state])

            # Go to the next state
            current_node = chance_node.children[next_state]

    @staticmethod
    def backup(stack: Deque):
        # Total Rollout Return (discounted)
        cummulative_reward = 0
        
        # Go in reverse over stack and update nodes
        while stack:
            current_node = stack.pop()

            if isinstance(current_node, DecisionNode):
                cummulative_reward += current_node.reward

            else:
                # Accumulate total returns
                current_node.visits += 1
                current_node.value += cummulative_reward
