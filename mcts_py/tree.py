
import random

from collections import deque
from typing import Hashable


@dataclass
class DecisionNode:
    state: Hashable  # State of simulation
    children: dict  # Dict of decision nodes


@dataclass
class ChanceNode:
    action: Hashable  # Taken action
    visits: int  # Number of visits
    value: float  # Cummulative value
    children: dict  # Dict of chanve nodes


class MCST:

    def __init__(self, num_iterationsm, simulation_runner):
        self.env = env
        self.num_iterationsm = num_iterationsm
        self.simulation_runner = simulation_runner

        self.root = None
        self.stack = None

    def build_tree(self, state: Hashable)
        # Reset tree
        self.root = DecisionNode(state, set())
        self.stack = deque()

        for i in range(self.num_iterations):
            self.expand(self.root)
            self.backup()

    def expand(self, node: DecisionNode)
        curr = node
        
        while not curr.state.is_end_of_game():
            # Get available actions
            available_actions = set(curr.state.enumerate_moves())
            # If all actions were tried at least once
            if curr.children.keys() == available_actions:
                # Go to the next state
                curr = dn


            else:
                # Sample random from untried actions
                # TODO: check if for Python 3.9 convertion to tuple is required
                act = random.sample(available_actions - curr.children.keys())
                # Advance simulation
                next_state = curr.state.apply_move(act)
                # Rollout simulation till the end using default policy
                value = self.simulation_runner(next_state)

                # Append nodes to the tree
                dn = DecisionNode(next_state, {})
                cn = ChanceNode(act, 0, 0, {act: dn})
                # Record nodes for backpropagation
                self.stack.append(cn)
                self.stack.append(dn)



    def backup(self)
        # Go in reverse over self.stack and update nodes
        pass