from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mctspy.simluator import MCTSSimulator

import random
import typing as t

from collections import deque, defaultdict
from dataclasses import dataclass


@dataclass
class DecisionNode:
    """ Decision Node of Monte Carlo Tree Search with multi-agent support.

    Parameteres
    -----------
    state: hashable
        State of a simulation.
    visits: int
        Total number of visits for Decision Node (i.e, sum of childrens' visits).
    children: dict
        Dictionary for children Chance Nodes.
    agent_id: hashable
        Id of the agent which has to take an action for multi-agent simullation (e.g. multiplayer game).
    """
    state: t.Hashable
    visits: int
    children: t.Dict
    agent_id: t.Hashable


@dataclass
class ChanceNode:
    """ Chance Node of Monte Carlo Tree Search with multi-agent support.

    Parameteres
    -----------
    action: hashable
        Taken Action.
    visits: int
        Number of visits.
    reward: float
        Last observed reward for the state-action pair.
    value: float
        Estiamte of a state-action pair value, i.e. Q(s, a) in RL notation.
    children: dict
        Dictionary for children Decision Nodes.
    agent_id: hashable
        Id of the agent who took the action for multi-agent simullation (e.g. multiplayer game).
    """
    action: t.Hashable
    visits: int
    reward: float
    value: float
    children: t.Dict
    agent_id: t.Hashable


class MCTS:

    def __init__(
        self,
        simulator: MCTSSimulator,  
        action_selection_policy: t.Callable[[DecisionNode], t.Hashable], 
        state_value_estimator: t.Callable[[t.Hashable], t.Dict[t.Hashable, float]],
        num_iterations: int,
    ) -> None:
        
        self.simulator = simulator
        self.action_selection_policy = action_selection_policy
        self.state_value_estimator = state_value_estimator
        self.num_iterations = num_iterations

        self.stack = deque()

    def build_tree(self, node: DecisionNode):
        for i in range(self.num_iterations):
            assert not self.stack

            value = MCTS.expand(
                node, 
                self.simulator, 
                self.stack, 
                self.action_selection_policy, 
                self.state_value_estimator
            )

            MCTS.backup(self.stack, value)

    @staticmethod
    def expand(
        node: DecisionNode,
        simulator: MCTSSimulator,
        stack: t.Deque, 
        action_selection_policy: t.Callable[[DecisionNode], t.Hashable], 
        state_value_estimator: t.Callable[[t.Hashable], t.Dict],
    ) -> float:
        """ Expand the tree.

        Function traverses the Tree by iterating over the Decision Nodes 
        until a Decision Node with at least one untried action or teminal 
        state is found. Provided action selection policy is used to select
        an action during the traverse.
        
        For the Decision Node with untried actions, one of those is sampled 
        uniformly. The state value estimator is then used to compute the value 
        of its successor state. For the Decision Node with terminal state the
        final value is computed by simulator. The state value is then reuturned.

        All Decision and Chance nodes are added to the stack, that is used
        for backing up the rewards and the terimal value (actual or estimated).

        Parameters
        ----------
        node: DecisionNode
            Root Decision Node.
        simulator: MCTSSimulator
            Simulator that follows the MCTS simulatio interface.
        action_selection_policy: callable(DecisionNode) -> hashable
            Policy that computes the action to take for the given Decision Node
        state_value_estimator: callable(hashable) -> dict
            Function that estimates the value for non-terminal Decision Node.
            It's used during the expansion of Desion Node with untried actions.

        Returns
        -------
        dict:
            Value of the terminal state for every agent.
            Returned by the simulator or estimated with value estimator.
        """
        current_node = node

        while not simulator.state_is_terminal(current_node.state):
            
            stack.append(current_node)
            available_actions = simulator.enumerate_actions(current_node.state)
            
            # Decision Node with untried actions is found
            if not current_node.children.keys() == available_actions:
                action = random.choice(tuple(available_actions - current_node.children.keys()))
                next_state, reward, *_ = simulator.step(current_node.state, action)
                
                chance_node = ChanceNode(action, 0, reward, 0.0, {}, current_node.agent_id)
                current_node.children[action] = chance_node
                stack.append(chance_node)

                # Return the estimate of a successor state value
                return state_value_estimator(next_state)
        
            action = action_selection_policy(current_node)
            next_state, reward, agent_id = simulator.step(current_node.state, action)
            
            chance_node = current_node.children[action]
            chance_node.reward = reward

            if next_state not in chance_node.children:
                chance_node.children[next_state] = DecisionNode(next_state, 0, {}, agent_id)

            stack.append(chance_node)
            current_node = chance_node.children[next_state]

        # Decision Node with terminal state is found. Return the value computed by simulator.
        return simulator.get_terminal_value(current_node.state)

    @staticmethod
    def backup(stack: t.Deque, terminal_value: t.Dict):
        """ Backup the terminal state value and recorded rewards.

        Parameters
        ----------
        """
        cummulative_rewards = defaultdict(int)
        cummulative_rewards.update(terminal_value)

        while stack:
            current_node = stack.pop()
            current_node.visits += 1

            if isinstance(current_node, ChanceNode):
                cummulative_rewards[current_node.agent_id] += current_node.reward
                current_node.value += cummulative_rewards[current_node.agent_id]
