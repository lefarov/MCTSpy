from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mctspy.simulator import SimulatorInterface, SimulatorInterfacePO

import random
import typing as t

from collections import deque, defaultdict
from dataclasses import dataclass


@dataclass
class DecisionNode:
    """ Decision Node of Monte Carlo Tree Search with multi-agent support.

    Parameteres
    -----------
    observation: hashable
        Observation of a simulation's state. The state itself in case of Fully-observed MDP.
    visits: int
        Total number of visits for Decision Node (i.e, sum of children's' visits).
    children: dict
        Dictionary for children Chance Nodes.
    agent_id: hashable
        Id of the agent which has to take an action for multi-agent simulation (e.g. multiplayer game).
    belief_state: list
        List of probably true system states
        (i.e. particles from https://papers.nips.cc/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf).
    """
    observation: t.Hashable
    visits: int
    children: t.Dict
    agent_id: t.Hashable
    belief_state: t.List[t.Hashable] = None
    history: t.Dict[t.Hashable, t.List] = None


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
        Estimate of a state-action pair value, i.e. Q(s, a) in RL notation.
    children: dict
        Dictionary for children Decision Nodes.
    agent_id: hashable
        Id of the agent who took the action for multi-agent simulation (e.g. multiplayer game).
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
        simulator: SimulatorInterface,  
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
        simulator: SimulatorInterface,
        stack: t.Deque, 
        action_selection_policy: t.Callable[[DecisionNode], t.Hashable], 
        state_value_estimator: t.Callable[[t.Hashable], t.Dict],
    ) -> float:
        """ Expand the tree.

        Function traverses the Tree by iterating over the Decision Nodes 
        until a Decision Node with at least one untried action or terminal 
        state is found. Provided action selection policy is used to select
        an action during the traverse.
        
        For the Decision Node with untried actions, one of those is sampled 
        uniformly. The state value estimator is then used to compute the value 
        of its successor state. For the Decision Node with terminal state the
        final value is computed by simulator. The state value is then returned.

        All Chance nodes are added to the stack, that is used for backing up 
        the rewards and the terimal value (actual or estimated).

    Parameters
        ----------
        node: DecisionNode
            Root Decision Node.
        simulator: MCTSSimulator
            Simulator that follows the MCTS simulation interface.
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

        while not simulator.state_is_terminal(current_node.observation):
            
            current_node.visits += 1
            available_actions = simulator.enumerate_actions(current_node.observation)
            
            # Decision Node with untried actions is found
            if not set(current_node.children.keys()) == available_actions:
                action = random.choice(tuple(available_actions - current_node.children.keys()))
                next_state, reward, *_ = simulator.step(current_node.observation, action)
                
                chance_node = ChanceNode(action, 1, reward, 0.0, {}, current_node.agent_id)
                current_node.children[action] = chance_node
                
                stack.append(chance_node)

                # Return the estimate of a successor state value
                return state_value_estimator(next_state)
        
            action = action_selection_policy(current_node)
            next_state, reward, next_agent_id = simulator.step(current_node.observation, action)
            
            chance_node = current_node.children[action]
            chance_node.reward = reward
            chance_node.visits += 1
            
            stack.append(chance_node)

            if next_state not in chance_node.children:
                chance_node.children[next_state] = DecisionNode(next_state, 0, {}, next_agent_id)

            current_node = chance_node.children[next_state]

        # Decision Node with terminal state is found. Return the value computed by simulator.
        return simulator.get_terminal_value(current_node.observation)

    @staticmethod
    def backup(stack: t.Deque, terminal_value: t.Dict):
        """ Backup the terminal state value and recorded rewards.

        Parameters
        ----------
        """
        cumulative_rewards = defaultdict(int)
        cumulative_rewards.update(terminal_value)

        while stack:
            current_node = stack.pop()
            cumulative_rewards[current_node.agent_id] += current_node.reward
            current_node.value += cumulative_rewards[current_node.agent_id]


class POMCP:

    def __init__(
            self,
            simulator: SimulatorInterface,
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

            value = POMCP.expand(
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
        simulator: SimulatorInterface,
        stack: t.Deque, 
        action_selection_policy: t.Callable[[DecisionNode], t.Hashable], 
        state_value_estimator: t.Callable[[t.Hashable], t.Dict],
    ) -> float:
        current_node = node

        # Sample one of the Belief state
        state = random.choice(current_node.belief_state)
        agent_histories = defaultdict(list)
        # Copy recorded node history up to this node
        agent_histories.update(node.history)

        while not simulator.state_is_terminal(state):
            
            current_node.visits += 1
            available_actions = simulator.enumerate_actions(state)
            agent_id = current_node.agent_id
            
            # Decision Node with untried actions is found
            if not current_node.children.keys() == available_actions:
                action = random.choice(tuple(available_actions - current_node.children.keys()))
                next_state, _, reward, *_ = simulator.step(state, action)
                
                chance_node = ChanceNode(
                    action=action, 
                    visits=1, 
                    reward=reward, 
                    value=0.0, 
                    children={}, 
                    agent_id=agent_id
                )

                current_node.children[action] = chance_node
                stack.append(chance_node)

                # Return the estimate of a successor state value
                return state_value_estimator(next_state)
        
            action = action_selection_policy(current_node)
            next_state, next_observation, reward, next_agent_id = simulator.step(state, action)
            next_observation = next_observation[next_agent_id]
            
            chance_node = current_node.children[action]
            chance_node.reward = reward
            chance_node.visits += 1
            
            stack.append(chance_node)

            # Add action and observation to the agent's history
            agent_histories[agent_id].extend((action, next_observation))
            # If we do switch the agents
            if next_agent_id != agent_id:
                # Add observation to the next agent's history
                agent_histories[next_agent_id].append(next_observation)

            # Add new Decision Node if not existing
            if next_observation not in chance_node.children:
                chance_node.children[next_observation] = DecisionNode(
                    observation=next_observation,
                    visits=0,
                    children={},
                    agent_id=next_agent_id,
                    belief_state=[],
                    history=agent_histories
                )

            current_node = chance_node.children[next_observation]

            # Add actual sampled state to the belief state
            # TODO: use heuristic to add possible particles (by perturbation) ?
            current_node.belief_state.append(next_state)
            # Transition into the next state
            state = next_state

        # Decision Node with terminal state is found. Return the value computed by simulator.
        return simulator.get_terminal_value(state)

    def backup(stack: t.Deque, terminal_value: t.Dict):
        """ Backup the terminal state value and recorded rewards.

        Parameters
        ----------
        """
        cumulative_rewards = defaultdict(int)
        cumulative_rewards.update(terminal_value)

        while stack:
            current_node = stack.pop()
            cumulative_rewards[current_node.agent_id] += current_node.reward
            current_node.value += cumulative_rewards[current_node.agent_id]
    