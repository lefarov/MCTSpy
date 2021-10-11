from __future__ import annotations
import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mctspy.simluator import SimulatorInterface

from mctspy.tree import DecisionNode, POMCP, ChanceNode

import random
from collections import defaultdict


__all__ = ["random_rollout_value", "pull_children_belief_state"]


def random_rollout_value(state, seed: int, env: SimulatorInterface):
    """ Rollout the environment till terminal state with random actions.
    """
    random.seed(seed)
    cumulative_reward = defaultdict(int)

    while not env.state_is_terminal(state):
        agent_id = env.get_current_agent(state)
        state, _, reward, _ = env.step(
            state, 
            random.choice(tuple(env.enumerate_actions(state)))
        )
        cumulative_reward[agent_id] += reward

    terminal_value = env.get_terminal_value(state)
    for agent, value in terminal_value.items():
        cumulative_reward[agent] += value

    return cumulative_reward


def pull_children_belief_state(node: DecisionNode, agent_id: int, last_observation: t.Hashable):
    """ Combine the particles from the children's belief states.
    
    This method should be used in cases when we can't decide which child
    should be promoted to the root and need to "re-estimate" the belief state.
    
    Example: Multi-agent POMDP, where you don't have an access to other agents'
    actions and resulted observations.\

    """

    belief_state = []
    __combine_particles_dfs(node, agent_id, belief_state, last_observation)

    if not belief_state:
        raise ValueError(
            "There's no child nodes of the given agent id. Check your tree depth."
        )

    return belief_state


def __combine_particles_dfs(node: DecisionNode, agent_id: int, particles: list,
                            last_observation: t.Optional[t.Hashable] = None):

    doesHistoryMatch = last_observation is None or node.history[agent_id][-1] == last_observation
    if isinstance(node, DecisionNode) and node.agent_id == agent_id and doesHistoryMatch:
        particles.extend(node.belief_state)
        return

    for child in node.children.values():
        __combine_particles_dfs(child, agent_id, particles)


