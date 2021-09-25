from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mctspy.simluator import SimulatorInterface

from mctspy.tree import DecisionNode

import random
from collections import defaultdict


__all__ = ["random_rollout_value", "pull_childrens_belief_state"]


def random_rollout_value(state, seed: int, env: SimulatorInterface):
    """ Rollout the environment till terminal state with random actions.
    """
    random.seed(seed)
    cummulative_reward = defaultdict(int)

    while not env.state_is_terminal(state):
        agent_id = env.get_current_agent(state)
        state, _, reward, _ = env.step(
            state, 
            random.choice(tuple(env.enumerate_actions(state)))
        )
        cummulative_reward[agent_id] += reward

    terminal_value = env.get_terminal_value(state)
    for agent, value in terminal_value.items():
        cummulative_reward[agent] += value

    return cummulative_reward


def pull_childrens_belief_state(node: DecisionNode, agent_id: int):
    """ Combine the particles from the children's belief states.
    
    This method should be used in cases when we can't decide which child
    should be promoted to the root and need to "re-estimate" the belief state.
    
    Example: Multi-agent POMDP, where you don't have an access to other agents'
    actions and resulted observations.
    """

    belief_state = []
    __combine_particles_dfs(node, agent_id, belief_state)

    if not belief_state:
        raise ValueError(
            "There're no child nodes of the given agent id. Check your tree depth."
        )

    return belief_state


def __combine_particles_dfs(node: DecisionNode, agent_id: int, particles: list):
    if isinstance(node, DecisionNode) and node.agent_id == agent_id:
        particles.extend(node.belief_state)
        return

    for child in node.children.values():
        __combine_particles_dfs(child, agent_id, particles)

    return