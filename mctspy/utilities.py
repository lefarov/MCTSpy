from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mctspy.simluator import SimulatorInterface


import random
from collections import defaultdict


def random_rollout_value(state, seed: int, env: SimulatorInterface):
    """ Rollout the environment till terminal state with random actions.
    """
    random.seed(seed)
    cummulative_reward = defaultdict(int)

    while not env.state_is_terminal(state):
        agent_id = env.get_current_agent(state)
        state, reward, next_agent_id = env.step(
            state, 
            random.choice(tuple(env.enumerate_actions(state)))
        )
        cummulative_reward[agent_id] += reward
        agent_id = next_agent_id

    terminal_value = env.get_terminal_value(state)
    for agent, value in terminal_value.items():
        cummulative_reward[agent] += value

    return cummulative_reward