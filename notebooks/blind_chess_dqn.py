import functools
import operator
import random

import reconchess
import torch
import numpy as np

from agents.blind_chess import TestQNet, QAgent, Transition
from utilities.replay_buffer import ReplayBufferList, HistoryRaplayBuffer


def policy_sampler(q_values: torch.Tensor, valid_action_indices, eps: float = 0.05) -> int:
    # TODO: Don't mask but punish illegal moves.
    action_q_masked = torch.full_like(q_values, fill_value=torch.finfo(q_values.dtype).min)
    action_q_masked[valid_action_indices] = q_values[valid_action_indices]

    if random.random() >= eps:
        action_index = torch.argmax(action_q_masked).item()
    else:
        action_index = random.choice(valid_action_indices)

    return action_index



def main():

    narx_memory_length = 12
    n_hidden = 256

    q_nets = [
        TestQNet(narx_memory_length, n_hidden),
        TestQNet(narx_memory_length, n_hidden),
    ]

    agents = [QAgent(net, functools.partial(policy_sampler, eps=0.5), narx_memory_length) for net in q_nets]
    replay_buffer = HistoryRaplayBuffer(1000, (8, 8, 13), tuple())

    for i in range(10):
        winner_color, win_reason, game_history = reconchess.play_local_game(agents[0], agents[1])

        replay_buffer.add(Transition.stack(agents[0].history))
        replay_buffer.add(Transition.stack(agents[1].history))

    replay_buffer.sample_batch(10, 100)




if __name__ == '__main__':
    main()
