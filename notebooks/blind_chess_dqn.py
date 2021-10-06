import functools
import operator
import random
import itertools

import chess
import reconchess
import torch
import numpy as np

from agents.blind_chess import TestQNet, QAgent, Transition
from utilities.replay_buffer import HistoryReplayBuffer


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
    n_steps = 10
    n_batches_per_step = 10
    n_games_per_step = 10
    batch_size = 10
    slice_size = 100

    q_nets = [
        TestQNet(narx_memory_length, n_hidden),
        TestQNet(narx_memory_length, n_hidden),
    ]

    agents = [QAgent(net, functools.partial(policy_sampler, eps=0.5), narx_memory_length) for net in q_nets]
    replay_buffer = HistoryReplayBuffer(1000, (8, 8, 13), tuple())

    for i_step in range(n_steps):
        for i_game in range(n_games_per_step):
            winner_color, win_reason, game_history = reconchess.play_local_game(agents[0], agents[1])
            # TODO: Adjust the rewards like in AlphaStar (propagate from the last step back).
            # TODO: Check that the agent's color is randomly selected.

            white_index = 0 if agents[0].color == chess.WHITE else 1
            black_index = 1 - white_index
            # history_length = min(len(agent.history) for agent in agents)
            # for i_move in range(history_length - 1):
            #     # TODO: Correct?
            #     agents[white_index].history[i_move].action_opponent = agents[black_index].history[i_move].action
            #     agents[black_index].history[i_move].action_opponent = agents[white_index].history[i_move].action

            padded_zipper = functools.partial(
                itertools.zip_longest, fillvalue=Transition(None, -1, None)
            )
            for transition_white, transition_black, transition_white_next in padded_zipper(
                agents[white_index].history[1::2],
                agents[black_index].history[1::2],
                agents[white_index].history[3::2],
            ):
                transition_white.action_opponent = transition_black.action
                transition_black.action_opponent = transition_white_next.action

            # TODO: Save opponent's actions to train the predictor.
            replay_buffer.add(Transition.stack(agents[0].history))
            replay_buffer.add(Transition.stack(agents[1].history))


        for i_batch in range(n_batches_per_step):
            batch_obs, batch_act, batch_rew, batch_obs_next = replay_buffer.sample_batch(batch_size, slice_size)

            batch_q = q_nets[0]()





if __name__ == '__main__':
    main()
