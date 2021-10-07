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

    narx_memory_length = 50
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
            # TODO: Implement sampling of the colors.

            white_index = 0 if agents[0].color == chess.WHITE else 1
            black_index = 1 - white_index

            # Zipper that will iterate until the end of the longest sequence and
            # pad missing data of shorter sequences with the transitions containing
            # default opponent action as the recorded action.
            padded_zipper = functools.partial(
                itertools.zip_longest, fillvalue=Transition(None, -1, None)
            )

            # Iterate over 3 transitions windows: (1) with Move actions of the white player
            # (2) Move actions of the black player and (3) white Move actions shifted by one timestep forward.
            for transition_white, transition_black, transition_white_next in padded_zipper(
                agents[white_index].history[1::2],
                agents[black_index].history[1::2],
                agents[white_index].history[3::2],
            ):
                transition_white.action_opponent = transition_black.action
                transition_black.action_opponent = transition_white_next.action

            replay_buffer.add(Transition.stack(agents[0].history))
            replay_buffer.add(Transition.stack(agents[1].history))


        for i_batch in range(n_batches_per_step):
            data = replay_buffer.sample_batch(batch_size, narx_memory_length)
            (
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                batch_act_opponent,
             ) = map(torch.as_tensor, data)

            state_val, sense_adv, move_adv, opponent_move  = q_nets[0](batch_obs)

            # Compute Q-values for selected move actions
            move_adv_selected = move_adv.gather(-1, batch_act.long().unsqueeze(-1))
            move_adv_mean = move_adv.mean(-1, keepdim=True)
            q_val_selected = state_val + move_adv_selected - move_adv_mean

            pass

if __name__ == '__main__':
    main()
