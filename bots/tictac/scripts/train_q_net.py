import random
from typing import *

import numpy as np
import torch
import torch.nn.functional
import wandb
from reconchess import play_local_game

from bots.blindchess.losses import q_loss
from bots.blindchess.play import play_local_game_batched
from bots.tictac.agent import RandomAgent, Transition
from bots.tictac.game import TicTacToe, Player, WinReason, Board
from bots.tictac.net import TicTacQNet


class Episode(NamedTuple):
    transitions: List[Transition]

    def __len__(self):
        return len(self.transitions)


class DataPoint(NamedTuple):
    transition_history: List[Transition]

    @property
    def transition_now(self):
        # The train transition is stored as the next to last in the history.
        return self.transition_history[-2]

    @property
    def transition_next(self):
        # The next (t + 1_ transition is stored as the last in the history.
        return self.transition_history[-1]

    @property
    def history_now(self):
        # All but the last, which is the next transition (t + 1).
        return self.transition_history[:-1]

    @property
    def history_next(self):
        # All but the first, which is too old for the history of the next transition.
        return self.transition_history[1:]


def obs_list_to_tensor(obs_list):
    # Stack, convert to a tensor and add a trivial channel dimension.
    return torch.tensor(np.stack([t.observation for t in obs_list])).unsqueeze(-1)


def action_to_one_hot(action):
    return torch.nn.functional.one_hot(torch.tensor(action), TicTacToe.BoardSize ** 2)


def main():

    steps_per_epoch = 21
    epoch_number = 10000
    games_per_epoch = 32

    net_memory_length = 3
    net_hidden_number = 128

    train_batch_size = 128
    train_lr = 1e-3
    train_weight_sense = 1.0

    train_data_mode = 'fixed-data'
    # train_data_mode = 'replay-buffer'
    # train_data_mode = 'fresh-data'
    fixed_data_epoch_number = 100

    wandb_description = 'larger-model-more-data-2'

    wandb.init(project="recon_tictactoe", entity="not-working-solutions", )
    wandb.run.name += '-' + wandb_description

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32

    obs_shape = (TicTacToe.BoardSize, TicTacToe.BoardSize, 1)
    act_shape = (1,)  # Action index.

    # Train on random agent data.
    agents = [RandomAgent(), RandomAgent()]

    q_net = TicTacQNet(net_memory_length, net_hidden_number).to(device)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=train_lr)

    replay_buffer = []  # type: List[Episode]

    for i_epoch in range(epoch_number):

        loss_epoch = 0.0

        # ---------------- Collect play data. ----------------
        episodes = []
        for i_game in range(games_per_epoch):
            winner_color, win_reason, _ = play_local_game(agents[0], agents[1], TicTacToe())

            # We only train on the white player's perspective for now.
            history_mine = agents[0].history
            # history_opp = agents[1].history

            episodes.append(Episode(history_mine))

        if train_data_mode == 'replay-buffer':
            replay_buffer.extend(episodes)
        elif train_data_mode == 'fresh-data':
            replay_buffer = episodes
        elif train_data_mode == 'fixed-data':
            if i_epoch < fixed_data_epoch_number:
                replay_buffer.extend(episodes)
        else:
            raise ValueError()

        # ---------------- Train the model. ----------------
        # Enumerate the steps by their global index for convenience.
        step_index = i_epoch * steps_per_epoch
        for i_step in range(step_index, step_index + steps_per_epoch):

            # --- Sample the train transitions with history.
            data_raw = []
            for i_sample in range(train_batch_size):
                # Sample an episode, weighted by episode length so all transitions are equally likely.
                episode = random.choices(replay_buffer, weights=[len(e) for e in replay_buffer], k=1)[0]

                # Sample a transition and extract its history.
                t_now = random.randint(0, len(episode) - 2)
                transition_history = []
                for t in range(t_now - net_memory_length + 1, t_now + 2):  # From n steps ago up to next.
                    transition_history.append(episode.transitions[t])

                data_raw.append(DataPoint(transition_history))

            # --- Convert into training arrays.
            data_obs = torch.empty((train_batch_size, net_memory_length, *obs_shape), dtype=dtype, device=device)
            data_obs_next = torch.empty((train_batch_size, net_memory_length, *obs_shape), dtype=dtype, device=device)
            data_act = torch.empty((train_batch_size, *act_shape), dtype=torch.int64, device=device)
            data_rew = torch.empty((train_batch_size, 1), dtype=dtype, device=device)
            data_done = torch.empty((train_batch_size, 1), dtype=dtype, device=device)
            data_is_move = torch.empty((train_batch_size, 1), dtype=dtype, device=device)

            for i_sample, point in enumerate(data_raw):
                data_obs[i_sample, ...] = obs_list_to_tensor(point.history_now)
                data_obs_next[i_sample, ...] = obs_list_to_tensor(point.history_next)
                data_act[i_sample] = point.transition_now.action
                data_rew[i_sample] = point.transition_now.reward
                data_done[i_sample] = point.transition_now.done
                data_is_move[i_sample] = int(point.transition_now.is_move)

            # --- Update the model.
            optimizer.zero_grad()

            q_sense_now, q_move_now = q_net(data_obs)
            q_sense_next, q_move_next = q_net(data_obs)

            loss_sense = torch.sum((1 - data_is_move) * q_loss(q_sense_now, q_sense_next, data_act, data_rew, data_done))
            loss_move  = torch.sum(     data_is_move  * q_loss(q_move_now,  q_move_next,  data_act, data_rew, data_done))

            loss_total = loss_move + train_weight_sense * loss_sense

            loss_total.backward()
            optimizer.step()

            loss_epoch += loss_total.item()

            wandb.log(step=i_step, data={
                "loss_total_step": loss_total.item(),
                "loss_move_step": loss_move.item(),
                "loss_sense_step": loss_sense.item(),
            })

            # print(f"Step: {i_step} | Total: {loss_total.item():.2f} "
            #       f"Move: {loss_move.item():.2f} Sense: {loss_sense.item():.2f}")

        loss_epoch /= steps_per_epoch

        step_index = ((i_epoch + 1) * steps_per_epoch - 1)  # Compute the last step index.
        wandb.log(step=step_index, data={"loss_total_epoch": loss_epoch})

        print(f"Epoch {i_epoch}  Loss: {loss_epoch}")


if __name__ == '__main__':
    main()
