import random
from typing import *

import numpy as np
import torch
import torch.nn.functional
import torchsummary
import wandb
from bots.tictac.play_custom import play_local_game

from bots.blindchess.losses import q_loss
from bots.blindchess.play import play_local_game_batched
from bots.tictac.agent import RandomAgent, QAgent
from bots.tictac.data_structs import Episode, DataPoint, Transition, DataTensors
from bots.tictac.game import TicTacToe, Player, WinReason, Board, Square
from bots.tictac.net import TicTacQNet


def main():

    steps_per_epoch = 21
    epoch_number = 10000
    games_per_epoch = 128

    net_memory_length = 1
    net_hidden_number = 512

    train_batch_size = 128
    train_lr = 1e-3
    train_weight_sense = 1.0

    eval_freq_epochs = 10
    eval_games = 1024

    # train_data_mode = 'fixed-data'
    # train_data_mode = 'replay-buffer'
    train_data_mode = 'fresh-data'
    fixed_data_epoch_number = 10

    wandb_description = 'fresh-data_true-state_fc-net-large'

    wandb.init(project="recon_tictactoe", entity="not-working-solutions", )
    wandb.run.name = wandb.run.name + '-' + wandb_description if wandb.run.name else wandb_description  # Can be 'None'.

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32

    q_net = TicTacQNet(net_memory_length, net_hidden_number).to(device)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=train_lr)

    # Train on random agent data.
    agents = [RandomAgent(), RandomAgent()]
    q_agent = QAgent(q_net)

    print("Built the Q-Net.")
    torchsummary.summary(q_net, (net_memory_length, *TicTacQNet.ObsShape))

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
                    if t >= 0:
                        transition_history.append(episode.transitions[t])
                    else:  # Pad the history with empty transitions.
                        transition_history.append(Transition.get_empty_transition())

                data_raw.append(DataPoint(transition_history))

            # --- Convert into training tensors.
            data = q_net.convert_transitions_to_tensors(data_raw)

            # --- Update the model.
            optimizer.zero_grad()

            q_sense_now, q_move_now = q_net(data.obs)
            q_sense_next, q_move_next = q_net(data.obs_next)

            # Mask the invalid actions.
            q_move_next[data.act_next_mask == 0] = torch.finfo(dtype).min

            # Q-sense is updated with the next q-move (and vice versa), because that's the next agent's action.
            loss_sense = torch.sum((1 - data.is_move) * q_loss(q_sense_now, q_move_next,  data.act, data.rew, data.done))
            loss_move  = torch.sum(     data.is_move  * q_loss(q_move_now,  q_sense_next, data.act, data.rew, data.done))

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

        # --- Winrate evaluation.
        if i_epoch % eval_freq_epochs == 0:
            win_count = 0
            for i_game in range(eval_games):
                winner_color, win_reason, _ = play_local_game(q_agent, agents[1], TicTacToe())
                if winner_color == Player.Cross:
                    win_count += 1

            winrate = win_count / eval_games
            print(f"Eval winrate: {winrate}")
            wandb.log(step=step_index, data={"winrate": winrate})

        print(f"Epoch {i_epoch}  Loss: {loss_epoch}")


if __name__ == '__main__':
    main()
