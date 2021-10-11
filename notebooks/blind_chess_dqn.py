import functools
import os

import torch
import random
import itertools
import wandb
import typing as t
import numpy as np

import chess
import reconchess

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from agents.blind_chess import TestQNet, QAgent, Transition, RandomBot, PIECE_VALUE
from utilities.replay_buffer import HistoryReplayBuffer


def move_proxy_reward(taken_move, requested_move):
    # If invalid move was selected
    if taken_move is None:
        return -0.5

    # If valid move was selected, but it was modified because of unknown opponent figure
    if taken_move != requested_move:
        return -0.01

    return 0.0


def capture_proxy_reward(piece: chess.Piece, lost: bool, weight=0.2):
    # If Opponent captured your piece.
    if lost:
        # Return negative piece value multiplied by the importance weight of sense action.
        return - PIECE_VALUE[piece.piece_type] * weight
    else:
        # Return value of a Pawn, since we don't know which piece we've captured.
        return PIECE_VALUE[chess.PAWN] * weight


def sense_proxy_reward(piece: chess.Piece, weight=0.0):
    # Return sensed piece value multiplied by the importance weight of sense action.
    return PIECE_VALUE[piece.piece_type] * weight


def policy_sampler(
    q_values: torch.Tensor,
    valid_action_indices,
    mask_invalid_actions=False,
    eps: float = 0.05
) -> int:
    
    action_q_masked = torch.full_like(q_values, fill_value=torch.finfo(q_values.dtype).min)
    action_q_masked[valid_action_indices] = q_values[valid_action_indices]

    # If we don't mask invalid actions, use original Q as masked values
    if not mask_invalid_actions:
        action_q_masked = q_values

    if random.random() >= eps:
        action_index = torch.argmax(action_q_masked).item()
    else:
        action_index = random.choice(valid_action_indices)

    return action_index


def q_selector(q_net: torch.nn.Module, action: str="move") -> t.Callable:
    # TODO: replace string by Enum
    # Select reqested output of the Q-net
    if action == "move":
        index = 2
    elif action == "sense":
        index = 1
    else:
        raise ValueError("Unknown action type.")

    def selector(obs):
        return q_net(obs)[index]

    return selector


def convert_to_tensor(array: np.ndarray, device) -> torch.Tensor:
    tensor = torch.as_tensor(array)
    # If tensor has only batch dimension, insert a singular dimension
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)

    # Torch indexing supports only int64
    if tensor.dtype == torch.int32:
        tensor = tensor.long()

    return tensor.to(device)


def q_loss(
    model: torch.nn.Module,
    model_target: torch.nn.Module,
    obs: torch.Tensor,
    act: torch.Tensor,
    rew: torch.Tensor,
    obs_next: torch.Tensor,
    done: torch.Tensor,
    discount: int=0.9,
    double_q: bool=True,
) -> torch.Tensor:
        # Compute Q values for observation at t
        q_values = model(obs)
        # Select Q values for chosen actions
        q_values_selected = q_values.gather(-1, act)

        with torch.no_grad():
            # Compute Q values for next observation using target model
            q_values_next = model_target(obs_next)

        if double_q:
            # Double Q idea: select the optimum action for the observation at t+1
            # using the trainable model, but compute it's Q value with target one
            q_values_next_estimate = model(obs_next)
            q_opt_next = q_values_next.gather(
                -1, torch.argmax(q_values_next_estimate, dim=-1, keepdim=True)
            )

        else:
            # Select max Q value for the observation at t+1
            q_opt_next = torch.max(q_values_next, dim=-1, keepdim=True)

        # Target estimate for Cumulative Discounted Reward
        q_values_target = rew + discount * q_opt_next * (1. - done)

        # Compute TD error
        loss = torch.nn.functional.smooth_l1_loss(
            q_values_selected, q_values_target
        )

        return loss


def main():

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    wandb.init(project="blind_chess", entity="not-working-solutions")

    narx_memory_length = 50
    replay_size = 10000
    batch_size = 512
    
    n_hidden = 64
    n_steps = 5000
    n_batches_per_step = 10
    n_games_per_step = 1

    n_test_games = 1
    evaluation_freq = 10
    
    # Frequency for updating target Q network
    target_q_update = 500

    # If you don't need learning rate annealing, set `le_end` equal to `lr_start`
    lr_start = 0.01
    lr_end = 0.0001
    # Weights of opponent's move prediction, move td and sense td errors.
    loss_weights = (1e-7, 1., 1.)
    gamma = 1
    gradient_clip = 100

    # Set to 0. if don't want to propagate terminal reward.
    reward_decay_factor = 0.0  # 1.05

    q_nets = [
        TestQNet(narx_memory_length, n_hidden).to(device),
        TestQNet(narx_memory_length, n_hidden).to(device),
    ]

    wandb.watch(q_nets[0])

    convert_to_tensor_on_device = functools.partial(convert_to_tensor, device=device)

    # We can also clone the first Q-net, but I'm not sure that it's necessary 
    q_net_target = TestQNet(narx_memory_length, n_hidden).to(device)
    q_net_target.eval()

    # Opponent move loss
    opponent_act_loss_func = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = Adam(q_nets[0].parameters(), lr=lr_start)
    lr_scheduler = CosineAnnealingLR(optimizer, n_steps, lr_end)

    # Report the size of the netowrks
    for stack in (q_nets[0].conv_stack, q_nets[0].fc_stack, q_nets[0]):
        net_size = 0
        for param in stack.parameters():
            net_size += param.numel()
        print(f"Stack size: {net_size}")

    # TODO: schedule exploration epsilon
    # agents = [
    #     QAgent(
    #         net,
    #         functools.partial(policy_sampler, eps=eps),
    #         narx_memory_length,
    #         capture_proxy_reward,
    #         move_proxy_reward,
    #         sense_proxy_reward,
    #     )
    #     for net, eps in zip(q_nets, (0.2, 1.0))
    # ]
    
    # TODO: check if this directory will be synced with the server.
    game_plots_path = os.path.abspath(
        os.path.join(wandb.run.dir, os.pardir, "games")
    ) 

    training_agent = QAgent(
        q_nets[0],
        functools.partial(policy_sampler, eps=0.2),
        narx_memory_length,
        device,
        capture_proxy_reward,
        move_proxy_reward,
        sense_proxy_reward,
    )

    test_agent = QAgent(
        q_nets[0],
        functools.partial(policy_sampler, eps=0.0),
        narx_memory_length,
        device,
        root_plot_direcotry=game_plots_path
    )

    random_agent = RandomBot(
        capture_proxy_reward, move_proxy_reward, sense_proxy_reward
    )

    test_opponent = RandomBot(
        capture_proxy_reward,
        move_proxy_reward,
        sense_proxy_reward,
        root_plot_direcotry=game_plots_path,
    )

    agents = [training_agent, random_agent]

    replay_buffer = HistoryReplayBuffer(replay_size, (8, 8, 13), tuple())

    for i_step in range(n_steps):
        print(f"Step {i_step + 1} / {n_steps}")

        print("Playing.")
        for i_game in range(n_games_per_step):
            winner_color, win_reason, game_history = reconchess.play_local_game(agents[0], agents[1])
            # TODO: Implement return estimation as in Apex-DQN.
            # TODO: Implement prioritized replay
            # TODO: Implement sampling of the colors.

            white_index = 0 if agents[0].color == chess.WHITE else 1
            black_index = 1 - white_index

            # Zipper that will iterate until the end of the longest sequence and
            # pad missing data of shorter sequences with the transitions containing
            # default opponent action as the recorded action.
            padded_zipper = functools.partial(
                itertools.zip_longest, fillvalue=Transition(None, -1, None)
            )

            # TODO: mirror the history

            # Iterate over 3 transitions windows: (1) with Move actions of the white player
            # (2) Move actions of the black player and (3) white Move actions shifted by one timestep forward.
            for transition_white, transition_black, transition_white_next in padded_zipper(
                agents[white_index].history[1::2],
                agents[black_index].history[1::2],
                agents[white_index].history[3::2],
            ):
                transition_white.action_opponent = transition_black.action
                transition_black.action_opponent = transition_white_next.action

            # TODO: think if we can use AlphaZero state value trick (overwrite all rewards with 1.)
            # Reward shaping: propagate the final reward to the preceeding timesteps with exponential decay.
            if reward_decay_factor != 0.0:
                length = max(len(agent.history) for agent in agents)
                for i in range(-1, -length -1, -1):
                    try:
                        for agent in agents:
                            agent.history[i-1].reward += agent.history[i].reward * (reward_decay_factor ** i)

                    # If we came to the end of the shortest history
                    except IndexError as e:
                        continue

            replay_buffer.add(Transition.stack(agents[0].history))
            # replay_buffer.add(Transition.stack(agents[1].history))

        # Report if our replay buffer is full
        wandb.log({"replay_is_full": int(replay_buffer.is_full)})

        print("Training.")
        for i_batch in range(n_batches_per_step):
            # Sample Move data
            data = replay_buffer.sample_batch(batch_size, narx_memory_length, "move")
            (
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                batch_act_opponent,
                batch_done,
             ) = map(convert_to_tensor_on_device, data)

            terminal_count = batch_done.count_nonzero().item()

            # Compute opponnet loss
            # TODO: can we do single prop through network?
            # TODO: adjust TD error scaling
            *_, pred_act_opponent = q_nets[0](batch_obs)
            opponent_act_loss = opponent_act_loss_func(
                pred_act_opponent, batch_act_opponent.squeeze(-1)
            )

            total_loss = loss_weights[0] * opponent_act_loss

            # Compute Move loss
            move_loss = q_loss(
                q_selector(q_nets[0], "move"),
                q_selector(q_net_target, "move"),
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                batch_done,
                discount=gamma
            )

            total_loss += loss_weights[1] * move_loss

            # Sample Sense data
            data = replay_buffer.sample_batch(batch_size, narx_memory_length, "sense")
            (
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                _,
                batch_done,
             ) = map(convert_to_tensor_on_device, data)

            terminal_count += batch_done.count_nonzero().item()

            # Compute Sense loss
            sense_loss = q_loss(
                q_selector(q_nets[0], "sense"),
                q_selector(q_net_target, "sense"),
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                batch_done,
                discount=gamma,
            )

            total_loss += loss_weights[2] * sense_loss
            # TODO: normalize losses

            # Optimize the model
            optimizer.zero_grad()
            total_loss.backward()
            
            # TODO: check the gradients and clip if needed
            # for _, param in q_nets[0].named_parameters():
            #     param.grad.data.clamp_(gradient_clip, gradient_clip)

            optimizer.step()

            wandb.log({
                "total_loss": total_loss,
                "move_loss": move_loss,
                "sense_loss": sense_loss,
                "opponent_act_loss": opponent_act_loss,
                "terminal_fraction": terminal_count / batch_size,
            })

        # Clone target network with specified frequency
        if i_step % target_q_update == 0:
            q_net_target.load_state_dict(q_nets[0].state_dict())

        # Evaluate our agent with greedy policy
        if i_step % evaluation_freq == 0:
            print("Evaluation.")
            win_rate = 0
            for i_test_game in range(n_test_games):
                # TODO: save game plots to WANDB
                # Set the plotting subdirectory for the current game
                test_tame_name = str(n_test_games * i_step + i_test_game)
                test_agent.plot_directory = f"agent_{test_tame_name}"
                test_opponent.plot_directory = f"opponent_{test_tame_name}"

                # TODO: Implement sampling of the colors.
                winner_color, win_reason, game_history = reconchess.play_local_game(test_agent, test_opponent)

                if winner_color == chess.WHITE:
                    win_rate += 1

            wandb.log({"win_rate": win_rate / n_test_games})

        # Update learning rate
        # lr_scheduler.step()


if __name__ == '__main__':
    main()
