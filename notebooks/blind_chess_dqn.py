import functools
import operator

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

from agents.blind_chess import TestQNet, QAgent, Transition, RandomBot
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


def convert_to_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(array)
    # If tensor has only batch dimension, insert a singular dimension
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)

    # Torch indexing supports only int64
    if tensor.dtype == torch.int32:
        tensor = tensor.long()

    return tensor


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

    wandb.init(project="blind_chess", entity="not-working-solutions")

    narx_memory_length = 50
    replay_size = 10000
    batch_size = 512
    n_hidden = 64
    n_steps = 10
    n_batches_per_step = 10
    n_games_per_step = 10
    n_test_games = 100
    # If you don't need learning rate annealing, set `le_end` equal to `lr_start`
    lr_start = 0.1
    lr_end = 0.001
    # Weights of opponent's move prediction, move td and sense td errors.
    loss_weights = (1e-7, 1., 1.)
    gamma = 1
    gradient_clip = 100

    # Set to 0. if don't want to propagate any reward.
    reward_decay_weight = 1.05

    q_nets = [
        TestQNet(narx_memory_length, n_hidden),
        TestQNet(narx_memory_length, n_hidden),
    ]

    wandb.watch(q_nets[0])

    # We can also clone the first Q-net, but I'm not sure that it's necessary 
    q_net_target = TestQNet(narx_memory_length, n_hidden)
    q_net_target.eval()

    # Opponent move loss
    opponent_act_loss_func = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = Adam(q_nets[0].parameters(), lr=lr_start)
    lr_scheduler = CosineAnnealingLR(optimizer, n_steps, lr_end)

    for stack in (q_nets[0].conv_stack, q_nets[0].fc_stack, q_nets[0]):
        net_size = 0
        for param in stack.parameters():
            net_size += param.numel()
        print(f"Stack size: {net_size}")

    agents = [QAgent(net, functools.partial(policy_sampler, eps=0.2), narx_memory_length) for net in q_nets]
    test_agent = QAgent(q_nets[0], functools.partial(policy_sampler, eps=0.0), narx_memory_length)
    random_agent = RandomBot()

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
            length = max(len(agent.history) for agent in agents)
            for i in range(-1, -length -1, -1):
                try:
                    for agent in agents:
                        agent.history[i-1].reward += agent.history[i].reward * (reward_decay_weight ** i)

                # If we came to the end of the shortest history
                except IndexError as e:
                    continue

            replay_buffer.add(Transition.stack(agents[0].history))
            replay_buffer.add(Transition.stack(agents[1].history))

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
             ) = map(convert_to_tensor, data)

            rew_count = batch_rew.count_nonzero().item()

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
                # TODO: implement explicit saving for `done`
                torch.where(batch_rew != 0, 1., 0.),
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
             ) = map(convert_to_tensor, data)

            # Compute Sense loss
            sense_loss = q_loss(
                q_selector(q_nets[0], "sense"),
                q_selector(q_net_target, "sense"),
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                # TODO: implement explicit saving for `done`
                torch.where(batch_rew != 0, 1., 0.),
                discount=gamma,
            )

            total_loss += loss_weights[2] * sense_loss
            # TODO: normalize losses
            # TODO: move everything to GPU

            # Optimize the model
            optimizer.zero_grad()
            total_loss.backward()
            
            # for _, param in q_nets[0].named_parameters():
            #     param.grad.data.clamp_(gradient_clip, gradient_clip)

            optimizer.step()

            wandb.log({
                "total_loss": total_loss,
                "move_loss": move_loss,
                "sense_loss": sense_loss,
                "opponent_act_loss": opponent_act_loss,
                "samples_with_reward": rew_count,
            })

        print("Evaluation.")
        win_rate = 0
        for i_test_game in range(n_test_games):
            # TODO: Implement sampling of the colors.
            winner_color, win_reason, game_history = reconchess.play_local_game(test_agent, random_agent)

            if winner_color == chess.WHITE:
                win_rate += 1

        wandb.log({
            "win_rate": win_rate / n_test_games,
            "annealed_lr": lr_scheduler.get_lr(),
        })

        # Update learning rate
        lr_scheduler.step()


if __name__ == '__main__':
    main()
