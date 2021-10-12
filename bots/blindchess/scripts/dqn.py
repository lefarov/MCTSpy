import functools
import itertools
import os
import time
import wandb

import chess
import reconchess
from reconchess import WinReason

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from bots.blindchess.play import BatchedAgentManagerSimple, play_local_game_batched
from bots.blindchess.buffer import HistoryReplayBuffer
from bots.blindchess.losses import q_loss
from bots.blindchess.agent import (
    TestQNet,
    QAgent,
    Transition,
    RandomBot,
    QAgentBatched,
)
from bots.blindchess.utilities import (
    move_proxy_reward, 
    capture_proxy_reward, 
    sense_proxy_reward, 
    egreedy_masked_policy_sampler, 
    q_selector, 
    convert_to_tensor,
)


def main():

    use_wandb = True

    narx_memory_length = 50
    replay_size = 10000
    batch_size = 2048
    
    n_hidden = 64
    n_steps = 5000
    n_batches_per_step = 10
    n_games_per_step = 10

    n_test_games = 128 * 10
    evaluation_freq = 100
    evaluation_batch_size = 128
    
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
    reward_decay_factor = 1.05  # 1.05

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if use_wandb:
        wandb.init(project="blind_chess", entity="not-working-solutions")

    q_nets = [
        TestQNet(narx_memory_length, n_hidden).to(device),
        TestQNet(narx_memory_length, n_hidden).to(device),
    ]

    if use_wandb:
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
    
    game_plots_path = os.path.abspath(
        os.path.join(wandb.run.dir if use_wandb else os.path.join(os.getcwd(), 'wandb'), os.pardir, "games")
    ) 

    training_agent = QAgent(
        q_nets[0],
        functools.partial(egreedy_masked_policy_sampler, eps=0.2),
        narx_memory_length,
        device,
        capture_proxy_reward,
        move_proxy_reward,
        sense_proxy_reward,
    )

    test_agent = QAgent(
        q_nets[0],
        functools.partial(egreedy_masked_policy_sampler, eps=0.0),
        narx_memory_length,
        device,
        root_plot_directory=game_plots_path
    )

    random_agent = RandomBot(
        capture_proxy_reward, move_proxy_reward, sense_proxy_reward
    )

    test_agent_batched = QAgentBatched(
        q_nets[0],
        functools.partial(egreedy_masked_policy_sampler, eps=0.0),
        narx_memory_length,
        device
    )

    random_bot_ctor = lambda: RandomBot(capture_proxy_reward, move_proxy_reward, sense_proxy_reward)
    random_agent_batched = BatchedAgentManagerSimple(random_bot_ctor)

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
        if use_wandb:
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

            if use_wandb:
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
            time_before = time.time()
            # TODO: save game plots to WANDB
            # Set the plotting subdirectory for the current game
            # test_tame_name = str(n_test_games * i_step + i_test_game)
            # test_agent.plot_directory = f"agent_{test_tame_name}"
            # test_opponent.plot_directory = f"opponent_{test_tame_name}"

            # TODO: Implement sampling of the colors.
            # winner_color, win_reason, game_history = reconchess.play_local_game(test_agent, test_opponent)
            results = play_local_game_batched(test_agent_batched, random_agent_batched,
                                              total_number=n_test_games, batch_size=evaluation_batch_size)

            print(f"Evaluation finished in {time.time() - time_before:.2f} s.")

            win_rate = 0
            for winner_color, win_reason, game_history in results:
                if winner_color == chess.WHITE and win_reason == WinReason.KING_CAPTURE:
                    win_rate += 1

            if use_wandb:
                wandb.log({"win_rate": win_rate / n_test_games})

        # Update learning rate
        # lr_scheduler.step()


if __name__ == '__main__':
    main()
