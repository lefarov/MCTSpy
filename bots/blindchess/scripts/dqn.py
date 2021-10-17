import os
import functools
import itertools
import time
import wandb
import tempfile

import chess
import reconchess
from reconchess import WinReason

import torch
from torch.optim import Adam

from bots.blindchess.play import DelegatingAgentManager, play_local_game_batched
from bots.blindchess.buffer import HistoryReplayBuffer
from bots.blindchess.losses import q_loss
from bots.blindchess.agent import (
    TestQNet,
    QAgent,
    Transition,
    RandomBot,
    QAgentManager,
)
from bots.blindchess.utilities import (
    move_proxy_reward, 
    capture_proxy_reward, 
    sense_proxy_reward, 
    egreedy_masked_policy_sampler, 
    q_selector, 
    convert_to_tensor,
)

# Available options are "online", "offline" or "disabled"
WANDB_MODE = "disabled"

CONFIG = {
    "narx_memory_length": 50,
    "replay_size": 50000,
    "batch_size": 1024,
    
    "n_hidden": 64,
    "n_steps": 5000,
    "n_batches_per_step": 10,
    "n_games_per_step": 128,
    "n_test_games": 128,
    
    "evaluation_freq": 100,
    "game_batch_size": 128,
    
    # Frequency for updating target Q network
    "target_q_update": 500,
    "lr": 0.01,
    # Weights of opponent's move prediction, move td and sense td errors.
    "loss_weights": (1e-7, 1., 1.),
    "gamma": 1.0,
    "gradient_clip": 100,

    # Set to 0. if don't want to propagate terminal reward.
    "reward_decay_factor": 1.05,  # 1.05
}


def main():
    # TODO: Implement return estimation as in Apex-DQN.
    # TODO: Implement prioritized replay.
    # TODO: Implement sampling of the colors.
    # TODO: make Q-network as an torch module.
    # TODO: mirror the history.
    # TODO: think if we can use AlphaZero state value trick (overwrite all rewards with 1.)

    wandb.init(
        project="blind_chess",
        entity="not-working-solutions",
        config=CONFIG,
        mode=WANDB_MODE,
    )

    conf = wandb.config

    plotting_dir = os.path.abspath(os.path.join(wandb.run.dir, os.pardir, "game"))
    if WANDB_MODE == "disabled":
        plotting_dir = tempfile.TemporaryDirectory()
    
    print(f"Root plotting direcotry: {plotting_dir}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Setup replay buffer
    replay_buffer = HistoryReplayBuffer(conf.replay_size, (8, 8, 13), tuple())
    data_converter = functools.partial(convert_to_tensor, device=device)

    # Trainable network
    q_net = TestQNet(conf.narx_memory_length, conf.n_hidden).to(device)
    # We can also clone trainable Q-net, but I'm not sure that it's necessary 
    q_net_target = TestQNet(conf.narx_memory_length, conf.n_hidden).to(device)
    q_net_target.eval()

    wandb.watch(q_net)
    
    # Report the size of the networks
    for stack in (q_net.conv_stack, q_net.fc_stack, q_net):
        net_size = 0
        for param in stack.parameters():
            net_size += param.numel()

        print(f"Stack size: {net_size}") 


    # Opponent move loss
    opponent_act_loss_func = torch.nn.CrossEntropyLoss()
    # Optimizer
    optimizer = Adam(q_net.parameters(), lr=conf.lr)

    def q_agent_factory(q_net, policy_sampler, device, *args, **kwargs):
        return QAgent(
            q_net,
            policy_sampler,
            conf.narx_memory_length,
            device,
            capture_proxy_reward,
            move_proxy_reward,
            sense_proxy_reward,
            *args,
            **kwargs,
        )

    train_agent_manager = QAgentManager(
        q_net,
        functools.partial(egreedy_masked_policy_sampler, eps=0.2),
        device,
        q_agent_factory,
    )

    test_agent_manager = QAgentManager(
        q_net,
        functools.partial(egreedy_masked_policy_sampler, eps=0.0),
        device,
        q_agent_factory,
    )
 
    def random_agent_factory(*args, **kwargs):
        return RandomBot(
            capture_proxy_reward,
            move_proxy_reward,
            sense_proxy_reward,
            *args,
            **kwargs
        )

    random_agent_manager = DelegatingAgentManager(random_agent_factory)

    for i_step in range(conf.n_steps):
        print(f"Step {i_step + 1} / {conf.n_steps}")

        print("Playing.")
        results = play_local_game_batched(
            train_agent_manager,
            random_agent_manager,
            total_number=conf.n_games_per_step,
            batch_size=conf.game_batch_size,
        )

        # Process the game results
        for _, _, _, white_player, black_player in results:
            
            # Zipper that will iterate until the end of the longest sequence and
            # pad missing data of shorter sequences with the transitions containing
            # default opponent action as the recorded action.
            padded_zipper = functools.partial(
                itertools.zip_longest, fillvalue=Transition(None, -1, None)
            )

            # Iterate over 3 transitions windows: (1) with Move actions of the white player
            # (2) Move actions of the black player and (3) white Move actions shifted by one timestep forward.
            for transition_white, transition_black, transition_white_next in padded_zipper(
                white_player.history[1::2],
                black_player.history[1::2],
                white_player.history[3::2],
            ):
                transition_white.action_opponent = transition_black.action
                transition_black.action_opponent = transition_white_next.action

            # Reward shaping: propagate the final reward to the preceding timesteps with exponential decay.
            if conf.reward_decay_factor != 0.0:
                length = max(len(white_player.history), len(black_player.history))
                
                for i in range(-1, -length -1, -1):
                    try:
                        discout = (conf.reward_decay_factor ** i)
                        white_player.history[i-1].reward += white_player.history[i].reward * discout
                        black_player.history[i-1].reward += black_player.history[i].reward * discout

                    # If we came to the end of the shortest history
                    except IndexError as e:
                        continue

            # Add player history to Replay Buffer.
            replay_buffer.add(Transition.stack(white_player.history))
            # replay_buffer.add(Transition.stack(black_player.history))

        # Report if our replay buffer is full
        wandb.log({"replay_is_full": int(replay_buffer.is_full)})

        print("Training.")
        for i_batch in range(conf.n_batches_per_step):
            # Sample Move data
            data = replay_buffer.sample_batch(conf.batch_size, conf.narx_memory_length, "move")
            (
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                batch_act_opponent,
                batch_done,
             ) = map(data_converter, data)

            terminal_count = batch_done.count_nonzero().item()

            # Compute opponent's loss
            # TODO: can we do single prop through network?
            # TODO: adjust TD error scaling
            *_, pred_act_opponent = q_net(batch_obs)
            opponent_act_loss = opponent_act_loss_func(
                pred_act_opponent, batch_act_opponent.squeeze(-1)
            )

            total_loss = conf.loss_weights[0] * opponent_act_loss

            # Compute Move loss
            move_loss = q_loss(
                q_selector(q_net, "move"),
                q_selector(q_net_target, "move"),
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                batch_done,
                discount=conf.gamma
            )

            total_loss += conf.loss_weights[1] * move_loss

            # Sample Sense data
            data = replay_buffer.sample_batch(conf.batch_size, conf.narx_memory_length, "sense")
            (
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                _,
                batch_done,
             ) = map(data_converter, data)

            terminal_count += batch_done.count_nonzero().item()

            # Compute Sense loss
            sense_loss = q_loss(
                q_selector(q_net, "sense"),
                q_selector(q_net_target, "sense"),
                batch_obs,
                batch_act,
                batch_rew,
                batch_obs_next,
                batch_done,
                discount=conf.gamma,
            )

            total_loss += conf.loss_weights[2] * sense_loss

            # Optimize the model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            wandb.log({
                "total_loss": total_loss,
                "move_loss": move_loss,
                "sense_loss": sense_loss,
                "opponent_act_loss": opponent_act_loss,
                "terminal_fraction": terminal_count / conf.batch_size,
            })

        # Clone target network with specified frequency
        if i_step % conf.target_q_update == 0:
            q_net_target.load_state_dict(q_net.state_dict())

        # Evaluate our agent with greedy policy
        if i_step % conf.evaluation_freq == 0:
            print("Evaluation.")
            time_before = time.time()

            results = play_local_game_batched(
                test_agent_manager,
                random_agent_manager,
                total_number=conf.n_test_games,
                batch_size=conf.game_batch_size,
            )

            print(f"Evaluation finished in {time.time() - time_before:.2f} s.")

            win_rate = 0
            for winner_color, win_reason, game_history, white_player, black_player in results:
                if winner_color == chess.WHITE and win_reason == WinReason.KING_CAPTURE:
                    win_rate += 1

            wandb.log({"win_rate": win_rate / conf.n_test_games})

            # Play one more game to write the plots
            test_agent = test_agent_manager.build_agent(root_plot_directory=plotting_dir.name)
            rand_agent = random_agent_manager.build_agent(root_plot_directory=plotting_dir.name)

            test_agent.plot_directory = f"agent_{i_step * conf.evaluation_freq}"
            rand_agent.plot_directory = f"opponent_{i_step * conf.evaluation_freq}"

            reconchess.play_local_game(test_agent, rand_agent)

    # Clean temporary plotting directory if wasn't cleaned previously ()
    plotting_dir.cleanup()


if __name__ == '__main__':
    main()
