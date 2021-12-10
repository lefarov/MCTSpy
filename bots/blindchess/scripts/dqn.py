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
from bots.blindchess.losses import CombinedLoss
from bots.blindchess.agent import (
    QAgent,
    Transition,
    RandomBot,
    QAgentManager,
)
from bots.blindchess.networks import TestQNet
from bots.blindchess.utilities import (
    EGreedyPolicy,
    move_proxy_reward, 
    capture_proxy_reward, 
    sense_proxy_reward,
    convert_to_tensor,
    EpsScheduler,
)

# Available options are "online", "offline" or "disabled"
WANDB_MODE = "online"

CONFIG = {
    "narx_memory_length": 1,
    "replay_size": 50000,
    "batch_size": 512,
    
    "n_hidden": 64,
    "n_steps": 500,
    "n_batches_per_step": 16,
    "n_games_per_step": 128,
    "n_test_games": 128,
    
    "evaluation_freq": 10,
    "game_batch_size": 128,

    # Frequency for updating target Q network
    "target_q_update": 10,
    "lr": 0.01,

    # Set to 0. if don't want to propagate terminal reward.
    "reward_decay_factor": 0.0,  # 1.05

    # Exploration
    "exploration": {
        "eps_base": 0.8,
        "eps_min": 0.05,
        "base_multiplier": 0.7,
        "schedule": [0.2, 0.5, 0.7, 0.9],
    },

    # Defintion of the loss
    "loss": {
        # Weights of opponent's move prediction, sense TD and move TD errors.
        "weights": (0.0, 1.0, 1.0),  # (1e-7, 1., 1.)
        "discount": 1.0,
        "double_q": True,
        # Mask Q values for the next observation in TD error.
        "mask_q": True,
    },
}


# Shapes
OBS_SHAPE = (8, 8, 13)
ACT_SHAPE = tuple()
MASK_SHAPE = (64 * 64,)

# Slices used to get sense and move data from the continuous history
SENSE_DATA_SLICE = slice(0, None, 2)
MOVE_DATA_SLICE = slice(1, None, 2)


def main():
    # TODO: Implement return estimation as in Apex-DQN (GAE).
    # TODO: Implement prioritized replay.
    # TODO: Implement sampling of the colors.
    # TODO: mirror the history.
    # TODO: think if we can use AlphaZero state value trick (overwrite all rewards with 1.)
    # TODO: add masks savings to the Random Bot
    # TODO: mask invalid en passant moves?

    """
    File "/home/max/projects/MCTSpy/bots/blindchess/agent.py", line 551, in _build_narx_batch
    RuntimeError: CUDA error: unspecified launch failure
    CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
    For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    wandb.init(
        project="blind_chess",
        entity="not-working-solutions",
        config=CONFIG,
        mode=WANDB_MODE,
    )

    conf = wandb.config

    plotting_dir = os.path.abspath(os.path.join(wandb.run.dir, os.pardir, "game"))
    if WANDB_MODE == "disabled":
        tempdir = tempfile.TemporaryDirectory()
        plotting_dir = tempdir.name
    
    print(f"Root plotting direcotry: {plotting_dir}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Setup replay buffer
    replay_buffer = HistoryReplayBuffer(conf.replay_size, OBS_SHAPE, ACT_SHAPE, MASK_SHAPE)
    data_converter = functools.partial(convert_to_tensor, device=device)
    # Slices representing data with move actions and sense action in the replay buffer
    action_slices = (SENSE_DATA_SLICE, MOVE_DATA_SLICE)

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

    # Losses definition
    combined_loss_func = CombinedLoss(
        model=q_net,
        model_target=q_net_target,
        opponent_loss=torch.nn.CrossEntropyLoss(),
        memory_length=conf.narx_memory_length,
        **conf.loss
    )

    # Exploration schedulers
    annealed_eps_scheduler = EpsScheduler(t_max=conf.n_steps, **conf.exploration)
    testing_eps_scheduler = EpsScheduler.constant_eps(0.0)

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
        EGreedyPolicy(annealed_eps_scheduler, masked=True),
        device,
        q_agent_factory,
    )

    test_agent_manager = QAgentManager(
        q_net,
        EGreedyPolicy(testing_eps_scheduler, masked=True),
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

        total_sense_moves = []

        # Process the game results
        for _, _, _, white_player, black_player in results:
            
            # Use zero as a dummy action. (A valid action is needed.)
            # TODO: Instead, we shouldn't train the opponent move head at the terminal transition.
            dummy_action = 0
            # Zipper that will iterate until the end of the longest sequence and
            # pad missing data of shorter sequences with the transitions containing
            # default opponent action as the recorded action.
            padded_zipper = functools.partial(
                itertools.zip_longest, fillvalue=Transition(None, dummy_action, None)
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
                
                for i in range(-2, -length -1, -1):
                    try:
                        discout = (conf.reward_decay_factor ** i)
                        white_player.history[i-1].reward += white_player.history[i].reward * discout
                        black_player.history[i-1].reward += black_player.history[i].reward * discout

                    # If we came to the end of the shortest history
                    except IndexError as e:
                        continue

            stacked_white_history = Transition.stack(white_player.history)
            
            # Record all sense actions
            total_sense_moves.extend(stacked_white_history.action[SENSE_DATA_SLICE])

            # Add player history to Replay Buffer.
            replay_buffer.add(stacked_white_history)
            # replay_buffer.add(Transition.stack(black_player.history))

        # Report if our replay buffer is full and current annealed epsilon
        wandb.log({
            "replay_is_full": int(replay_buffer.is_full),
            "current_eps": annealed_eps_scheduler.eps,
            "played_sense_actions": wandb.Histogram(total_sense_moves),
        })

        print("Training.")
        for i_batch in range(conf.n_batches_per_step):
            info_dict = dict()

            # Sample Combined data
            data = replay_buffer.sample_batch(conf.batch_size, conf.narx_memory_length, action_slices)
            (
                batch_obs,
                batch_act,
                batch_rew,
                batch_done,
                batch_obs_next,
                batch_act_next_mask,
                batch_act_opponent,
            ) = map(data_converter, data)

            batch_act[combined_loss_func.sense_ind]

            info_dict.update(
                terminal_transition_fraction=batch_done.count_nonzero().item() / conf.batch_size,
                reward_transition_fraction=batch_rew.count_nonzero().item() / conf.batch_size,
                sampled_sense_actions=wandb.Histogram(batch_act[combined_loss_func.sense_ind].cpu()),
            )

            # Compute Move loss and Opponent Move prediction Loss
            total_loss = combined_loss_func(
                batch_obs,
                batch_act,
                batch_rew,
                batch_done,
                batch_obs_next,
                batch_act_next_mask,
                batch_act_opponent,
                info_dict
            )

            # Optimize the model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log training Info dict
            wandb.log(info_dict)

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
            test_agent = test_agent_manager.build_agent(root_plot_directory=plotting_dir)
            rand_agent = random_agent_manager.build_agent(root_plot_directory=plotting_dir)

            test_agent.plot_directory = f"agent_{i_step * conf.evaluation_freq}"
            rand_agent.plot_directory = f"opponent_{i_step * conf.evaluation_freq}"

            reconchess.play_local_game(test_agent, rand_agent)

        # Update epsilons for e-greedy constant
        annealed_eps_scheduler.update_eps()

    # Clean temporary plotting directory if wasn't cleaned previously ()
    tempdir.cleanup()


if __name__ == '__main__':
    main()
