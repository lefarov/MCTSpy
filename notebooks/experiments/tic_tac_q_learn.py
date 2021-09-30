import functools
import operator
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import *

import numpy as np
import matplotlib.pyplot as plt

from simulations.tic_tac import TicTac, TicTacState
from simulations.tic_tac import TicTacAction

TQFunc = Dict[Tuple[TicTacState, TicTacAction], float]


def eps_policy(q_func: TQFunc, state: TicTacState, actions: Iterable[TicTacAction],
               eps: float = 0.05) -> Dict[TicTacAction, float]:
    pdf = {}  # Technically not PDF, just weighting.
    q_values = {}
    for action in actions:
        q_values[action] = q_func[(state, action)]
        pdf[action] = eps  # Init with eps.

    best_action = max(q_values.items(), key=operator.itemgetter(1))[0]

    pdf[best_action] = 1.0

    return pdf


def q_agent(q_func: TQFunc, game: TicTac, state: TicTacState):
    return max(list(game.enumerate_actions(state)), key=lambda a: q_func[(state, a)])


def main():
    episode_number = 10000
    eval_round_number = 1000

    lr = 0.005
    discount_rate = 0.9
    out_path = Path(os.environ['DEV_OUT_PATH']) / 'scripts' / 'tic_tac_dqn'

    game = TicTac()
    min_reward, max_reward = -1, 1

    # todo How are tabulated Q-funcs initialized?
    q_init_func = lambda: random.random() * (max_reward - min_reward) + min_reward
    q_func = defaultdict(q_init_func)  # type: TQFunc

    policy_eps_trained = functools.partial(eps_policy, q_func, eps=0.1)
    policy_random = lambda s, actions: [1.0 for _ in actions]
    # policy_greedy = functools.partial(eps_policy, q_func, eps=0)

    # replay_buffer = []
    # replay_buffer_max_size = 1024

    policies = [policy_random, policy_eps_trained]
    trained_agent_id = 1

    for i_episode in range(episode_number):
        # print(f"Episode #{i_episode + 1}")
        first_player_agent = int(random.randint(0, 1))
        player_to_agent = {0: first_player_agent, 1: 1 - first_player_agent}
        agent_to_player = dict(map(reversed, player_to_agent.items()))

        state, player_id = game.get_initial_state()
        available_actions = list(game.enumerate_actions(state))
        policy = policies[player_to_agent[player_id]]
        action = random.choices(available_actions, k=1, weights=policy(state, available_actions))[0]

        # Play a game and collect all state, actions and rewards along the way.
        sar_buffer = []
        reached_terminal = False
        while not reached_terminal:

            next_state, rewards, next_player_id = game.step(state, action)

            sar_buffer.append((state, action, rewards, player_id))

            reached_terminal = game.state_is_terminal(next_state)
            if not reached_terminal:
                available_actions = list(game.enumerate_actions(next_state))
                policy = policies[player_to_agent[next_player_id]]
                next_action = random.choices(available_actions, k=1, weights=policy(next_state, available_actions))[0]
            else:
                next_action = None  # Just a dummy action out of the terminal state
                # Insert a state for the other player using the same rewards,
                # so the last rewards are given to the player.
                # todo this is a silly hack, better add the rewards from one player's turn to the other.
                sar_buffer.append((next_state, next_action, rewards, next_player_id))

            state = next_state
            action = next_action
            player_id = next_player_id

        # Go over the SARSA pairs for our agent and update the Q function.
        for t, (state, action, rewards, player_id) in enumerate(sar_buffer):
            if agent_to_player[trained_agent_id] != player_id:
                continue  # todo We're throwing away useful data.

            if t + 2 < len(sar_buffer):
                t_next = t + 2
                next_state, next_action, next_rewards, next_player_id = sar_buffer[t_next]
            else:
                # Dummy values for the terminal state.
                next_state, next_action, next_rewards, next_player_id = None, None, None, -1

            q_current = q_func[(state, action)]
            q_next = q_func[(next_state, next_action)]

            reward = rewards[agent_to_player[trained_agent_id]]
            q_func[(state, action)] += lr * (reward + discount_rate * q_next - q_current)
            # print("Update {lr * (reward + discount_rate * q_next - q_current)}")

        if i_episode % int(episode_number / 10) == 0:
            print(f"Plot {i_episode}")
            fig, ax = plt.subplots()
            init_state, _ = game.get_initial_state()
            q_values = [q_func[(init_state, a)] for a in game.enumerate_actions(init_state)]
            ax.matshow(np.array(q_values).reshape((3, 3)), vmin=-1, vmax=1)

        # === This code accidentally assumed a one-player game, lol.
        # reached_terminal = False
        # while reached_terminal:
        #
        #     next_state, rewards, next_player_id = game.step(state, action)
        #     available_actions = list(game.enumerate_actions(next_state))
        #
        #     reached_terminal = game.state_is_terminal(next_state)
        #
        #     if not reached_terminal:
        #         next_action = random.choices(available_actions, k=1, weights=policy_eps_trained(next_state, available_actions))[0]
        #     else:
        #         next_action = None  # Just a dummy action out of the terminal state.
        #
        #     q_current = q_func[(state, action)]
        #     q_next = q_func[(next_state, next_action)]
        #
        #     q_func[(state, action)] += lr * (rewards[___] + discount_rate * q_next - q_current)
        #
        #     state = next_state
        #     action = next_action
        #     player_id = next_player_id

    out_path.mkdir(exist_ok=True, parents=True)
    with open(out_path / 'q_table.pcl', 'wb') as file:
        pickle.dump(dict(q_func), file)

    random_agent = lambda s: random.choice(list(game.enumerate_actions(s)))
    our_agent = functools.partial(q_agent, q_func, game)
    agents = [random_agent, our_agent]

    wins = 0
    draws = 0
    for i_game in range(eval_round_number):
        first_player_agent = int(random.randint(0, 1))
        player_to_agent = {0: first_player_agent, 1: 1 - first_player_agent}
        agent_to_player = dict(map(reversed, player_to_agent.items()))

        state, player_id = game.get_initial_state()

        while not game.state_is_terminal(state):
            action = agents[player_to_agent[player_id]](state)
            state, _, player_id = game.step(state, action)

        terminal_values = game.get_terminal_value(state)
        wins += int(terminal_values[agent_to_player[1]] == 1)
        draws += int(terminal_values[agent_to_player[1]] == 0)

    print(f"Wins: {wins / eval_round_number * 100:.1f}% Draws: {draws / eval_round_number * 100:.1f}%")

    plt.show()


if __name__ == '__main__':
    main()
