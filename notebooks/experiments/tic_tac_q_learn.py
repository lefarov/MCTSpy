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


def evaluate_vs_random_agent(q_func, game, eval_round_number):
    # Prepare the random and the argmax(Q) agents.
    random_agent = lambda s: random.choice(list(game.enumerate_actions(s)))
    our_agent = functools.partial(q_agent, q_func, game)
    agents = [random_agent, our_agent]

    wins = 0
    draws = 0
    for i_game in range(eval_round_number):
        # Decide who goes first this match.
        first_player_agent = int(random.randint(0, 1))
        player_to_agent = {0: first_player_agent, 1: 1 - first_player_agent}
        agent_to_player = dict(map(reversed, player_to_agent.items()))

        # Play a match.
        state, player_id = game.get_initial_state()
        while not game.state_is_terminal(state):
            action = agents[player_to_agent[player_id]](state)
            state, _, player_id = game.step(state, action)

        # Count the results.
        terminal_values = game.get_terminal_value(state)
        wins += int(terminal_values[agent_to_player[1]] == 1)
        draws += int(terminal_values[agent_to_player[1]] == 0)

    return draws, wins


def main():
    episode_number = 100000
    eval_round_number = 1000

    lr = 0.25
    discount_rate = 0.9
    eps = 0.02

    out_path = Path(os.environ['DEV_OUT_PATH']) / 'scripts' / 'tic_tac_dqn'

    game = TicTac()
    min_reward, max_reward = -1, 1

    # Define the state that will be visualized.
    init_state, _ = game.get_initial_state()
    vis_state = game.step(game.step(game.step(init_state, 0)[0], 4)[0], 1)[0]  # Moves: 0, 4, 1. Next must be 2.

    # todo How are tabulated Q-funcs initialized?
    q_init_func = lambda: random.random() * (max_reward - min_reward) + min_reward
    q_func = defaultdict(q_init_func)  # type: TQFunc

    # Build the policies. We will play against a random agent.
    policy_eps_trained = functools.partial(eps_policy, q_func, eps=eps)
    policy_random = lambda s, actions: [1.0 for _ in actions]
    # policy_greedy = functools.partial(eps_policy, q_func, eps=0)
    policies = [policy_random, policy_eps_trained]
    trained_agent_id = 1

    for i_episode in range(episode_number):
        # Decide who goes first this match/episode.
        first_player_agent = int(random.randint(0, 1))
        player_to_agent = {0: first_player_agent, 1: 1 - first_player_agent}
        agent_to_player = dict(map(reversed, player_to_agent.items()))

        # Init state and the first action.
        state, player_id = game.get_initial_state()
        available_actions = list(game.enumerate_actions(state))
        policy = policies[player_to_agent[player_id]]
        action = random.choices(available_actions, k=1, weights=policy(state, available_actions))[0]

        # Play a game and collect all states, actions and rewards along the way.
        sar_buffer = []
        reached_terminal = False
        while not reached_terminal:

            next_state, rewards, next_player_id = game.step(state, action)
            sar_buffer.append((state, action, rewards, player_id))

            if i_episode % int(episode_number / 10) == 0:
                print(state)
                print(f"Action: {action} Rewards: {rewards}")

            reached_terminal = game.state_is_terminal(next_state)
            if not reached_terminal:
                available_actions = list(game.enumerate_actions(next_state))
                policy = policies[player_to_agent[next_player_id]]
                next_action = random.choices(available_actions, k=1, weights=policy(next_state, available_actions))[0]
            else:
                next_action = None  # Just a dummy action out of the terminal state
                # Insert a state for the other player using the same rewards,
                # so the last rewards are given to the player.
                # todo this is a silly hack, better add the rewards from one player's turn to the other's.
                sar_buffer.append((next_state, next_action, rewards, next_player_id))

            state = next_state
            action = next_action
            player_id = next_player_id

        # Go over the SARSA pairs for our agent and update the Q function.
        for t, (state, action, rewards, player_id) in enumerate(sar_buffer):
            # Ignore states of our opponent.
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
            # print(f" Update by {lr * (reward + discount_rate * q_next - q_current)} Reward {reward}")
            # print("Update {lr * (reward + discount_rate * q_next - q_current)}")

        if i_episode % int(episode_number / 10) == 0:
            print(f"Plot {i_episode}")
            fig, ax = plt.subplots()
            q_values = np.zeros((TicTac.BoardSize * TicTac.BoardSize))
            for a in game.enumerate_actions(vis_state):
                q_values[a] = q_func[(vis_state, a)]
            ax.matshow(q_values.reshape((TicTac.BoardSize, TicTac.BoardSize)), vmin=-1, vmax=1)

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

    print(f"Q-values: {len(q_func)}")

    out_path.mkdir(exist_ok=True, parents=True)
    with open(out_path / 'q_table.pcl', 'wb') as file:
        pickle.dump(dict(q_func), file)

    draws, wins = evaluate_vs_random_agent(q_func, game, eval_round_number)

    print(f"Wins: {wins / eval_round_number * 100:.1f}% Draws: {draws / eval_round_number * 100:.1f}%")

    plt.show()


if __name__ == '__main__':
    main()
