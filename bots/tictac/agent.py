import operator
import os
import random
import typing as t
from typing import List, Optional

import numpy as np

import reconchess
import torch

from bots import tictac
from bots.tictac import WinReason
from bots.tictac.data_structs import Transition
from bots.tictac.net import TicTacQNet


TPolicySampler = t.Callable[[np.ndarray, List[int]], int]


def greedy_policy(q_vals: torch.Tensor, valid_actions: List[int]):
    q_vals_indexed = list(enumerate(q_vals))
    q_vals_valid = [q_vals_indexed[i] for i in valid_actions]

    # Return the original index corresponding to the largest q value.
    return max(q_vals_valid, key=operator.itemgetter(1))[0]


class PlayerWithBoardHistory(reconchess.Player):
    """ Player subclass that maintains board state and records its history."""

    def __init__(
        self,
        root_plot_directory=None,
    ) -> None:

        self.board = None
        self.color = None

        self._root_plot_directory = root_plot_directory
        self._plot_directory = None
        self._plot_index = 0

        self.history = []  # type: t.List[Transition]

    @property
    def plot_directory(self):
        """Plot directory for current game."""
        return self._plot_directory

    @plot_directory.setter
    def plot_directory(self, directory):
        if self._root_plot_directory is None:
            raise ValueError(
                "Cannot set plotting directory for the agent with None root plotting directory"
            )

        self._plot_directory = os.path.join(self._root_plot_directory, directory)
        os.makedirs(self._plot_directory)

        self._plot_index = 0

    def handle_game_start(
        self, color: tictac.Player, board: tictac.Board, opponent_name: str
    ):
        # Initialize board and color
        self.board = board
        self.color = tictac.Player(int(color))

        self.history = []

    def handle_opponent_move_result(
        self, captured_my_piece: bool, capture_square: t.Optional[tictac.Square]
    ):
        pass

    def handle_sense_result(
        self, sense_result: t.Tuple[int, t.Optional[tictac.Square]]
    ):
        self.board[sense_result[0]] = sense_result[1]

    def handle_move_result(
            self,
            requested_move: t.Optional[int],
            taken_move: t.Optional[int],
            captured_opponent_piece: bool,
            capture_square: t.Optional[int]
    ):
        if taken_move == requested_move:
            self.board[requested_move] = self.color
        elif self.board[requested_move] != self.color:
            self.board[requested_move] = tictac.Player((int(self.color) + 1) % 2)

    def handle_game_end(
            self,
            winner_color: t.Optional[tictac.Player],
            win_reason: t.Optional[tictac.WinReason],
            game_history: None
    ):
        if win_reason == WinReason.MatchThree:
            reward = 1 if winner_color == self.color else -1
        elif win_reason == WinReason.Draw:
            reward = 0
        else:
            raise ValueError()

        # Add reward
        self.history[-1].reward = reward
        self.history[-1].done = 1.0

        # Append last dummy transition to correctly handle the reward sampling.
        # (If the reward is in the last transition, it will always be in the Q(t+1) term and
        #  will never enter the TD loss as 'r'. Thus, we'll never learn the value of victory.)
        self.history.append(
            Transition(
                self.board.to_array(),
                action=0,  # Some valid action, not important.
                reward=0.0,
                done=0.0,
                is_move=False
                # action_mask=np.ones(TicTacToe.BoardSize ** 2),
            )
        )


class RandomAgent(PlayerWithBoardHistory):

    def choose_sense(self, sense_actions: List[int], move_actions: List[int], seconds_left: float) -> \
            Optional[int]:
        sense = random.choice(sense_actions)
        self.history.append(Transition(self.board.to_array(), sense, reward=0.0, is_move=False))

        return sense

    def choose_move(self, move_actions: List[int], seconds_left: float) -> Optional[int]:
        move = random.choice(move_actions)

        self.history.append(
            Transition(self.board.to_array(), move, reward=0, is_move=True)
        )

        return move


class QAgent(PlayerWithBoardHistory):

    def __init__(self, q_net: TicTacQNet, policy_sampler: TPolicySampler = greedy_policy):
        super().__init__()

        self.q_net = q_net
        self.policy_sampler = policy_sampler

    def choose_sense(self, sense_actions: List[int], move_actions: List[int], seconds_left: float) -> \
            Optional[int]:

        q_sense, q_move = self._call_q_net()
        sense = self.policy_sampler(q_sense, sense_actions)

        self.history.append(Transition(self.board.to_array(), sense, reward=0.0, is_move=False))

        return sense

    def choose_move(self, move_actions: List[int], seconds_left: float) -> Optional[int]:
        q_sense, q_move = self._call_q_net()
        move = self.policy_sampler(q_sense, move_actions)

        self.history.append(Transition(self.board.to_array(), move, reward=0.0, is_move=True))

        return move

    def _call_q_net(self):
        recent_obs_history = [t.observation for t in self.history[-self.q_net.narx_memory_length + 1:]]
        # Append the freshest board state to the input -- the opponent has made a move that might affect it.
        recent_obs_history.append(self.board.to_array())
        obs_tensors = self.q_net.obs_list_to_tensor(recent_obs_history)

        q_sense, q_move = self.q_net(obs_tensors.unsqueeze(0))

        return q_sense[0], q_move[0]
