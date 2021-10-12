import functools
import operator
import os
import random
from typing import List, Optional

import chess
import chess.engine
import chess.svg
import torch
import typing as t
import numpy as np
from dataclasses import dataclass

from reconchess import (
    Color,
    Player,
    Square,
    GameHistory,
    WinReason,
)

from bots.blindchess.utilities import index_to_move, move_to_index, board_to_onehot, PIECE_INDEX
from bots.blindchess.play import BatchedAgentManager


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    action_opponent: int = -1
    done: float = 0.0

    def __iter__(self):
        yield self.observation
        yield self.action
        yield self.reward
        yield self.action_opponent
        yield self.done

    @classmethod
    def stack(cls, transitions: t.Sequence):
        """Convert a sequence of namedtuples into a namedtuples of sequences."""

        def stacking_map(transitions):
            for items in zip(*transitions):
                yield np.stack(items)
            
        return cls(*stacking_map(transitions))


class PlayerWithBoardHistory(Player):
    """ Player subclass that maintains board state and records its history."""

    def __init__(
        self,
        capture_reward_func=None,
        move_reward_func=None,
        sense_reward_func=None,
        root_plot_directory=None,
    ) -> None:

        self.board = None
        self.color = None

        self.capture_reward_func = capture_reward_func
        self.move_reward_func = move_reward_func
        self.sense_reward_func = sense_reward_func

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
        self, color: Color, board: chess.Board, opponent_name: str
    ):
        # Initialize board and color
        self.board = board
        # TODO: insure correct color with context manager?
        self.board.turn = color
        self.color = color

        self.history = []

        self.save_board_to_svg()

    def handle_opponent_move_result(
        self, captured_my_piece: bool, capture_square: t.Optional[Square]
    ):
        if captured_my_piece:
            piece = self.board.remove_piece_at(capture_square)
            if self.capture_reward_func is not None:
                self.history[-1].reward += self.capture_reward_func(piece, lost=True)
    

    def handle_sense_result(
        self, sense_result: t.List[t.Tuple[Square, t.Optional[chess.Piece]]]
    ):
        # TODO: Remove the pieces from their old locations (if known).
        for square, piece in sense_result:
            if piece is not None:
                self.board.set_piece_at(square, piece)
                if self.sense_reward_func is not None and piece.color == self.color:
                    self.history[-1].reward += self.sense_reward_func(piece)

    def handle_move_result(
        self, 
        requested_move: t.Optional[chess.Move], 
        taken_move: t.Optional[chess.Move],
        captured_opponent_piece: bool, 
        capture_square: t.Optional[Square]
    ):
        if captured_opponent_piece:
            piece = self.board.remove_piece_at(capture_square)
            if self.capture_reward_func is not None:
                self.history[-1].reward += self.capture_reward_func(piece, lost=False)

        if taken_move is not None:
            self.board.push(taken_move)
            self.board.turn = self.color
        
        if self.move_reward_func is not None:
            self.history[-1].reward += self.move_reward_func(taken_move, requested_move)

        self.save_board_to_svg(requested_move)

    def handle_game_end(
        self, 
        winner_color: t.Optional[Color], 
        win_reason: t.Optional[WinReason],
        game_history: GameHistory
    ):
        if win_reason == WinReason.KING_CAPTURE:
            reward = 1 if winner_color == self.color else -1
        elif win_reason == WinReason.RESIGN:
            # Currently, we only resign when going over the move limit. Both players take a penalty.
            reward = -1

        self.history[-1].reward = reward
        self.history[-1].done = 1.0

        self.save_board_to_svg()

    def save_board_to_svg(self, lastmove=None):
        """Plot the board state with the last requested move."""

        if self._root_plot_directory is not None:
            svg_board = chess.svg.board(self.board, lastmove=lastmove)
            svg_path = os.path.join(self.plot_directory, f"_{self._plot_index}.svg")

            with open(svg_path, "w") as f:
                f.write(svg_board)

            self._plot_index += 1


class RandomBot(PlayerWithBoardHistory):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_sense(
        self,
        sense_actions: t.List[Square], 
        move_actions: t.List[chess.Move], 
        seconds_left: float
    ) -> t.Optional[Square]:
        sense = random.choice(sense_actions)
        self.history.append(Transition(board_to_onehot(self.board), sense, reward=0))

        return sense

    def choose_move(
        self, move_actions: t.List[chess.Move], seconds_left: float
    ) -> t.Optional[chess.Move]:
        # TODO: implement None action selection 
        # move = random.choice(move_actions + [None])
        move = random.choice(move_actions)
        self.history.append(
            Transition(board_to_onehot(self.board), move_to_index(move), reward=0)
        )

        return move


STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'


class TroutBot(Player):
    """
    TroutBot uses the Stockfish chess engine to choose moves. In order to run TroutBot you'll need to download
    Stockfish from https://stockfishchess.org/download/ and create an environment variable called STOCKFISH_EXECUTABLE
    that is the path to the downloaded Stockfish executable.

    Copied from https://reconchess.readthedocs.io/en/latest/bot_create.html.
    """

    def __init__(self):
        self.board = None
        self.color = None
        self.my_piece_captured_square = None

        # make sure stockfish environment variable exists
        if STOCKFISH_ENV_VAR not in os.environ:
            raise KeyError(
                'TroutBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
                    STOCKFISH_ENV_VAR))

        # make sure there is actually a file
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

        # initialize the stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)

    def handle_game_start(
        self, color: Color, board: chess.Board, opponent_name: str
    ):
        self.board = board
        self.color = color

    def handle_opponent_move_result(
        self, captured_my_piece: bool, capture_square: t.Optional[Square]
    ):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

    def choose_sense(
        self, 
        sense_actions: t.List[Square], 
        move_actions: t.List[chess.Move], 
        seconds_left: float
    ) -> t.Optional[Square]:
        # if our piece was just captured, sense where it was captured
        if self.my_piece_captured_square:
            return self.my_piece_captured_square

        # if we might capture a piece when we move, sense where the capture will occur
        future_move = self.choose_move(move_actions, seconds_left)
        if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
            return future_move.to_square

        # otherwise, just randomly choose a sense action, but don't sense on a square where our pieces are located
        for square, piece in self.board.piece_map().items():
            if piece.color == self.color:
                sense_actions.remove(square)
        return random.choice(sense_actions)

    def handle_sense_result(
        self, sense_result: t.List[t.Tuple[Square, t.Optional[chess.Piece]]]
    ):
        # add the pieces in the sense result to our board
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)

    def choose_move(
        self, move_actions: t.List[chess.Move], seconds_left: float
    ) -> t.Optional[chess.Move]:
        # if we might be able to take the king, try to
        enemy_king_square = self.board.king(not self.color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                return chess.Move(attacker_square, enemy_king_square)

        # otherwise, try to move with the stockfish chess engine
        try:
            self.board.turn = self.color
            self.board.clear_stack()
            result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
            return result.move
        except chess.engine.EngineTerminatedError:
            print('Stockfish Engine died')
        except chess.engine.EngineError:
            print('Stockfish Engine bad state at "{}"'.format(self.board.fen()))

        # if all else fails, pass
        return None

    def handle_move_result(
        self, 
        requested_move: t.Optional[chess.Move], 
        taken_move: t.Optional[chess.Move],
        captured_opponent_piece: bool, 
        capture_square: t.Optional[Square]
    ):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)

    def handle_game_end(
        self, 
        winner_color: t.Optional[Color], 
        win_reason: t.Optional[WinReason],
        game_history: GameHistory
    ):
        try:
            # if the engine is already terminated then this call will throw an exception
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass


class QAgent(PlayerWithBoardHistory):

    def __init__(
        self,
        q_net,
        policy_sampler,
        narx_memory_length,
        device,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.memory_length = narx_memory_length
        self.nanrx_memory = None

        self.q_net = q_net
        self.device = device
        self.policy_sampler = policy_sampler

    def handle_game_start(
        self, color: Color, board: chess.Board, opponent_name: str
    ):
        super().handle_game_start(color, board, opponent_name)
        
        # Initialize NARX memory with the shape [L, 8, 8, 13],
        # where L is the memory lenght, 8x8 is the board dimensions
        # and 13 is the one-hot-encoded piece representation
        self.nanrx_memory = np.tile(board_to_onehot(self.board), (self.memory_length, 1, 1, 1))

    def add_to_memory(self, board_onehot):
        assert isinstance(board_onehot, np.ndarray)
        assert board_onehot.shape == (8, 8, 13)

        # TODO: benchmark and search for faster implementations if needed
        # Shift memory by 1 postition to the right
        self.nanrx_memory = np.roll(self.nanrx_memory, 1, axis=0)
        # Overwrite the first observation
        self.nanrx_memory[0, :] = board_onehot

    def choose_sense(
        self, 
        sense_actions: t.List[Square], 
        move_actions: t.List[chess.Move], 
        seconds_left: float
    ) -> t.Optional[Square]:
        # Add latest state of observation to the NARX memory
        self.add_to_memory(board_to_onehot(self.board))

        with torch.no_grad():
            # Compute state Value and Q-value for every sense action 
            q_net_input = torch.as_tensor(
                self.nanrx_memory, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # Add the batch dim.
           
            _, sense_q, *_ = self.q_net(q_net_input)

            sense_index = self.policy_sampler(sense_q.squeeze(0), list(range(64)))

        self.history.append(Transition(board_to_onehot(self.board), sense_index, reward=0))

        return sense_actions[sense_index]

    def choose_move(
        self, move_actions: t.List[chess.Move], seconds_left: float
    ) -> t.Optional[chess.Move]:
        # TODO: allow for None move actions
        # Add latest state of observation to the NARX memory
        self.add_to_memory(board_to_onehot(self.board))

        # Transform chess Moves into their indices in action Space
        moves_indices = list(map(move_to_index, move_actions))

        with torch.no_grad():
            # Compute state Value and Q-value for every move action
            q_net_input = torch.as_tensor(
                self.nanrx_memory, dtype=torch.float32, device=self.device,
            ).unsqueeze(0)  # Add the batch dim.

            _, _, move_q, *_ = self.q_net(q_net_input)
            
            move_index = self.policy_sampler(move_q.squeeze(0), moves_indices)

        # Convert index of an action to chess Move
        move = index_to_move(move_index)
        if move not in move_actions:
            # TODO: what should we do with underpomotions
            # move.promotion = chess.QUEEN
            move = None


        # assert move in set(move_actions)

        self.history.append(Transition(board_to_onehot(self.board), move_index, reward=0))

        return move


class TestQNet(torch.nn.Module):

    def __init__(self, narx_memory_length, n_hidden, channels_per_layer: t.Optional[t.List[int]] = None):
        super().__init__()

        self.narx_memory_length = narx_memory_length
        self.n_hidden = n_hidden
        self.channels_per_layer = channels_per_layer or [64, 128, 256]

        # Board convolution backbone:
        # 3D convolution layer is applied to a thensor with shape (N,C​,D​,H​,W​)
        # where N - batch size, C (channels) - one-hot-encoding of a piece,
        # D (depth) - history length, H and W are board dimentions (i.e. 8x8).

        self.conv_stack = torch.nn.Sequential(
            # torch.nn.Conv3d(
            #     in_channels=len(PIECE_INDEX),
            #     out_channels=self.channels_per_layer[0],
            #     kernel_size=(3, 3, 3),
            #     # stride=(2, 2, 2)
            # ),
            # torch.nn.ReLU(),
            # torch.nn.Conv3d(
            #     in_channels=self.channels_per_layer[0],
            #     out_channels=self.channels_per_layer[1],
            #     kernel_size=(3, 3, 3),
            #     # stride=(2, 2, 2)
            # ),
            # torch.nn.ReLU(),
            # torch.nn.Conv3d(
            #     in_channels=self.channels_per_layer[1],
            #     out_channels=self.channels_per_layer[2],
            #     kernel_size=(3, 3, 3),
            #     # stride=(2, 2, 2)
            # ),
            # torch.nn.ReLU()
            torch.nn.Conv3d(
                in_channels=len(PIECE_INDEX),
                out_channels=64,
                kernel_size=(5, 5, 5),
                stride=(3, 3, 3)
            ),
            torch.nn.ReLU(),
            # torch.nn.Conv3d(
            #     in_channels=64,
            #     out_channels=128,
            #     kernel_size=(3, 3, 3),
            #     stride=(3, 3, 3)
            # ),
            # torch.nn.ReLU()
        )

        dummy_input = torch.zeros((1, len(PIECE_INDEX), self.narx_memory_length, 8, 8))
        fc_input_size = functools.reduce(operator.mul, self.conv_stack(dummy_input).shape)

        self.fc_stack = torch.nn.Sequential(
            torch.nn.Linear(fc_input_size, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
        )

        # Player heads
        self.fc_state_val = torch.nn.Linear(self.n_hidden, 1)
        self.fc_sense_adv = torch.nn.Linear(self.n_hidden, 64)
        self.fc_move_adv = torch.nn.Linear(self.n_hidden, 64 * 64)
        # Opponent heads
        self.fc_opponent_move = torch.nn.Linear(self.n_hidden, 64 * 64)

    def forward(self, board_memory: torch.Tensor):
        # Re-align board memory to fit the shape described in init
        # (B, T, H, W, C) -> (B, C, T, H, W)
        x = board_memory.permute(0, 4, 1, 2, 3)

        x = self.conv_stack(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc_stack(x)

        # Compute heads
        state_val = torch.nn.functional.relu(self.fc_state_val(x))
        
        sense_adv = torch.nn.functional.relu(self.fc_sense_adv(x))
        sense_q = state_val + sense_adv - sense_adv.mean(-1, keepdim=True)
        
        move_adv = torch.nn.functional.relu(self.fc_move_adv(x))
        move_q = state_val + move_adv - move_adv.mean(-1, keepdim=True)

        opponent_move = torch.nn.functional.relu(self.fc_opponent_move(x))

        return state_val, sense_q, move_q, opponent_move


class QAgentBatched(BatchedAgentManager):

    def __init__(
        self,
        q_net,
        policy_sampler,
        narx_memory_length,
        device
    ):

        self.q_net = q_net
        self.policy_sampler = policy_sampler
        self.narx_memory_length = narx_memory_length
        self.device = device

    def build_agent(self) -> QAgent:
        return QAgent(
            self.q_net,
            self.policy_sampler,
            self.narx_memory_length,
            self.device
        )

    def choose_move_batched(self,
                            agents: List[QAgent],
                            move_action_lists: List[List[chess.Move]]) -> Optional[List[chess.Move]]:

        narx_memory_batch = self._build_narx_batch(agents)

        with torch.no_grad():
            # Compute state Value and Q-value for every move action
            _, _, move_q_batch, *_ = self.q_net(narx_memory_batch)

        move_batch = []
        for agent, moves, move_q in zip(agents, move_action_lists, move_q_batch):

            # Transform chess Moves into their indices in action Space
            moves_indices = list(map(move_to_index, moves))
            move_index = self.policy_sampler(move_q, moves_indices)

            # Convert index of an action to chess Move
            move = index_to_move(move_index)
            if move not in move_action_lists:
                # TODO: what should we do with under-promotions
                # move.promotion = chess.QUEEN
                move = None

            # assert move in set(move_actions)

            agent.history.append(Transition(board_to_onehot(agent.board), move_index, reward=0))

            move_batch.append(move)

        return move_batch

    def choose_sense_batched(self, agents: List[QAgent],
                             sense_action_lists: List[List[Square]],
                             move_action_lists: List[List[chess.Move]]) -> List[Optional[Square]]:

        narx_memory_batch = self._build_narx_batch(agents)

        with torch.no_grad():
            # Compute state Value and Q-value for every sense action
            _, sense_q_batch, *_ = self.q_net(narx_memory_batch)

        sense_batch = []
        for agent, senses, sense_q in zip(agents, sense_action_lists, sense_q_batch):
            sense_index = agent.policy_sampler(sense_q, list(range(64)))

            agent.history.append(Transition(board_to_onehot(agent.board), sense_index, reward=0))

            sense_batch.append(senses[sense_index])

        return sense_batch

    def _build_narx_batch(self, agents):
        # TODO: allow for None move actions
        # Add latest state of observation to the NARX memory
        narx_memory_batch = torch.empty((len(agents), *agents[0].nanrx_memory.shape),
                                        dtype=torch.float32, device=self.device)

        for i, agent in enumerate(agents):
            agent.add_to_memory(board_to_onehot(agent.board))
            narx_memory_batch[i, ...] = torch.as_tensor(agent.nanrx_memory,
                                                        dtype=narx_memory_batch.dtype, device=self.device)
        return narx_memory_batch
