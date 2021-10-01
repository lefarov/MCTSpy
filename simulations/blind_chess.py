import typing as t

import chess
import json
import enum
import typing
import numpy as np

from collections import namedtuple, defaultdict
from reconchess import (
    LocalGame,
    GameHistoryEncoder,
    GameHistoryDecoder,
    Player,
    play_turn, WinReason,
)

from mctspy.simluator import SimulatorInterface


def capture_reward(piece: chess.Piece, game: LocalGame):
    """Simple capturing reward."""
    return 1


""" Namedtuple for the game state

board: chess._BoardState
    Hashable true board state.
history: str
    JSON dump of the game history.
start_time: datetime
    Start time of the current turn.
timers: str
    JSON dump of a dictionary with the number of seconds left for each player.

"""
GameState = namedtuple(
    "GameState", ("board", "history", "start_time", "timers")
)

GameAction = namedtuple("GameAction", ("type", "action"))


class BlindChessSP:
    """Statefull (but resetable) simulator for single-player Blind Chess.
    
    Opponent's agent is soldered into the simulator.
    
    TODO:
    1. Manage boards' stack of moves.
    2. Make sure that turns are preserved for two boards.
    3. Play for different color.
    4. Play simultoneounsly for two players.
    """

    game_history_attr_name = "_LocalGame__game_history"

    def __init__(
        self, 
        opponent: Player,
        reward_func,
        player_color: bool=chess.WHITE, 
        second_per_player: float=900
    ) -> None:

        self.opponent = opponent
        self.reward_fuc = reward_func

        self.game = LocalGame(second_per_player)
        self.sense_action = True

        # Handle game start
        self.observed_board = self.game.board.copy()
        self.player_color = player_color

        self.opponent.handle_game_start(
            not self.player_color, self.game.board.copy(), "MCTS Bot"
        )

        # Start the game
        self.game.start()

    def reset(self, state: GameState, observation: chess._BoardState):
        """Reset the game to the given state."""
        # restore the observable and true boards
        observation.restore(self.observed_board)
        state.board.restore(self.game.board)

        # Restore the history
        restored_history = json.loads(state.history, cls=GameHistoryDecoder)
        setattr(self.game, self.game_history_attr_name, restored_history)

        # Restore the sutrat time of the current turn
        self.game.current_turn_start_time = state.start_time

        # Restore timers
        timers = json.loads(state.timers)
        self.game.seconds_left_by_color.update(timers)

        # Restore the turn
        self.game.turn = self.player_color

        # Restore the opponent
        self.opponent.handle_game_start(
            not self.player_color, self.game.board.copy(), "MCTS Bot"
        )

        # Restore the current action type
        self.sense_action = True

    def get_state(self):
        """Get the complete game state."""
        # Dump the game history to JSON
        history = getattr(self.game, self.game_history_attr_name)
        history_dump = json.dumps(history, cls=GameHistoryEncoder)

        # Dump timers to JSON
        timers = json.dumps(self.game.seconds_left_by_color)

        return GameState(
            board=self.game.board._board_state(),
            history=history_dump,
            start_time=self.game.current_turn_start_time,
            timers=timers,
        )

    def get_observation(self):
        """Get the current observation."""
        return self.observed_board._board_state()

    def step(self, action):
        """Execute action.
        
        TODO: move action should switch the turn, sens action shouldn't

        """
        reward = 0

        # Check the type of the current action
        if self.sense_action:
            assert isinstance(action, int)

            # Apply sense action and add the result to the observable board
            for square, piece in self.game.sense(action):
                self.observed_board.set_piece_at(square, piece)

        else:
            assert isinstance(action, chess.Move)

            # Apply move action and update the board's stack
            _, taken_move, capture_square = self.game.move(action)
            if taken_move is not None:
                self.observed_board.push(taken_move)
                # Push will switch the color of the board, so we reset it back
                self.observed_board.turn = self.player_color

            # Remove captured figure from the observable board and compute reward
            if capture_square is not None:
                captured_piece = self.observed_board.remove_piece_at(capture_square)
                reward = self.reward_fuc(captured_piece, self.game)

            # End player's turn
            self.game.end_turn()
        
            # Opponent move
            # TODO: will it work if state is terminal?
            play_turn(self.game, self.opponent, end_turn_last=False)

            # Update observable board based on opponent move
            capture_square = self.game.opponent_move_results()
            if capture_square is not None:
                self.observed_board.remove_piece_at(capture_square)
                # TODO: add penalty for captured figures

        # Switch the action type
        self.sense_action = not self.sense_action

        return (
            self.get_state(), self.get_observation(), reward, self.player_color
        )

    def is_terminal(self):
        return self.game.is_over()

    def enumerate_actions(self):
        if self.sense_action:
            return self.game.sense_actions()
        else:
            return self.game.move_actions()

    def get_agent_num(self):
        return 1

    def get_current_agent(self):
        return self.player_color

    def get_terminal_value(self):
        value = 0

        winner_color = self.game.get_winner_color()
        if winner_color == self.player_color:
            value = 100.0

        return {self.player_color: value}


class BlindChessActionType(enum.IntEnum):
    Sense = 0
    Move = 1


MPGameState = namedtuple(
    "MPGameState", 
    ("true_board", "white_board", "black_board", "turn", "action_type")
)

MPGameAction = namedtuple("MPGameAction", ("sense", "move"))


class BlindChessMP(SimulatorInterface):
    """ Opponent should be played by the same MCTS tree

    TODO: 
    1. make sure time cannot be exceeded (this should be controlled by MCTS)
    2. opponenet does get the full state back (i.e. observation == full state)?
    3. opponent only does move actions?
    """

    def __init__(
        self,
        white_name: str = "MCTSwhite",
        black_name: str = "MCTSblack",
        seconds_per_player: int = 900,
    ) -> None:

        self.game = LocalGame(seconds_per_player)
        self.game.store_players(white_name, black_name)

        self.observed_boards = {
            True: self.game.board.copy(), False: self.game.board.copy()
        }

        # Set correct turns on boards
        for color, board in self.observed_boards.items():
            board.turn = color

        self.game.start()

    def reset(self, state: MPGameState):
        """Reset the game to the given state.
        """
        state.true_board.restore(self.game.board)
        state.white_board.restore(self.observed_boards[True])
        state.black_board.restore(self.observed_boards[False])

        # Restore the turn
        self.game.turn = state.turn

    def _get_state(self, action_type):
        # TODO: use FEN strings also for state?
        return MPGameState(
            true_board=self.game.board._board_state(),
            white_board=self.observed_boards[True]._board_state(),
            black_board=self.observed_boards[False]._board_state(),
            turn=self.game.turn,
            action_type=action_type,
        )

    def step(self, state: MPGameState, action: MPGameAction, reset: bool=False):
        # TODO: make it a decorator
        # Reset observable and true boards to the current state
        if reset:
            self.reset(state)

        reward = 0
        if state.action_type == BlindChessActionType.Sense:
            assert isinstance(action.sense, int)

            # Apply sense action and add the result to the observable board
            for square, piece in self.game.sense(action.sense):
                self.observed_boards[state.turn].set_piece_at(square, piece)

            action_type = BlindChessActionType.Move

        elif state.action_type == BlindChessActionType.Move:
            assert isinstance(action.move, chess.Move)

            # Apply move action and update the board's stack
            _, taken_move, capture_square = self.game.move(action.move)

            # If captured any piece
            # TODO: handle the surrigate reward for capturing the piece
            if capture_square is not None:
                # Remove it from observable boards
                for _, board in self.observed_boards.items():
                    board.remove_piece_at(capture_square)

            if taken_move is not None:
                self.observed_boards[state.turn].push(taken_move)

            # TODO: make it a decorator
            # Make sure that turns didn't change on observed boards
            for color, board in self.observed_boards.items():
                board.turn = color

            # End player's turn
            self.game.end_turn()

            action_type = BlindChessActionType.Sense

        else:
            raise ValueError("Unsupported action type.")

        state = self._get_state(action_type)
        observation = {True: state.white_board, False: state.black_board}

        return self._get_state(action_type), observation, reward, self.game.turn

    def enumerate_actions(self, state, reset=True):
        if reset:
            self.reset(state)

        if state.action_type == BlindChessActionType.Sense:
            return {MPGameAction(a, None) for a in self.game.sense_actions()}
        else:
            return {MPGameAction(None, a) for a in self.game.move_actions()}

    def get_initial_state(self):
        return self._get_state(BlindChessActionType.Sense), self.game.turn

    def get_agent_num(self):
        return 2

    def get_current_agent(self, state: MPGameState, reset=True):
        if reset:
            self.reset(state)

        return self.game.turn

    def state_is_terminal(self, state: MPGameState, reset=True):
        if reset:
            self.reset(state)

        return self.game.is_over()

    def get_terminal_value(self, state: MPGameState, reset=True):
        if reset:
            self.reset(state)

        if self.game.get_win_reason() == WinReason.TIMEOUT:
            return {True: 0, False: 0}

        winner_color = self.game.get_winner_color()
        return {winner_color: 1, not winner_color: -1}


# I know how dict-comprehension works, I just don't like how it looks
PIECE_INDEX = {" ": 0}
for i, symbol in enumerate(chess.UNICODE_PIECE_SYMBOLS.keys()):
    PIECE_INDEX[symbol] = i + 1


def board_state_to_npboard(board_state: chess._BoardState, piece_index: typing.Dict):
    board = chess.Board.empty()
    board_state.restore(board)

    return board_to_npboard(board, piece_index)


def fen_to_npboard(fen, piece_index=PIECE_INDEX):
    board = chess.Board.empty()
    board.set_fen(fen)

    return board_to_npboard(board, piece_index)


def board_to_npboard(board, piece_index=PIECE_INDEX):
    board_index = np.zeros((64, ), dtype=np.int32)
    board_onehot = np.zeros((64, len(piece_index)))

    for square, piece in board.piece_map().items():
        board_index[square] = piece_index[piece.symbol()]
        board_onehot[square][piece_index[piece.symbol()]] = 1

    return board_index.reshape(8, 8), board_onehot.reshape(8, 8, -1)


def action_to_npaction(action: chess.Move) -> np.ndarray:
    action_onehot = np.zeros(64 * 64)
    action_onehot[action.from_square * 64 + action.to_square] = 1

    return action_onehot


def onehot_action_to_move(action: int):
    
    
    file_index = action // 8
    rank_index = action % 8

    return f"{}"
