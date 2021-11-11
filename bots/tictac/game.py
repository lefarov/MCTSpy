import copy
from datetime import datetime
from typing import Optional

import numpy as np
import reconchess.game
import typing as t

from enum import IntEnum

from reconchess import LocalGame


class ActionType(IntEnum):
    Unknown = 0
    Sense = 1
    Move = 2


class Player(IntEnum):
    Cross = 1
    Nought = 0   # Has to match the reconchess interface, which uses bools.


class Square(IntEnum):
    Empty = -1
    Cross = Player.Cross
    Nought = Player.Nought


class WinReason(IntEnum):
    Draw = 0
    MatchThree = 1


class Board:
    Size: int = 3
    
    def __init__(self) -> None:
        self._board = [Square.Empty] * (TicTacToe.BoardSize ** 2)  # type: t.List[Square]

    def __setitem__(self, key, value: t.Union[Square, int]):
        self._board[key] = Square(value)

    def __getitem__(self, square: int) -> Square:
        return Square(self._board[square])

    def __repr__(self):
        s = ''
        reprDict = {Square.Cross: 'X', Square.Nought: 'O', Square.Empty: ' '}
        size = TicTacToe.BoardSize
        for i in range(size):
            s += ''.join(map(reprDict.get, self._board[i * size: (i + 1) * size]))
            s += '\n'

        return s

    def to_array(self):
        return np.array(self._board).reshape((Board.Size, Board.Size))

    def copy(self) -> 'Board':
        return copy.deepcopy(self)

    def _repr_svg_(self):
        pass


class TicTacToe(LocalGame):
    BoardSize: int = 3

    def __init__(self, seconds_per_player: float = 900):
        super().__init__(seconds_per_player)

        self.turn = Player.Cross
        self.board = Board()

        self.seconds_left_by_player = {
            Player.Cross: seconds_per_player, Player.Nought: seconds_per_player
        }

        self.current_turn_start_time = None
        self.move_results = None

        self._is_finished = False
        self._resignee = None

    def start(self):
        self.current_turn_start_time = datetime.now()

    def end(self):
        self.seconds_left_by_player[self.turn] = self.get_seconds_left()
        self._is_finished = True

    def store_players(self, white_name, black_name):
        pass

    def resign(self):
        self._resignee = self.turn

    def get_seconds_left(self) -> float:
        if not self._is_finished and self.current_turn_start_time:
            elapsed_since_turn_start = (datetime.now() - self.current_turn_start_time).total_seconds()
            return self.seconds_left_by_player[self.turn] - elapsed_since_turn_start
        else:
            return self.seconds_left_by_player[self.turn]

    def sense_actions(self) -> t.List[int]:
        # We can sense every square.
        return list(range(TicTacToe.BoardSize ** 2))

    def move_actions(self) -> t.List[int]:
        # Return all positions except for the ones that player already holds.
        # Avoids pointless moves, makes the game converge.
        return [i for i, square in enumerate(self.board) if square != self.turn]

    def opponent_move_results(self) -> t.Optional[int]:
        return self.move_results

    def sense(self, square: t.Optional[int]) -> t.Tuple[int, Square]:

        sense_result = None
        if square is not None and not self._is_finished:
            if square not in self.sense_actions():
                raise ValueError(f"TicTacToe::sense({square}): {square} is not a valid square.")

            sense_result = self.board[square]

        return square, sense_result

    def move(self, requested_square: t.Optional[int]) -> t.Tuple[t.Optional[int], t.Optional[int], t.Optional[int]]:

        used_square = None
        if self.board[requested_square] == Square.Empty and not self._is_finished:
            self.board[requested_square] = self.turn
            used_square = requested_square

        return requested_square, used_square, None

    def end_turn(self):
        elapsed = datetime.now() - self.current_turn_start_time
        self.seconds_left_by_player[self.turn] -= elapsed.total_seconds()

        self.turn = Player((self.turn + 1) % 2)
        self.current_turn_start_time = datetime.now()

    def get_game_history(self):
        return None

    def get_winner_color(self) -> Optional[Player]:
        winnerId = None
        size = TicTacToe.BoardSize
        for playerId in (Player.Cross, Player.Nought):
            for i in range(size):
                isRow = all(self.board[i * size + j] == playerId for j in range(size))
                isCol = all(self.board[j * size + i] == playerId for j in range(size))

                if isRow or isCol:
                    winnerId = playerId
                    break

            isDiag = all(self.board[j * size + j] == playerId for j in range(size))
            isDiagInv = all(self.board[j * size + size - j - 1] == playerId for j in range(size))

            if isDiag or isDiagInv:
                winnerId = playerId
                break

        return winnerId

    def get_win_reason(self) -> Optional[WinReason]:
        winner_id = self.get_winner_color()
        return WinReason.MatchThree if winner_id is not None else WinReason.Draw

    def is_over(self) -> bool:
        return self.get_winner_color() is not None or not any(x == Square.Empty for x in self.board)
