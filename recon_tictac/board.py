import copy

import numpy as np
import typing as t

from recon_tictac.enums import Player, WinReason, Square
from recon_tictac.render import render_board


class Board:
    # TODO: add iteration over the board cells
    Size: int = 3
    Shape: t.Tuple[int, int] = (3, 3)

    def __init__(self) -> None:
        self._board = [Square.Empty] * (self.Size ** 2)  # type: t.List[Square]

    def __setitem__(self, key, value: t.Union[Square, int]):
        self._board[key] = Square(value)

    def __getitem__(self, square: int) -> Square:
        return Square(self._board[square])

    def __iter__(self):
        return iter(self._board)

    def __repr__(self):
        s = ""
        repr_dict = {Square.Cross: "X", Square.Nought: "O", Square.Empty: " "}
        for i in range(self.Size):
            s += "".join(map(repr_dict.get, self._board[i * self.Size: (i + 1) * self.Size]))
            s += "\n"

        return s

    def to_array(self):
        return np.array(self._board).reshape((Board.Size, Board.Size))

    def copy(self) -> 'Board':
        return copy.deepcopy(self)

    def is_game_over(self):
        return self.get_winner() is not None or not any(x == Square.Empty for x in self._board)

    def get_winner(self):
        # TODO: optimize it a bit :)
        winner_id = None
        for player_id in (Player.Cross, Player.Nought):
            for i in range(self.Size):
                is_row = all(self._board[i * self.Size + j] == player_id for j in range(self.Size))
                is_col = all(self._board[j * self.Size + i] == player_id for j in range(self.Size))

                if is_row or is_col:
                    winner_id = player_id
                    break

            is_diag = all(self._board[j * self.Size + j] == player_id for j in range(self.Size))
            is_diag_inv = all(self._board[j * self.Size + self.Size - j - 1] == player_id for j in range(self.Size))

            if is_diag or is_diag_inv:
                winner_id = player_id
                break

        return winner_id

    def get_win_reason(self) -> t.Optional[WinReason]:
        winner_id = self.get_winner()
        return WinReason.MatchThree if winner_id is not None else WinReason.Draw

    def _repr_svg_(self):
        return render_board(self).asSvg()
