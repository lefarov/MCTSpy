import typing as t

from datetime import datetime
from reconchess import LocalGame as LocalReconChessGame

from recon_tictac import Player, WinReason, Square, Board


class LocalGame(LocalReconChessGame):

    def __init__(self, seconds_per_player: float = 900):
        super().__init__(seconds_per_player)

        self.turn = Player.Nought
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
        return list(range(self.board.Size ** 2))

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

    def get_winner_color(self) -> t.Optional[Player]:
        return self.board.get_winner()

    def get_win_reason(self) -> t.Optional[WinReason]:
        return self.board.get_win_reason()

    def is_over(self) -> bool:
        return self.board.is_game_over()
