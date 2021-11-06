import copy
import datetime
import reconchess.game
import typing as t

from enum import IntEnum


class ActionType(IntEnum):
    Unknown = 0
    Sense = 1
    Move = 2


class Player(IntEnum):
    Cross = 0
    Nought = 1


class Square(IntEnum):
    Empty = -1
    Cross = Player.Cross
    Nought = Player.Nought


class Board:
    
    def __init__(self) -> None:
        self._board = [TicTacToe.EmptyCell] * (TicTacToe.BoardSize ** 2)

    def __getitem__(self, square: int) -> Square:
        return self._board[square]

    def __repr__(self):
        s = ''
        reprDict = {Player.Cross: 'X', Player.Nought: 'O', TicTacToe.EmptyCell: ' '}
        size = TicTacToe.BoardSize
        for i in range(size):
            s += ''.join(map(reprDict.get, self.grid[i * size: (i + 1) * size]))
            s += '\n'

        return s

    def _repr_svg_(self):
        pass


class TicTacToe(reconchess.game.Game):
    BoardSize: int = 3
    EmptyCell: int = -1

    def __init__(self, seconds_per_player: float = 900):
        self.turn = Player.Cross
        # Shortcat for accessing board from the game object
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
        self.seconds_left_by_color[self.turn] = self.get_seconds_left()
        self._is_finished = True

    def store_players(self, white_name, black_name):
        pass

    def resign(self):
        self._resignee = self.turn

    def get_seconds_left(self) -> float:
        if not self._is_finished and self.current_turn_start_time:
            elapsed_since_turn_start = (datetime.now() - self.current_turn_start_time).total_seconds()
            return self.seconds_left_by_color[self.turn] - elapsed_since_turn_start
        else:
            return self.seconds_left_by_color[self.turn]

    def sense_actions(self) -> t.List[int]:
        # We can sense every square.
        return list(range(TicTacToe.BoardSize ** 2))

    def move_actions(self) -> t.List[int]:
        # Return all positions except for the ones that player already holds.
        # Avoids pointless moves, makes the game converge.
        return [i for i, square in enumerate(self.state.board) if square != self.state.nextAgentId]

    def opponent_move_results(self) -> t.Optional[int]:
        return self.move_results

    def sense(self, square: t.Optional[int]) -> t.Tuple[int, Square]:
        
        sense_result = None
        if square is not None and not self._is_finished:
            if square not in self.sense_actions():
                raise ValueError(f"TicTacToe::sense({square}): {square} is not a valid square.")

            sense_result = self.board[square]

        return square, sense_result

    def move(self, requested_square: t.Optional[int]) \
        -> t.Tuple[t.Optional[int], t.Optional[int], t.Optional[int]]:
        
        used_square = None
        if self.board[requested_square] == TicTacToe.EmptyCell and not self._is_finished:
            self.board[requested_square] = self.turn
            used_square = requested_square

        return requested_square, used_square, None

    def end_turn(self):
        elapsed = datetime.now() - self.current_turn_start_time
        self.seconds_left_by_color[self.turn] -= elapsed.total_seconds()

        self.turn = Player((self.turn + 1) % 2)
        self.current_turn_start_time = datetime.now()

    def get_game_history(self):
        return None

    def is_over(self) -> bool:
        pass

    def step(self, state: TicTacState, action: int) -> t.Tuple[TicTacState, int, float, int]:
        # We have list in our state
        newState = copy.deepcopy(state)
        coord = action

        # --- Handle the 'sense' action.
        if newState.nextActionType == ActionType.Sense:
            observation = newState.board[coord]
            newState.nextActionType = ActionType.Move

            return newState, observation, 0, newState.nextAgentId

        # --- Handle the 'move' action.

        # Place a new piece, unless the cell is occupied.
        if state.board[coord] == TicTac.EmptyCell:
            newState.board[coord] = state.nextAgentId

        # Report what's in the cell now.
        observation = newState.board[coord]
        # Advance the counters.
        newState.nextAgentId = (state.nextAgentId + 1) % 2
        newState.nextActionType = ActionType.Sense

        # Check for the win condition.
        size = TicTac.BoardSize
        for agentId in (0, 1):
            for i in range(size):
                isRow = all(newState.board[i * size + j] == agentId for j in range(size))
                isCol = all(newState.board[j * size + i] == agentId for j in range(size))

                if isRow or isCol:
                    newState.winnerId = agentId
                    break

            isDiag = all(newState.board[j * size + j] == agentId for j in range(size))
            isDiagInv = all(newState.board[j * size + size - j - 1] == agentId for j in range(size))

            if isDiag or isDiagInv:
                newState.winnerId = agentId
                break

        # Win = 1, Draw = 0, Loss = -1
        reward = (1 if newState.winnerId == state.nextAgentId else -1) if newState.winnerId is not None else 0

        return newState, observation, reward, newState.nextAgentId

    def state_is_terminal(self, state: TicTacState) -> bool:
        return state.winnerId is not None or not any(x == TicTac.EmptyCell for x in state.board)

    def get_terminal_value(self, state: TicTacState) -> t.Dict[t.Hashable, float]:
        return {
            0: int(state.winnerId == 0),
            1: int(state.winnerId == 1)
        }
