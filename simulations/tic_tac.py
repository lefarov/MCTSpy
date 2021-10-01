import copy
import typing as t
from dataclasses import dataclass, field


TicTacAction = int


@dataclass
class TicTacState:
    board: t.List[int] = field(default_factory=lambda: [TicTac.EmptyCell] * (TicTac.BoardSize ** 2))

    nextAgentId: int = 0
    winnerId: t.Optional[int] = None

    def __repr__(self):
        s = ''
        reprDict = {0: 'X', 1: 'O', TicTac.EmptyCell: ' '}
        size = TicTac.BoardSize
        for i in range(size):
            s += ''.join(map(reprDict.get, self.board[i * size: (i + 1) * size]))
            s += '\n'

        return s

    def __hash__(self):
        return hash((tuple(self.board), self.nextAgentId, self.winnerId))


class TicTac:
    BoardSize: int = 3
    EmptyCell: int = -1

    def step(self, state: TicTacState, action: int) -> t.Tuple[TicTacState, t.Dict[int, float], int]:
        # We have a list in our state.
        newState = copy.deepcopy(state)
        coord = action

        # Place a new piece, unless the cell is occupied.
        if state.board[coord] == TicTac.EmptyCell:
            newState.board[coord] = state.nextAgentId

        # Advance the counter.
        newState.nextAgentId = (state.nextAgentId + 1) % 2

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
        rewards = {}
        for player_id in (0, 1):
            rewards[player_id] = (1 if newState.winnerId == player_id else -1) if newState.winnerId is not None else 0

        return newState, rewards, newState.nextAgentId

    def state_is_terminal(self, state: TicTacState) -> bool:
        return state.winnerId is not None or not any(x == TicTac.EmptyCell for x in state.board)

    def enumerate_actions(self, state: TicTacState) -> t.Set:
        # Return only the empty positions
        # Avoids pointless moves, makes the game converge.
        return set(i for i, x in enumerate(state.board) if x == TicTac.EmptyCell)

    def get_initial_state(self) -> t.Tuple[TicTacState, int]:
        state = TicTacState()

        return state, state.nextAgentId

    def get_agent_num(self) -> int:
        return 2

    def get_current_agent(self, state: TicTacState) -> int:
        return state.nextAgentId

    def get_terminal_value(self, state: TicTacState) -> t.Dict[t.Hashable, float]:
        # todo This is duplicated, fix.
        result = {}
        for player_id in (0, 1):
            result[player_id] = (1 if state.winnerId == player_id else -1) if state.winnerId is not None else 0

        return result
