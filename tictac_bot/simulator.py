import copy
import typing as t
from dataclasses import dataclass, field
from enum import IntEnum

from mctspy.simulator import SimulatorInterface


class TicTacActionType(IntEnum):
    Unknown = 0
    Sense = 1
    Move = 2


@dataclass
class TicTacState:
    board: t.List[int] = field(default_factory=lambda: [TicTac.EmptyCell] * (TicTac.BoardSize ** 2))

    nextActionType: TicTacActionType = TicTacActionType.Sense
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


class TicTac(SimulatorInterface):
    BoardSize: int = 3
    EmptyCell: int = -1

    def step(self, state: TicTacState, action: int) -> t.Tuple[TicTacState, int, float, int]:
        # We have list in our state
        newState = copy.deepcopy(state)
        coord = action

        # --- Handle the 'sense' action.
        if newState.nextActionType == TicTacActionType.Sense:
            observation = newState.board[coord]
            newState.nextActionType = TicTacActionType.Move

            return newState, observation, 0, newState.nextAgentId

        # --- Handle the 'move' action.

        # Place a new piece, unless the cell is occupied.
        if state.board[coord] == TicTac.EmptyCell:
            newState.board[coord] = state.nextAgentId

        # Report what's in the cell now.
        observation = newState.board[coord]
        # Advance the counters.
        newState.nextAgentId = (state.nextAgentId + 1) % 2
        newState.nextActionType = TicTacActionType.Sense

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

    def enumerate_actions(self, state: TicTacState) -> t.Set:
        # Return all positions except for the ones that player already holds.
        # Avoids pointless moves, makes the game converge.
        return set(i for i, x in enumerate(state.board) if x != state.nextAgentId)

    def get_initial_state(self) -> t.Tuple[TicTacState, int]:
        state = TicTacState()
        return state, state.nextAgentId

    def get_agent_num(self) -> int:
        return 2

    def get_current_agent(self, state: TicTacState) -> int:
        return state.nextAgentId

    def get_terminal_value(self, state: TicTacState) -> t.Dict[t.Hashable, float]:
        return {
            0: int(state.winnerId == 0),
            1: int(state.winnerId == 1)
        }
