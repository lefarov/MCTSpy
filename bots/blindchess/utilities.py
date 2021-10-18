import random
import torch
import typing as t

import chess
import numpy as np


# Chess pieces values according to https://www.chess.com/terms/chess-piece-value
PIECE_VALUE = {
    chess.PAWN: 1/8,
    chess.KNIGHT: 3/8,
    chess.BISHOP: 3/8,
    chess.ROOK: 5/8,
    chess.QUEEN: 8/8,
    chess.KING: 8/8,
}


PIECE_INDEX = {" ": 0}
# I know how dict-comprehension works, I just don't like how it looks
for i, symbol in enumerate(chess.UNICODE_PIECE_SYMBOLS.keys()):
    PIECE_INDEX[symbol] = i + 1


def index_to_move(action: int):
    return chess.Move(from_square=action // 64, to_square=action % 64)


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def move_to_onehot(move: chess.Move) -> np.ndarray:
    move_onehot = np.zeros(64 * 64)
    move_onehot[move.from_square * 64 + move.to_square] = 1

    return move_onehot


def board_to_onehot(board: chess.Board, piece_index=PIECE_INDEX):
    board_onehot = np.zeros((64, len(piece_index)))

    for square, piece in board.piece_map().items():
        board_onehot[square][piece_index[piece.symbol()]] = 1

    return board_onehot.reshape(8, 8, -1)


def board_to_index_encoding(board: chess.Board, piece_index=PIECE_INDEX):
    board_index = np.zeros((64, ), dtype=np.int32)

    for square, piece in board.piece_map().items():
        board_index[square] = piece_index[piece.symbol()]

    return board_index.reshape(8, 8)


def board_state_to_index_encoding(board_state: chess._BoardState, piece_index: t.Dict):
    board = chess.Board.empty()
    board_state.restore(board)

    return board_to_index_encoding(board, piece_index)


def fen_to_index_encoding(fen, piece_index=PIECE_INDEX):
    board = chess.Board.empty()
    board.set_fen(fen)

    return board_to_index_encoding(board, piece_index)


def move_proxy_reward(taken_move, requested_move):
    # If invalid move was selected
    if taken_move is None:
        return -0.5

    # If valid move was selected, but it was modified because of unknown opponent figure
    if taken_move != requested_move:
        return -0.01

    return 0.0


def capture_proxy_reward(piece: chess.Piece, lost: bool, weight=0.0):
    # If Opponent captured your piece.
    if lost:
        # Return negative piece value multiplied by the importance weight of sense action.
        return - PIECE_VALUE[piece.piece_type] * weight
    else:
        # Return value of a Pawn, since we don't know which piece we've captured.
        return PIECE_VALUE[chess.PAWN] * weight


def sense_proxy_reward(piece: chess.Piece, weight=0.0):
    # Return sensed piece value multiplied by the importance weight of sense action.
    return PIECE_VALUE[piece.piece_type] * weight


def egreedy_masked_policy_sampler(
    q_values: torch.Tensor,
    valid_action_indices,
    mask_invalid_actions=True,
    eps: float = 0.05
) -> int:

    action_q_masked = torch.full_like(q_values, fill_value=torch.finfo(q_values.dtype).min)
    action_q_masked[valid_action_indices] = q_values[valid_action_indices]

    # If we don't mask invalid actions, use original Q as masked values
    if not mask_invalid_actions:
        action_q_masked = q_values

    if random.random() >= eps:
        action_index = torch.argmax(action_q_masked).item()
    else:
        action_index = random.choice(valid_action_indices)

    return action_index


def q_selector(q_net: torch.nn.Module, action: str="move") -> t.Callable:
    # Select requested output of the Q-net
    if action == "move":
        index = 2
    elif action == "sense":
        index = 1
    else:
        raise ValueError("Unknown action type.")

    def selector(obs):
        return q_net(obs)[index]

    return selector


def convert_to_tensor(array: np.ndarray, device) -> torch.Tensor:
    tensor = torch.as_tensor(array)
    # If tensor has only batch dimension, insert a singular dimension
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)

    # Torch indexing supports only int64
    if tensor.dtype == torch.int32:
        tensor = tensor.long()

    return tensor.to(device)
