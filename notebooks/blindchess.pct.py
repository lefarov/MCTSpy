# %% imports
import numpy as np

import chess
from chess import (
    SQUARES,  # Whites are at the bottom of the board
    SQUARES_180,  # Squares rotaed 180 degrees (blacks are at the bottom)
    SQUARE_NAMES,
)

from reconchess import LocalGame
# %%

game = LocalGame()

piece = game.board.piece_at(SQUARES_180[0])
piece.__hash__()

# copying the board ?
board = game.board
board_copy = game.board.copy()

board.push(chess.Move.from_uci("g1f3"))

# `_BoardState` is hashable
board_state = board._board_state()
board_move_stack = tuple(board.move_stack)

s = set()
s.add(board_state)
s.add(board_move_stack)

# Doesn't restore the hisroty !!!
board_state.restore(board_copy)
board_copy.move_stack = list(board_move_stack) 

# %% Board to observation


# %% [markdown]
# 1. Can we save restore board? (board as a state of MCTS)?
#   1.1 Manually go over the figures and restore the board
# 2. How should we deal with actual board / visible board?
