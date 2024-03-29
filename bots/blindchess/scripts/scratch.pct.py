#%% imports
import torch
import json
import random
import numpy as np

import chess
from chess import (
    SQUARES,  # Whites are at the bottom of the board
    SQUARES_180,  # Squares rotaed 180 degrees (blacks are at the bottom)
    SQUARE_NAMES,
)

from reconchess import (
    LocalGame, 
    GameHistory, 
    GameHistoryEncoder, 
    GameHistoryDecoder,
)

from bots.blindchess.simulator import (
    BlindChessMP, 
    MPGameAction
)
from bots.blindchess.utilities import board_to_onehot, board_to_index_encoding, board_state_to_index_encoding, fen_to_index_encoding, \
    PIECE_INDEX

#%%

game = LocalGame()
game_hist = GameHistory()

# Use a dump of the game history as a part of the state
# TODO: access and restory games history
# Save game history
history_json = json.dumps(game_hist, cls=GameHistoryEncoder)
game_hist_restore = json.loads(history_json, cls=GameHistoryDecoder)

game.start()
game.__game_history = game_hist_restore

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

#%% Board to observation


#%% [markdown]
# 1. Can we save restore board? (board as a state of MCTS)?
#   1.1 Manually go over the figures and restore the board
# 2. How should we deal with actual board / visible board?

#%% [markdown]
# ## Step down manually through MCTS simulator for Blind Chess
from bots.blindchess.simulator import BlindChessSP, capture_reward
from bots.blindchess.agent import RandomBot, TroutBot

#%%
opponent = RandomBot()
sim = BlindChessSP(opponent, capture_reward)

#%%
sense_actions = sim.enumerate_actions()
sense = random.choice(sense_actions)
state, obs, rew, player_id = sim.step(sense)

# Assert that action type has changed
assert not sim.sense_action

#%%
move_actions = sim.enumerate_actions()
move = random.choice(move_actions)
state, obs, rew, player_id = sim.step(move)

#%%
opponent2 = RandomBot()
sim2 = BlindChessSP(opponent2, capture_reward)
#%%
sim2.reset(state, obs)
#%%


#%% [markdown]
# ## Multiplayer Game

#%%
sim = BlindChessMP()
state = sim.get_initial_state()

#%% Make two moves by each player
for i in range(4):
    sense_actions = sim.enumerate_actions(state)
    sense = random.choice(sense_actions)
    state, obs, rew, player_id = sim.step(
        state, MPGameAction(sense=sense, move=None)
    )

    move_actions = sim.enumerate_actions(state)
    move = random.choice(move_actions)
    state, obs, rew, player_id = sim.step(
        state, MPGameAction(sense=None, move=move)
    )

#%% Check the observations and the true board state
board = chess.Board()
state.true_board.restore(board)

#%% Construct observation
obs_embedding = torch.nn.Embedding(len(PIECE_INDEX), 4)
board_ind, board_ohe = board_to_index_encoding(sim.game.board)
# Learnable embeddings for board observation
# TODO: can we pretrain it?
emb = obs_embedding(torch.as_tensor(board_ind, dtype=torch.long))
#%%

act_embedding = torch.nn.Embedding()
