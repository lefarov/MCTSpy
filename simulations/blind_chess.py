import chess
import json

from collections import namedtuple
from reconchess import (
    LocalGame,
    GameHistoryEncoder, 
    GameHistoryDecoder,
    Player,
    play_turn,
)


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


class BlindChessMP:
    """ Opponent should be played by the same MCTS tree

    TODO: 
    1. opponenet does get the full state back (i.e. observation == full state)
    2. opponent only does move actions
    """

    def __init__(
        self,
        white_name: str = "MCTSwhite",
        black_name: str = "MCTSblack",
        seconds_per_player: int = 900,
    ) -> None:

        self.game = LocalGame(seconds_per_player)
        self.game.store_players(white_name, black_name)

        self.white_board = self.game.board.copy()
        self.black_board = self.game.board.copy()
        self.boards_dict = {True: self.white_board, False: self.black_board}

        self.game.start()

    def reset(
        self,
        state: GameState,
        observation_white: chess._BoardState,
        observation_black: chess._BoardState,
    ):
        """Reset the game to the given state.
        
        TODO: should we add move stack to the observation?
        """
        # restore the observable and true boards
        observation_white.restore(self.white_board)
        observation_black.resotre(self.black_board)
        state.board.restore(self.game.board)

        # Restore the history
        restored_history = json.loads(state.history, cls=GameHistoryDecoder)
        setattr(
            self.game,
            f"_{self.game.__class__.__name__}__game_history",
            restored_history
        )

        # Restore the sutrat time of the current turn
        self.game.current_turn_start_time = state.start_time

        # Restore timers
        timers = json.loads(state.timers)
        self.game.seconds_left_by_color.update(timers)

        # Restore the turn
        self.game.turn = state.turn

    def get_state(self):
        """Get the complete game state."""
        # Dump the game history to JSON
        history = getattr(self.game, f"_{self.game.__class__.__name__}__game_history",)
        history_dump = json.dumps(history, cls=GameHistoryEncoder)

        # Dump timers to JSON
        timers = json.dumps(self.game.seconds_left_by_color)

        return GameState(
            board=self.game.board._board_state(),
            history=history_dump,
            start_time=self.game.current_turn_start_time,
            timers=timers,
            turn=self.game.turn
        )

    def step(self, action: GameAction):
        reward = 0

        if action.type == "sense":
            assert isinstance(action.action, int)

            # Apply sense action and add the result to the observable board
            for square, piece in self.game.sense(action.action):
                self.boards_dict[self.game.turn].set_piece_at(square, piece)
        
        elif action.type == "move":
            assert isinstance(action.action, chess.Move)

            # Apply move action and update the board's stack
            _, taken_move, capture_square = self.game.move(action.action)
            if taken_move is not None:
                self.observed_board.push(taken_move)
                # Push will switch the color of the board, so we reset it back
                self.observed_board.turn = self.player_color

        else:
            raise ValueError("Unsupported action type.")
