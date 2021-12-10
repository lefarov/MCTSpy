from copy import deepcopy

import chess
from reconchess.types import *
from reconchess.player import Player
from reconchess.game import Game
from reconchess.history import GameHistory

import recon_tictac
import recon_tictac.enums
from recon_tictac.interfaces.reconchess.game import LocalGame


def play_local_game(
        nought_player: Player,
        cross_player: Player,
        game: LocalGame = None,
        seconds_per_player: float = 900
) -> Tuple[Optional[Color], Optional[WinReason], GameHistory]:
    """
    Plays a game between the two players passed in. Uses :class:`LocalGame` to run the game, and just calls
    :func:`play_turn` until the game is over: ::

        while not game.is_over():
            play_turn(game, player)

    :param nought_player: The Nought :class:`Player`.
    :param cross_player: The Cross :class:`Player`.
    :param game: The :class:`LocalGame` object to use.
    :param seconds_per_player: The time each player has to play. Only used if `game` is not passed in.
    :return: The results of the game, also passed to each player via :meth:`Player.handle_game_end`.
    """
    players = [nought_player, cross_player]

    if game is None:
        game = LocalGame(seconds_per_player=seconds_per_player)

    nought_name = nought_player.__class__.__name__
    cross_name = cross_player.__class__.__name__
    game.store_players(nought_name, cross_name)

    nought_player.handle_game_start(bool(recon_tictac.enums.Player.Nought), game.board.copy(), nought_name)
    cross_player.handle_game_start(bool(recon_tictac.enums.Player.Cross), game.board.copy(), cross_name)
    game.start()

    while not game.is_over():
        play_turn(game, players[game.turn], end_turn_last=True)

    game.end()
    winner_color = game.get_winner_color()
    win_reason = game.get_win_reason()
    game_history = game.get_game_history()

    nought_player.handle_game_end(winner_color, win_reason, game_history)
    cross_player.handle_game_end(winner_color, win_reason, game_history)

    return winner_color, win_reason, game_history


def play_turn(game: Game, player: Player, end_turn_last=False):
    """
    Coordinates playing a turn for `player` in `game`. Does the following sequentially:

    #. :func:`notify_opponent_move_results`
    #. :func:`play_sense`
    #. :func:`play_move`

    See :func:`play_move` for more info on `end_turn_last`.

    :param game: The :class:`Game` that `player` is playing in.
    :param player: The :class:`Player` whose turn it is.
    :param end_turn_last: Flag indicating whether to call :meth:`Game.end_turn` before or after :meth:`Player.handle_move_result`
    """
    sense_actions = game.sense_actions()
    move_actions = game.move_actions()

    notify_opponent_move_results(game, player)

    play_sense(game, player, sense_actions, move_actions)

    play_move(game, player, move_actions, end_turn_last=end_turn_last)


def notify_opponent_move_results(game: Game, player: Player):
    """
    Passes the opponents move results to the player. Does the following sequentially:

    #. Get the results of the opponents move using :meth:`Game.opponent_move_results`.
    #. Give the results to the player using :meth:`Player.handle_opponent_move_result`.

    :param game: The :class:`Game` that `player` is playing in.
    :param player: The :class:`Player` whose turn it is.
    """
    opt_capture_square = game.opponent_move_results()
    player.handle_opponent_move_result(opt_capture_square is not None, opt_capture_square)


def play_sense(game: Game, player: Player, sense_actions: List[Square], move_actions: List[chess.Move]):
    """
    Runs the sense phase for `player` in `game`. Does the following sequentially:

    #. Get the sensing action using :meth:`Player.choose_sense`.
    #. Apply the sense action using :meth:`Game.sense`.
    #. Give the result of the sense action to player using :meth:`Player.handle_sense_result`.

    :param game: The :class:`Game` that `player` is playing in.
    :param player: The :class:`Player` whose turn it is.
    :param sense_actions: The possible sense actions for `player`.
    :param move_actions: The possible move actions for `player`.
    """
    # TODO ASAP DEBUG ONLY
    player.board._board = deepcopy(game.board._board)

    sense = player.choose_sense(sense_actions, move_actions, game.get_seconds_left())
    sense_result = game.sense(sense)
    player.handle_sense_result(sense_result)


def play_move(game: Game, player: Player, move_actions: List[chess.Move], end_turn_last=False):
    """
    Runs the move phase for `player` in `game`. Does the following sequentially:

    #. Get the moving action using :meth:`Player.choose_move`.
    #. Apply the moving action using :meth:`Game.move`.
    #. Ends the current player's turn using :meth:`Game.end_turn`.
    #. Give the result of the moveaction to player using :meth:`Player.handle_move_result`.

    If `end_turn_last` is True, then :meth:`Game.end_turn` is called last instead of before
    :meth:`Player.handle_move_result`.

    :param game: The :class:`Game` that `player` is playing in.
    :param player: The :class:`Player` whose turn it is.
    :param move_actions: The possible move actions for `player`.
    :param end_turn_last: Flag indicating whether to call :meth:`Game.end_turn` before or after :meth:`Player.handle_move_result`
    """
    move = player.choose_move(move_actions, game.get_seconds_left())
    requested_move, taken_move, opt_enemy_capture_square = game.move(move)

    if not end_turn_last:
        game.end_turn()

    player.handle_move_result(requested_move, taken_move,
                              opt_enemy_capture_square is not None, opt_enemy_capture_square)

    # TODO ASAP DEBUG ONLY
    player.board._board = deepcopy(game.board._board)

    if end_turn_last:
        game.end_turn()
