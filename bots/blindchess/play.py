from typing import *
from abc import abstractmethod

import chess
from reconchess.types import *
from reconchess.player import Player
from reconchess.game import Game, LocalGame, RemoteGame
from reconchess.history import GameHistory


class PlayerBatched:

    @abstractmethod
    def get_subplayer(self, game_index: int) -> Player:
        pass

    @abstractmethod
    def choose_move_batched(self, subplayer_indices: List[int], move_actions: List[List[chess.Move]]) -> Optional[List[chess.Move]]:
        pass

    @abstractmethod
    def choose_sense_batched(self, subplayer_indices: List[int], sense_actions: List[List[Square]], move_actions: List[List[chess.Move]]) -> \
            List[Optional[Square]]:
        pass


class PlayerBatchedWrapper(PlayerBatched):

    def __init__(self, batch_size: int, player_ctor: Callable[[], Player]):
        self.subplayers = [player_ctor() for _ in range(batch_size)]

    def get_subplayer(self, game_index: int) -> Player:
        return self.subplayers[game_index]

    def choose_move_batched(self, subplayer_indices: List[int],
                            move_actions: List[List[chess.Move]]) -> Optional[List[chess.Move]]:
        return [self.subplayers[i_subplayer].choose_move(move_actions[i], 666)
                for i, i_subplayer in enumerate(subplayer_indices)]

    def choose_sense_batched(self, subplayer_indices: List[int],
                             sense_actions: List[List[Square]], move_actions: List[List[chess.Move]]) -> \
            List[Optional[Square]]:
        return [self.subplayers[i_subplayer].choose_sense(sense_actions[i], move_actions[i], 666)
                for i, i_subplayer in enumerate(subplayer_indices)]


def play_local_game_batched(white_player: PlayerBatched, black_player: PlayerBatched, game_number: int = 32,
                            move_limit: int = 100, seconds_per_player: float = 900) -> List[Tuple[Optional[Color], Optional[WinReason], GameHistory]]:

    players = [black_player, white_player]

    games = [LocalGame(seconds_per_player=seconds_per_player) for _ in range(game_number)]

    for i, game in enumerate(games):
        white_name = white_player.__class__.__name__
        black_name = black_player.__class__.__name__
        game.store_players(white_name, black_name)

        white_player.get_subplayer(i).handle_game_start(chess.WHITE, game.board.copy(), black_name)
        black_player.get_subplayer(i).handle_game_start(chess.BLACK, game.board.copy(), white_name)
        game.start()

    move_count = 0
    while True:
        games_not_over_indices = [i for i, game in enumerate(games) if not game.is_over()]
        assert all(games[i].turn == games[games_not_over_indices[0]].turn for i in games_not_over_indices)

        if not games_not_over_indices:
            break
        if move_count > move_limit:
            for game in games:
                if not game.is_over():
                    game.resign()
            break

        play_turn_batched(games, games_not_over_indices, players[games[0].turn], end_turn_last=True)
        move_count += 1

    results = []
    for i, game in enumerate(games):
        game.end()
        winner_color = game.get_winner_color()
        win_reason = game.get_win_reason()
        game_history = game.get_game_history()

        white_player.get_subplayer(i).handle_game_end(winner_color, win_reason, game_history)
        black_player.get_subplayer(i).handle_game_end(winner_color, win_reason, game_history)

        results.append((winner_color, win_reason, game_history))

    return results


def play_turn_batched(games: List[Game], games_not_over_indices: List[int], player: PlayerBatched, end_turn_last=False):

    sense_actions_batch, move_actions_batch = [], []
    for i_game in games_not_over_indices:
        game = games[i_game]
        sense_actions_batch.append(game.sense_actions())
        move_actions_batch.append(game.move_actions())
        notify_opponent_move_results(game, player.get_subplayer(i_game))

    # --- play_sense(game, player, sense_actions, move_actions) ---
    sense_batch = player.choose_sense_batched(games_not_over_indices, sense_actions_batch, move_actions_batch)
    for sense, i_game in zip(sense_batch, games_not_over_indices):
        if games[i_game].is_over():
            continue

        sense_result = games[i_game].sense(sense)
        player.get_subplayer(i_game).handle_sense_result(sense_result)

    # --- play_move(game, player, move_actions, end_turn_last=end_turn_last) ---
    move_batch = player.choose_move_batched(games_not_over_indices, move_actions_batch)
    for move, i_game in zip(move_batch, games_not_over_indices):
        game = games[i_game]
        if game.is_over():
            continue

        requested_move, taken_move, opt_enemy_capture_square = game.move(move)

        if not end_turn_last:
            game.end_turn()

        player.get_subplayer(i_game).handle_move_result(requested_move, taken_move,
                                                        opt_enemy_capture_square is not None, opt_enemy_capture_square)

        if end_turn_last:
            game.end_turn()


def notify_opponent_move_results(game: Game, player: Player):
    opt_capture_square = game.opponent_move_results()
    player.handle_opponent_move_result(opt_capture_square is not None, opt_capture_square)

