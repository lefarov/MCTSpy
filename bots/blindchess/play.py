from dataclasses import dataclass
from typing import *
from abc import abstractmethod

import chess
from reconchess.types import *
from reconchess.player import Player
from reconchess.game import Game, LocalGame, RemoteGame
from reconchess.history import GameHistory


BatchedAgent = Player
MatchResult = Tuple[Optional[Color], Optional[WinReason], GameHistory]


class BatchedAgentManager:

    @abstractmethod
    def build_agent(self) -> BatchedAgent:
        pass

    @abstractmethod
    def choose_move_batched(self,
                            agents: List[BatchedAgent],
                            move_action_lists: List[List[chess.Move]]) -> Optional[List[chess.Move]]:
        pass

    @abstractmethod
    def choose_sense_batched(self,
                             agents: List[BatchedAgent],
                             sense_actions: List[List[Square]],
                             move_action_lists: List[List[chess.Move]]) -> List[Optional[Square]]:
        pass


class DelegatingAgentManager(BatchedAgentManager):

    def __init__(self, player_ctor: Callable[[], Player]):
        self.player_ctor = player_ctor

    def build_agent(self, *args, **kwargs) -> BatchedAgent:
        return self.player_ctor(*args, **kwargs)

    def choose_move_batched(self,
                            agents: List[BatchedAgent],
                            move_action_lists: List[List[chess.Move]]) -> Optional[List[chess.Move]]:

        return [a.choose_move(moves, 666) for a, moves in zip(agents, move_action_lists)]

    def choose_sense_batched(self,
                             agents: List[BatchedAgent],
                             sense_actions: List[List[Square]],
                             move_action_lists: List[List[chess.Move]]) -> List[Optional[Square]]:

        return [a.choose_sense(senses, moves, 666)
                for a, senses, moves in zip(agents, sense_actions, move_action_lists)]


@dataclass
class MatchInProgress:
    game: LocalGame
    players: Tuple[BatchedAgent, BatchedAgent]
    move_count: int = 0

    def get_curr_player(self):
        return self.players[1 - int(self.game.turn)]


def play_local_game_batched(white_manager: BatchedAgentManager,
                            black_manager: BatchedAgentManager,
                            total_number: int = 128,
                            batch_size: int = 32,
                            move_limit: int = 100,
                            seconds_per_player: float = 900) -> List[MatchResult]:

    managers = (white_manager, black_manager)

    #  for _ in range(batch_size)
    matches_in_progress = []  # type: List[MatchInProgress]
    match_results = []  # type: List[MatchResult]

    while len(match_results) < total_number:

        # Start new matches to fill up the batch.
        while len(matches_in_progress) < min(batch_size, total_number - len(match_results)):
            # TODO: insert predefined amount of moves when creating the game
            game = LocalGame(seconds_per_player=seconds_per_player)
            white_player = white_manager.build_agent()
            black_player = black_manager.build_agent()

            white_name = white_manager.__class__.__name__
            black_name = black_manager.__class__.__name__
            game.store_players(white_name, black_name)

            white_player.handle_game_start(chess.WHITE, game.board.copy(), black_name)
            black_player.handle_game_start(chess.BLACK, game.board.copy(), white_name)

            game.start()

            matches_in_progress.append(MatchInProgress(game, (white_player, black_player)))

        # Advance all the current matches by one turn.
        play_turn_batched(matches_in_progress, managers)

        # Record the results of the games that finished.
        for i_match, match in enumerate(matches_in_progress.copy()):  # Copy so we can edit the list.
            game = match.game
            if match.move_count >= move_limit:
                game.resign()

            if game.is_over():
                game.end()
                winner_color = game.get_winner_color()
                win_reason = game.get_win_reason()
                game_history = game.get_game_history()

                for player in match.players:
                    player.handle_game_end(winner_color, win_reason, game_history)

                match_results.append((winner_color, win_reason, game_history, *match.players))

                matches_in_progress.remove(match)

    return match_results


def play_turn_batched(matches: List[MatchInProgress], managers: Tuple[BatchedAgentManager, BatchedAgentManager]):
    white_matches, black_matches = [], []
    for match in matches:
        container = white_matches if match.game.turn else black_matches
        container.append(match)

    # Handle white and black parts of the batch separately, since the agents are different.
    for manager, manager_matches in zip(managers, (white_matches, black_matches)):
        if len(manager_matches) == 0:
            continue

        sense_actions_batch, move_actions_batch = [], []
        players = []

        for match in manager_matches:
            sense_actions_batch.append(match.game.sense_actions())
            move_actions_batch.append(match.game.move_actions())
            players.append(match.get_curr_player())

            # todo Hide in the manager. Do this for all the agent methods.
            notify_opponent_move_results(match.game, match.get_curr_player())

        # --- play_sense(game, player, sense_actions, move_actions) ---
        sense_batch = manager.choose_sense_batched(players, sense_actions_batch, move_actions_batch)
        for sense, match, player in zip(sense_batch, manager_matches, players):
            sense_result = match.game.sense(sense)
            player.handle_sense_result(sense_result)

        # --- play_move(game, player, move_actions, end_turn_last=end_turn_last) ---
        move_batch = manager.choose_move_batched(players, move_actions_batch)
        for move, match, player in zip(move_batch, manager_matches, players):
            requested_move, taken_move, opt_enemy_capture_square = match.game.move(move)

            player.handle_move_result(requested_move, taken_move,
                                      opt_enemy_capture_square is not None, opt_enemy_capture_square)

            # The original code used the 'end_last_turn' flag to decide when to call this. Removed it.
            match.game.end_turn()

            match.move_count += 1


def notify_opponent_move_results(game: Game, player: Player):
    opt_capture_square = game.opponent_move_results()
    player.handle_opponent_move_result(opt_capture_square is not None, opt_capture_square)

