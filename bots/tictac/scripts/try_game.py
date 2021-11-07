from reconchess import play_local_game

from bots.blindchess.play import play_local_game_batched
from bots.tictac.agent import RandomAgent
from bots.tictac.game import TicTacToe, Player, WinReason


def main():
    game = TicTacToe()

    game.start()

    game.move(0)
    game.end_turn()
    game.move(1)
    game.end_turn()
    game.move(4)
    game.end_turn()
    game.move(5)
    game.end_turn()
    game.move(8)
    game.end_turn()

    print(game.board)

    assert game.is_over()
    assert game.get_winner_color() == Player.Cross
    assert game.get_win_reason() == WinReason.MatchThree

    game.end()

    # ==========

    game = TicTacToe()
    game.start()

    game.move(0)
    game.end_turn()
    game.move(1)
    game.end_turn()
    game.move(2)
    game.end_turn()
    game.move(5)
    game.end_turn()
    game.move(3)
    game.end_turn()
    game.move(6)
    game.end_turn()
    game.move(4)
    game.end_turn()
    game.move(8)
    game.end_turn()
    game.move(7)

    print(game.board)

    assert game.is_over()
    assert game.get_winner_color() is None
    assert game.get_win_reason() == WinReason.Draw

    # ========

    game = TicTacToe()

    agents = [RandomAgent(), RandomAgent()]

    winner_color, win_reason, _ = play_local_game(agents[0], agents[1], game)

    pass


if __name__ == '__main__':
    main()
