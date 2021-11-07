from reconchess import play_local_game

from bots.blindchess.play import play_local_game_batched
from bots.tictac.agent import RandomAgent
from bots.tictac.game import TicTacToe, Player, WinReason

def main():

    for i_episode in range(10):

        for i_game in range(10):
            play_local_game()


if __name__ == '__main__':
    main()
