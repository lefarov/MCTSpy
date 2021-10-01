import reconchess

from agents.blind_chess import TestQNet, QAgent


def main():

    narx_memory_length = 12
    n_hidden = 256

    q_nets = [
        TestQNet(narx_memory_length, n_hidden),
        TestQNet(narx_memory_length, n_hidden),
    ]

    agents = [QAgent(net, None, narx_memory_length) for net in q_nets]

    winner_color, win_reason, game_history = reconchess.play_local_game(agents[0], agents[1])



if __name__ == '__main__':
    main()
