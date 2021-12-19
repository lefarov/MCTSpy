import os

import torch
import tempfile
import wandb

from recon_tictac import Player, plotting_mode
from recon_tictac.interfaces.reconchess.play import play_local_game
from recon_tictac.interfaces.reconchess.game import LocalGame

from bots.tictac.agent import RandomAgent, QAgent
from bots.tictac.net import TicTacQNet


def main():
    net_memory_length = 1
    net_hidden_number = 512

    with tempfile.TemporaryDirectory() as plotting_dir:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        q_net = TicTacQNet(net_memory_length, net_hidden_number).to(device)

        # TODO: load directly from WANDB
        q_net.load_state_dict(torch.load(os.path.expanduser("~/Downloads/model_1000.pt")))
        
        q_agent_eval = QAgent(q_net)
        random_agent = RandomAgent()

        # Enter the plotting context
        with plotting_mode():
            # Set plotting directories for player and opponent (they'll be created if non-existant)
            q_agent_eval.plot_directory = os.path.join(plotting_dir, f"player")
            random_agent.plot_directory = os.path.join(plotting_dir, f"opponent")

            game = LocalGame()
            # Let Crosses make the first move
            game.turn = Player.Cross

            play_local_game(q_agent_eval, random_agent, game)
            pass


if __name__ == '__main__':
    main()
