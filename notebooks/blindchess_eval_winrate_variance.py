import random

import matplotlib.pyplot as plt
import chess
from reconchess import WinReason

from bots.blindchess.play import DelegatingAgentManager
from bots.blindchess.play import play_local_game_batched
from bots.blindchess.agent import RandomBot
from bots.blindchess.utilities import capture_proxy_reward, move_proxy_reward, sense_proxy_reward


def random_agent_factory(*args, **kwargs):
    return RandomBot(
        capture_proxy_reward,
        move_proxy_reward,
        sense_proxy_reward,
        *args,
        **kwargs
    )


batch_size = 128
sample_number = 1024
sample_sizes = [2 ** x for x in range(5, 10)]


random_agent_manager = DelegatingAgentManager(random_agent_factory)

data = []
for sample_size in sample_sizes:
    print(f"Testing sample size {sample_size}...")
    game_results = play_local_game_batched(
        # train_agent_manager,
        random_agent_manager,
        random_agent_manager,
        total_number=sample_size * 4,
        batch_size=batch_size
    )

    finished_games = [r for r in game_results if r[1] == WinReason.KING_CAPTURE]
    wins = [int(r[0] == chess.WHITE) for r in finished_games]

    samples = [sum(random.sample(wins, sample_size)) / sample_size for _ in range(sample_number)]
    data.append(samples)

fig, ax = plt.subplots()
ax.boxplot(data)
ax.set_xticklabels(list(map(str, sample_sizes)))
ax.set_ylim([-0.2, 1.2])

plt.show()
