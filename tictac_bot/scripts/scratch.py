import random

from tictac_bot.simulator import *


game = TicTac()


for i in range(1000):
    state, _ = game.get_initial_state()
    while not game.state_is_terminal(state):
        actions = game.enumerate_actions(state)
        nextAction = random.choice(tuple(actions))
        print(f"Next action player #{state.nextAgentId}: {nextAction}")
        state, obs, reward, _ = game.step(state, nextAction)
        print(f"Reward: {reward} Obs: {obs}")
        print(state)

    print(f"Winner: {state.winnerId}")
