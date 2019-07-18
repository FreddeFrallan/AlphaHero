import Main.Environments.Connect4.Constants as Consts
import numpy as np


def simulateAction(grid, player, action):
    deepCopy = np.copy(grid)
    playAction(deepCopy, player, action)
    return deepCopy


# Can this be done in numpy instead of manual loop?
# Could possibly be optimized with np.argmax hack
def playAction(state, player, action):
    for y in range(Consts.HEIGHT):
        if (state[y][action] != 0):
            y -= 1
            break

    state[y][action] = player
