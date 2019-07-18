import numpy as np


def getPossibleActions(state, player=1):
    return np.where(state[0] == 0)[0]
