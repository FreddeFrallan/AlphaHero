import numpy as np


def actionToCoord(action):
    y = int(action / 3)
    x = action % 3
    return y, x


def coordToAction(y, x):
    return y * 3 + x


def createEmptyState():
    return np.zeros((3, 3))
