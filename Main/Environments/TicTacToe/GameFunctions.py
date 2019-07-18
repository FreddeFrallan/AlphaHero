from Main.Environments.TicTacToe import Utils
import numpy as np


def simulateAction(state, action, player):
    copyBoard = np.copy(state)
    y, x = Utils.actionToCoord(action)
    copyBoard[y][x] = player
    return copyBoard


def getPossibleActions(state):
    emptyCells = np.where(state == 0)
    return [Utils.coordToAction(emptyCells[0][i], emptyCells[1][i])
            for i in range(len(emptyCells[0]))]


def evaluateBoard(state):
    sums = []
    sums.extend(np.sum(state, axis=-1))  # Sums Row
    sums.extend(np.sum(state, axis=0))  # Sums Cols
    sums.append(np.sum(np.diag(state)))  # Sums Diag 0 - 9
    sums.append(np.sum(np.diag(np.fliplr(state))))  # Sums Diag 6 - 2

    if (np.max(sums) == 3):
        return 1
    if (np.min(sums) == -3):
        return 0
    return 0.5
