from Main.Environments.Connect4 import Constants, Utils
from Tests.Environments.Connect4 import testCasesRawEvaluate
from unittest import TestCase
import numpy as np


def convState2NormalState(convState):
    normState = np.zeros((Constants.HEIGHT, Constants.WIDTH))

    # Re-create Board
    setCoordsToValue(normState, np.where(convState[:, :, 0] > 0), -1)  # P2
    setCoordsToValue(normState, np.where(convState[:, :, 1] > 0), 0)  # 0
    setCoordsToValue(normState, np.where(convState[:, :, 2] > 0), 1)  # P1

    print(convState[:, :, 0])
    print(np.where(convState[:, :, 0] > 0))
    print(convState[:, :, 1])
    print(convState[:, :, 2])
    print(normState)

    p1Turn = convState[:, :, 3]
    p2Turn = convState[:, :, 4]
    # Check current Player, if 0, somethings wrong
    currentPlayer = 0
    if (np.any(p1Turn > 0) and np.any(p2Turn > 0)):
        currentPlayer = 0
    elif (np.any(p1Turn > 0)):
        currentPlayer = 1
    elif (np.any(p2Turn > 0)):
        currentPlayer = -1

    return normState, currentPlayer


def setCoordsToValue(state, coords, value):
    for i in range(len(coords[0])):
        y, x = coords[0][i], coords[1][i]
        state[y][x] = value


class TestInvertState(TestCase):
    def testState2ConvState(self):
        for case in testCasesRawEvaluate.TEST_CASES:
            board = np.array(case[0])
            for p in [-1, 1]:
                postBoard, player = convState2NormalState(Utils.state2ConvState(board, p))
                assert (p == player)
                assert (np.array_equal(board, postBoard))
