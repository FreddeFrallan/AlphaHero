import Main.Environments.Connect4.Connect4Bitmaps as bitMaps
from Main.Environments.Connect4 import Utils, SimulateAction, GetPossibleActions, TerminalEvaluation
import numpy as np


def simulateAndAssert(bitBoard, action, numpyState):
    player = Utils.getCurrentPlayerFromState(numpyState)
    simBoard = bitMaps.simulateAction(bitBoard, player, action)
    numpySimBoard = SimulateAction.simulateAction(numpyState, player, action)
    assert np.array_equal(simBoard.toNumpyState(), numpySimBoard)
    return simBoard


def possibleActionsAndAssert(bitBoard, numpyState):
    possActions = bitMaps.getPossibleActions(bitBoard, 1338)
    numpyPossActions = GetPossibleActions.getPossibleActions(numpyState, 1338)
    for i in range(len(possActions)):
        assert possActions[i] == numpyPossActions[i]

    return possActions


def terminalEvaluationAndAssert(bitBoard, numpyState):
    terminal, score = bitMaps.terminalEvaluation(bitBoard)
    numpyTerminal, numpyScore = TerminalEvaluation.terminalEvaluation(numpyState)
    assert terminal == numpyTerminal, str(terminal) + ", " + str(numpyTerminal)
    assert score == numpyScore, str(score) + ", " + str(numpyScore)
    return terminal, score


def mirrorAndAssert(bitBoard, boardNumpy):
    mirrored = bitBoard.mirror()
    numpyMirrored = np.fliplr(boardNumpy)
    assert np.array_equal(mirrored.toNumpyState(), numpyMirrored)
    return mirrored
