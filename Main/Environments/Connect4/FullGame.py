import Main.Environments.Connect4.Constants as Consts
from Main.Environments.Connect4 import TerminalEvaluation, SimulateAction, GetPossibleActions, Utils
from Main.Training import SelfPlayAgent
from Main.Training.Connect4 import MemoryBuffers
import numpy as np

CURRENT_ROUND = 0  # Used for conveniently setting the policy sensitivity in LabelGenerator


def dealOutRewards(p1, p2, finalScore):
    p1.getReward(finalScore)
    p2.getReward(finalScore)


def displayGame(board, currentPlayer):
    print(board) if currentPlayer == 0 else print(board * -1)


# Plays a complete game, assuming that only valid moves are made
# Board is always inverted & passed so that the player thinks it's playing as nr.1
# ****OBS**** assumes that BoardEvaluation has been done **********
def playGame(player1, player2, rounds=1, printGame=False):
    global CURRENT_ROUND  # Used for conveniently setting the policy sensitivity

    players = [player1, player2]
    playerSigns = [1, -1]
    finalScores = []
    for r in range(rounds):
        board = Utils.createStartBoard()
        currentPlayer = 0
        currentRound = 0

        terminal, finalScore = TerminalEvaluation.terminalEvaluation(board)
        while (terminal == False):
            move = players[currentPlayer].makeMove(board)
            assert (move in GetPossibleActions.getPossibleActions(board))
            board = SimulateAction.simulateAction(board, playerSigns[currentPlayer], move)

            if (SelfPlayAgent.CURRENT_MODEL is not None):
                assert (np.array_equal(board, SelfPlayAgent.CURRENT_ROOT.state))

            if (printGame):
                print(board)
            currentPlayer = (currentPlayer + 1) % 2  # Increment to the next player

            terminal, finalScore = TerminalEvaluation.terminalEvaluation(board)
            currentRound += 1
            CURRENT_ROUND = currentRound

        dealOutRewards(player1, player2, finalScore)
        finalScores.append(finalScore)

        # For statistics
        lastPlayer = players[(currentPlayer + 1) % 2]
        # MemoryBuffers.addPreGameOverPolicyPredictionToBuffer(lastPlayer.lastPolicyPrediction)
        # MemoryBuffers.addPreGameOverPostPolicyToBuffer(lastPlayer.lastPolicy)
        # MemoryBuffers.addToRewardBuffer(finalScore)
        # MemoryBuffers.addGameLengthToBuffer(currentRound)

    return finalScores

