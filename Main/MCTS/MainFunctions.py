from Main import Hyperparameters
import numpy as np


class Node:
    def __init__(self, state, action=None, currentPlayer=1, policyEstimation=1):
        self.expanded = False
        self.terminal = False
        self.action = action
        self.state = state
        self.children = []
        self.visits = 0
        self.score = 0.5
        self.currentPlayer = currentPlayer
        self.policyEstimation = policyEstimation
        self.policyEstimatedFromThisNode = None  # Only contains the valid moves


def createNewMCTS(state, currentPlayer):
    return Node(state, currentPlayer=currentPlayer)


def _expandNode(game, node, policy):
    node.expanded = True
    nextPlayer = node.currentPlayer * -1
    node.children = [Node(game.simulateAction(node.state, node.currentPlayer, a),
                          action=a, currentPlayer=nextPlayer, policyEstimation=policy[a]) for a in
                     game.getPossibleActions(node.state, node.currentPlayer)]


def _backprop(path, evalScore):
    for n in path:
        n.score += evalScore


# ***************************************************
# ****** Exploration - Exploitation formulas ******
# ***************************************************

# Uses pre computation in the form of only calculating the np.log(iteration) once per search iteration
def _UCB(node, logIterations, currentPlayer):
    exploitationScore = (node.score / (node.visits + 1))
    if (currentPlayer == -1):
        exploitationScore = 1 - exploitationScore

    explorationScore = Hyperparameters.EXPLORATION_CONSTANT * np.sqrt(logIterations / (node.visits + 1))
    return exploitationScore + explorationScore


# This is only performed in the pre-computation of all the possible UCB scores
def calculateExplorationScore(iteration, nodeVisits):
    return Hyperparameters.EXPLORATION_CONSTANT * np.sqrt(np.log(iteration) / (nodeVisits + 1))


# Optimized version that uses the pre-computed dictionary
def _UCB2(node, iteration, currentPlayer):
    exploitationScore = (node.score / (node.visits + 1))
    if (currentPlayer == -1):
        exploitationScore = 1 - exploitationScore

    explorationScore = PRE_COMPUTED_EXPLORATION_FACTOR[iteration][node.visits]
    return exploitationScore + explorationScore


# ***** Improves the speed of the UCB function with 33% *****
# Nested Dictionary for the Exploration rate of the UCB
# First lookup is the current iteration
# Second lookup is the visit count on the current node
PRE_COMPUTED_EXPLORATION_FACTOR = {}


def createPreComputationTableForUCB2():
    for i in range(Hyperparameters.MCTS_SIMULATIONS_PER_MOVE + 1):
        temp = {}
        for j in range(Hyperparameters.MCTS_SIMULATIONS_PER_MOVE + 1):
            temp[j] = calculateExplorationScore(i, j)
        PRE_COMPUTED_EXPLORATION_FACTOR[i] = temp
    print("Pre Computation of UCB-score complete...")


# We should be able to do some serious optimizations here, by caching values and other things
def _PUCT(node):
    childVisitSum = node.visits - 1
    exploration = node.policyEstimatedFromThisNode * Hyperparameters.EXPLORATION_CONSTANT * np.sqrt(childVisitSum)
    exploration /= [c.visits + 1 for c in node.children]

    qValues = np.array([c.score / max(1, c.visits) for c in node.children])
    if (node.currentPlayer != 1):
        qValues = 1 - qValues
    return qValues + exploration
