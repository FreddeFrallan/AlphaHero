from Main.Environments.Connect4 import Utils
import time
import numpy as np
from Main.MCTS import MainFunctions


# from numba import jit


def performSearchOnTime(game, state, thinkTime, currentPlayer, root=None):
    if (root is None):
        root = MainFunctions.createNewMCTS(state, currentPlayer)  # if root is None else root

    t1 = time.time()
    while (time.time() - t1 < thinkTime):
        simulateMCTS(game, root, root.visits + 1)

    return root


def performSearch(game, state, amountOfIterations, currentPlayer, root=None):
    if (root is None):
        root = MainFunctions.createNewMCTS(state, currentPlayer)  # if root is None else root

    while (root.visits < amountOfIterations):
        simulateMCTS(game, root, root.visits + 1)

    return root


def simulateMCTS(game, root, iteration):
    node = root
    path = []

    while (node.expanded and node.terminal == False):
        node.visits += 1
        path.append(node)
        # node = node.children[np.argmax(
        # [MainFunctions._UCB2(c, iteration, node.currentPlayer) for c in node.children])]
        node = node.children[np.argmax(MainFunctions._PUCT(node))]

    if (node.terminal):
        # Should we perhaps re-consider this ? We could in theory skip adding visits and score to terminal nodes
        # and only return the original score, for clarity and computation
        path.append(node)
        MainFunctions._backprop(path, node.score / node.visits)
        node.visits += 1
        return

    # We have not yet been to this node and should evaluate
    node.visits += 1
    node.terminal, terminalScore = game.evaluateTerminal(node.state)

    if (node.terminal == False):
        evalScore, policy = game.evaluateState(node.state, node.currentPlayer)

        # Only set policy to legal moves
        legalMoves = game.getPossibleActions(node.state, 1)
        '''
        legalMovesOld = gameOld.getPossibleActions(node.state.toNumpyState(), 1)
        for i in range(len(legalMoves)):
            assert(legalMoves[i] == legalMovesOld[i])
        '''

        if (len(legalMoves) != 7):
            legalPolicy = Utils.createNormalizedLegalPolicy(policy, legalMoves)
            node.policyEstimatedFromThisNode = legalPolicy / np.sum(legalPolicy)  # Set & Normalize
        else:
            node.policyEstimatedFromThisNode = policy

        MainFunctions._expandNode(game, node, policy)
    else:
        evalScore = terminalScore

    node.score = evalScore
    MainFunctions._backprop(path, evalScore)


def getBestActionFromNode(node, game):
    if (node.visits == 1):
        legalMoves = game.getPossibleActions(node.state, node.currentPlayer)
        return np.argmax(Utils.createNormalizedLegalPolicy(node.policyEstimatedFromThisNode, legalMoves))
    action = max(node.children, key=lambda x: x.visits).action

    return action


def getCertaintyVector(node):
    if (node.visits == 1):
        return node.policyEstimatedFromThisNode

    visits = [child.visits for child in node.children]
    return visits / np.sum(visits)
