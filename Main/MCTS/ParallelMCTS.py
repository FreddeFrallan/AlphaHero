import numpy as np
from Main.Environments.Connect4 import Utils
from Main.MCTS import MainFunctions as Funcs


# Since we're using PUCT we make no use of the current iteration
def simulateMCTS(game, root, iteration):
    node = root
    path = []

    while (node.expanded and node.terminal == False):
        node.visits += 1
        path.append(node)
        node = node.children[np.argmax(Funcs._PUCT(node))]

    if (node.terminal):
        path.append(node)
        Funcs._backprop(path, node.score / node.visits)
        node.visits += 1
        return None, []  # We can perform another Tree-search before eval

    # We have not yet been to this node and should evaluate
    node.visits += 1
    node.terminal, node.score = game.evaluateTerminal(node.state)
    if (node.terminal):
        Funcs._backprop(path, node.score)
        return None, []  # We can perform another Tree-search before eval

    return node, path


def simulatePostEvaluate(game, node, path, evalScore, policy):
    Funcs._expandNode(game, node, policy)

    # Only extract policy for legal moves, since it comes from a softmax layer there should be no risk of the legal
    #  policy summing to zero.
    legalMoves = game.getPossibleActions(node.state, 1338)
    if (len(legalMoves) != 7):
        legalPolicy = Utils.createNormalizedLegalPolicy(policy, legalMoves)
        node.policyEstimatedFromThisNode = legalPolicy / np.sum(legalPolicy)  # Set & Normalize
    else:
        node.policyEstimatedFromThisNode = policy

    node.score = evalScore
    Funcs._backprop(path, evalScore)
