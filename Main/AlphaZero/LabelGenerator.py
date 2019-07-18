import numpy as np
from Main.Environments.Connect4 import Constants, GetPossibleActions, Utils


# Instead of recursively returning elements we just append them to an global list
def generateLabels(root):
    # validateTree(root)
    # return Utils.state2ConvState(root.state, root.currentPlayer), _createPolicyLabel(root)
    return Utils.bitBoard2ConvState(root.state), _createPolicyLabel(root)

def generateQLabels(root):
    # return Utils.state2ConvState(root.state, root.currentPlayer), _createValueLabels(root), _createPolicyLabel(root)
    return Utils.bitBoard2ConvState(root.state), _createValueLabels(root), _createPolicyLabel(root)


# Assumes that every child has been visited
def _createPolicyLabel(node):
    qValues = np.zeros(Constants.AMOUNT_OF_POSSIBLE_ACTIONS)

    for c in node.children:
        qValues[c.action] = c.visits

    qValues /= np.sum(qValues)  # Normalize
    return qValues


# Assumes that the node has been visited
def _createValueLabels(node):
    return node.score / max(1, node.visits)

# Only for debugging!
def validateTree(node):
    # possibleActions = GetPossibleActions.getPossibleActions(node.state)
    import Main.Environments.Connect4.Connect4Bitmaps as bitMaps
    possibleActions = bitMaps.getPossibleActions(node.state)
    for c in node.children:
        assert (c.action in possibleActions)
        validateTree(c)
