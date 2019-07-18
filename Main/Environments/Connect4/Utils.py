import Main.Environments.Connect4.Constants as Consts
import numpy as np

from Main import Hyperparameters


def convertAlbotState2Numpy(state):
    '''
    arr = np.zeros((6, 7), dtype=np.int)
    for y in range(6):
        for x in range(7):
            element = state.grid[y][x]
            arr[y][x] = element

    return arr
    '''
    return np.array(state.grid)


def createStartBoard():
    return np.zeros(Consts.BOARD_SHAPE)


# Creates a 5 slice one-hot representation
# 0 - Opponent
# 1 - Empty Cells
# 2 - Player
# 3 - If player turn
# 4 - If Opponent turn
def state2ConvState(state, currentPlayer):
    if (Hyperparameters.USE_CONMPRESSED_BOARD_REPRESENTATION):
        return compressedState2ConvState(state, currentPlayer)

    convState = np.zeros((Consts.HEIGHT, Consts.WIDTH, 5))
    convSlides = [np.where(state == i) for i in [-1, 0, 1]]

    # Can probably be optimized with some clever numpy indexing
    # Perhaps convState[convSlides[i][0]][convSlides[i][1]] = 1
    for i in range(len(convSlides)):
        for j in range(len(convSlides[i][0])):
            y = convSlides[i][0][j]
            x = convSlides[i][1][j]
            convState[y][x][i] = 1

    # Add CurrentPlayer Slice
    if (currentPlayer == 1):
        convState[:, :, 3] = 1
    else:
        convState[:, :, 4] = 1

    return convState


# Creates a 3 slice one-hot representation
# 0 - Opponent
# 1 - Player
# 2 - If player turn
def compressedState2ConvState(state, currentPlayer):
    convState = np.zeros((Consts.HEIGHT, Consts.WIDTH, 3))
    convSlides = [np.where(state == i) for i in [-1, 1]]

    # Can probably be optimized with some clever numpy indexing
    # Perhaps convState[convSlides[i][0]][convSlides[i][1]] = 1
    for i in range(len(convSlides)):
        for j in range(len(convSlides[i][0])):
            y = convSlides[i][0][j]
            x = convSlides[i][1][j]
            convState[y][x][i] = 1

    # Add CurrentPlayer Slice
    if (currentPlayer == 1):
        convState[:, :, 1] = 1
    return convState


def bitBoard2ConvState(bitBoard):
    numpyState = bitBoard.toNumpyState()
    return state2ConvState(numpyState, getCurrentPlayerFromState(numpyState))


def createMirroredStateAndPolicy(states, polices):
    # Mirror states & Polices
    s = np.array(states)
    p = np.array(polices)
    mirrorP = np.zeros(p.shape)
    mirrorS = np.zeros(s.shape)

    for x in range(3):  # Loop until mid Col
        mirrorS[:, :, x] = s[:, :, Consts.WIDTH - 1 - x]
        mirrorS[:, :, Consts.WIDTH - 1 - x] = s[:, :, x]

        mirrorP[:, x] = p[:, Consts.WIDTH - 1 - x]
        mirrorP[:, Consts.WIDTH - 1 - x] = p[:, x]

    mirrorS[:, :, 3] = s[:, :, 3]
    mirrorP[:, 3] = p[:, 3]

    # This can perhaps be compressed into one Numpy operations instead of iteration?
    return [mirrorS[i] for i in range(s.shape[0])], [mirrorP[i] for i in range(p.shape[0])]


def createMirrorState(state):
    mirrorS = np.zeros(state.shape)

    for x in range(3):  # Loop until mid Col
        mirrorS[:, x] = state[:, Consts.WIDTH - 1 - x]
        mirrorS[:, Consts.WIDTH - 1 - x] = state[:, x]

    # Add the mid Col
    mirrorS[:, 3] = state[:, 3]

    return mirrorS


# Is supposed to return a vector with the same length as legalMoves, so that it can easily be broadcasted later
def createNormalizedLegalPolicy(policy, legalMoves):
    legalLen = len(legalMoves)
    legalPolicy = np.zeros(legalLen)
    legalOnes = np.ones(legalLen)
    for i in range(legalLen):
        legalPolicy[i] = policy[legalMoves[i]]

    legalSum = np.sum(legalPolicy)
    if (legalSum == 0):
        return legalOnes / legalLen
    else:
        return legalPolicy / legalSum  # return & Normalize


def compareState(s1, s2):
    for y in range(s1.shape[0]):
        for x in range(s1.shape[1]):
            if (s1[y][x] != s2[y][x]):
                return False
    return True


def getCurrentPlayerFromState(state):
    if (len(np.where(state == 0)[0]) % 2 == 0):
        return 1
    else:
        return -1
