import Main.Environments.Connect4.Constants as Const
from Main.Environments.Connect4 import Utils
import numpy as np

POSSIBLE_WIN_STATES = []
POSSIBLE_SEQS_OF_4 = []

ONE_IN_ROW_SCORE = 1
TWO_IN_ROW_SCORE = 10
THREE_IN_ROW_SCORE = 50

MAX_SCORE_APPROX = 400  # 282.5 max on 100k, 271.5 max on 1M
PREDICT_MODEL = None


def setModel(model):
    global PREDICT_MODEL
    PREDICT_MODEL = model


def modelPredict(state, currentPlayer):
    mirrored = np.random.random() > 0.5
    if (mirrored):
        state = Utils.createMirrorState(state)

    # state = state.toNumpyState()
    convState = Utils.state2ConvState(state, currentPlayer)
    evaluation, policy = PREDICT_MODEL.predict(np.array([convState]))
    evaluation = evaluation[0][0]
    policy = policy[0]

    if (mirrored):
        return evaluation, policy[::-1]
    return evaluation, policy


def modelPredictBitBoard(state, currentPlayer):
    return modelPredict(state.toNumpyState(), currentPlayer)


def initRawEvaluateState():
    rowSeqs = []
    for y in range(Const.HEIGHT):
        rowSeqs.append([(y, x) for x in range(Const.WIDTH)])

    colSeqs = []
    for x in range(Const.WIDTH):
        colSeqs.append([(y, x) for y in range(Const.HEIGHT)])

    diagSeqs = []
    for X in range(4):
        for delta in [(1, 0), (-1, Const.HEIGHT - 1)]:  # arg0: deltaY  arg1:startY
            tempDiag = []
            y = delta[1]
            x = X
            while (legalPos(y, x)):
                tempDiag.append((y, x))
                x += 1
                y += delta[0]
            diagSeqs.append(tempDiag)
    # Hardcode missed diags
    diagSeqs.append([(3, 0), (2, 1), (1, 2), (0, 3)])
    diagSeqs.append([(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)])
    diagSeqs.append([(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)])
    diagSeqs.append([(2, 0), (3, 1), (4, 2), (5, 3)])

    global POSSIBLE_WIN_STATES
    POSSIBLE_WIN_STATES += rowSeqs
    POSSIBLE_WIN_STATES += colSeqs
    POSSIBLE_WIN_STATES += diagSeqs
    # debugSeq("Rows", rowSeqs)
    # debugSeq("Cols", colSeqs)
    # debugSeq("Diags", diagSeqs)

    global POSSIBLE_SEQS_OF_4
    for seq in POSSIBLE_WIN_STATES:
        POSSIBLE_SEQS_OF_4 += _generateSequencesOf4(seq)


def legalPos(y, x):
    return 0 <= y < Const.HEIGHT and 0 <= x < Const.WIDTH


def debugSeq(name, seq):
    print(name)
    for s in seq:
        print(s)


def rawEvaluateState(state):
    global POSSIBLE_WIN_STATES

    for seq in POSSIBLE_WIN_STATES:
        currentPlayer = 0
        sequenceCounter = 0
        for coord in seq:
            y, x = coord
            if (state[y][x] == currentPlayer):
                sequenceCounter += 1
            else:
                currentPlayer = state[y][x]
                sequenceCounter = 1

            if (currentPlayer != 0 and sequenceCounter == 4):
                return 1 if currentPlayer == 1 else 0

    return 0.5


def generateHeuristicLabels(states):
    labels = [heuristicEvaluateState(s) for s in states]
    maxElement = np.max(np.abs(labels))
    print(maxElement)
    labels /= maxElement
    return labels


def heuristicEvaluateState(state):
    score = 0
    for seq in POSSIBLE_SEQS_OF_4:
        score += _calcSeqOf4Score(seq, state)

    # normalize to [0, 1] where 0 is enemy win, and 1 player win
    score /= (MAX_SCORE_APPROX * 2)  # now [-0.5, 0.5]
    score += 0.5  # mean = 0.5

    return score


def _generateSequencesOf4(seq):
    sequences = []
    length = len(seq)
    for i in range(length - 3):  # [0, 6-3] = [0, 3]
        sequences.append(seq[i:i + 4])  # i, i+1, i+2, i+3

    return sequences


def _calcSeqOf4Score(seq, state):
    intervalPlayer = 0
    amount = 0

    for i in range(4):
        y, x = seq[i]
        if (state[y][x] == 0):
            continue

        if (intervalPlayer == 0):
            intervalPlayer = state[y][x]
            amount += 1
        elif (intervalPlayer == state[y][x]):
            amount += 1
        else:  # two players have a piece in this seq of 4
            return 0

    if (amount == 0):
        return 0
    elif (amount == 1):
        score = ONE_IN_ROW_SCORE
    elif (amount == 2):
        score = TWO_IN_ROW_SCORE
    else:
        score = THREE_IN_ROW_SCORE

    return score * intervalPlayer
