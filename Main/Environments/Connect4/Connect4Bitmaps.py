from Main import MachineSpecificSettings
from Main.Environments.Connect4 import Utils
from RootDir import ROOT_DIR
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
from PositionFile import POSITION

'''
class POSITION(Structure):
    _fields_ = [
        ('current_position', c_uint64),
        ('mask', c_uint64),
        ('moves', c_uint),
        ('score', c_float)
    ]
'''


# lib = None


def init():
    try:
        global lib
        if (MachineSpecificSettings.IS_UNIX_MACHINE):
            lib = cdll.LoadLibrary(ROOT_DIR + '/Libs/Connect4BitmapLibrary.so')
        else:
            lib = cdll.LoadLibrary(ROOT_DIR + '/Libs/Connect4BitmapLibrary.dll')
        print("BitmapLibrary loaded!")

    except:
        print("**Could not load Connect4 Library!**")

    _setTypes()


def _setTypes():
    lib.init()

    lib.getPossibleActions.argtypes = [POSITION, POINTER(c_int32)]
    lib.getPossibleActions.restype = None

    lib.boardToArray.argtypes = [POSITION, ndpointer(c_int, flags="C_CONTIGUOUS")]
    lib.boardToArray.restype = None

    lib.simulateAction.argtypes = [POSITION, c_int, POINTER(POSITION)]
    # lib.simulateAction.restype = POSITION
    lib.simulateAction.restype = None

    '''
    lib.terminalEvaluation.argtypes = [POSITION]
    lib.terminalEvaluation.restype = c_float
    '''

    lib.setBoardFromArray.argtypes = [POINTER(POSITION), ndpointer(c_int, flags="C_CONTIGUOUS"), c_int]
    lib.setBoardFromArray.restype = None

    lib.playAction.argtypes = [POINTER(POSITION), c_int]
    lib.playAction.restype = None

    lib.getKey.argtypes = [POSITION]
    lib.getKey.restype = c_uint64

    lib.getMirroredState.argtypes = [POSITION, POINTER(POSITION)]
    lib.getMirroredState.restype = None


# maybe allocate int8 instead?
def getPossibleActions(bitBoard, currentPlayerNotUsed):
    buffer = (c_int32 * 8)()
    lib.getPossibleActions(bitBoard.pos, buffer)

    possibleActions = []
    i = 0
    while buffer[i] != -1:
        if (buffer[i] == -1):
            break
        possibleActions.append(buffer[i])
        i += 1

    return possibleActions


def simulateAction(bitBoard, currentPlayerNotUsed, action):
    pos = POSITION(0, 0, 0, 1337)
    lib.simulateAction(bitBoard.pos, action, byref(pos))
    return BitBoard(pos)


def terminalEvaluation(bitBoard):
    # terminalScore = lib.terminalEvaluation(bitBoard.obj)
    terminalScore = bitBoard.pos.score
    if (terminalScore == 1337):  # Magic number representing non-terminal
        return False, 0.5
    else:
        return True, terminalScore


class BitBoard(object):

    def __init__(self, position=None, numpyState=None):

        if (position is None and numpyState is None):
            self.pos = POSITION(0, 0, 0, 1337)
        elif (numpyState is None):
            self.pos = position
        else:
            pos = POSITION(0, 0, 0, 1337)
            numMoves = len(np.where(numpyState != 0)[0])
            lib.setBoardFromArray(byref(pos), numpyState, numMoves)
            self.pos = pos

    def __hash__(self):
        return lib.getKey(self.pos)

    def __eq__(self, other):
        return lib.getKey(self.pos) == lib.getKey(other.pos)

    def toNumpyState(self):
        buffer = np.empty((6, 7), dtype=np.intc)
        lib.boardToArray(self.pos, buffer)
        return buffer

    def mirror(self):
        pos = POSITION(0, 0, 0, 1337)
        lib.getMirroredState(self.pos, byref(pos))
        return BitBoard(pos)

    def toConvState(self, currentPlayer):
        return Utils.state2ConvState(self.toNumpyState(), currentPlayer)

    # Should be immutable for eq and hash


def playRandomGame():
    import Tests.Environments.Connect4.AssertBitBoards as tests
    for i in range(10000):
        board = BitBoard()
        boardNumpy = board.toNumpyState()
        while (True):
            mirrored = tests.mirrorAndAssert(board, boardNumpy)
            print(boardNumpy)
            # terminal, terminalScore = terminalEvaluation(board)
            terminal, terminalScore = tests.terminalEvaluationAndAssert(board, boardNumpy)
            # print("Terminal: ", terminal)
            # print("TerminalScore: ", terminalScore)
            if (terminal):
                break
            # possActions = getPossibleActions(board, 1338)
            possActions = tests.possibleActionsAndAssert(board, boardNumpy)
            # print("Possible actions: ", possActions)
            # board = simulateAction(board, 1338, possActions[np.random.choice(len(possActions))])
            board = tests.simulateAndAssert(board, possActions[np.random.choice(len(possActions))], boardNumpy)
            boardNumpy = board.toNumpyState()
            board = BitBoard(numpyState=boardNumpy)
            print(board.toNumpyState())
