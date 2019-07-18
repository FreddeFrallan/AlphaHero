from Main.Environments.Connect4 import Constants, Utils
from Tests.Environments.Connect4 import testCasesRawEvaluate
from unittest import TestCase
import numpy as np



class TestCreateMirroredStateAndPolicy(TestCase):
    def testMirrorState(self):
        AMOUNT_OF_TESTS_PER_CASE = 10
        for case in testCasesRawEvaluate.TEST_CASES:
            board = np.array(case[0])
            for p in [-1, 1]:
                convState = Utils.state2ConvState(board, p)
                convStates = [convState for i in range(AMOUNT_OF_TESTS_PER_CASE)]
                randomPolices = [np.random.random(7) for i in range(AMOUNT_OF_TESTS_PER_CASE)]

                mirrorStates, mirrorPolices = Utils.createMirroredStateAndPolicy(convStates, randomPolices)
                reMirrorStates, reMirrorPolices = Utils.createMirroredStateAndPolicy(mirrorStates, mirrorPolices)

                for i in range(len(randomPolices)):
                    assert np.array_equal(randomPolices[i], reMirrorPolices[i])

                for m in reMirrorStates:
                    assert np.array_equal(convState, m)

