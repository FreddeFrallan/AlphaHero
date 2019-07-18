import numpy as np
from unittest import TestCase

from Main.Environments.Connect4.EvaluateState import *
from Tests.Environments.Connect4.testCasesRawEvaluate import TEST_CASES


class TestRawEvaluateState(TestCase):
    def test_rawEvaluateState(self):
        initRawEvaluateState()
        for t in TEST_CASES:
            state, expectedScore = t
            evalScore = rawEvaluateState(np.array(state))
            if(expectedScore != evalScore):
                print(np.array(state))
                self.fail("Expected: {}, Got: {}".format(expectedScore, evalScore))


