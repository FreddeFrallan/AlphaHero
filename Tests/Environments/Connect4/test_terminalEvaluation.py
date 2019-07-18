from unittest import TestCase

import numpy as np

from Main.Environments.Connect4.EvaluateState import initRawEvaluateState, rawEvaluateState
from Tests.Environments.Connect4.testCasesTerminalEvaluation import TEST_CASES
from Main.Environments.Connect4.TerminalEvaluation import terminalEvaluation


class TestTerminalEvaluation(TestCase):
    def test_terminalEvaluation(self):
        initRawEvaluateState()
        for t in TEST_CASES:
            state, expectedEval = t
            terminalEval, _  = terminalEvaluation(np.array(state))
            if (expectedEval != terminalEval):
                print(np.array(state))
                self.fail("Expected: {}, Got: {}".format(expectedEval, terminalEval))
