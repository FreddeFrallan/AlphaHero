from unittest import TestCase
import Tests.Training.Connect4.testCasesLabelGenerator as TestCases
import Main.AlphaZero.LabelGenerator as LabelGenerator
import numpy as np

class TestGenerateLabels(TestCase):

    def test_terminalEvaluation(self):
        for c in TestCases.CASES:
            root, stateInputs, valueLabels, policyLabels = c
            states, values, polices = LabelGenerator.generateLabels(root, 0)
            print(states)
            print(values)
            print(polices)

            #Check Amounts
            if(len(stateInputs) != len(states)):
                self.fail("Expected {} states, Got {}".format(len(stateInputs), len(states)))
            if (len(valueLabels) != len(values)):
                self.fail("Expected {} values, Got {}".format(len(valueLabels), len(values)))
            if (len(policyLabels) != len(polices)):
                self.fail("Expected {} polices, Got {}".format(len(policyLabels), len(polices)))

            # Check states
            for i in range(len(states)):
                if (stateInputs[i] != states[i]):
                    self.fail("At Index {} Expected state: {}, Got: {}".format(
                        i, stateInputs[i], states[i]))

            # Check Values
            for i in range(len(values)):
                if (valueLabels[i] != values[i]):
                    self.fail("At Index {} Expected value: {}, Got: {}".format(
                        i, valueLabels[i], values[i]))

            # Check Polices
            for p in range(len(polices)):
                pol = polices[p]
                polLabel = policyLabels[p]

                for i in range(len(polices[i])):
                    if (np.around(polLabel[i], 5) != np.around(pol[i], 5)):
                        self.fail("At Index {} Expected policy: {}, Got: {}".format(
                            i, polLabel[i], pol[i]))

