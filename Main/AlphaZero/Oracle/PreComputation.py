from Main.Environments.Connect4 import Utils
from Main.Training.Connect4 import MemoryBuffers
import numpy as np
import time


# states is a list of bitboards
def _countDistinctStates(states):
    distinct = {}
    for s in states:
        mState = s.mirror()

        sExists = (s in distinct)
        mExists = (mState in distinct)

        if (sExists == False and mExists == False):
            distinct.update({s: [s, 1]})
        elif (sExists):
            distinct[s][-1] += 1
        elif (mExists):
            distinct[mState][-1] += 1

    return distinct


def _countDistinctStatesAsList(states):
    distinct = _countDistinctStates(states)
    return [distinct[k] for k in distinct.keys()]


def computePredictionTable(model):
    t1 = time.time()
    if (len(MemoryBuffers.STATES_VISITED) == 0):
        return {}

    distinctStates = _countDistinctStates(MemoryBuffers.STATES_VISITED)
    distinctStates = [distinctStates[k][0] for k in distinctStates.keys()]  # List-form
    distinctStates.extend([d.mirror() for d in distinctStates])  # Add mirrored states

    convStates = [Utils.bitBoard2ConvState(d) for d in distinctStates]
    predictions = model.predict(np.array(convStates))
    preComputeTable = {}
    for i in range(len(distinctStates)):
        preComputeTable.update({distinctStates[i]: (predictions[0][i][0], predictions[1][i])})

    print("Pre computation complete: {} states,  {}".format(len(distinctStates), time.time() - t1))
    return preComputeTable
