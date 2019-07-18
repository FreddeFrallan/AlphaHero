from Main import Hyperparameters
import numpy as np
import os
'''
# Replay buffers for learning
REPLAY_STATE_BUFFER = []
REPLAY_VALUE_BUFFER = []
REPLAY_POLICY_BUFFER = []

CURRENT_MODEL_VERSION = 0

# Contains a list where every entry is dict containing: states, their weight and their value
DISTINCT_REPLAY_BUFFER = []


# ************** Training Data
def addLabelsToReplayBuffer(states, valueLabels, policyLabels):
    global REPLAY_STATE_BUFFER, REPLAY_VALUE_BUFFER, REPLAY_POLICY_BUFFER

    distinct = {}
    for i in range(len(states)):
        s = states[i]
        e = valueLabels[i]
        p = policyLabels[i]
        assert np.around(np.sum(p), 1) == 1

        key = s.tostring()
        if ((key in distinct) == False):
            distinct[key] = [s, [e], [p], 1]
        else:
            temp = distinct[key]
            temp[1].append(e)
            temp[2].append(p)
            temp[3] += 1

    generation = {}
    for k in distinct.keys():
        s, E, P, c = distinct[k]  # state, evalList, policyList, counter
        e = np.mean(E)
        p = np.mean(P, axis=0)
        assert np.around(np.sum(p), 1) == 1
        generation[k] = [s, e, p, c]
    DISTINCT_REPLAY_BUFFER.append(generation)

    # Apply distinct sliding window
    if (CURRENT_MODEL_VERSION < Hyperparameters.SLIDING_WINDOW_TURNS_TO_FULL):
        popTurn = np.ceil(
            1 / (Hyperparameters.DISTINCT_REPLAY_BUFFER_LENGTH / Hyperparameters.SLIDING_WINDOW_TURNS_TO_FULL))
        if (CURRENT_MODEL_VERSION % (popTurn + 1) == popTurn):
            del DISTINCT_REPLAY_BUFFER[0]

    # Keep bufferSize
    while (len(DISTINCT_REPLAY_BUFFER) > Hyperparameters.DISTINCT_REPLAY_BUFFER_LENGTH):
        del DISTINCT_REPLAY_BUFFER[0]


class TrainingData:
    def __init__(self, states, values, labels):
        self.states = states
        self.values = values
        self.labels = labels


def storeTrainingDataToDisk():
    import pickle, time
    writeTime = time.time()
    try:
        with open(Hyperparameters.TRAINING_DATA_PATH, 'wb') as handle:
            pickle.dump(DISTINCT_REPLAY_BUFFER, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Training Data stored to file: {} sec".format(time.time() - writeTime))
    except Exception as e:
        print(e)
        print("Could not store Training data to file, trying split...")


def loadOldTrainingDataFromDisk():
    global DISTINCT_REPLAY_BUFFER
    import pickle

    if (os.path.isfile(Hyperparameters.TRAINING_DATA_PATH) == False):
        print("Unable to locate stored Training Data")
        return
    with open(Hyperparameters.TRAINING_DATA_PATH, 'rb') as handle:
        oldData = pickle.load(handle)
        DISTINCT_REPLAY_BUFFER = oldData

    print("Training Data loaded from file, Samples: ", len(DISTINCT_REPLAY_BUFFER))


def _getAllEntriesMatchingKey(key):
    entries = []
    for gen in DISTINCT_REPLAY_BUFFER:
        if (key in gen):
            entries.append(gen[key])
    return entries


def getDistinctTrainingData():
    distinctStates, distinctValues, distinctPolices = [], [], []
    checkedKeys = {}

    allWeights, allEvals, allPolices = [], [], []
    distinctAmounts = []
    distinctWeights = []
    for gen in DISTINCT_REPLAY_BUFFER:
        for k in gen.keys():
            if (k in checkedKeys):  # We have already added this key
                continue

            entries = _getAllEntriesMatchingKey(key=k)
            state = entries[0][0]
            distinctStates.append(state)

            amount = 0
            weightSum = 0
            for ent in entries:
                allEvals.append(ent[1])
                allPolices.append(ent[2])
                allWeights.append(ent[3])
                weightSum += ent[3]
                amount += 1

            distinctAmounts.append(amount)
            distinctWeights.append(weightSum)

    amount = len(allWeights)
    allWeights = np.array(allWeights, ndmin=2).reshape(amount, 1)
    allEvals = np.array(allEvals, ndmin=2).reshape(amount, 1)
    allPolices = np.array(allPolices, ndmin=2).reshape(amount, 7)

    allEvals *= allWeights
    allPolices *= allWeights

    counter = 0
    for i in range(len(distinctAmounts)):
        d = distinctAmounts[i]
        wSum = distinctWeights[i]
        distinctValues.append(np.sum(allEvals[counter: counter + d] / wSum))
        distinctPolices.append(np.sum(allPolices[counter: counter + d] / wSum, axis=0))
        counter += d

    return distinctStates, distinctValues, distinctPolices


def clearReplayBuffer():
    global DISTINCT_REPLAY_BUFFER
    DISTINCT_REPLAY_BUFFER.clear()


def getAmountOfStoredDataPoints():
    return np.sum(len(gen.keys()) for gen in DISTINCT_REPLAY_BUFFER)
'''
