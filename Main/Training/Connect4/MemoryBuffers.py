from Main import Hyperparameters
import numpy as np
import random, os

# Replay buffers for learning
REPLAY_STATE_BUFFER = []
REPLAY_VALUE_BUFFER = []
REPLAY_POLICY_BUFFER = []
REPLAY_WEIGHTS_BUFFER = []

# Buffers for visited states
STATES_VISITED = []

# For statistics
POST_MCTS_POLICY_BUFFER = []
VALUE_PREDICTION_BUFFER = []
POLICY_PREDICTION_BUFFER = []

PRE_GAME_OVER_POST_MCTS_POLICY_BUFFER = []
PRE_GAME_OVER_PREDICTION_POLICY_BUFFER = []

ROUNDS_BUFFER = []
REWARD_BUFFER = []
SAVED_NETWORK_PREDICTIONS_BUFFER = []

CURRENT_MODEL_VERSION = 0


# ************** Training Data
def addLabelsToReplayBuffer(states, valueLabels, policyLabels):
    global REPLAY_STATE_BUFFER, REPLAY_VALUE_BUFFER, REPLAY_POLICY_BUFFER

    REPLAY_STATE_BUFFER.extend(states)
    REPLAY_VALUE_BUFFER.extend(valueLabels)
    REPLAY_POLICY_BUFFER.extend(policyLabels)

    if (CURRENT_MODEL_VERSION + 1 < Hyperparameters.SLIDING_WINDOW_TURNS_TO_FULL):
        maxBufferSize = (1 + CURRENT_MODEL_VERSION) * int(
            Hyperparameters.REPLAY_BUFFER_LENGTH / Hyperparameters.SLIDING_WINDOW_TURNS_TO_FULL)
    else:
        maxBufferSize = Hyperparameters.REPLAY_BUFFER_LENGTH

    maxBufferSize = min(maxBufferSize, Hyperparameters.REPLAY_BUFFER_LENGTH)

    overLimit = len(REPLAY_STATE_BUFFER) - maxBufferSize
    if (overLimit > 0):
        del REPLAY_STATE_BUFFER[:overLimit]
        del REPLAY_VALUE_BUFFER[:overLimit]
        del REPLAY_POLICY_BUFFER[:overLimit]

def getTrainingData():
    sampleSize = min(Hyperparameters.SAMPLES_PER_TRAINING_BATCH, len(REPLAY_STATE_BUFFER))

    possibleIndices = list(np.arange(len(REPLAY_STATE_BUFFER)))
    states, valueLabels, policyLabels = [], [], []
    for i in random.sample(possibleIndices, sampleSize):
        states.append(REPLAY_STATE_BUFFER[i])
        valueLabels.append(REPLAY_VALUE_BUFFER[i])
        policyLabels.append(REPLAY_POLICY_BUFFER[i])

    return np.array(states), np.array(valueLabels), np.array(policyLabels)


def getAllTrainingData():
    return np.array(REPLAY_STATE_BUFFER), np.array(REPLAY_VALUE_BUFFER), np.array(REPLAY_POLICY_BUFFER), np.array(REPLAY_WEIGHTS_BUFFER)


class TrainingData:
    def __init__(self, states, values, labels):
        self.states = states
        self.values = values
        self.labels = labels


def storeTrainingDataToDisk():
    import pickle, time
    writeTime = time.time()
    try:
        storeClass = TrainingData(REPLAY_STATE_BUFFER, REPLAY_VALUE_BUFFER, REPLAY_POLICY_BUFFER)
        with open(Hyperparameters.TRAINING_DATA_PATH, 'wb') as handle:
            pickle.dump(storeClass, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Training Data stored to file: {} sec".format(time.time() - writeTime))
    except Exception as e:
        try:
            print(e)
            print("Could not store Training data to file, trying split...")

            storeStates = TrainingData(REPLAY_STATE_BUFFER, [], [])
            storeEvals = TrainingData([], REPLAY_VALUE_BUFFER, [])
            storePolices = TrainingData([], [], REPLAY_POLICY_BUFFER)
            with open(Hyperparameters.TRAINING_DATA_PATH + "S", 'wb') as handle:
                pickle.dump(storeStates, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(Hyperparameters.TRAINING_DATA_PATH + "E", 'wb') as handle:
                pickle.dump(storeEvals, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(Hyperparameters.TRAINING_DATA_PATH + "P", 'wb') as handle:
                pickle.dump(storePolices, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("Split successfull")
        except Exception as e:
            print(e)
            print("Split failed")


def loadOldTrainingDataFromDisk():
    global REPLAY_STATE_BUFFER, REPLAY_VALUE_BUFFER, REPLAY_POLICY_BUFFER
    import pickle

    if (os.path.isfile(Hyperparameters.TRAINING_DATA_PATH) == False):
        print("Unable to locate stored Training Data")
        return
    with open(Hyperparameters.TRAINING_DATA_PATH, 'rb') as handle:
        storeClass = pickle.load(handle)

    REPLAY_STATE_BUFFER = storeClass.states
    REPLAY_VALUE_BUFFER = storeClass.values
    REPLAY_POLICY_BUFFER = storeClass.labels
    print("Training Data loaded from file, Samples: ", len(REPLAY_STATE_BUFFER))


def getDistinctTrainingData():
    distinct = {}
    for i in range(len(REPLAY_STATE_BUFFER)):
        s = REPLAY_STATE_BUFFER[i]
        e = REPLAY_VALUE_BUFFER[i]
        p = REPLAY_POLICY_BUFFER[i]
        key = s.tostring()
        if ((key in distinct) == False):
            distinct[key] = [s, [e], [p]]
        else:
            distinct[key][1].append(e)
            distinct[key][2].append(p)

    distinctStates = []
    distinctValues = []
    distinctPolices = []
    for k in distinct.keys():
        temp = distinct[k]
        distinctStates.append(temp[0])
        distinctValues.append(np.mean(temp[1]))
        distinctPolices.append(np.mean(temp[2], axis=0))

    return distinctStates, distinctValues, distinctPolices


def clearReplayBuffer():
    global REPLAY_STATE_BUFFER, REPLAY_VALUE_BUFFER, REPLAY_POLICY_BUFFER, REPLAY_WEIGHTS_BUFFER
    REPLAY_STATE_BUFFER.clear()
    REPLAY_VALUE_BUFFER.clear()
    REPLAY_POLICY_BUFFER.clear()
    REPLAY_WEIGHTS_BUFFER.clear()


# ***************** Statistics
def appendToBuffer(newValue, theList, maxSize):
    if (len(theList) >= maxSize):
        del theList[0]
    theList.append(newValue)


def setValuePredictionBuffer(data):
    global VALUE_PREDICTION_BUFFER
    VALUE_PREDICTION_BUFFER = data


def addToValuePredictionBuffer(data):
    global VALUE_PREDICTION_BUFFER
    VALUE_PREDICTION_BUFFER.extend(data)


def setMostVisitedStates(data):
    global STATES_VISITED
    STATES_VISITED = data


def getAmountOfStoredDataPoints():
    return len(REPLAY_VALUE_BUFFER)
