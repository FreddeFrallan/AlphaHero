import h5py

from Main.Training.Connect4 import MemoryBuffers
from Main import Hyperparameters
import numpy as np
import time
import os

CURRENT_CONTENDER_VERSION = 0


def evaluateModelAgainstTrainingData(model):
    inStates, valueLabels, policyLabels = MemoryBuffers.getAllTrainingData()
    _, valueLoss, policyLoss = model.evaluate(np.array(inStates), [np.array(valueLabels), np.array(policyLabels)]
                                              , verbose=2, shuffle=True)
    print("ValueLoss: {}  PolicyLoss: {}".format(valueLoss, policyLoss))


def fitAndEvaluateModel(model, currentModelVersion):
    global CURRENT_CONTENDER_VERSION
    # This should already be done, but just to be safe. There's no danger in overriding this,
    # since every other worker should already have the latest weights and won't update since we don't increase the
    # currentModelVersion
    model.save_weights(Hyperparameters.CURRENT_MODEL_WEIGHTS_PATH, overwrite=True)

    t1 = time.time()
    # Change to the contender version before training
    if (os.path.isfile(Hyperparameters.CONTENDER_MODEL_WEIGHTS_PATH)):
        print("Loading old contender model...")
        model.load_weights(Hyperparameters.CONTENDER_MODEL_WEIGHTS_PATH)

    # In the paper the sample training data from their training buffer, here we just run through all our current samples
    #inStates, valueLabels, policyLabels = MemoryBuffers.getAllTrainingData()
    inStates, valueLabels, policyLabels = MemoryBuffers.getDistinctTrainingData()
    model.fit(np.array(inStates), [np.array(valueLabels), np.array(policyLabels)],
              epochs=Hyperparameters.EPOCHS_PER_TRAINING, batch_size=Hyperparameters.MINI_BATCH_SIZE, verbose=2,
              shuffle=True)

    print("Training finished: {} ms".format(time.time() - t1))

    return currentModelVersion + 1
