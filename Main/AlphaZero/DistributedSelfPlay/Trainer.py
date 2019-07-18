import datetime
import os
import time

from Main import MachineSpecificSettings, Hyperparameters
from Main.AlphaZero.DistributedSelfPlay import Connection, FitModel
from Main.AlphaZero import Utils
from Main.Training.Connect4 import MemoryBuffers

STATUS_TRAIN_DATA = "trainData"
STATUS_INIT_MODEL = "initModel"


def _getModelPath():
    return os.path.abspath("TrainerModel")


def _writeModelToDiskAsBytes(modelAsBytes):
    tempFilePath = _getModelPath()
    f = open(tempFilePath, 'wb')
    f.write(modelAsBytes)
    f.close()
    return tempFilePath


def _readModelFromDisk():
    f = open(_getModelPath(), 'rb')
    temp = f.read()
    f.close()
    return temp


def startTrainer(port, gpuSettings):
    connection = Connection.Connection(ip='localhost', port=port, server=False)
    status, data = connection.readMessage()
    assert status == STATUS_INIT_MODEL

    modelAsBytes, trainerSettings = data
    modelAbsPath = _writeModelToDiskAsBytes(modelAsBytes)
    Hyperparameters.REPLAY_BUFFER_LENGTH = trainerSettings[0]
    Hyperparameters.SLIDING_WINDOW_TURNS_TO_FULL = trainerSettings[1]

    # Used for naming the runtime analasys log
    if ("Y" in input("Use old training data (Y/N):").upper()):
        MemoryBuffers.loadOldTrainingDataFromDisk()
    startTime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H;%M;%S")

    while (True):
        status, data = connection.readMessage()

        if (status == STATUS_TRAIN_DATA):
            t1 = time.time()
            modelVersion, states, values, policies, weights = data
            MemoryBuffers.CURRENT_MODEL_VERSION = modelVersion
            MemoryBuffers.addLabelsToReplayBuffer(states, values, policies)
            MemoryBuffers.storeTrainingDataToDisk()

            FitModel.fitModel(modelAbsPath, gpuSettings, MemoryBuffers.CURRENT_MODEL_VERSION, startTime)
            trainedModelAsBytes = _readModelFromDisk()

            print("Training finished:", time.time() - t1)
            connection.sendMessage("Finished", (trainedModelAsBytes,))


def _getLearningRate(generation):
    for a in Hyperparameters.LEARNING_RATE_SCHEDULE:
        cycleNumber, lr = a
        if (generation < cycleNumber):
            return lr

    _, finalLr = Hyperparameters.LEARNING_RATE_SCHEDULE[-1]
    return finalLr


def loopingTrainer(port, gpuSettings):
    connection = Connection.Connection(ip='localhost', port=port, server=False)
    status, data = connection.readMessage()
    assert status == STATUS_INIT_MODEL

    modelAsBytes, trainerSettings = data
    modelAbsPath = _writeModelToDiskAsBytes(modelAsBytes)
    Hyperparameters.REPLAY_BUFFER_LENGTH = trainerSettings[0]
    Hyperparameters.SLIDING_WINDOW_TURNS_TO_FULL = trainerSettings[1]

    # Used for naming the runtime analasys log
    if ("Y" in input("Use old training data (Y/N):").upper()):
        MemoryBuffers.loadOldTrainingDataFromDisk()

    import os, StartInit
    StartInit.init()
    print("Starting Trainer GPU-Settings: {}".format(gpuSettings))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuSettings
    from Main.AlphaZero import NeuralNetworks
    import numpy as np
    import keras

    MachineSpecificSettings.setupHyperparameters()
    singleModel = keras.models.load_model(modelAbsPath)

    if (MachineSpecificSettings.AMOUNT_OF_GPUS > 1):
        trainingModel = NeuralNetworks.createMultipleGPUModel(singleModel)
    else:
        trainingModel = singleModel

    while (True):
        status, data = connection.readMessage()
        print("Got msg:", status)

        if (status == STATUS_TRAIN_DATA):
            t1 = time.time()
            modelVersion, states, values, policies, weights = data
            Utils.storeWeightsToFile(weights, "RandomWeights.dat")
            keras.backend.set_value(trainingModel.optimizer.lr, _getLearningRate(modelVersion))

            MemoryBuffers.CURRENT_MODEL_VERSION = modelVersion
            MemoryBuffers.addLabelsToReplayBuffer(states, values, policies, weights)

            inStates, valueLabels, policyLabels = MemoryBuffers.getDistinctTrainingData()
            s = np.array(inStates)
            v = np.array(valueLabels)
            p = np.array(policyLabels)
            dataProcessingTime = time.time() - t1
            print("Data preprocessing finished: {}".format(dataProcessingTime))
            print("Using LR:", keras.backend.get_value(trainingModel.optimizer.lr))
            trainingModel.fit(np.array(s), [np.array(v), np.array(p)],
                              epochs=Hyperparameters.EPOCHS_PER_TRAINING, batch_size=Hyperparameters.MINI_BATCH_SIZE,
                              verbose=2,
                              shuffle=True)

            singleModel.save(modelAbsPath, overwrite=True)
            singleModel.save(Hyperparameters.MODELS_SAVE_PATH + str(modelVersion + 1))

            trainedModelAsBytes = _readModelFromDisk()
            print("Training finished:", time.time() - t1)
            connection.sendMessage("Finished", (trainedModelAsBytes,))

            MemoryBuffers.storeTrainingDataToDisk()
