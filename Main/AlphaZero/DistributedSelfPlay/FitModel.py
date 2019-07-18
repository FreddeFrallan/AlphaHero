from Main.Training.Connect4 import MemoryBuffers
from Main import Hyperparameters, MachineSpecificSettings
import multiprocessing as mp
import time


def _fitModelProc(modelPath, useMultipleGPUs, gpuSettings, modelGeneration, inStates, replayEval, replayPolicy,
                  startTime):
    import os
    print("Starting Trainer GPU-Settings: {}".format(gpuSettings))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuSettings
    from Main.AlphaZero import NeuralNetworks
    import numpy as np
    import keras

    # Set the startTime so we append to the same log
    # RuntimeAnalysis.ANALYSIS_STARTTIME = startTime

    MachineSpecificSettings.setupHyperparameters()
    singleModel = keras.models.load_model(modelPath)

    if (useMultipleGPUs):
        trainingModel = NeuralNetworks.createMultipleGPUModel(singleModel)
    else:
        trainingModel = singleModel

    trainingModel.fit(np.array(inStates), [np.array(replayEval), np.array(replayPolicy)],
                      epochs=Hyperparameters.EPOCHS_PER_TRAINING, batch_size=Hyperparameters.MINI_BATCH_SIZE, verbose=2,
                      shuffle=True)

    singleModel.save(modelPath, overwrite=True)
    singleModel.save(Hyperparameters.MODELS_SAVE_PATH + str(modelGeneration + 1))

    print("Training finished proc")
    keras.backend.clear_session()
    print("Session cleared")


def fitModel(modelAbsPath, gpuSettings, modelGeneration, startTime):
    import numpy as np
    print("Stored data points: ", MemoryBuffers.getAmountOfStoredDataPoints())
    t1 = time.time()
    inStates, valueLabels, policyLabels = MemoryBuffers.getDistinctTrainingData()

    s = np.array(inStates)
    v = np.array(valueLabels)
    p = np.array(policyLabels)
    dataProcessingTime = time.time() - t1
    print("Data preprocessing finished: {}".format(dataProcessingTime))

    if (MachineSpecificSettings.REMOTE_WORKER_AND_TRAINER and dataProcessingTime < 5):
        print("Waiting for GPU")
        time.sleep(5 - dataProcessingTime)

    multipleGPUs = MachineSpecificSettings.AMOUNT_OF_GPUS > 1
    proc = mp.Process(target=_fitModelProc,
                      args=(modelAbsPath, multipleGPUs, gpuSettings, modelGeneration, s, v, p, startTime))
    proc.start()
    proc.join()


# *********************************''

def loopingOracle(modelPath, useMultipleGPUs, gpuSettings, modelGeneration, inStates, replayEval, replayPolicy,
                  startTime):
    import os
    print("Starting Trainer GPU-Settings: {}".format(gpuSettings))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuSettings
    from Main.AlphaZero import NeuralNetworks
    import numpy as np
    import keras

    MachineSpecificSettings.setupHyperparameters()
    singleModel = keras.models.load_model(modelPath)

    if (useMultipleGPUs):
        trainingModel = NeuralNetworks.createMultipleGPUModel(singleModel)
    else:
        trainingModel = singleModel

    trainingModel.fit(np.array(inStates), [np.array(replayEval), np.array(replayPolicy)],
                      epochs=Hyperparameters.EPOCHS_PER_TRAINING, batch_size=Hyperparameters.MINI_BATCH_SIZE, verbose=2,
                      shuffle=True)

    singleModel.save(modelPath, overwrite=True)
    singleModel.save(Hyperparameters.MODELS_SAVE_PATH + str(modelGeneration + 1))

    print("Training finished proc")
    keras.backend.clear_session()
    print("Session cleared")


# ***********************************


def benchmark():
    import RootDir

    print("Loading training data...")
    MemoryBuffers.loadOldTrainingDataFromDisk()
    absPath = RootDir.getAbsolutePath(input("ModelName: "))
    gpuSettings = input("Gpu Settings: ")
    t1 = time.time()

    dStates, dEvals, dPolics = MemoryBuffers.getDistinctTrainingData()
    print("Data pre-processing finished:", time.time() - t1)

    useMultipleModels = MachineSpecificSettings.AMOUNT_OF_GPUS > 1
    _fitModelProc(absPath, useMultipleModels, gpuSettings, 0, dStates, dEvals, dPolics, t1)
    print("Full training finished:", time.time() - t1)


if (__name__ == '__main__'):
    benchmark()
