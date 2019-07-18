from Main.AlphaZero import NeuralNetworks
from Main.Training.Connect4 import MemoryBuffers, FitAndEvaluateModel, SelfPlayWorker
from Main import Hyperparameters, MachineSpecificSettings
from Main.Training import WorkersPool
from Main.Analysis.Connect4 import RuntimeAnalysis
from Main.AlphaZero.Oracle import PredictionOracle, PreComputation
import time
import keras


# Creates the network, either from a saved file or from scratch.
def init(modelCounter=0):
    if (modelCounter == 0):
        model = NeuralNetworks.createResidualNetwork((6, 7, 5), Hyperparameters.FILTERS_PER_CONV_LAYER,
                                                     Hyperparameters.LAYERS_PER_RESIDUAL_BLOCK,
                                                     Hyperparameters.AMOUNT_OF_RESIDUAL_BLOCKS)

        model.save(Hyperparameters.MODELS_SAVE_PATH + str(modelCounter))
    else:  # Load an older model and start from there
        model = keras.models.load_model(Hyperparameters.MODELS_SAVE_PATH + str(modelCounter))
        # model = keras.models.load_model("../../../Z-Models/Model" + str(modelCounter))

    model.save(Hyperparameters.CURRENT_MODEL_PATH, overwrite=True)
    parallelModel = None if MachineSpecificSettings.AMOUNT_OF_GPUS <= 1 else NeuralNetworks.createMultipleGPUModel(
        model)

    model.summary()
    return model, parallelModel, modelCounter


# Self-play workers generates new labels, using the Oracle algorithm
def _initSelfPlay():
    pool = WorkersPool.WorkersPool(Hyperparameters.AMOUNT_OF_SELF_PLAY_WORKERS, SelfPlayWorker.testPoolFunc)
    print("Self-Play Workers Initialized")
    return pool


# Plays X amount of self-play games in parallel. The currentModelVersion is sent along so that the workers only update
# if they have an older version
def selfPlay(selfPlayPool, model):
    t1 = time.time()

    computeTable = PreComputation.computePredictionTable(
        model) if Hyperparameters.USE_PREDICTION_CACHE else {}
    selfPlayPool.runJobs(
        [(Hyperparameters.AMOUNT_OF_GAMES_PER_WORKER, computeTable)] * Hyperparameters.AMOUNT_OF_SELF_PLAY_WORKERS)

    PredictionOracle.runPredictionOracle(model, selfPlayPool)  # Enter Oracle mode on the main thread

    print("Self play finished: {} ms".format(time.time() - t1))
    MemoryBuffers.storeTrainingDataToDisk()


def alphaZeroPipeline():
    # Setup specific settings for this machine
    MachineSpecificSettings.setupHyperparameters()
    model, parallelModel, currentModelVersion = init(modelCounter=0)
    selfPlayPool = _initSelfPlay()
    for i in range(10):
        print("10")

    #MemoryBuffers.loadOldTrainingDataFromDisk()

    while (True):
        print("\n******\nModel Version:", currentModelVersion)
        MemoryBuffers.CURRENT_MODEL_VERSION = currentModelVersion  # Used for the MemoryBuffer Sliding-Window
        selfPlay(selfPlayPool, model) # Can not be optimized with parallel model... ?
        RuntimeAnalysis.dumpRuntimeAnalysis(model, currentModelVersion)
        print("Replay", len(MemoryBuffers.REPLAY_STATE_BUFFER))

        if (MachineSpecificSettings.AMOUNT_OF_GPUS > 1):
            currentModelVersion = FitAndEvaluateModel.fitAndEvaluateModel(parallelModel, currentModelVersion)
        else:
            currentModelVersion = FitAndEvaluateModel.fitAndEvaluateModel(model, currentModelVersion)

        # Save model after every cycle, since we don't evaluate. Important that it's the non-parallel model that's saved
        model.save(Hyperparameters.MODELS_SAVE_PATH + str(currentModelVersion))


if (__name__ == '__main__'):
    alphaZeroPipeline()
