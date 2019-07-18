from Main.AlphaZero.DistributedSelfPlay import Connection, SelfPlay, Trainer
from Main.Training.Connect4 import MemoryBuffers
from Main import Hyperparameters
import multiprocessing as mp
import os



def _initModelProc(generation, path):
    from Main.AlphaZero import NeuralNetworks
    import keras, os
    import numpy as np

    # Let's always use the same network
    np.random.seed(42)
    from tensorflow import set_random_seed
    set_random_seed(43)

    if (generation == 0):
        if (Hyperparameters.USE_CONMPRESSED_BOARD_REPRESENTATION):
            boardRepr = (6, 7, 3)
        else:
            boardRepr = (6, 7, 5)
        model = NeuralNetworks.createResidualNetwork(boardRepr, Hyperparameters.FILTERS_PER_CONV_LAYER,
                                                     Hyperparameters.LAYERS_PER_RESIDUAL_BLOCK,
                                                     Hyperparameters.AMOUNT_OF_RESIDUAL_BLOCKS)
    else:  # Load an older model and start from there
        absPath = Hyperparameters.MODELS_SAVE_PATH + str(generation)
        print("Abs Path", absPath)
        print(os.path.isfile(absPath))
        model = keras.models.load_model(absPath)

    model.save(path, overwrite=True)
    model.summary()
    print("Model initalized at: {}".format(path))


def _initModel(generation):
    p = mp.Process(target=_initModelProc, args=(generation, _currentModelAbsPath()))
    p.start()
    p.join()
    p.terminate()
    print("Main model initialized")


def _getCurrentModelBytes():
    f = open(_currentModelAbsPath(), 'rb')
    data = f.read()
    f.close()
    return data


def _currentModelAbsPath():
    return os.path.abspath(Hyperparameters.CURRENT_MODEL_PATH)


def startOverlord(trainerPort, remoteWorkerPorts):
    import StartInit
    StartInit.init()
    print("Enter what model you wish to use, enter 0 if you wish to generate a new model.")
    modelGeneration = int(input("Model Generation:"))
    _initModel(modelGeneration)

    # Send model and trainer settings to the trainer
    trainerConnection = Connection.Connection(ip='localhost', port=trainerPort, server=True)
    trainerSettings = (Hyperparameters.REPLAY_BUFFER_LENGTH, Hyperparameters.SLIDING_WINDOW_TURNS_TO_FULL)
    trainerConnection.sendMessage(Trainer.STATUS_INIT_MODEL, (_getCurrentModelBytes(), trainerSettings))

    # Connect to all the remote workers
    remoteWorkerConnections = [Connection.Connection(ip='localhost', port=p, server=True) for p in remoteWorkerPorts]
    print("All connections we're made")
    MemoryBuffers.CURRENT_MODEL_VERSION = modelGeneration
    modelAsBytes = _getCurrentModelBytes()

    try:
        while (True):
            SelfPlay.selfPlay(remoteWorkerConnections, modelAsBytes, MemoryBuffers.CURRENT_MODEL_VERSION)
            # SEND DATA TO trainerConnection
            trainerConnection.sendMessage(Trainer.STATUS_TRAIN_DATA,
                                          (MemoryBuffers.CURRENT_MODEL_VERSION, MemoryBuffers.REPLAY_STATE_BUFFER,
                                           MemoryBuffers.REPLAY_VALUE_BUFFER, MemoryBuffers.REPLAY_POLICY_BUFFER,
                                           MemoryBuffers.REPLAY_WEIGHTS_BUFFER))
            # WAIT for trainer
            _, data = trainerConnection.readMessage()
            modelAsBytes = data[0]
            MemoryBuffers.CURRENT_MODEL_VERSION += 1
    except Exception as e:
        print(e)

    for c in remoteWorkerConnections:
        c.close()
    input("End of Main program...")


if (__name__ == '__main__'):
    _initModel(0)
