from Main.AlphaZero.DistributedSelfPlay import Constants, Connection, RemoteDataManager, RemoteSelfPlayPool
from Main import Hyperparameters, MachineSpecificSettings
from Main.AlphaZero.Oracle import PredictionOracle
import multiprocessing as mp
import os, time, gc


def _initSelfPlay(overlordConnection, remoteWorkerID, computeTable):
    toOraclePipe = mp.Queue()
    dataManager = RemoteDataManager.DataManager(overlordConnection, toOraclePipe,
                                                MachineSpecificSettings.AMOUNT_OF_WORKERS)
    abortFlag = dataManager.abortFlag
    toDataManagerPipe = dataManager.sendPipe
    endWorkersPipe = dataManager.workersFinishedPipe

    pool = RemoteSelfPlayPool.WorkersPool(MachineSpecificSettings.AMOUNT_OF_WORKERS, toOraclePipe, toDataManagerPipe,
                                          abortFlag, computeTable, remoteWorkerID)

    return dataManager, pool, toOraclePipe, endWorkersPipe


def _selfPlayProc(overlordConnection, remoteWorkerID, modelAbsPath, MCTSIterations, argMaxLimit, gpuSetting,
                  terminateQueue):
    try:
        import os, threading
        print("StartingOracle GPU-Settings: {}".format(gpuSetting))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuSetting
        import keras, tensorflow as tf, keras.backend as K

        print("\nRemote Worker ID", remoteWorkerID)

        # Init parameters
        Hyperparameters.MCTS_SIMULATIONS_PER_MOVE = MCTSIterations
        Hyperparameters.POLICY_THRESHOLD = argMaxLimit
        MachineSpecificSettings.setupHyperparameters()

        # Init model, manager & Workers
        model = keras.models.load_model(modelAbsPath)
        if (MachineSpecificSettings.IS_UNIX_MACHINE == False):
            model._make_predict_function()

        computeTable = {}
        dataManager, selfPlayPool, toOraclePipe, endPipe = _initSelfPlay(overlordConnection, remoteWorkerID,
                                                                         computeTable)
        print("Self-Play Workers Initialized!")

        # Start self-play
        t1 = time.time()
        selfPlayPool.start()
        oracleThread = threading.Thread(target=PredictionOracle.runPredictionOracle,
                                        args=(model, selfPlayPool, toOraclePipe, K, tf)).start()

        # Wait to exit, either by oracle or Datamanager
        print(endPipe.get())

        # Kill the remaining workers
        try:
            oracleThread.terminate()
        except:
            pass
        dataManager.killDataManager()

        del model
        keras.backend.clear_session()
        for i in range(15): gc.collect()
        print("Self-Play finished: ", time.time() - t1)

    except Exception as e:
        print("ERROR")
        print(e)
    finally:
        print("Session cleared...")
        terminateQueue.put("RemoteWorker Cleared")


def _selfPlay(connection, remoteWorkerID, modelAsBytes, MCTSIterations, argMaxLimit, gpuSetting):
    # In the case we have several workers on the same machine, we store them at different files.
    # This can be solved by introducing the "Oracle Layer", allowing one remoteWorker to have several oracles
    tempFilePath = os.path.abspath("TempModel{}".format(remoteWorkerID))
    f = open(tempFilePath, 'wb')
    f.write(modelAsBytes)
    f.close()

    terminateQ = mp.Queue()
    p = mp.Process(target=_selfPlayProc,
                   args=(connection, remoteWorkerID, tempFilePath, MCTSIterations, argMaxLimit, gpuSetting, terminateQ))
    p.start()
    print(terminateQ.get())
    p.terminate()


def startRemoteWorker(port, gpuSettings):
    ip = "localhost"
    print("GPU-Settings: {}".format(gpuSettings))
    connection = Connection.Connection(ip, port, False)

    while (True):
        t1 = time.time()
        msg, data = connection.readMessage()
        print("Read msg time:", time.time() - t1)

        if (msg == Constants.RemoteProtocol.KILL_WORKER):
            print("Killing remote worker...")
            break
        elif (msg == Constants.RemoteProtocol.START_SELF_PLAY):
            remoteWorkerID, kerasModelAsBytes, MCTSIterations, argMaxLimit = data
            print("Starting self-play!")
            _selfPlay(connection, remoteWorkerID, kerasModelAsBytes, MCTSIterations, argMaxLimit, gpuSettings)


if (__name__ == '__main__'):
    startRemoteWorker()
