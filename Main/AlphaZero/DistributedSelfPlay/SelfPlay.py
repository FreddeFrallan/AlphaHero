from Main.AlphaZero.DistributedSelfPlay import Constants
from Main.Training.Connect4 import MemoryBuffers
from Main import Hyperparameters
import multiprocessing as mp
import numpy as np
import time


def _waitForWorker(connection, dumpPipe):
    gamesCollected = 0
    collectingDataFromWorker = True
    while (collectingDataFromWorker):
        msg, data = connection.readMessage()

        dumpPipe.put((msg, data))
        if (msg == Constants.RemoteProtocol.DUMP_VISITED_STATES_TO_OVERLORD):
            collectingDataFromWorker = False
        elif (msg == Constants.RemoteProtocol.DUMP_REPLAY_DATA_TO_OVERLORD):
            amountOfGames = data[0]
            gamesCollected += amountOfGames

    print("Worker Finished: {}   Amount of Games: {}".format(connection.id, gamesCollected))


def _stopRemoteWorkers(connections):
    print("Aborting remoteWorkers")
    for c in connections:
        c.sendMessage(Constants.RemoteProtocol.OVERLORD_REPLAY_BUFFER_FULL, ("",))


def _replayWatcher(connections, dumpPipe):
    print("Starting replay watcher")
    collectedGamesThisCycle = 0
    MemoryBuffers.clearReplayBuffer()
    startTimeSelfPlay = time.time()

    while (True):
        msg, data = dumpPipe.get()
        if (msg == Constants.RemoteProtocol.DUMP_REPLAY_DATA_TO_OVERLORD):
            amountOfGames, states, evals, polices, weights = data
            MemoryBuffers.addLabelsToReplayBuffer(states, evals, polices)
            collectedGamesThisCycle += amountOfGames

            cycleProgressMsg = "{} / {}".format(collectedGamesThisCycle, Hyperparameters.AMOUNT_OF_NEW_GAMES_PER_CYCLE)
            elapsedTime = np.around(time.time() - startTimeSelfPlay, 3)
            elapsedTimeMsg = "Time: {}".format(elapsedTime)
            gamesPerSecondMsg = "Games/Sec: {}".format(np.around(collectedGamesThisCycle / elapsedTime, 3))

            print(cycleProgressMsg + "\t\t" + elapsedTimeMsg + "\t\t" + gamesPerSecondMsg)

            if (collectedGamesThisCycle >= Hyperparameters.AMOUNT_OF_NEW_GAMES_PER_CYCLE):
                _stopRemoteWorkers(connections)
                return


def _getCurrentArgMaxLevel(modelGeneration):
    for a in Hyperparameters.ARG_MAX_SCHEDULE:
        cycleNumber, argMaxLevel = a
        if (modelGeneration < cycleNumber):
            return argMaxLevel

    _, finalArgMaxLevel = Hyperparameters.ARG_MAX_SCHEDULE[-1]
    return finalArgMaxLevel


def selfPlay(workerConnections, modelAsBytes, modelGeneration):
    t1 = time.time()

    argMaxLevel = _getCurrentArgMaxLevel(modelGeneration)
    workerCounter = 0
    for c in workerConnections:
        c.sendMessage(Constants.RemoteProtocol.START_SELF_PLAY,
                      (workerCounter, modelAsBytes, Hyperparameters.MCTS_SIMULATIONS_PER_MOVE, argMaxLevel))
        workerCounter += 1
    print("Sending out models finished:", time.time() - t1)

    dumpPipe = mp.Queue()
    procs = [mp.Process(target=_waitForWorker, args=(c, dumpPipe)) for c in workerConnections]
    for p in procs:
        p.start()

    _replayWatcher(workerConnections, dumpPipe)

    print("Self-Play finished: {}".format(time.time() - t1))
