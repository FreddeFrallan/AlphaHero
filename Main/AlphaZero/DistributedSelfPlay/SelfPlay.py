from Main.AlphaZero.DistributedSelfPlay import Constants
from Main.Training.Connect4 import MemoryBuffers
from Main import Hyperparameters
import multiprocessing as mp
import numpy as np
import time

'''
Listen for data from the Remote Worker and forward it to the Replay Watcher.
Every worker will continue to work until the pre-determined number of games has been collected.

After the Remote Workers have been aborted by the Replay Watcher, the will message the listener and the listener quits
'''
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


# Collect data from all listeners and upon reaching a pre-determined number of games abort all Remote Workers
# As the main data is stored at the Looping Trainer we clear the Replay Buffer at the start
def _replayWatcher(connections, dumpPipe):
    print("Starting replay watcher")
    collectedGamesThisCycle = 0
    MemoryBuffers.clearReplayBuffer()
    startTimeSelfPlay = time.time()

    while (True):
        msg, data = dumpPipe.get()  # Data passed from a listener

        if (msg == Constants.RemoteProtocol.DUMP_REPLAY_DATA_TO_OVERLORD):
            amountOfGames, states, evals, polices, weights = data
            MemoryBuffers.addLabelsToReplayBuffer(states, evals, polices)
            collectedGamesThisCycle += amountOfGames

            # Display a formatted message
            cycleProgressMsg = "{} / {}".format(collectedGamesThisCycle, Hyperparameters.AMOUNT_OF_NEW_GAMES_PER_CYCLE)
            elapsedTime = np.around(time.time() - startTimeSelfPlay, 3)
            elapsedTimeMsg = "Time: {}".format(elapsedTime)
            gamesPerSecondMsg = "Games/Sec: {}".format(np.around(collectedGamesThisCycle / elapsedTime, 3))
            print(cycleProgressMsg + "\t\t" + elapsedTimeMsg + "\t\t" + gamesPerSecondMsg)

            # Upon receving sufficent number of games we send a message to all Remote Workers to abort
            if (collectedGamesThisCycle >= Hyperparameters.AMOUNT_OF_NEW_GAMES_PER_CYCLE):
                _stopRemoteWorkers(connections)
                return


'''
*** CURRENTLY INNACTIVATED ***
The argmax scheduele deceides at what point in a game we start playing deterministicly according to the policy .
'''


def _getCurrentArgMaxLevel(modelGeneration):
    for a in Hyperparameters.ARG_MAX_SCHEDULE:
        cycleNumber, argMaxLevel = a
        if (modelGeneration < cycleNumber):
            return argMaxLevel

    _, finalArgMaxLevel = Hyperparameters.ARG_MAX_SCHEDULE[-1]
    return finalArgMaxLevel


'''
Broadcast the current: (Network Parameters, MCTS simulations per move, ArgMax schedule) to all Remote Workers. 
Then start a listener for every worker that collects game data.
These listeners forwards the collected data to the Replay Watcher

Finishes after a fixed number of games.  
'''


def selfPlay(workerConnections, modelAsBytes, modelGeneration):
    t1 = time.time()  # Only used for displaying elapsed time to the user

    argMaxLevel = _getCurrentArgMaxLevel(modelGeneration)
    workerCounter = 0
    for c in workerConnections:
        c.sendMessage(Constants.RemoteProtocol.START_SELF_PLAY,
                      (workerCounter, modelAsBytes, Hyperparameters.MCTS_SIMULATIONS_PER_MOVE, argMaxLevel))
        workerCounter += 1
    print("Sending out models finished:", time.time() - t1)

    # Start a listener for every remote worker
    dumpPipe = mp.Queue()
    procs = [mp.Process(target=_waitForWorker, args=(c, dumpPipe)) for c in workerConnections]
    for p in procs:
        p.start()

    # Wait until all listeners have reported that they have finished, then stop all Remote Workers
    _replayWatcher(workerConnections, dumpPipe)
    print("Self-Play finished: {}".format(time.time() - t1))
