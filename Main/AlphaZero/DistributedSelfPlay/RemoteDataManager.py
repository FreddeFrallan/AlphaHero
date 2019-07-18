from Main.AlphaZero.DistributedSelfPlay import Constants as C
from Main.AlphaZero.Oracle import PreComputation
from Main.Training.Connect4 import MemoryBuffers
from multiprocessing.sharedctypes import Value
from Main import MachineSpecificSettings, Hyperparameters
import multiprocessing as mp
import ctypes


class DataManager:

    def __init__(self, overlordConnection, toOraclePipe, amountOfLocalWorkers):
        self.abortFlag = Value(ctypes.c_bool, False, lock=False)
        self.sendPipe = mp.Queue()
        self.workersFinishedPipe = mp.Queue()

        self.listenerProc = mp.Process(target=listenToOverlord,
                                       args=(overlordConnection, self.abortFlag, toOraclePipe))
        self.senderProc = mp.Process(target=sendToOverlord,
                                     args=(
                                     overlordConnection, self.sendPipe, amountOfLocalWorkers, self.workersFinishedPipe))

        self.listenerProc.start()
        self.senderProc.start()

    def killDataManager(self):
        try:
            self.listenerProc.terminate()
            self.senderProc.terminate()
        except:
            pass


def listenToOverlord(overlordConnection, abortFlag, toOraclePipe):
    while (abortFlag.value == False):
        overlordMsg = overlordConnection.readMessage()
        msg = overlordMsg[0]

        if (msg == C.RemoteProtocol.OVERLORD_REPLAY_BUFFER_FULL):
            abortFlag.value = True
            toOraclePipe.put((C.LocalWorkerProtocol.SELF_PLAY_OVER,))  # Send abort msg to oracle
            break
    print("Ending listen thread")


def sendToOverlord(overlordConnection, localPipe, amountOfWorkers, endPipe):
    # Needed in the end when we wish to count the bitmaps
    import time
    time.sleep(3)
    print("Starting init")
    import StartInit
    StartInit.init()

    runningCycle = True
    amountOfCollectedGames = 0
    amountOfCollectedWorkers = 0
    collectedVisitedStates = []

    while (runningCycle):
        tupleMsg = localPipe.get()
        msgType = tupleMsg[0]

        if (msgType == C.LocalWorkerProtocol.DUMP_TO_REPLAY_BUFFER):
            _, amountOfGames, states, evals, polices, weights = tupleMsg
            MemoryBuffers.addLabelsToReplayBuffer(states, evals, polices)
            amountOfCollectedGames += amountOfGames

            if (amountOfCollectedGames >= MachineSpecificSettings.GAMES_BATCH_SIZE_TO_OVERLORD):
                print("Sending to oracle from dataworker")
                dStates, dEvals, dPolices, dWeights = MemoryBuffers.getAllTrainingData()
                dumpMsg = (amountOfCollectedGames, dStates, dEvals, dPolices, dWeights)
                overlordConnection.sendMessage(C.RemoteProtocol.DUMP_REPLAY_DATA_TO_OVERLORD, dumpMsg)

                amountOfCollectedGames = 0
                MemoryBuffers.clearReplayBuffer()

        elif (msgType == C.LocalWorkerProtocol.DUMP_MOST_VISITED_STATES):
            amountOfCollectedWorkers += 1
            _, states = tupleMsg

            if (amountOfCollectedWorkers >= amountOfWorkers):
                print("collected states from all local workers: ", len(collectedVisitedStates))
                sendMostVisitedStatesToOverlord(overlordConnection, collectedVisitedStates)
                print("Sent message to all workers")
                runningCycle = False

    endPipe.put("Ending by datamanager")
    print("Ending sending thread")


# Until we have fixed hashing this seems like a bad idea
def sendMostVisitedStatesToOverlord(overlordConnection, collectedStates):
    if (Hyperparameters.USE_PREDICTION_CACHE):
        overlordConnection.sendMessage(C.RemoteProtocol.DUMP_VISITED_STATES_TO_OVERLORD, [])
    else:
        overlordConnection.sendMessage(C.RemoteProtocol.DUMP_VISITED_STATES_TO_OVERLORD, [])
