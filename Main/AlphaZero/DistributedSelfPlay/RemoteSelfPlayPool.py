from Main.AlphaZero.DistributedSelfPlay import Constants as C
from Main.Training.Connect4 import SelfPlayWorker
from Main import Hyperparameters
import multiprocessing as mp


class WorkersPool:

    def __init__(self, amountOfProcs, oracleSendPipe, dataDumpPipe, abortFlag, computeTable, remoteWorkerID):
        self.procPipes = []
        self.procs = []
        func = SelfPlayWorker.distributedSelfPlayWorker

        for i in range(amountOfProcs):
            settings = {
                C.LocalWorkerProtocol.MCTS_ITERATIONS: Hyperparameters.MCTS_SIMULATIONS_PER_MOVE,
                C.LocalWorkerProtocol.REMOTE_WORKER_ID: remoteWorkerID,
                C.LocalWorkerProtocol.LOCAL_WORKER_ID: i
            }

            workerListenPipe = mp.Queue(maxsize=0)
            args = (settings, oracleSendPipe, workerListenPipe, dataDumpPipe, abortFlag, computeTable)
            self.procs.append(mp.Process(target=func, args=args))
            self.procPipes.append(workerListenPipe)

    def broadcastMsg(self, msg):
        for i in range(len(self.procs)):
            self.sendMessageToProc(i, msg)

    def forceClose(self):
        for p in self.procs:
            p.terminate()

    def joinAllWorkers(self):
        for p in self.procs:
            p.join()

    def start(self):
        for p in self.procs:
            p.start()

    def sendMessageToProc(self, id, msg):
        self.procPipes[id].put(msg)
