from multiprocessing.sharedctypes import Value
import multiprocessing as mp
from ctypes import c_bool




class WorkersPool:

    def __init__(self, amountOfProcs, func):
        self.procs = []
        self.procQues = []
        self.qToChild = mp.Queue(maxsize=0)
        self.qFromChild = mp.Queue(maxsize=0)
        self.doJobFlag = Value(c_bool, False, lock=False)
        for i in range(amountOfProcs):
            childProc = mp.Queue(maxsize=0)
            self.procs.append(mp.Process(target=func, args=(self.qToChild, self.qFromChild, childProc, self.doJobFlag, i)))
            self.procQues.append(childProc)

        for p in self.procs:
            p.start()

    def close(self):
        for p in self.procs:
            p.terminate()

    def abortCurrentJobs(self):
        self.doJobFlag.value = False

    # Custom mapping function that returns the arguments in the correct order
    def runAndCollectJobs(self, args):
        self.doJobFlag.value = False
        for i in range(len(args)):
            self.qToChild.put(args[i] + (i,))
        res = [self.qFromChild.get() for a in args]
        res.sort(key=lambda x: x[-1])
        self.doJobFlag.value = False

        return [r[:-1] for r in res]

    def runJobs(self, args):
        self.doJobFlag.value = True
        for i in range(len(args)):
            self.qToChild.put(args[i] + (i,))

    def listenOnQ(self):
        return self.qFromChild.get()

    def sendMsgToChild(self, msg, childID):
        self.procQues[childID].put(msg)
