from Main.AlphaZero.DistributedSelfPlay import SSHTunnel
from Main.AlphaZero.DistributedSelfPlay.SSHTunnel import HOST_IP_INDEX, HOST_PORT_INDEX, HOST_USERNAME_INDEX, SELF_PORTS
from Main.AlphaZero.DistributedSelfPlay import FitModel

trainers = SSHTunnel.TRAINERS
workers = SSHTunnel.WORKERS
hosts = SSHTunnel.HOSTS


def _promptTrainerPort():
    print("Trainers:")
    for trainerName in trainers:
        print("\t" + str(trainerName))

    while (True):
        trainer = input("Choose trainer: ")
        if (str.isdigit(trainer)):
            print("You entered raw port:", int(trainer))
            return int(trainer), False
        if (trainer not in trainers):
            print("Trainer " + str(trainer) + " not found!")
        else:
            print("Trainer " + str(trainer) + " added! (port " + str(trainers[trainer]) + ")")
            break

    return int(trainers[trainer]), True


def _promptWorkerPorts(overlord=True):
    print("Workers:")
    for workerName in workers:
        print("\t" + str(workerName))

    if(overlord):
        print("Choose remote workers. Stop adding workers by typing \"q\"")
    workerPorts = []

    while (True):
        promptMessage = "Add worker: " if overlord else "Choose your name: "
        worker = input(promptMessage)
        if (worker == 'q'):
            break
        if(str.isdigit(worker)):
            print("You entered raw port:", int(worker))
            if(overlord):
                workerPorts.append(int(worker))
            else:
                return int(worker), False
        elif (worker not in workers):
            print("Worker " + str(worker) + " not found!")
        else:
            port = workers[worker]
            if (not overlord):
                return port, True
            workerPorts.append(int(port))
            print("Worker " + str(worker) + " added! (port " + str(port) + ")")

    return workerPorts, False


def _promptHost():
    print("Hosts:")
    for hostName in hosts:
        print("\t" + str(hostName))

    while (True):
        host = input("Choose host: ")
        if (host not in hosts):
            print("Host " + str(host) + " not found!")
        else:
            print("Host " + str(host) + " added! (" + str(hosts[host]) + ")")
            break

    return hosts[host]


def _isInteger(var):
    return type(var) == int or str.isdigit(var)


def startOverlord():
    trainerPort, _ = _promptTrainerPort()

    workerPorts, _ = _promptWorkerPorts(True)

    from Main.AlphaZero.DistributedSelfPlay import DistributedAlphaZeroOverlord
    DistributedAlphaZeroOverlord.startOverlord(trainerPort, workerPorts)


def startRemoteWorker():
    port, useSSHTunnel = _promptWorkerPorts(False)

    if (port not in SELF_PORTS and useSSHTunnel):  # and not _isInteger(port)):
        host = _promptHost()
        SSHTunnel.createSSHTunnel(host, port)

    gpu = input("Gpu: ")
    print("Starting imports for Remote Worker...")
    from Main.AlphaZero.DistributedSelfPlay import RemoteWorker
    RemoteWorker.startRemoteWorker(port, gpu)


def startTrainer():
    port, useSSH = _promptTrainerPort()

    if (port not in SELF_PORTS and useSSH):
        host = _promptHost()
        SSHTunnel.createSSHTunnel(host, port)

    gpu = input("Gpu(s): ")

    from Main.AlphaZero.DistributedSelfPlay import Trainer
    Trainer.startTrainer(port, gpu)

def startLoopingTrainer():
    port, useSSH = _promptTrainerPort()

    if (port not in SELF_PORTS and useSSH):
        host = _promptHost()
        SSHTunnel.createSSHTunnel(host, port)

    gpu = input("Gpu(s): ")

    from Main.AlphaZero.DistributedSelfPlay import Trainer
    Trainer.loopingTrainer(port, gpu)


if (__name__ == "__main__"):
    print("Modes for remote execute:")
    print("\'r\'/\'w\' - Remote Worker")
    print("\'o\'/\'d\' - Remote Overlord ")
    print("\'t\' - Remote Trainer")
    print("\'bt\' - Benchmark Trainer")
    print("\'lt\' - Looping Trainer")

    while (True):
        mode = input("\nMode: ").strip().lower()
        if (mode == 'o' or mode == 'd'):
            startOverlord()
        if (mode == 'r' or mode == 'w'):
            startRemoteWorker()
        if (mode == 't'):
            startTrainer()
        if (mode == 'bt'):
            FitModel.benchmark()
        if (mode == 'lt'):
            startLoopingTrainer()
