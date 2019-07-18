import Main.Hyperparameters as MainParam
import os

AMOUNT_OF_GPUS = 1
BATCH_SIZE = 512
AMOUNT_OF_WORKERS = 6
GAMES_BATCH_SIZE_TO_REMOTE_WORKER = 1  # To overlord
GAMES_PER_WORKER = 50
OPTIMIZED_GRAPH_MAX_BATCH = GAMES_PER_WORKER

GAMES_BATCH_SIZE_TO_OVERLORD = 30


IS_UNIX_MACHINE = os.name != 'nt'
USE_LOW_PRIORITY_ON_WORKERS = False

# Only adds a small wait on the trainer before it starts using the GPU.
# Should only be set to false if the machine is ONLY trainer
REMOTE_WORKER_AND_TRAINER = True

def setupHyperparameters():
    print("Setting up Machine Dependent Hyperparameters...")
    MainParam.MINI_BATCH_SIZE = BATCH_SIZE
    MainParam.AMOUNT_OF_SELF_PLAY_WORKERS = AMOUNT_OF_WORKERS
    MainParam.AMOUNT_OF_GAMES_PER_WORKER = GAMES_PER_WORKER

    print("New Settings...")
    print("Batch Size: ", BATCH_SIZE)
    print("Amount of Workers: ", AMOUNT_OF_WORKERS)
    print("Games per worker: ", GAMES_PER_WORKER)
