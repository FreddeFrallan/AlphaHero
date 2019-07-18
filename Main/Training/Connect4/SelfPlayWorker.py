from Main.AlphaZero.Oracle import OracleCommands, PreComputation
from Main.Environments.Connect4 import Utils, Connect4Bitmaps
from Main.Training.Connect4 import ParallelFullGame, ParallelSelfPlayAgent
from Main import Hyperparameters, MachineSpecificSettings
from Main.AlphaZero.DistributedSelfPlay import Constants
from Main.Training import GameFunctions
import time
import numpy as np

COMPUTE_TABLE = {}


def initGameFuncs():
    # EvaluateState.initRawEvaluateState()  # Init Connect4 TerminalEval
    # return GameFunctions.GameFunctions(SimulateAction.simulateAction, GetPossibleActions.getPossibleActions,
    #                                    Evaluations.modelEval, TerminalEvaluation.terminalEvaluation)
    return GameFunctions.GameFunctions(Connect4Bitmaps.simulateAction, Connect4Bitmaps.getPossibleActions,
                                       None, Connect4Bitmaps.terminalEvaluation,
                                       generateStartState=Connect4Bitmaps.BitBoard)


def initSelfPlayAgents(gameFuncs):
    agent1 = ParallelSelfPlayAgent.SelfPlayAgent(gameFuncs, Hyperparameters.MCTS_SIMULATIONS_PER_MOVE, playerID=1,
                                                 dataAugmentationFunc=Utils.createMirroredStateAndPolicy,
                                                 computeTable=COMPUTE_TABLE)
    agent2 = ParallelSelfPlayAgent.SelfPlayAgent(gameFuncs, Hyperparameters.MCTS_SIMULATIONS_PER_MOVE, playerID=-1,
                                                 dataAugmentationFunc=Utils.createMirroredStateAndPolicy,
                                                 computeTable=COMPUTE_TABLE)
    return agent1, agent2


def _computeAndSendMostVisitedStates(searchedStates, qSend, workerID):
    distinctStates = PreComputation._countDistinctStatesAsList(searchedStates)
    distinctStates.sort(key=lambda x: x[-1], reverse=True)

    amountOfPassedGames = min(Hyperparameters.PRE_COMPUTED_PREDICTION_MOST_VISITED_THRESHOLD,
                              len(distinctStates))
    topDistinct = [distinctStates[i][0] for i in range(amountOfPassedGames)]
    qSend.put((OracleCommands.DUMP_MOST_VISITED_STATES, topDistinct, workerID))


def distributedSelfPlayWorker(workerSettings, oracleSend, oracleListen, dataDumpPipe, abortFlag, computeTable):
    global COMPUTE_TABLE
    import StartInit
    StartInit.init()
    if (MachineSpecificSettings.IS_UNIX_MACHINE and MachineSpecificSettings.USE_LOW_PRIORITY_ON_WORKERS):
        import os
        os.nice(1)

    # Setup Hyperparameters
    amountOfGames = MachineSpecificSettings.GAMES_PER_WORKER
    Hyperparameters.MCTS_SIMULATIONS_PER_MOVE = workerSettings[Constants.LocalWorkerProtocol.MCTS_ITERATIONS]
    remoteWorkerID = workerSettings[Constants.LocalWorkerProtocol.REMOTE_WORKER_ID]
    localWorkerID = workerSettings[Constants.LocalWorkerProtocol.LOCAL_WORKER_ID]

    # Set unique random seed for worker
    seed = (remoteWorkerID * 109 + localWorkerID + 1) * 17
    np.random.seed(seed)
    COMPUTE_TABLE = computeTable
    gameFuncs = initGameFuncs()

    ParallelFullGame.playParallelGames(gameFuncs, amountOfGames, oracleSend, oracleListen, dataDumpPipe, abortFlag,
                                       localWorkerID)
