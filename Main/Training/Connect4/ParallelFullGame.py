from Main.AlphaZero.DistributedSelfPlay import Constants as C
from Main.Environments.Connect4 import Connect4Bitmaps
from Main.Training.Connect4 import SelfPlayWorker
from Main.AlphaZero.Oracle import OracleCommands
from Main import MachineSpecificSettings
from Main.MCTS import MainFunctions
import time


class ParallelGame:

    def __init__(self, gameFuncs):
        self.gameFuncs = gameFuncs
        self.currentRound = 0
        self.currentPlayerIndex = 0
        self.agents = SelfPlayWorker.initSelfPlayAgents(gameFuncs)
        self.agents[0].opponent = self.agents[1]
        self.agents[1].opponent = self.agents[0]

        # Create parallel Search Tree
        firstPlayer = self.agents[0].playerID
        # self.agents[0].currentRoot = MainFunctions.createNewMCTS(Utils.createStartBoard(), firstPlayer)
        # self.firstRoot = MainFunctions.createNewMCTS(Connect4Bitmaps.BitBoard(), firstPlayer)
        self.firstRoot = MainFunctions.createNewMCTS(gameFuncs.genereateStartState(), firstPlayer)
        self.agents[0].currentRoot = self.firstRoot
        self.policyLabels = []
        self.stateInputs = []
        self.evalLabels = []

        # Init the search
        self.startNewSearch()

    def getCurrentPlayer(self):
        return self.agents[self.currentPlayerIndex]

    def setNextPlayersTurn(self):
        self.currentRound += 1
        self.currentPlayerIndex = (self.currentPlayerIndex + 1) % 2

    def isTerminal(self):
        return self.getCurrentPlayer().currentRoot.terminal

    def giveRewardAndCollectData(self):
        #terminal, evalScore = Connect4Bitmaps.terminalEvaluation(self.getCurrentPlayer().currentRoot.state)
        terminal, evalScore = self.gameFuncs.evaluateTerminal(self.getCurrentPlayer().currentRoot.state)
        gameScore = self.getCurrentPlayer().currentRoot.score / self.getCurrentPlayer().currentRoot.visits
        assert (evalScore == gameScore)

        for a in self.agents:
            a.getReward(evalScore)
            self.stateInputs.extend(a.localStatesBuffer)
            self.evalLabels.extend(a.stateRewardsBuffer)
            self.policyLabels.extend(a.localPolicyBuffer)

    def startNewSearch(self):
        self.getCurrentPlayer().startNewSearch(self.currentRound)

    def getOracleData(self, value, policy):
        self.getCurrentPlayer().getEvalData(value[0], policy)

    def getSearchedStates(self):
        return self.agents[0].statesEvaluated + self.agents[1].statesEvaluated

    def getSavedEvals(self):
        return self.agents[0].savedEvals + self.agents[1].savedEvals

    def getPerformedEvals(self):
        return self.agents[0].performedEvals + self.agents[1].performedEvals


def _collectEvalStates(currentGames):
    evalStates = []
    evalGames = []
    endedGames = []
    for game in currentGames:
        status, data = _getEvalFromGame(game)

        if (status == "GameOver"):
            endedGames.append(game)
            continue

        evalStates.append(data)
        evalGames.append(game)

    return evalStates, evalGames, endedGames


def _getEvalFromGame(game):
    status, data = game.getCurrentPlayer().performSearchIteration()

    if (status == "MakeMove"):
        game.setNextPlayersTurn()
        if (game.isTerminal()):
            game.giveRewardAndCollectData()
            return "GameOver", None
        else:
            game.startNewSearch()
            return _getEvalFromGame(game)

    if (status == "Eval"):
        return "Eval", data


def _dumpFinishedGames(finishedGames, qSend):
    states, evalLabels, policyLabels, weights, searchedStates = [], [], [], [], []
    for g in finishedGames:
        gStates = g.stateInputs
        amountOfRegularDataPoints = len(gStates)

        states.extend(gStates)
        evalLabels.extend(g.evalLabels)
        policyLabels.extend(g.policyLabels)
        searchedStates.extend(g.getSearchedStates())
        weights.extend([1] * amountOfRegularDataPoints)

    print("Sending batch to q")
    qSend.put(
        (C.LocalWorkerProtocol.DUMP_TO_REPLAY_BUFFER, len(finishedGames), states, evalLabels, policyLabels, weights))


# Runs "amount" of games in parallel on a separate process. The evaluations are expected to be made my an oracle
# using the oracleSend & oracleListen queues.
def playParallelGames(gameFuncs, amountOfGames, oracleSend, oracleListen, dataDumpPipe, abortFlag, localWorkerID):
    games = [ParallelGame(gameFuncs) for i in range(amountOfGames)]
    newlyFinishedGames = []
    oldFinishedGames = []

    # DEBUG - Timekeeping
    oracleWaitTime = []
    simulateTime = []
    performedEvals = []

    while (abortFlag.value == False):
        t1 = time.time()
        # Refill games
        for i in range(amountOfGames - len(games)):
            games.append(ParallelGame(gameFuncs))

        # Collect predictions
        evalStates, evalGames, endedGames = _collectEvalStates(games)
        assert (len(evalStates) == len(evalGames))

        # Remove finished games and dump data to server if batch sized
        for g in endedGames:
            newlyFinishedGames.append(g)
            games.remove(g)

        if (len(newlyFinishedGames) >= MachineSpecificSettings.GAMES_BATCH_SIZE_TO_REMOTE_WORKER):
            _dumpFinishedGames(newlyFinishedGames, dataDumpPipe)
            oldFinishedGames.extend(newlyFinishedGames)
            newlyFinishedGames.clear()

        # Check we're still running self-play
        if (abortFlag.value):
            break
        simulateTime.append(time.time() - t1)
        t1 = time.time()

        # Send predictions to oracle
        oracleSend.put((OracleCommands.EVAL_NEW_DATA, len(evalStates), evalStates, localWorkerID))
        compactMsg = oracleListen.get()

        status = compactMsg[0]
        if (status == C.LocalWorkerProtocol.SELF_PLAY_OVER):
            print("Got abort direct msg")
            break

        _, amount, oraclePrediction = compactMsg

        oracleWaitTime.append(time.time() - t1)

        assert (amount == len(evalStates))
        performedEvals.append(amount)  # DEBUG stats

        # Insert data into current games and local buffer
        for i in range(len(evalGames)):
            games[i].getOracleData(oraclePrediction[0][i], oraclePrediction[1][i])

    searchedStates = []
    for g in oldFinishedGames:
        searchedStates.extend(g.getSearchedStates())
    print("Sending visited games:", localWorkerID)
    dataDumpPipe.put((C.LocalWorkerProtocol.DUMP_MOST_VISITED_STATES, searchedStates))
