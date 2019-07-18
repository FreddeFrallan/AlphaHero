from Main import Hyperparameters
from Main.MCTS import ParallelMCTS
from Main.AlphaZero import LabelGenerator
from Main.Environments.Connect4 import Utils
import numpy as np

'''
Unlike previous implementations where we let self-play agents go against eachother using the "FullGame" interface.
Two agents simple share a game tree, where they both do search and acts as the current game
'''


class SelfPlayAgent():

    def __init__(self, gameFuncs, simulationsPerMove, playerID=1, dataAugmentationFunc=None, opponent=None,
                 computeTable=None):
        self.dataAugmentationFunc = dataAugmentationFunc
        self.simulationsPerMove = simulationsPerMove
        self.stateRewardsBuffer = []
        self.localStatesBuffer = []
        self.localPolicyBuffer = []
        self.gameFuncs = gameFuncs
        self.playerID = playerID
        self.opponent = opponent
        self.currentRound = 0
        self.invertState = False

        self.mirroredState = None
        self.currentNode = None
        self.currentPath = None
        self.currentRoot = None  # Is shared between the two self players

        self.statesEvaluated = []
        self.computeTable = computeTable
        self.savedEvals = 0
        self.performedEvals = 0

    def startNewSearch(self, currentRound):
        self.currentRound = currentRound
        self.addDirichletNoiseToRoot(self.currentRoot)

    # At the start of every new move we add some noise to the policy prediction
    def addDirichletNoiseToRoot(self, root):
        noise = np.random.dirichlet(Hyperparameters.DIRICHLET_NOISE_PARAM * np.ones(len(root.children)))
        for i in range(len(noise)):
            root.children[i].policyEstimation += noise[i]

    # Performs a MCTS until we can either make a move or have state that we require the oracle to evaluate
    def performSearchIteration(self):
        node, path = None, []
        while (node is None and self.currentRoot.visits < self.simulationsPerMove):
            node, path = ParallelMCTS.simulateMCTS(self.gameFuncs, self.currentRoot, self.currentRoot.visits + 1)

        # We can perform a move
        if (node is None and self.currentRoot.visits >= self.simulationsPerMove):
            return "MakeMove", self.makeMove()

        # We have reached an eval Node
        self.currentNode = node
        self.currentPath = path

        '''
        # Remove mirroring of states
        self.invertState = np.random.random() >= 0.5
        if (self.invertState):
            self.mirroredState = node.state.mirror()

        currentState = self.mirroredState if self.invertState else node.state
        '''

        currentState = node.state
        #numpyState = currentState.toNumpyState()
        # Check the pre-compute table
        if (Hyperparameters.USE_PREDICTION_CACHE):
            self.statesEvaluated.append(node.state)

            # Only queries the dict once:
            res = self.computeTable.get(currentState, None)
            # if (currentState in self.computeTable):
            if (res is not None):
                # print("Found precomputed predict!")
                value, policy = res  # self.computeTable[currentState]
                self.getEvalData(value, policy)
                self.savedEvals += 1
                return self.performSearchIteration()

        self.performedEvals += 1

        return "Eval", currentState.toConvState(node.currentPlayer)#Utils.state2ConvState(numpyState, node.currentPlayer)

    def makeMove(self):
        # Create Training Labels from current Tree
        inState, evalLabel, policyLabel = LabelGenerator.generateQLabels(self.currentRoot)
        self._addToLocalGameBuffer(inState, evalLabel, policyLabel)

        if (self.currentRound >= Hyperparameters.POLICY_THRESHOLD):
            selectedMove = np.argmax(policyLabel)
        else:
            selectedMove = np.random.choice(np.arange(0, len(policyLabel)), p=policyLabel)

        self.setNewCurrentRoot(self.currentRoot, selectedMove)
        return selectedMove

    def getEvalData(self, value, policy):
        if (Hyperparameters.USE_PREDICTION_CACHE):
            keyState = self.mirroredState if self.invertState else self.currentNode.state

            # inserts keyState and (value, policy) if not exists, else just returns old value, which we do not use
            # hopefully, this works
            self.computeTable.setdefault(keyState, (value, policy))

        if (self.invertState):
            policy = policy[::-1]  # Re-invert policy
        ParallelMCTS.simulatePostEvaluate(self.gameFuncs, self.currentNode, self.currentPath, value, policy)

    def _addToLocalGameBuffer(self, state, eval, policy):
        self.localStatesBuffer.append(state)
        self.localPolicyBuffer.append(policy)
        if (Hyperparameters.USE_Q_LABELS):
            self.stateRewardsBuffer.append(eval)

    # Effectively playing the move in the current game as the tree is shared with the opponent
    def setNewCurrentRoot(self, root, move):
        for c in root.children:
            if (c.action == move):
                self.opponent.currentRoot = c

    # When a game is over we get a reward, we can then label all of the labels created from the current game
    def getReward(self, reward):
        augmentedStates, augmentedPolices = self.dataAugmentationFunc(self.localStatesBuffer, self.localPolicyBuffer)

        self.localStatesBuffer.extend(augmentedStates)
        self.localPolicyBuffer.extend(augmentedPolices)

        if (Hyperparameters.USE_Q_LABELS):  # IF we use the Q-Labels, we simply duplicate our current eval labels
            self.stateRewardsBuffer.extend(self.stateRewardsBuffer)

            # TESTING, take the mean of the Q-Value & Z-Value
            self.stateRewardsBuffer = (np.array(self.stateRewardsBuffer) + reward) / 2
        else:
            self.stateRewardsBuffer = [reward] * len(self.localStatesBuffer)

    # This is currently not used as we simply create new agents for every self-play turn.
    # It's possible we could gain a minor speedup by re-using old agents and clearing their buffers...
    def clearLocalBuffers(self):
        global CURRENT_ROOT
        CURRENT_ROOT = None
        del self.localStatesBuffer
        del self.stateRewardsBuffer
        del self.localPolicyBuffer
        self.localStatesBuffer = []
        self.stateRewardsBuffer = []
        self.localPolicyBuffer = []
