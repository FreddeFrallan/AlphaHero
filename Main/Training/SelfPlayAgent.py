from Main import Hyperparameters
from Main.MCTS import SequentialMCTS
from Main.AlphaZero import LabelGenerator
from Main.Environments.Connect4 import Utils, GetPossibleActions, FullGame
import numpy as np

CURRENT_MODEL = None
CURRENT_ROOT = None
CURRENT_ORACLE_LISTEN = None
CURRENT_ORACLE_SEND = None
WORKER_ID = None


def setNewCurrentRoot(root, move):
    global CURRENT_ROOT
    for c in root.children:
        if (c.action == move):
            CURRENT_ROOT = c
            break


class SelfPlayAgent():

    def __init__(self, gameFuncs, simulationsPerMove, playerID=1, dataAugmentationFunc=None):
        self.dataAugmentationFunc = dataAugmentationFunc
        self.simulationsPerMove = simulationsPerMove
        self.stateRewardsBuffer = []
        self.localStatesBuffer = []
        self.localPolicyBuffer = []
        self.gameFuncs = gameFuncs
        self.playerID = playerID

    def makeMove(self, state):
        global CURRENT_ROOT
        if (CURRENT_ROOT is not None):
            self.addDirichletNoiseToRoot(CURRENT_ROOT)

        root = SequentialMCTS.performSearch(self.gameFuncs, state, self.simulationsPerMove,
                                            self.playerID, CURRENT_ROOT)

        # Bug testing
        assert (Utils.compareState(root.state, state))
        assert (root.currentPlayer == self.playerID)
        assert (root.children[0].currentPlayer == -self.playerID)
        moves = GetPossibleActions.getPossibleActions(state)
        for c in root.children:
            assert(c.action in moves)

        # Create Training Labels from current Tree
        inState, policyLabel = LabelGenerator.generateLabels(root)
        self.addToLocalGameBuffer(inState, policyLabel)

        if (FullGame.CURRENT_ROUND >= Hyperparameters.POLICY_THRESHOLD):
            selectedMove = np.argmax(policyLabel)
        else:
            selectedMove = np.random.choice(np.arange(0, len(policyLabel)), p=policyLabel)

        setNewCurrentRoot(root, selectedMove)
        return selectedMove

    def addDirichletNoiseToRoot(self, root):
        noise = np.random.dirichlet(Hyperparameters.DIRICHLET_NOISE_PARAM * np.ones(len(root.children)))
        for i in range(len(noise)):
            root.children[i].policyEstimation += noise[i]

    def addToLocalGameBuffer(self, state, policy):
        self.localStatesBuffer.append(state)
        self.localPolicyBuffer.append(policy)

    # When a game is over we get a reward, we can then label all of the labels created from the current game
    def getReward(self, reward):
        augmentedStates, augmentedPolices = self.dataAugmentationFunc(self.localStatesBuffer, self.localPolicyBuffer)

        self.localStatesBuffer.extend(augmentedStates)
        self.stateRewardsBuffer = [reward] * len(self.localStatesBuffer)
        self.localPolicyBuffer.extend(augmentedPolices)

    # For some reason it seems to make a difference to the GC if we delete the lists first... ?
    def clearLocalBuffers(self):
        global CURRENT_ROOT
        CURRENT_ROOT = None
        del self.localStatesBuffer
        del self.stateRewardsBuffer
        del self.localPolicyBuffer
        self.localStatesBuffer = []
        self.stateRewardsBuffer = []
        self.localPolicyBuffer = []
