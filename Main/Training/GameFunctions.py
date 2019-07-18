class GameFunctions:

    def __init__(self, simulateAction, getPossibleActions, evalState, terminal, generateStartState=None, switch=True):
        self.genereateStartState = generateStartState
        self.getPossibleActions = getPossibleActions
        self.simulateAction = simulateAction
        self.evaluateTerminal = terminal
        self.evaluateState = evalState
        self.switchPlayer = switch
