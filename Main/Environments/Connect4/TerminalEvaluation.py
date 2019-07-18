import Main.Environments.Connect4.EvaluateState as Eval
import Main.Environments.Connect4.GetPossibleActions as Actions


def terminalEvaluation(state):
    evalScore = Eval.rawEvaluateState(state)
    if (evalScore == 0.5):
        if (len(Actions.getPossibleActions(state, 1)) == 0):
            return True, evalScore
        return False, evalScore
    else:
        return True, evalScore
