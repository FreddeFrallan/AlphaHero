from Main.Environments.Connect4 import Connect4Bitmaps


# Should only be imported once (per process)!


def init():
    print("StartInit.py init() called!")
    # EvaluateState.initRawEvaluateState()
    Connect4Bitmaps.init()
