import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def getAbsolutePath(path):
    if (path[0] != '\\' or path[0] != '/'):
        path = "/" + path
    return ROOT_DIR + path
