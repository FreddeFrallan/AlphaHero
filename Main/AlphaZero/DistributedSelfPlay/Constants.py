from enum import Enum

BUFFER_SIZE = 4096
HEADER_MSG_SIZE = 16
HEADER_ENDIAN_TYPE = 'little'

SSH_WORKER_FILE_NAME = "AlphaZeroSSHWorker.py"
SSH_WORKER_LOG_FILE_NAME = "SSHWorkerLogFIle.log"
SSH_MAX_BUFFER_SIZE = 512
SSH_MAGIC_SENTENCE = b"ThisIsTheStartOfTheAlphaZeroProgram\n"
SSH_MAGIC_SENTENCE_SIZE = len(SSH_MAGIC_SENTENCE)
SSH_END_MESSAGE = "END_OF_SSH_SESSION"
#SSH_FULL_HEADER_SIZE = HEADER_MSG_SIZE + SSH_MAGIC_NUMBER_SIZE

class RemoteProtocol(Enum):
    KILL_WORKER = 0
    START_SELF_PLAY = 1
    END_SELF_PLAY = 2
    SELF_PLAY_WORKER_FINISHED = 3
    OVERLORD_REPLAY_BUFFER_FULL = 4
    REMOTE_WORKER_FINISHED = 5
    DUMP_REPLAY_DATA_TO_OVERLORD = 6
    DUMP_VISITED_STATES_TO_OVERLORD = 7


class LocalWorkerProtocol(Enum):
    MCTS_ITERATIONS = "MCTS"
    REMOTE_WORKER_ID = "RID"
    LOCAL_WORKER_ID = "LID"
    DUMP_TO_REPLAY_BUFFER = "DUMP"
    DUMP_MOST_VISITED_STATES = "DUMP_VISITED"
    SELF_PLAY_OVER = "SELF_PLAY_OVER"

