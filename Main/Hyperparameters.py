import RootDir


# *************** NetworkStructure
CONV_NETS = [70, 70, 70, 70]
DENSE_NETS = [512, 200]
L2_REGULARIZATION = 0
USE_CONMPRESSED_BOARD_REPRESENTATION = False

# *************** Rez-Net
FILTERS_PER_CONV_LAYER = 64
LAYERS_PER_RESIDUAL_BLOCK = 2
AMOUNT_OF_RESIDUAL_BLOCKS = 20

# *************** Search
EXPLORATION_CONSTANT = 3
MCTS_SIMULATIONS_PER_MOVE = 30
THINK_TIME_PER_MOVE = 9
DIRICHLET_NOISE_PARAM = 1  # I have no fucking clue what to pick here

# *************** Self-Play
USE_PREDICTION_CACHE = True
PRE_COMPUTED_PREDICTION_MOST_VISITED_THRESHOLD = 3500
PRE_COMPUTATION_MOVE_LIMIT = 43
AMOUNT_OF_NEW_GAMES_PER_CYCLE = 300

CURRENT_MODEL_PATH = "CurrentModel"
CURRENT_MODEL_WEIGHTS_PATH = "CurrentModelWeights"
AMOUNT_OF_SELF_PLAY_WORKERS = 4
AMOUNT_OF_GAMES_PER_WORKER = 200 * 16 # This is configured in MachineSpecific settings

POLICY_THRESHOLD = 43  # How many turns in the game until we start picking deterministic moves
# List where every entry denots: (Cycles, stepsUntilArgMax)
# So entry '(15, 20)', shows that if cycle number 14, they will argmax after move 20
# The last entry will be held forever
ARG_MAX_SCHEDULE = [(10, 10), (15, 20), (20, 43)]


# *************** Training
USE_Q_LABELS = True
LEARNING_RATE_SCHEDULE = [(10, 0.005), (15, 0.0035), (20, 0.0025), (25, 0.001), (30, 0.0005)]

averageGameLength = 22
generationsInBuffer = 20
augmentationfactor = 2
REPLAY_BUFFER_LENGTH = 2300000
DISTINCT_REPLAY_BUFFER_LENGTH = 20

SLIDING_WINDOW_TURNS_TO_FULL = 35
SAMPLES_PER_TRAINING_BATCH = 2048
MINI_BATCH_SIZE = 512
MODELS_SAVE_PATH = RootDir.getAbsolutePath("/OldModels/Model")
CONTENDER_MODEL_WEIGHTS_PATH = RootDir.getAbsolutePath("/Main/AlphaZero/Connect/ContenderModelWeights")
TRAINING_DATA_PATH = RootDir.getAbsolutePath("TrainingData.data")
EPOCHS_PER_TRAINING = 2

LEARNING_RATE = 0.005
MOMENTUM = 0.9

# ****************************** OLD ******************************

# *************** Evaluation
AMOUNT_OF_EVAL_WORKERS = 4
AMOUNT_OF_RANDOM_EVALUATIONS = 15000
AMOUNT_OF_VERSUS_EVALUATIONS = 40

EVAL_POLICY_THRESHOLD = 0
EVAL_POLICY_SENSITIVITY = 100
THRESHOLD_FOR_UPDATING_CURRENT_MODEL = 0.6
RANDOM_EVAL_WITH_SEARCH = False

# *************** TrueSkill LEAGUE - Evaluation
TRUE_SKILL_POLICY_THRESHOLD = 0  # In the paper they always evaluate with a deterministic policy
AMOUNT_OF_TRUE_SKILL_LEAGUE_WORKERS = 16
AMOUNT_OF_FULL_GAME_SCHEMAS_PER_UPDATE = 4
OLD_MODELS_PATH = RootDir.getAbsolutePath("/OldModels")

TRUE_SKILL_LEAGUE_PLOT_EXTRACT_FACTOR = 100
TRUE_SKILL_LEAGUE_PLOT_SUB_VALUE = 40

TRUE_SKILL_NETWORK_PLAYER = "NetworkPlayer"
TRUE_SKILL_UNBIASED_PLAYER = "UnBiasedPlayer"
TRUE_SKILL_HEURISTIC_PLAYER = "HeuristicPlayer"

# ***************Statistics
# Runtime Analysis
RUNTIME_ANALYSIS_FILE_PATH = RootDir.getAbsolutePath("RuntimeAnalysis/")
RUNTIME_ANALYSIS_RECENT_LENGTH = 10000

# Once per evaluation
POLICY_PREDICTION_BUFFER_LENGTH = 10000
VALUE_PREDICTION_BUFFER_LENGTH = 10000

# Once per move
POST_POLICY_BUFFER_LENGTH = 1000
SAVED_NETWORK_PREDICTIONS__BUFFER_LENGTH = 1000

# Once per Game stats
ROUNDS_BUFFER_LENGTH = 100
PRE_GAME_OVER_POLICY_BUFFER_LENGTH = 100
REWARD_BUFFER_LENGTH = 100