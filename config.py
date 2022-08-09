import platform
platform.platform()
#global variables and possibly more
TRAJECTORIES_DIR = "./Trajectories"

if platform.system().lower()=="linux":
    DATA_DIR = "/mnt/sda1/AmitF/Diffusion_Data"
else:
    DATA_DIR = "./Data"
IMAGING_PARAMETERS_PATH = "./Resources/Parameters/imaging_parameters_table.csv"
MODEL_PARAMETERS_PATH = "./Resources/Parameters/model_parameters_table.csv"
OUTPUTS_DIR = "./Outputs/files"
IMG4VID_DIR = "./Outputs/img4vid"

DEFAULT_NEIGHBOR_THRESHOLD_UM = 4. #for find_neighbors
BREAK_COLLIDING_TRAJECTORIES_WIN_SIZE_FRAMES = 9
MAX_ALLOWED_COLLISIONS_IN_WIN = 7
MIN_BROKEN_DF_LENGTH = 50

MAX_FRAME_BEFORE_UNDERSAMPLING = 6144

PADDING_INTERPOLATION_MAX_FRAMES = 7

MAX_PROCESSORS_LIMIT = 20 #None for no limit

BAYESIAN_MAX_K_EVENTS = 6
BAYESIAN_MIN_STATE_DURATION_FRAMES = 10

#For Viterbi algorithm
MAX_AVAILABLE_STATES_TO_KEEP = 100
FORGET_FAR_TETHER_POINTS_THRESHOLD_RATIO = 50.
