from sacred import Ingredient
from sacred.settings import SETTINGS

model_ingredient = Ingredient('model')
dataset_ingredient = Ingredient('dataset', ingredients=[model_ingredient])
train_ingredient = Ingredient('train', ingredients=[dataset_ingredient])
collect_ingredient = Ingredient('collect', ingredients=[model_ingredient])
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

@model_ingredient.config
def cfg_model():
    # name of the model (will be saved in "$RLBC_MODELS/name")
    name = ''
    # name of the architecture
    archi = 'resnet_18_narrow32'
    # mode, flat or skills
    mode = 'flat'
    # number of frames taken as input
    num_frames = 3
    # number of scalar signals taken as input
    num_signals = 0
    # dimension of signal
    dim_signal = 7
    # type of conv layers normalization
    normalization = 'batchnorm'
    # model input type
    input_type = 'depth'
    # model action space
    action_space = 'tool_lin'
    # timesteps of actions in the future to predict
    steps_action = (1, 10, 20, 30)
    # number of skill heads
    num_skills = 1
    # device to load the model on
    device = 'cuda'
    # flag to resume training
    resume = True
    # flag to resume training using stored optimizer
    load_optimizer = True
    # epoch to resume training
    epoch = 'current'


@dataset_ingredient.config
def cfg_dataset():
    # name of the dataset (will be saved in "$RLBC_DATA/name")
    name = ''
    # max number of demos to train on
    max_demos = None
    # number of cameras used during data loading
    num_cameras = 1
    # name of data augmentation to apply
    image_augmentation = ''
    # name of the signals to load
    signal_keys = ['target_position']
    # list of signals dimension
    signal_lengths = [2]
    # flag to load mask, needed for data augmentation
    load_masks = True


@train_ingredient.config
def cfg_train():
    # gripper loss coefficient
    lam_grip = 0.1
    # master pretraining loss coefficient
    lam_master = 0.0
    # mini-batch size
    batch_size = 64
    # optimizer learning rate
    learning_rate = 1e-3
    # number of epochs to train the model
    epochs = 101
    # number of workers to load data
    workers = 16
    # number of epochs between two evaluations
    eval_interval = 4
    # proportion of the dataset withold for evaluation
    eval_proportion = 0.05
    # first epoch to start evaluation
    eval_first_epoch = 0


@collect_ingredient.config
def cfg_collect():
    # folder to save data or report (will be saved in "$RLBC_DATA/folder")
    folder = ''
    # agent type: script, bc or rl
    agent = 'script'
    # dir of the pickle file containing pre-recorded demos if the agent is replay
    replay_dir = ''
    # database type: demos, video or evaluation
    db_type = 'demos'
    # environment to record or evaluate on
    env = 'UR5-PickCamEnv-v0'
    # starting seed
    seed = 0
    # number of episodes to record
    episodes = 1
    # override environment max number of steps
    max_steps = -1
    # skills timescale or a list of them
    timescale = 60
    # list of skills sequence
    skill_sequence = []
    # first epoch to evaluate:
    first_epoch = None
    # last epoch to evaluate
    last_epoch = None
    # epochs interval between two evaluations
    iter_epoch = 2
    # number of workers to use
    workers = 1
    # flag to rewrite the dataset
    rewrite = True
    # flag to record trajectories even when they are failed
    # used for skills data collection
    record_failed = False
    # flag for skill data collection
    skill_collection = False
    # flag to use dask
    dask = False
    # flag to render the environment, used for debug
    render = False
    # flag to stop data collection when the environment returns done
    enforce_stop_when_done = False
    # flag to overlay attention maps on top of depth maps
    attention_maps = False
    # flag to add data augmentation to collected images
    # the augmentation flag is used only for BC agent. RL agent reads the RL args
    image_augmentation = ''
