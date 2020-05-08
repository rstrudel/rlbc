import datetime
from sacred import Ingredient

# flake8: noqa
general_ingredient = Ingredient('general')
ppo_ingredient = Ingredient('ppo')
bc_ingredient = Ingredient('bc')
hierarchy_ingredient = Ingredient('hierarchy')
log_ingredient = Ingredient('log')
train_ingredient = Ingredient('train', ingredients=[
    general_ingredient, ppo_ingredient, bc_ingredient, hierarchy_ingredient, log_ingredient])


@general_ingredient.config
def cfg_general():
    # environment to train on
    env_name = 'UR5-BowlCamEnv-v0'
    # random seed
    seed = 1
    # whether to render the training
    render = False
    # whether to stop the execution in the very beginning for debug
    pudb = False
    # how many training CPU processes to use
    num_processes = 8
    # the envs will be queried using the specified batch size
    dask_batch_size = None
    # number of frames to train
    num_train_timesteps = 30e7
    # episodes max length
    max_length = None
    # which device to run the experiments on: cuda or cpu
    device = 'cuda'
    # type of input for the conv nets
    input_type = 'depth'
    # number of last actions to pass to the agent
    action_memory = 0


@ppo_ingredient.config
def cfg_ppo():
    # learning rate
    lr = 7e-4
    # number of ppo epochs
    ppo_epoch = 5
    # number of batches for ppo
    num_mini_batch = 8
    # the ppo batch size
    num_master_steps_per_update = 10
    # discount factor for rewards
    gamma = 0.99
    # RMSprop/Adam optimizer epsilon
    eps = 1e-5
    # value loss coefficient
    value_loss_coef = 1.
    # entropy term coefficient
    entropy_coef = 0.05
    # max norm of gradients
    max_grad_norm = 0.5
    # ppo clip parameter
    clip_param = 0.2


@bc_ingredient.config
def cfg_bc():
    # name of the pretrained model with bc skills (should be stored in "$RLBC_MODELS/name")
    bc_model_name = None
    # epoch of the pretrained model with bc skills (should be stored in "$RLBC_MODELS/name")
    bc_model_epoch = None
    # which data augmentation to use for the frames
    augmentation = ''
    # mime action space definition (only used for non-rlbc setup)
    mime_action_space = None


@hierarchy_ingredient.config
def cfg_hierarchy():
    # a list of timescales corresponding to each skill or the timescale value
    timescale = None
    # number of skills used in the hierarchy
    num_skills = 4
    # set vision based master head type
    master_type = 'conv'
    # master number of channels
    master_num_channels = 64
    # master conv filters size
    master_size_conv_filters = 3
    # use the expert scripts defined in the environment (instead of the bc skills)
    use_expert_scripts = False


@log_ingredient.config
def cfg_log():
    # folder to save the model (will be saved in "$RLBC_MODELS/name")
    folder = 'default'
    # log interval, one log per n updates
    log_interval = 1
    # save interval, one save per n updates
    save_interval = 2
    # whether to write ALL environments gifs to "$RLBC_MODELS/name/gif"
    write_gifs = False
