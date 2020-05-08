from sacred import Ingredient
from sacred.settings import SETTINGS
from configs.bc import train_ingredient

sim2real_ingredient = Ingredient('sim2real', ingredients=[train_ingredient])
SETTINGS.CONFIG.READ_ONLY_CONFIG = False


@sim2real_ingredient.config
def cfg_sim2real():
    # name of the model (will be saved in "$RLBC_DATA/name")
    name = None
    # name of the training dataset (will be loaded from "$RLBC_DATA/name")
    trainset_name = None
    # name of the validation dataset (will be loaded from "$RLBC_DATA/name")
    evalset_name = None
    # number of demos in the training dataset
    max_demos_train = None
    # number of demos in the validation dataset
    max_demos_eval = None
    # maximum number of primitive transformations in the augmentation function
    num_transforms = 8
    # exploration coefficient of MCTS
    exploration_cst = 0.5
    # name of the score to use in MCTS
    score_name = 'median_score'
    # weather to resume MCTS training if a checkpoint exists
    resume = True
    # which dask cluster to use
    cluster = 'local'
    # number of gpus to use. on the local cluster num_gpus > 1 is not allowed
    num_gpus = 1
