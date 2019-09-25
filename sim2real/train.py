import os
import numpy as np

from sacred import Experiment
from dask import distributed

from configs.sim2real import sim2real_ingredient
from bc.utils import misc
from bc.utils import cluster as cluster_utils

from sim2real import utils
from sim2real.mcts import MonteCarloTree

ex = Experiment('sim2real', ingredients=[sim2real_ingredient])


def make_tree(mcts_dir, num_transforms, exploration_cst, score_name, resume):
    tree_path = os.path.join(mcts_dir, 'mcts.pkl')
    if resume and os.path.exists(tree_path):
        print('Load MCTS from {}'.format(tree_path))
    else:
        tree_path = None

    mc_tree = MonteCarloTree(
        max_transforms=num_transforms,
        c_exp=exploration_cst,
        pickle_path=tree_path,
        backprop_score_name=score_name)
    assert mc_tree.max_transforms == num_transforms

    return mc_tree


def get_score(mc_tree, train, dataset, model, trainset_dir,
              evalset_dir, mcts_dir, max_demos_train, max_demos_eval,
              iter_mcts, log_id):
    assert model['device'] == 'cuda', 'model.device should be set to cuda'
    print('Getting the score for MCTS iteration {} (worker {})'.format(iter_mcts, log_id))
    np.random.seed(iter_mcts)
    utils.set_up_worker_logging(log_id, mcts_dir)
    net, optimizer = utils.create_network(
        model['model_dir'], model['epoch'], model['device'], train['learning_rate'])
    train_loader, eval_loader = utils.create_loaders(
        train, dataset, model, trainset_dir, evalset_dir, max_demos_train,
        max_demos_eval)

    aug_path, leaf_node = mc_tree.sample_path()
    utils.set_augmentation_function(train_loader, aug_path)

    policy_scores = utils.evaluate_augmentation(
        net, optimizer, train_loader, eval_loader, iter_mcts, train['epochs'],
        train['eval_first_epoch'], train['eval_interval'])
    return aug_path, policy_scores


def train_mcts(mcts_dir, num_transforms, train, dataset, model, trainset_dir,
               evalset_dir, max_demos_train, max_demos_eval, cluster,
               num_gpus, exploration_cst, score_name, resume, **unused_kwargs):
    # run all the transformations and save them in <mcts_dir>/transformations
    utils.test_transformations(train, dataset, model, trainset_dir, evalset_dir, mcts_dir)

    print('Initializing GPU workers...')
    client = cluster_utils.make_client(
        cluster, 'gpu', num_gpus, mcts_dir, no_nanny=True)

    print('Starting the training...')
    mc_tree = make_tree(mcts_dir, num_transforms, exploration_cst, score_name, resume)
    iter_start = mc_tree.iterations

    print('MCTS of depth {} and exploration constant {}'.format(
        num_transforms, exploration_cst))

    args_worker = [
        train, dataset, model, trainset_dir,
        evalset_dir, mcts_dir, max_demos_train, max_demos_eval
    ]
    args_workers_all = [[arg] * num_gpus for arg in [mc_tree] + args_worker]
    # log_ids will be the list of iter_mcts as well (in the beginning)
    log_ids = list(range(iter_start, iter_start + num_gpus))
    futures = client.map(get_score, *args_workers_all, log_ids, log_ids)
    jobs_queue = distributed.as_completed(futures)

    for iter_mcts, future in enumerate(jobs_queue):
        try:
            path, policy_scores = future.result()
            mc_tree.add_path(path, policy_scores)
            mc_tree.save(mcts_dir)
            print('MCTS iteration {}'.format(iter_start + iter_mcts))
            print('\tpath {}\n\tscore {:.3f}\n\terror {:.3f}cm\n'.format(
                path, policy_scores[score_name],
                (1 - policy_scores[score_name]) * 10))
        except:
            print('WARNING: one of the MCTS worker died, I will gracefully ignore it')

        iter_mcts_future = iter_start + iter_mcts + num_gpus
        log_id = iter_mcts_future % num_gpus
        new_future = client.submit(get_score, mc_tree, *args_worker, iter_mcts_future, log_id)
        jobs_queue.add(new_future)

        if iter_mcts > 0 and iter_mcts % 10 == 0:
            utils.print_mcts_score(mc_tree)


@ex.automain
def main(sim2real, train, dataset, model):
    model, dataset, sim2real = misc.update_arguments(model=model, dataset=dataset, sim2real=sim2real)
    utils.set_up_training(sim2real['mcts_dir'])
    train_mcts(train=train, dataset=dataset, model=model, **sim2real)
