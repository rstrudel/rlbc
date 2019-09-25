import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy
from termcolor import colored

from bc.model import log
from bc.model import utils as bc_utils
from sim2real import transformations


def set_up_training(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    log.init_writers(save_dir)


def test_transformations(train, dataset, model, trainset_dir, evalset_dir, save_dir):
    train_loader, _ = create_loaders(
        train, dataset, model, trainset_dir, evalset_dir, None, None)
    transformations.test(train_loader.dataset, save_dir)


def set_up_worker_logging(log_id, save_dir):
    # master process does not log anything anymore
    log.init_writers(os.path.join(save_dir, 'worker{}'.format(log_id)))


def create_network(model_dir, epoch, device, learning_rate):
    net, _ = bc_utils.load_model(model_dir, epoch, device)
    optimizer = bc_utils.create_optimizer(net, None, learning_rate)
    return net, optimizer


def create_loaders(train, dataset, model, trainset_dir, evalset_dir,
                   max_demos_train, max_demos_eval):
    # create the train dataset
    trainset_dict, evalset_dict = deepcopy(dataset), deepcopy(dataset)
    trainset_dict['image_augmentation'] = 'random'
    trainset_dict['dataset_dir'] = trainset_dir
    if max_demos_train is not None:
        trainset_dict['max_demos'] = max_demos_train
    train_loader = bc_utils.make_loader(
        dataset=trainset_dict,
        model=model,
        eval_proportion=0,
        batch_size=train['batch_size'],
        workers=train['workers'])[0]

    # create the eval dataset
    # evalset should be a real dataset, no masks are available
    evalset_dict['image_augmentation'] = ''
    evalset_dict['load_masks'] = False
    evalset_dict['dataset_dir'] = evalset_dir
    # real dataset will (always) have 1 camera only
    evalset_dict['num_cameras'] = 1
    if max_demos_eval is not None:
        evalset_dict['max_demos'] = max_demos_eval
    eval_loader = bc_utils.make_loader(
        dataset=evalset_dict,
        model=model,
        eval_proportion=0,
        batch_size=train['batch_size'],
        workers=train['workers'])[0]
    return train_loader, eval_loader


def train_network(net, optimizer, loader):
    # set the scalars means and stds for human readable errors
    net.statistics['train'] = loader.dataset.dataset.get_statistics()
    net.statistics['gt'] = net.statistics['train']
    train_losses = {}
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        batch_loss = net.compute_loss(*batch, eval=False)
        optimizer.step()
        bc_utils.append_losses(train_losses, batch_loss)
    return train_losses


def evaluate_augmentation(net, optimizer, train_loader, eval_loader, iter_mcts,
                          num_epochs_train, eval_first_epoch, eval_interval):
    eval_losses_epochs = []
    for epoch_net in tqdm(range(num_epochs_train)):
        train_losses = train_network(net, optimizer, train_loader)
        log.mean(
            train_losses,
            iter_mcts * num_epochs_train + epoch_net,
            stage='train')
        if epoch_net >= eval_first_epoch and epoch_net % eval_interval == 0:
            # set the scalars means and stds for human readable errors
            net.statistics['gt'] = eval_loader.dataset.dataset.get_statistics()
            eval_losses = bc_utils.run_evaluation(net, eval_loader)
            eval_losses_epochs.append(eval_losses)
            log.mean(
                eval_losses,
                iter_mcts * num_epochs_train + epoch_net,
                stage='eval')
    eval_scores = compute_mcts_score(eval_losses_epochs)
    return eval_scores


def error2score(error):
    # set the score range in [0, 1]
    return 1 - min(error / 10, 1)


def compute_mcts_score(losses_list):
    # losses list contain <num_epochs_eval> lists
    # each list contains a dict with loss_key and loss_value_list for each batch
    # we want to get get an error in cm
    loss_key = 'target_position_error'
    error_epochs_avg = np.array(
        [np.mean(loss[loss_key]) for loss in losses_list])
    assert error_epochs_avg.size != 0
    # min_error_epochs = error_epochs_avg.min()
    scores = {}
    scores['median_score'] = error2score(np.median(error_epochs_avg))
    scores['max_score'] = error2score(np.min(error_epochs_avg))
    return scores


def set_augmentation_function(loader, aug_path):
    dataset = loader.dataset.dataset
    dataset._frames.set_augmentation(aug_path)


def print_mcts_score(mc_tree):
    history = mc_tree._history
    mc_tree.root
    paths, scores = [], []
    for epoch in history:
        paths.append(epoch[0])
        scores.append(epoch[1]['max_score'])

    best_score, best_path = 0, None
    best_scores = []
    for it, (score, path) in enumerate(zip(scores, paths)):
        if score > best_score:
            best_score, best_path = score, path
            print('[iteration {}] New best path is {}, score {}'.format(it, path, score))
        best_scores.append(best_score)

    print(colored('\nAfter {} iterations of MCTS, the best policy is {} with the error {} cm'.format(
        len(scores), best_path, (1 - best_score) * 10), 'green'))
