import torch
import torch.optim as optim

import os
import numpy as np
import json
import glob

from configs.bc import train_ingredient
from bc.dataset import ImitationDataset, RegressionDataset
from bc.dataset.utils import Subset
from bc.model import FlatPolicy, SkillsPolicy, FilmPolicy, Regression


@train_ingredient.capture
def make_loader(model, dataset, batch_size, workers, eval_proportion):
    """ Create a dataset (train or train+eval) loader(s). """
    dataset_extra_args = dict(
        num_frames=model['num_frames'],
        channels=model['input_type'],
        action_space=model['action_space'],
        steps_action=model['steps_action'],
        num_signals=model['num_signals'],
        num_skills=model['num_skills'])
    if model['mode'] == 'regression':
        dataset_class = RegressionDataset
    else:
        dataset_class = ImitationDataset

    im_dataset = dataset_class(**dataset, **dataset_extra_args)
    train_dataset, eval_dataset = split_dataset(im_dataset, eval_proportion)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        num_workers=workers,
        shuffle=True,
        drop_last=False)
    if eval_dataset is not None:
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size,
            num_workers=workers,
            shuffle=True,
            drop_last=False)
    else:
        eval_loader = None

    env_name = im_dataset._frames.infos['env_name']
    return (train_loader, eval_loader, env_name, im_dataset.get_statistics())


@train_ingredient.capture
def make_net(model,
             dataset,
             learning_rate,
             epochs,
             lam_grip,
             lam_master,
             env_name,
             statistics):
    """ Create a network using the given arguments. """
    model_dir = model['model_dir']
    epoch = 0
    if os.path.exists(model_dir) and model['resume'] and find_max_epoch(
            model_dir) is not None:
        epoch = find_max_epoch(model_dir)
        print('Load network{} from the savedir {}'.format(
            ' and optimizer' if model['load_optimizer'] else '', model_dir))
        print('Current epoch is {}'.format(epoch))
        net, optimizer_state_dict = load_model(model_dir, epoch, model['device'])
        if not model['load_optimizer']:
            optimizer_state_dict = None
    else:
        network_args = dict(
            env_name=env_name,
            statistics=statistics,
            lam_grip=lam_grip,
            lam_master=lam_master,
            **model,
            **dataset)
        network_class = mode_to_network_class(model['mode'])
        net = network_class(**network_args)
        optimizer_state_dict = None

    optimizer = create_optimizer(net, optimizer_state_dict, learning_rate)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    print('Model directory: {}\nModel arguments: {}'.format(model_dir, net.args))

    return net, optimizer, epoch, model_dir


@train_ingredient.capture
def write_info(model, dataset, batch_size, learning_rate, epochs):
    """ Save some information about the trainin to a json. """
    file_info = open(os.path.join(model['model_dir'], 'info.json'), 'w')
    dic_info = {
        'cameras': dataset['num_cameras'],
        'dataset': dataset['dataset_dir'],
        'max_demos': dataset['max_demos'],
        'resume': model['resume'],
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }
    file_info.write(json.dumps(dic_info, cls=NumpyEncoder))
    file_info.close()


def load_model(path, epoch, device=None):
    """ Load the model from the specified path. """
    assert path is not None and path != ''

    model_name = 'model_{}.pth'.format(epoch)
    model_path = os.path.join(path, model_name)
    assert os.path.exists(model_path), 'model at {} does not exist'.format(
        model_path)

    if str(device) == 'cpu':
        loaded_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage)
    else:
        loaded_dict = torch.load(model_path)

    model = loaded_dict['model']
    model.net = model.net.module
    model.to(device)

    return model, loaded_dict['optimizer_state_dict']


def save_model(path, epoch, model, optimizer):
    """ Save the model to the specified path. """
    assert os.path.exists(path)

    # cannot pickle model.device, set it to None before saving
    device = model.device
    model.device = None
    dict_model = dict(
        model=model,
        args=model.args,
        statistics=model.statistics,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch)

    model_name = 'model_{}.pth'.format(epoch)
    model_path = os.path.join(path, model_name)
    torch.save(dict_model, model_path)

    model.device = device

    # create symlink to last saved model
    model_symlink = os.path.join(path, 'model_current.pth')
    if os.path.islink(model_symlink):
        os.unlink(model_symlink)
    os.symlink(model_path, model_symlink)


def create_optimizer(net, optimizer_state_dict, learning_rate, device='cuda'):
    """ Create (or load) the Adam optimizer for network training """
    # define optimizer
    optimizer = optim.Adam([{
        'params': net.net.parameters(),
        'initial_lr': learning_rate
    }])
    # load optimizer checkpoint if available
    if optimizer_state_dict is not None:
        target_device = 'cpu' if device == 'cpu' else 'cuda'
        # load the optimizer weights
        optimizer.load_state_dict(optimizer_state_dict)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = getattr(v, target_device)()
    return optimizer


def find_max_epoch(path):
    """ Find the last checkpoint of a trained network. """
    checkpoints = glob.glob(os.path.join(path, 'model_*.pth'))
    if len(checkpoints) == 0:
        return None
    max_epoch = 0
    for checkpoint_path in checkpoints:
        epoch_str = checkpoint_path.split('/model_')[-1].replace('.pth', '')
        try:
            max_epoch = max(int(epoch_str), max_epoch)
        except:
            pass
    return max_epoch


def append_losses(losses_all, losses_batch):
    """ Losses are stored as a dictionary of lists. Add the batch loss to the list. """
    for loss_name, loss_value in losses_batch.items():
        if loss_name in losses_all:
            losses_all[loss_name].append(loss_value.item())
        else:
            losses_all[loss_name] = [loss_value.item()]


def run_evaluation(net, loader):
    """ Run the evaluation of the network. """
    net.net.eval()
    losses_eval = {}
    for i, batch in enumerate(loader):
        with torch.no_grad():
            losses_batch = net.compute_loss(*batch, eval=True)
            append_losses(losses_eval, losses_batch)
    net.net.train()
    return losses_eval


def split_dataset(dataset, eval_proportion, shuffle=False):
    """ Split a dataset into non-overlapping sets (train and eval). """
    split_sizes = [1. - eval_proportion, eval_proportion]
    split_frames = []
    split_demos = []
    num_demos = dataset.get_num_demos()
    split_num_demos = [int(fraction * num_demos) for fraction in split_sizes]
    split_num_demos[0] += num_demos - sum(split_num_demos)
    num_instances = len(dataset)
    demos = list(range(num_demos))
    if shuffle:
        np.random.shuffle(demos)
    start_idx = 0
    for split_idx in range(len(split_sizes)):
        if split_sizes[split_idx] == 0:
            split_frames.append(None)
            continue
        split_frames.append([])
        split_demos.append(range(start_idx, start_idx + split_num_demos[split_idx]))
        for demo_idx in split_demos[split_idx]:
            demo_slice = dataset.get_demo_frame_idxs(demos[demo_idx])
            split_frames[split_idx].extend(
                list(range(demo_slice.start, demo_slice.stop)))
        start_idx += split_num_demos[split_idx]
        # Check if the split indices are unique
        assert len(set(split_frames[split_idx])) == len(split_frames[split_idx])

    if eval_proportion > 0:
        # Check that splits do not intersect
        for split_idx in range(len(split_frames)):
            for split_idx2 in range(split_idx + 1, len(split_frames)):
                assert len(set(split_frames[split_idx]).intersection(split_frames[split_idx2])) == 0
        assert sum([len(s) for s in split_frames]) == num_instances

    split_datasets = [Subset(dataset, split) if split is not None else None for split in split_frames]
    return split_datasets


def mode_to_network_class(mode):
    """ Map a string name of the network to a class. """
    if mode in ['flat', 'signals']:
        return FlatPolicy
    elif 'skills' in mode:
        return SkillsPolicy
    elif 'film' in mode:
        return FilmPolicy
    elif mode == 'regression':
        return Regression
    else:
        raise NotImplementedError('mode {} is unknown'.format(mode))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
