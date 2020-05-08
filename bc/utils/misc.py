import torch
import random
import socket
import os
import numpy as np


def get_device(device):
    assert device in (
        'cpu', 'cuda'), 'device {} should be in (cpu, cuda)'.format(device)
    if socket.gethostname() == 'gemini' or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda' if device == 'cuda' else "cpu"
    return device


def seed_exp(seed, device='cuda'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)


def update_arguments(model=None, dataset=None, collect=None, sim2real=None):
    """ User provides the arguments in a user-friendly way.
    This function takes care of converting them to the format used by the repo. """

    def update_model_args(model):
        if model is None:
            return None
        # convert the input_type argument from a string to a tuple
        if isinstance(model['input_type'], (tuple, list)):
            return
        input_type_str2list = {
            'rgb': ('rgb', ),
            'depth': ('depth', ),
            'rgbd': ('depth', 'rgb')
        }
        assert model['input_type'] in input_type_str2list
        model['input_type'] = input_type_str2list[model['input_type']]
        # get the full paths using the user-speicified settings
        model['model_dir'] = os.path.join(os.environ['RLBC_MODELS'], model['name'])
        model.pop('name')
        return model

    def update_dataset_args(dataset):
        if dataset is None:
            return None
        dataset['dataset_dir'] = os.path.join(os.environ['RLBC_DATA'], dataset['name'])
        dataset.pop('name')
        signal_keys_updated = []
        for signal_key in dataset['signal_keys']:
            signal_keys_updated.append(('state', signal_key))
        dataset['signal_keys'] = signal_keys_updated
        return dataset

    def update_collect_args(collect):
        if collect is None:
            return None
        collect['collect_dir'] = os.path.join(os.environ['RLBC_DATA'], collect['folder'])
        collect.pop('folder')
        return collect

    def update_sim2real_args(sim2real):
        if sim2real is None:
            return None
        sim2real['mcts_dir'] = os.path.join(os.environ['RLBC_MODELS'], sim2real['name'])
        sim2real['trainset_dir'] = os.path.join(os.environ['RLBC_DATA'], sim2real['trainset_name'])
        sim2real['evalset_dir'] = os.path.join(os.environ['RLBC_DATA'], sim2real['evalset_name'])
        sim2real.pop('name')
        return sim2real

    model = update_model_args(model)
    dataset = update_dataset_args(dataset)
    collect = update_collect_args(collect)
    sim2real = update_sim2real_args(sim2real)

    return [args for args in (model, dataset, collect, sim2real) if args is not None]
