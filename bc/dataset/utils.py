import lmdb
import os
import shutil
import glob

import pickle as pkl
import numpy as np

from PIL import Image
from io import BytesIO
from tqdm import tqdm

from torch.utils.data import Dataset

from bc.dataset.keys import Keys
from bc.dataset.scalars import Scalars
from bc.utils import Report


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def create_report(collect_args, net_path):
    report_name = os.path.basename(os.path.normpath(net_path))
    if report_name.startswith('seed'):
        report_dir_name = os.path.basename(
            os.path.dirname(os.path.normpath(net_path)))
        report_name = '{}_{}'.format(report_dir_name, report_name)
    report_path = os.path.join('{}/{}.rep'.format(collect_args['collect_dir'],
                                                  report_name))
    if os.path.exists(report_path):
        print('Report {} already exists.'.format(report_path))
        report = Report(path=report_path)
    else:
        report = Report(collect_args['env'], net_path)
        if not os.path.exists(collect_args['collect_dir']):
            os.mkdir(collect_args['collect_dir'])
        report.save(report_path)

    trained_epochs = get_trained_epoch(net_path)
    assert len(trained_epochs
               ) > 0, 'There are no net checkpoints in {}'.format(net_path)
    # check which seeds were already evaluated and which are required to evalute
    seeds_to_be_evaluated = set(range(collect_args['seed'],
                                      collect_args['seed'] + collect_args['episodes']))
    evaluated_epochs = []
    for epoch, seeds_successes in report.entries['epochs'].items():
        seeds_evaluated = set([x[0] for x in seeds_successes])
        if len(seeds_to_be_evaluated.difference(seeds_evaluated)) == 0:
            # all the required seeds were already evaluated
            evaluated_epochs.append(epoch)
    # evaluated_epochs = list(report.get_success_rate().keys())
    unevaluated_epochs = [
        epoch for epoch in trained_epochs if epoch not in evaluated_epochs
    ]
    assert len(unevaluated_epochs
               ) > 0, 'All the epochs of {} are already evaluated'.format(
                   net_path)

    if collect_args['first_epoch'] is None:
        first_epoch = trained_epochs[0]
    else:
        first_epoch = collect_args['first_epoch']
        assert first_epoch in trained_epochs, '{} not in {}'.format(
            first_epoch, trained_epochs)
    if collect_args['last_epoch'] is None:
        last_epoch = trained_epochs[-1] + 1
    else:
        last_epoch = collect_args['last_epoch']

    epochs_range = [
        epoch for epoch in unevaluated_epochs if epoch in list(
            range(first_epoch, last_epoch, collect_args['iter_epoch']))
    ]
    assert len(
        epochs_range
    ) > 0, 'Epochs of {} between {} and {} are already evaluated'.format(
        net_path, first_epoch, last_epoch)
    return report, report_path, epochs_range


def get_trained_epoch(net_path):
    net_epoch_paths = glob.glob(os.path.join(net_path, 'model_[0-9]*.pth'))
    net_epochs = [
        int(path.split('model_')[-1].replace('.pth', ''))
        for path in net_epoch_paths
    ]
    return sorted(net_epochs)


def gather_dataset(dataset_path):
    """
    Write demonstration datasets into one global dataset
    """
    scalars_dic = {}
    # gather trajectory datasets in one dataset
    list_dataset_dir = os.listdir(dataset_path)
    db_global = lmdb.open(dataset_path, 150 * 1024**3, writemap=True)
    keys_dataset = []
    with db_global.begin(write=True) as txn_w:
        for dataset_file in tqdm(list_dataset_dir):
            # locate lmdb dataset folders, write content into a global dataset
            # from the dataset folder, also locate pickle file containing scalar values
            path_lmdb_dataset = os.path.join(dataset_path, dataset_file)
            if os.path.isdir(path_lmdb_dataset)\
                 and os.path.exists(os.path.join(path_lmdb_dataset, 'data.mdb')):
                path_scalars_dataset = os.path.join(dataset_path, dataset_file+'.pkl')
                assert os.path.exists(path_scalars_dataset), 'Missing scalars pickle file: {}'.\
                    format(path_scalars_dataset)
                # update scalars
                scalars_demo = pkl.load(open(path_scalars_dataset, 'rb'))
                scalars_dic.update(scalars_demo)
                # update lmdb dataset
                db_demo = lmdb.open(path_lmdb_dataset, readonly=True)
                with db_demo.begin(write=False) as txn_r:
                    for key, value in txn_r.cursor():
                        keys_dataset.append(key.decode('ascii'))
                        txn_w.put(key, value)
                # remove scalars and lmdb dataset
                del scalars_demo
                db_demo.close()
                shutil.rmtree(path_lmdb_dataset)
                os.remove(path_scalars_dataset)

    # update and save list of keys
    path_keys = os.path.join(dataset_path, '_keys_')
    if '_keys_' not in list_dataset_dir:
        keys = Keys(keys=keys_dataset)
    else:
        keys = pkl.load(open(path_keys, 'rb'))
        keys.update(keys_dataset)
    pkl.dump(keys, open(path_keys, 'wb'))

    # update and save scalars
    path_scalars = os.path.join(dataset_path, 'scalars.pkl')
    if 'scalars.pkl' not in list_dataset_dir:
        scalars = Scalars(scalars_dic)
    else:
        scalars = pkl.load(open(path_scalars, 'rb'))
        scalars.update(scalars_dic)
    pkl.dump(scalars, open(path_scalars, 'wb'))


def compress_images(obs):
    """ Store png compressed image as bytes format. """
    im_keys = [
        k for k in obs.keys() if 'rgb' in k or 'depth' in k or 'mask' in k
    ]
    for im_key in im_keys:
        # mask min value is -1 by default
        # offset mask values so that min value is set to 0
        if 'mask' in im_key:
            assert obs[im_key].min() >= -1
            obs[im_key] += 1
        im = Image.fromarray(obs[im_key])
        im_buf = BytesIO()
        im.save(im_buf, format='PNG')
        obs[im_key] = im_buf.getvalue()


def decompress_images(dic_buffer, channels, camera_idx):
    """ Decompress dictionnary of compressed images. """
    dic_frame = {}
    for channel in channels:
        channel_cam = channel + str(camera_idx)
        bytes_frame = BytesIO(dic_buffer[channel_cam])
        frame = np.asarray(Image.open(bytes_frame)).copy()
        dic_frame[channel] = frame
        # was offset by 1 for compression
        if 'mask' in channel:
            dic_frame[channel] -= 1

    return dic_frame


def process_trajectory(states, actions, steps, seed, compression=False):
    assert len(states) == len(
        actions), 'Length of actions and states not the same'
    assert len(states) == len(steps), 'Length of states and steps not the same'
    if not states:
        return

    im_keys = ['rgb', 'depth', 'mask']
    frames = {}
    scalars = {}

    for i, (state, action, step) in enumerate(zip(states, actions, steps)):
        data_im_keys = [k for k in state.keys() if any(im_key in k for im_key in im_keys)]

        # store jpeg compressed image as bytes format
        if compression:
            compress_images(state)
        for im_key in data_im_keys:
            if (seed, step) not in frames:
                frames[(seed, step)] = {}
            frames[(seed, step)][im_key] = state[im_key]
            state.pop(im_key)

        # store scalar values
        scalars[(seed, step)] = {'action': action, 'state': state}

    return frames, scalars
