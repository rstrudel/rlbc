import lmdb
from io import BytesIO
import pickle as pkl
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import datetime

from bc.dataset.keys import Keys
from bc.dataset import utils


class DatasetReader:
    def __init__(self, path, channels):
        self._path = path
        self._env = None
        self._txn = None
        self._channel = None
        self.keys = None
        self.infos = json.load(open(os.path.join(path, 'info.json')))

        self._channels = []

        self._camera_idx = 0
        self.set_channels(channels)
        self._load_db()

    def __len__(self):
        return len(self.keys)

    # return stacked frames of shape NxHxW
    # for each color channel where N is the slice size
    def __getitem__(self, idx):
        keys = self.keys[idx]
        if isinstance(idx, int):
            keys = [keys]
            stack_frames = self._get_frames(keys)
            stack_frames = stack_frames[0]
        else:
            stack_frames = self._get_frames(keys)

        return stack_frames

    def _get_frames(self, keys):
        stack_frames = []
        for key in keys:
            if key not in self.keys:
                seed, step = self.keys.key2demo(key)
                assert ValueError('Demo {} Step {} is not in\
                the dataset.'.format(seed, step))
            else:
                key = key.encode('ascii')
                buffer = self._txn.get(key)
                dic_frame = self._buffer2frames(buffer)
                stack_frames.append(dic_frame)

        return stack_frames

    def _buffer2frames(self, buffer):
        dic_buffer = pkl.load(BytesIO(buffer))
        dic_frame = utils.decompress_images(dic_buffer, self._channels,
                                            self._camera_idx)
        return dic_frame

    def set_channels(self, channels):
        assert isinstance(channels, (tuple, list))
        for channel in channels:
            assert channel in ['depth', 'rgb', 'mask']
        self._channels = channels

    def set_camera(self, camera_idx):
        self._camera_idx = camera_idx

    # open lmdb environment and transaction
    # load keys from cache
    def _load_db(self):
        path = self._path

        self._env = lmdb.open(
            self._path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self._txn = self._env.begin(write=False)

        cache_file = os.path.join(path, '_keys_')
        if os.path.isfile(cache_file):
            self.keys = pkl.load(open(cache_file, 'rb'))
        else:
            print('Loading dataset keys...')
            with self._env.begin(write=False) as txn:
                keys = [key.decode('ascii') for key, _ in tqdm(txn.cursor())]
            self.keys = Keys(keys=keys)
            pkl.dump(self.keys, open(cache_file, 'wb'))

        if not self.keys:
            raise ValueError('Empty dataset.')


class DatasetWriter(object):
    def __init__(self, path, env_name, rewrite=False):
        self._map_size = 200 * 1024**3
        self._path = path
        self._db = None
        self._dic_traj_dataset = {}
        if not (os.path.exists(path)):
            os.mkdir(path)
        self._keys = []
        self._keys_file = os.path.join(path, '_keys_')

        info = dict(
            env_name=env_name,
            timestamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
        info_path = os.path.join(path, 'info.json')
        if not os.path.exists(info_path):
            with open(info_path, 'w') as f:
                json.dump(info, f)
        else:
            if not rewrite:
                raise NameError("Dataset {} already exists.".format(path))

    # do not initialize during __init__ to avoid pickling error when using MPI
    def init(self):
        self._db = lmdb.open(self._path, map_size=self._map_size)

    def close(self):
        self.write_keys()
        if self._db is not None:
            self._db.close()
            self._db = None

    def write_keys(self):
        keys = Keys(self._keys)
        pkl.dump(keys, open(self._keys_file, 'wb'))

    # jpeg_compression=False, in dataset/collector.py,
    # the images are already compressed by the worker to save RAM,
    # thus s[im_key] is already a set of bytes after image compression
    def write_frames(self, frames):
        if self._db is None:
            self.init()

        with self._db.begin(write=True) as txn:
            for key, value in frames.items():
                seed, step = key
                key = 'S{:06}/T{:06}'.format(seed, step)
                self._keys.append(key)
                buffer = BytesIO()
                pkl.dump(value, buffer)
                txn.put(key.encode('ascii'), buffer.getvalue())
