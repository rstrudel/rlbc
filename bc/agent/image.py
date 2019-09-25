from bc.agent import utils
from bc.dataset import Frames, Signals
from bc.utils.misc import get_device
from sim2real.augmentation import Augmentation

import random
import torch
import numpy as np


class ImageAgent:
    def __init__(self, path, epoch, max_steps, device='cuda'):
        self._path = path
        self._epoch = epoch
        self._device = get_device(device)
        self.max_steps = max_steps
        self.model = None
        self._augmentation = None

        # queues for image and signal stackory
        self._stack_frames = None
        self._stack_signals = None

        self._count_steps = 0

    def reset(self):
        self._count_steps = 0
        self._stack_frames = None

    def _load_model(self, path, epoch):
        pass

    def set_augmentation(self, augmentation_str):
        assert isinstance(augmentation_str, str)
        self._augmentation = Augmentation(augmentation_str)

    def update_stacks(self, obs, stack_frames, stack_signals):
        channels = self.args['input_type']
        num_channels = Frames.sum_channels(channels)
        num_frames = self.args['num_frames']
        num_signals = self.args[
            'num_signals'] if 'num_signals' in self.args else 0
        dim_signal = self.args['dim_signal'] if 'dim_signal' in self.args else 0

        # sample a transformation if it has not been done before
        assert self._augmentation is not None
        if self._augmentation._policy is None:
            assert 'depth0' in obs
            self._augmentation.sample_sequence(img_size=obs['depth0'].shape)

        # frames dic to tensor
        items = list(obs.items())
        obs_im = {}
        for key, value in items:
            for channel in channels:
                if channel in key:
                    obs_im[channel] = obs[key].copy()
        frames = Frames.dict_to_tensor(
            frames=[obs_im],
            channels=channels,
            num_channels=num_channels,
            augmentation_str='',
            augmentation=self._augmentation)

        # stack of frames
        if stack_frames is None:
            stack_frames = Frames.adjust_shape(frames, num_frames, channels)
        else:
            stack_frames = utils.add_to_stack(frames, stack_frames,
                                              num_channels)

        # stack of signals
        if num_signals > 0:
            obs_sig = {'state': obs}
            signal = Signals.transform(
                obs_sig, [('state', key_sig) for key_sig in self.keys_signal],
                self.model.statistics)
            if stack_signals is None:
                stack_signals = Signals.adjust_shape(signal[None, :],
                                                     num_signals)
                stack_signals = stack_signals.view(num_signals,
                                                   dim_signal).numpy()
            else:
                stack_signals = utils.add_to_stack(signal, stack_signals)

        return stack_frames, stack_signals

    @property
    def args(self):
        return self.model.args

    def seed_exp(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self._device == 'cuda':
            torch.cuda.manual_seed(seed)
        torch.set_num_threads(1)
