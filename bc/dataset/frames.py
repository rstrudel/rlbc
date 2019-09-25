import torch
import numpy as np

from bc.dataset.dataset_lmdb import DatasetReader
from sim2real.augmentation import Augmentation
from sim2real.transformations import ImageTransform

CHANNEL2SPAN = {'depth': 1, 'rgb': 3, 'mask': 1}


class Frames:
    def __init__(self,
                 path,
                 channels=('depth', ),
                 limit='',
                 max_demos=None,
                 augmentation='',
                 output_size=224):
        """
        Args:
        path: path of the dataset
        channels: channels to load - rgb, depth, mask
        limit: critera to limit first and last idx of getitem
        based of the set index of reference
        mode: type of array returned by frames
        max_demos: maximum number of demos to load frames from
        augmentation: augmentation to be applied to frames
        """
        assert isinstance(channels, (list, tuple))
        for channel in channels:
            assert channel in CHANNEL2SPAN.keys()
        self.channels = tuple(sorted(channels))
        self._dataset = DatasetReader(path, self.channels)
        self.infos = self._dataset.infos
        self.keys = self._dataset.keys
        self._limit = limit
        self.keys.set_query_limit(limit)
        assert isinstance(augmentation, str)
        self._augmentation = Augmentation(augmentation)
        self._im_keys = ['rgb', 'depth', 'mask']
        self._output_size = output_size
        channels_no_mask = [c for c in channels if c != 'mask']
        self._num_channels = self.sum_channels(channels_no_mask)

        if max_demos is not None:
            self.keys.set_max_demos(max_demos)

        self._seed_augmentation = None

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        db = self._dataset
        channels = self.channels
        num_channels = self._num_channels
        augmentation = self._augmentation
        output_size = self._output_size

        idx_min, idx_max = self._user2dbidx(idx)
        frames = db[idx_min:idx_max]

        # convert dic of observation to homogeneous array
        frames, masks = self.dict_cat(frames, channels, num_channels)
        # convert to tensor
        frames = torch.tensor(frames)
        masks = torch.tensor(masks)

        # map array from [0, 255] to [0, 1]
        frames = self.unit_range(frames)

        # augment images, works only for depth images
        # each sample_sequence generates a new augmentation sequence
        # the transformation is consistent across frames
        img_size = frames.size()[1:]
        augmentation.sample_sequence(img_size=img_size)
        if 'rgb' not in channels:
            frames = augmentation(frames, masks)

        # crop the image to fixed size
        # if name is not '', do a random crop else do a center crop
        centered_crop = augmentation.name == ''
        params_crop = ImageTransform.sample_params(
            name_transformation='cropping',
            magn=(output_size, output_size),
            img_size=img_size)
        frames = Augmentation.crop(frames, params_crop, centered_crop)

        # maps array from [0, 1] to [-1, 1]
        frames = self.normalize(frames)

        return frames

    @staticmethod
    def dict_cat(frames, channels, num_channels):
        """
        Concatenate dictionnary of frames split by channels into an array
        frames: list of dictionnary containing depth, rgb, mask keys
        channels: channels to be concatenated
        num_channels: number of channels per frame
        """
        channels = [c for c in channels if 'mask' not in c]

        size = frames[0][channels[0]].shape[0]
        stack_frames = np.zeros((num_channels * len(frames), size, size),
                                dtype=np.uint8)
        stack_masks = np.zeros((len(frames), size, size), dtype=int)
        idx_stack = 0
        for idx_frame, frame in enumerate(frames):
            for channel in channels:
                channel_span = CHANNEL2SPAN[channel]
                channel_im = frame[channel]
                if channel_span > 1:
                    # put the last dimension of rgb image (numpy way) to the first one (torch way)
                    channel_im = np.swapaxes(
                        np.swapaxes(channel_im, 2, 1), 1, 0)
                stack_frames[idx_stack:idx_stack + channel_span] = channel_im
                idx_stack += channel_span
            if 'mask' in frame:
                stack_masks[idx_frame] = frame['mask']

        return stack_frames, stack_masks

    def set_augmentation(self, path):
        self._augmentation.set_augmentation(path)

    @staticmethod
    def unit_range(frames):
        """
        frames: uint8 array or torch tensor in [0, 255]
        return: float array in [0, 1]
        """
        if type(frames) is np.ndarray:
            unit_frames = frames.astype(float)
        elif type(frames) is torch.Tensor:
            unit_frames = frames.float()
        # inplace operations
        unit_frames /= 255
        return unit_frames

    @staticmethod
    def normalize(frames):
        """
        frames: uint8 array in [0, 1]
        return: float array in [-1, 1]
        """
        # inplace operations
        frames -= 0.5
        frames /= 0.5
        return frames

    @staticmethod
    def dict_to_tensor(frames,
                       channels,
                       num_channels,
                       output_size=(224, 224),
                       augmentation_str='',
                       augmentation=None):
        """
        Convert dictionnary of observation to normalized tensor,
        augment the images on the way if an augmentation is passed
        frames: dictionnary of observations (mime, mujoco, ...)
        return: torch tensor in [-1, 1]
        """
        frames, masks = Frames.dict_cat(frames, channels, num_channels)
        frames = torch.tensor(frames)
        masks = torch.tensor(masks)
        frames = Frames.unit_range(frames)
        if augmentation is None:
            augmentation = Augmentation(augmentation_str)
            augmentation.sample_sequence(frames.size()[1:])
        if 'rgb' not in channels:
            frames = augmentation(frames, masks)
        # crop is centered if there are not augmentation set
        centered_crop = augmentation_str == ''
        img_size = frames.size()[1:]
        params_crop = ImageTransform.sample_params(
            name_transformation='cropping',
            magn=output_size,
            img_size=img_size)
        frames = Augmentation.crop(frames, params_crop, centered_crop)
        frames = Frames.normalize(frames)
        return frames

    @staticmethod
    def adjust_shape(x, num_frames, channels):
        """
        x: torch tensor with potentially missing num_frames
        return: array where first frame is repeated to match num_frames size
        """
        assert isinstance(channels, (tuple, list))
        channels2num = {'depth': 1, 'rgb': 3, 'mask': 0}
        num_channels = 0
        for channel in channels:
            num_channels += channels2num[channel]

        x_chan = x.shape[0]
        assert x_chan % num_channels == 0
        if x_chan != num_frames * num_channels:
            missing_frames = int(num_frames - x_chan / num_channels)
            m = x[:num_channels].repeat(missing_frames, 1, 1)
            x = torch.cat((m, x), dim=0)

        return x

    @staticmethod
    def sum_channels(channels):
        """Sum of the span of channels"""
        num_channels = 0
        for channel in channels:
            num_channels += CHANNEL2SPAN[channel]
        return num_channels

    def set_camera(self, camera_idx):
        self._dataset.set_camera(camera_idx)

    def get_num_demos(self):
        return self.keys.get_num_demos()

    def get_demo_indices(self, demo_idx):
        """Return (t, t+T) if demo starts at timestep"""
        return self.keys.get_demo_indices(demo_idx)

    def get_mask(self, idx):
        assert 'mask' in self.channels
        assert isinstance(idx, int)
        frames = self._dataset[idx]
        return frames['mask']

    def _user2dbidx(self, idx):
        """convert user index to idx_min, idx_max within dataset range"""
        keys = self.keys
        if isinstance(idx, slice):
            start, end, step = idx.indices(len(self))
            # step bigger than 1 not handled
            assert step == 1
            # make sure all the frames come from the same demo
            idx_min, idx_max = keys.get_idx_min_max(start, end)
        elif isinstance(idx, int):
            idx_min, idx_max = idx, idx + 1
        else:
            raise TypeError('{} is an unvalid index type.'.format(type(idx)))

        return idx_min, idx_max
