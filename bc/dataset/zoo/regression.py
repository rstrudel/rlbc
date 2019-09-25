import torch
from torch.utils.data import Dataset

from bc.dataset import Frames, Signals

# these mask numbers assume the usage of Paris or Grenoble setup (with the cage)
MASKS = {'background': -1, 'robot': 0, 'table': 1, 'cage': 2, 'cube': 3}


class RegressionDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 max_demos,
                 num_cameras,
                 channels,
                 image_augmentation,
                 signal_keys=None,
                 signal_lengths=None,
                 load_masks=True,
                 **unused_kwargs):

        channels = list(channels)
        if image_augmentation or load_masks:
            channels += ['mask']
        self.load_masks = load_masks
        self._num_cameras = num_cameras

        # define frames
        frames = Frames(
            path=dataset_dir,
            channels=channels,
            max_demos=max_demos,
            augmentation=image_augmentation)
        self._frames = frames

        # define signals
        if signal_keys is None:
            signal_keys = [('state', 'target_position')]
        if signal_lengths is None:
            signal_lengths = [2]
        assert len(signal_keys) == len(signal_lengths)
        self._signal_keys, self._signals = [], []
        for signal_key in signal_keys:
            self._signals.append(Signals(dataset_dir, [signal_key]))
            self._signals[-1].keys.set_max_demos(max_demos)
            self._signal_keys.append(signal_key[-1])
        self._signal_lengths = signal_lengths

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        if self.load_masks:
            # check the cube visible part in the image
            # if it is too small, sample another image
            mask = self._frames.get_mask(int(idx))
            cube_visible_pixels = (mask == MASKS['cube']).sum()
            if cube_visible_pixels < 30:
                new_idx = int(torch.randint(len(self), (1, )))
                return self.__getitem__(new_idx)

        # sample camera
        camera_idx = int(torch.randint(high=self._num_cameras, size=(1, )))
        self._frames.set_camera(camera_idx)

        im = self._frames[idx:idx + 1]
        ground_truth = {}
        for signal_key, signal_length, signal in zip(
                self._signal_keys, self._signal_lengths, self._signals):
            ground_truth[signal_key] = torch.tensor(
                signal[idx][0]).float()[:signal_length]
        return im.float(), ground_truth, {}, {}

    def get_statistics(self):
        return self._signals[0].scalars.get_statistics()

    def get_num_demos(self):
        return self._frames.get_num_demos()

    def get_demo_frame_idxs(self, demo_idx):
        idx_beg, idx_end = self._frames.get_demo_indices(demo_idx)
        return slice(idx_beg, idx_end)
