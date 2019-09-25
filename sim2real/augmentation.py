import torch
from sim2real import transformations


class Augmentation:
    def __init__(self, name, path=''):
        self.name = name
        self._policy = None
        self._path = ['identity', 1, 1]

    def sample_sequence(self, img_size):
        path = self.sample_path()
        blocks = self._path2blocks(path)
        policy = transformations.path2policy(blocks, img_size, sampling='fixed')
        self._policy = policy

    def sample_path(self):
        name = self.name
        if name == '':
            path = ['identity', 1, 1]
        elif name == 'kinect1_learned':
            path = [
                'black_noise', 0.03, 2 / 3, 'identity', 1, 1, 'cutout', 1,
                2 / 3, 'identity', 1, 1, 'posterize', 6, 1 / 3, 'sharpness',
                0.5, 2 / 3, 'remove_object', ('table', ), 1, 'white_noise',
                0.08, 1
            ]
        elif name == 'kinect1_handcrafted':
            path = [
                'affine', (0.05, 0.02), 1,
                'scale', 0.1, 1, 'white_noise', 0.04, 2 / 3, 'remove_object',
                ('cage', ), 2 / 3, 'remove_object', ('background', ), 2 / 3,
                'remove_object', ('table', ), 1 / 6, 'black_noise', 0.01, 1
            ]
        elif name == 'kinect2_learned':
            path = [
                'remove_object', ('background', 'cage'), 1 / 3, 'scale', 0.05, 1,
                'affine', (5, 0.04), 2 / 3, 'black_noise', 0.03, 2 / 3
            ]
        elif name == 'random':
            path = self._path
        else:
            raise ValueError('Unkown augmentation path: {}'.format(name))

        return path

    def set_augmentation(self, path):
        assert self.name == 'random'
        self._path = path

    def _path2blocks(self, path):
        blocks = []
        assert len(path) % 3 == 0
        for i in range(int(len(path) // 3)):
            blocks.append(path[i * 3:(i + 1) * 3])
        return blocks

    def __call__(self, frames, masks):
        assert self._policy is not None, 'Call sample_sequence function first'
        frames_aug = torch.zeros_like(frames)
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            frame_aug = self._policy(frame[None, :], mask[None, :])
            frame_aug = torch.clamp(frame_aug, 0, 1)
            frames_aug[i] = frame_aug
        return frames_aug

    @staticmethod
    def crop(frames, output_size, centered_crop):
        return transformations.ImageTransform.cropping(
            frames, output_size, centered=centered_crop)
