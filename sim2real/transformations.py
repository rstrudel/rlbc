import torch
import os
import shutil
import functools
import numpy as np

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as F

MASKS = {'background': -1, 'robot': 0, 'table': 1, 'cage': 2}
PROBS = [1 / 3, 2 / 3, 1]


class ImageTransform:
    """Image transformation attributes: name, magnitude, probability"""

    # static attribute
    transforms = {
        'identity': {
            'magnitude': [1],
            'proba': [1],
        },
        'white_noise': {
            'magnitude': [0.04, 0.08],
            'proba': PROBS,
        },
        'black_noise': {
            'magnitude': [0.01, 0.03],
            'proba': PROBS,
        },
        'edge_noise': {
            'magnitude': [(['table', 'robot'], 2), (['table', 'robot'], 3),
                          (['table', 'robot'], 4)],
            'proba':
            PROBS,
        },
        'remove_object': {
            'magnitude': [('table', ), ('background', 'cage'),
                          ('background', 'cage', 'table')],
            'proba':
            PROBS,
        },
        'scale': {
            'magnitude': [0.03, 0.05],
            'proba': PROBS,
        },
        'sharpness': {
            'magnitude': [0.5, 1.],
            'proba': PROBS,
        },
        'cutout': {
            'magnitude': [1, 3, 5],
            'proba': PROBS
        },
        'invert': {
            'magnitude': [1],
            'proba': PROBS
        },
        'posterize': {
            'magnitude': [5, 6],
            'proba': PROBS,
        },
        'affine': {
            'magnitude': [(5, 0.04), (9, 0.07)],
            'proba': PROBS,
        },
        'contrast': {
            'magnitude': [0.5, 2.],
            'proba': PROBS,
        },
        'autocontrast': {
            'magnitude': [1],
            'proba': PROBS
        },
        'equalize': {
            'magnitude': [1],
            'proba': PROBS
        },
    }

    def __init__(self, transform_name):
        self.attribute_child = {
            'name': 'magnitude',
            'magnitude': 'proba',
            'proba': 'name'
        }
        assert transform_name in self.transforms

        self.name = transform_name
        self.transform = self.transforms[self.name]

    def attribute2range(self, attribute):
        assert attribute in ['name', 'magnitude', 'proba'], attribute
        if attribute == 'name':
            return sorted(self.transforms)
        else:
            return self.transform[attribute]

    def attribute2child(self, attribute):
        return self.attribute_child[attribute]

    # transformations is a list of (operation, magnitude_range, probability_range)
    # probability is always in [0, 1]
    # frame and mask are tensors with the shapes (1 x 224 x 224)
    @staticmethod
    def sample_params(name_transformation, magn, img_size):
        assert hasattr(ImageTransform, name_transformation)
        params = {}
        if name_transformation == 'affine':
            degree, translate = magn
            ret = transforms.RandomAffine.get_params(
                degrees=(-degree, degree),
                translate=(translate, translate),
                scale_ranges=None,
                shears=None,
                img_size=img_size)
            params['affine'] = ret
        elif name_transformation == 'white_noise':
            params['amplitude'] = torch.rand(1)
        elif name_transformation == 'scale':
            a_min, a_max = 1 - magn, 1 + magn
            b_min, b_max = -magn, magn
            alpha, beta = torch.rand((2,))
            a = (1 - alpha) * a_min + alpha * a_max
            b = (1 - beta) * b_min + beta * b_max
            params['a'], params['b'] = a, b
        elif name_transformation == 'remove_object':
            masks_list = magn
            remove_tosses = torch.rand(len(masks_list))
            masks_int = []
            for mask_str, toss in zip(masks_list, remove_tosses):
                if float(toss) < 1 / 2:
                    masks_int.append(int(MASKS[str(mask_str)]))
            params['objects_to_remove'] = masks_int
        elif name_transformation == 'cropping':
            th, tw = magn
            params['output_size'] = th, tw
            w, h = img_size
            # magn is equal to output size
            if w == tw and h == th:
                params['cropping'] = 0, 0, h, w
            else:
                i = np.random.randint(0, h - th)
                j = np.random.randint(0, w - tw)
                params['cropping'] = i, j, th, tw
        return params

    @staticmethod
    def identity(frame, mask, magn, unused_params):
        # identity, no transformation applied to frame
        return frame, mask

    @staticmethod
    def white_noise(frame, unused_mask, unused_magn, params):
        # magnitude of 0.04 should do this:
        # frame += 0.04 * torch.rand(1) * 2 * (torch.rand(img.shape) - 0.5)
        frame = frame + params['amplitude'] * 2 * (torch.rand(frame.shape) - 0.5)
        return frame, unused_mask

    @staticmethod
    def black_noise(frame, unused_mask, magn, unused_params):
        # randomly put pixels to 1
        # magnitude of 0.01 should do this:
        # mask_bernoulli = torch.bernoulli(0.01 * torch.ones_like(frame))
        # frame[mask_bernoulli == 1] = 1
        mask_bernoulli = torch.bernoulli(magn * torch.ones_like(frame))
        frame[mask_bernoulli == 1] = 1
        return frame, unused_mask

    @staticmethod
    def scale(frame, unused_mask, unused_magn, params):
        # magnitude of 0.1 should do this:
        # frame *= 0.9 + 0.2 * torch.rand(1)
        # frame = (1+2*(torch.rand(1)-0.5) * magn) * frame  # + magn * torch.rand(1)
        # frame = a * frame + b
        frame = params['a'] * frame
        return frame, unused_mask

    @staticmethod
    def remove_object(frame, mask, unused_magn, params):
        # set the pixels of the frame with the mask == magn to 1 (max value)
        for mask_int in params['objects_to_remove']:
            frame[mask == mask_int] = 1
        return frame, mask

    @staticmethod
    def cutout(frame, unused_mask, magn, unused_params):
        # cut out {0, ..., magn} random rectangles from the image
        num_times = torch.randint(int(magn) + 1, (1, )).type(torch.int)
        for _ in range(num_times):
            size = int(frame.shape[1])
            x_pos, y_pos = torch.randint(size, (2, )).type(torch.int)
            x_size, y_size = torch.randint(48, (2, )).type(torch.int)
            x_size = min(x_pos + x_size, size) - x_pos
            y_size = min(y_pos + y_size, size) - y_pos
            if x_size > 0 and y_size > 0:
                frame[0, y_pos:y_pos + y_size, x_pos:x_pos +
                      x_size] = torch.rand(1)
        return frame, unused_mask

    @staticmethod
    def edge_noise(frame, mask, magn, unused_params):
        # randomly put pixels on object edges to 1
        # magn defines the object on which to apply edge noise
        # magn_noise defines the max side of the removed rectangles at the edge
        # magn_noise = 4
        masks_list, magn_noise = magn
        pixels_edge = np.zeros((0, 2))
        for mask_str in masks_list:
            mask_noise = MASKS[mask_str]
            size = frame.shape[1]
            mask_object = Image.fromarray(
                ((mask == mask_noise)[0] * 255).numpy().astype(np.uint8))
            im_edge = np.asarray(mask_object.filter(ImageFilter.FIND_EDGES))
            pixels_edge = np.vstack((pixels_edge,
                                     np.array(np.where(im_edge == 255)).T))
            num_pixels_edge = pixels_edge.shape[0]
            sizes_noise = np.random.randint(0, magn_noise,
                                            (num_pixels_edge, 2))
            yx_mins = np.clip(pixels_edge - sizes_noise / 2, 0,
                              size).astype(np.uint8)
            yx_maxs = np.clip(pixels_edge + sizes_noise / 2, 0,
                              size).astype(np.uint8)
        for yx_min, yx_max in zip(yx_mins, yx_maxs):
            y_min, x_min = yx_min
            y_max, x_max = yx_max
            if y_max > y_min and x_max > x_min:
                frame[0, y_min:y_max, x_min:x_max] = 1
        return frame, mask

    @staticmethod
    def black_image(frame, unused_mask, magn, unused_params):
        # dummy
        frame = torch.zeros_like(frame)
        return frame, unused_mask

    @staticmethod
    def _op_pil(op_name, frame, magn=None):
        # apply PIL.ImageOps.op_name to the frame with the given magnitude
        frame_numpy = (frame * 255).cpu().numpy().astype(np.uint8)[0]
        op_pil = getattr(ImageOps, op_name)
        frame_pil = (Image.fromarray(frame_numpy), magn)
        if magn is not None:
            frame_pil = op_pil(Image.fromarray(frame_numpy), magn)
        else:
            frame_pil = op_pil(Image.fromarray(frame_numpy))
            frame = torch.tensor(
                np.array(frame_pil)).type_as(frame)[None] / 255
        return frame

    @staticmethod
    def _enhance_pil(enhance_name, frame, magn):
        # apply PIL.ImageEnhance.enhance_name to the frame with the given magnitude
        # final magn should be randomized in [1, 1 + magn]
        magn_min, magn_max = 1, 1 + magn
        magn = (magn_max - magn_min) * torch.rand((1, )) + magn_min
        frame_numpy = (frame * 255).cpu().numpy().astype(np.uint8)[0]
        pil_op = getattr(ImageEnhance,
                         enhance_name)(Image.fromarray(frame_numpy))
        frame_pil = pil_op.enhance(magn)
        frame = torch.tensor(np.array(frame_pil)).type_as(frame)[None] / 255
        return frame

    @staticmethod
    def posterize(frame, unused_mask, magn, unused_params):
        # binarize values of frame into magn bits
        magn = torch.randint(
            low=int(magn), high=9, size=(1, )).type(torch.int).item()
        return ImageTransform._op_pil('posterize', frame, magn), unused_mask

    @staticmethod
    def autocontrast(frame, unused_mask, unused_magn, unused_params):
        return ImageTransform._op_pil('autocontrast', frame), unused_mask

    @staticmethod
    def invert(frame, unused_mask, unused_magn, unused_params):
        return 1 - frame, unused_mask

    @staticmethod
    def equalize(frame, unused_mask, unused_magn, unused_params):
        return ImageTransform._op_pil('equalize', frame), unused_mask

    @staticmethod
    def sharpness(frame, unused_mask, magn, unused_params):
        # change the sharpness of the image
        # 0 is a strong blur, 1 is the original image, 2 is a strong sharpness
        return ImageTransform._enhance_pil('Sharpness', frame, magn), unused_mask

    @staticmethod
    def brightness(frame, unused_mask, magn, unused_params):
        # change the brightness of the image
        # 0 is gray image, 1 is the original image, 2 is a strong brightness
        return ImageTransform._enhance_pil('Brightness', frame, magn), unused_mask

    @staticmethod
    def contrast(frame, unused_mask, magn, unused_params):
        # change the contrast of the image
        # 0 is a gray image, 1 is the original image, 2 is a strong contrast
        return ImageTransform._enhance_pil('Contrast', frame, magn), unused_mask

    @staticmethod
    def affine(frame, mask, unused_magn, params):
        # random affine translation and rotation of the frame and mask
        # get parameters of the transform first and apply it
        # to ensure same transform is applied to frame and mask
        frame = F.to_pil_image(frame)
        # mask = F.to_pil_image(mask+1)
        frame = F.affine(frame, *params['affine'], resample=False, fillcolor=255)
        # mask = F.affine(mask, *ret, resample=False, fillcolor=255)
        frame = F.to_tensor(frame)
        return frame, mask

    @staticmethod
    def cropping(frames, params, centered=False, rgb=False):
        # random crop of the frame and mask
        # size is the final size of the crop
        w, h = params['output_size']
        frames_crop = torch.zeros(len(frames), w, h)
        for i, frame in enumerate(frames):
            frame = F.to_pil_image(frame[None, :])
            if centered:
                frame = F.center_crop(frame, (w, h))
            else:
                frame = F.crop(frame, *params['cropping'])
            frame = F.to_tensor(frame)
            frames_crop[i] = frame
        return frames_crop

    @staticmethod
    def resize(frame, size):
        """
        frames: tensor in [0, 1] of size (h, w)
        return: tensor in [0, 1] of size (size, size)
        """
        frame = F.to_pil_image(frame)
        frame = F.resize(frame, size)
        frame = F.to_tensor(frame)
        return frame


def compose(frame, mask, lambda_funcs):
    for func in lambda_funcs:
        frame, mask = func(frame, mask)
    return frame


def apply_transform(frame, mask, func, magnitude, params, proba):
    rand = float(torch.rand(1))
    if rand < proba:
        return func(frame, mask, magnitude, params)
    else:
        return frame, mask


def sample_from_probas(probas):
    fixed_choice = [np.random.binomial(1, p) for p in probas]
    return fixed_choice


def path2policy(path, img_size, sampling='stochastic'):
    """
    Converts path given by mcts of the form [name, magn, proba, name, ...]
    to a pytorch image transformation
    """
    probas = [block[2] for block in path]
    if sampling == 'fixed':
        probas = sample_from_probas(probas)
    lambda_funcs = []
    for block, proba in zip(path, probas):
        name, magnitude, _ = block
        # sample fixed parameters for each transformations
        params_transform = ImageTransform.sample_params(name, magnitude, img_size)
        func = functools.partial(
            apply_transform,
            func=getattr(ImageTransform, name),
            magnitude=magnitude,
            params=params_transform,
            proba=proba)
        lambda_funcs.append(func)
    policy = functools.partial(compose, lambda_funcs=lambda_funcs)
    return policy


def test(dataset, save_dir):
    """ Test transformations and save them. """
    transforms = ImageTransform.transforms
    save_dir = os.path.join(save_dir, 'transforms_test')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    for name, magns_probs in transforms.items():
        print('checking {}...'.format(name))
        dataset.dataset._frames.set_augmentation(['identity', 1, 1])
        frame_idx = np.random.randint(len(dataset))
        frame_orig = dataset[frame_idx][0].clone()
        for magn in magns_probs['magnitude']:
            aug_path = [name, magn, 1]
            dataset.dataset._frames.set_augmentation(aug_path)
            frame_aug = dataset[frame_idx][0]
            frame = np.vstack(
                (((frame_orig[0] * 0.5 + 0.5).numpy() * 255).astype(np.uint8),
                 ((frame_aug[0].numpy() * 0.5 + 0.5) * 255).astype(np.uint8)))
            im_orig_aug = Image.fromarray(frame)
            fig_name = '{}_m{}.png'.format(name, magn)
            im_orig_aug.save(os.path.join(save_dir, fig_name))
    print('Saved transformed frames to {}'.format(save_dir))
