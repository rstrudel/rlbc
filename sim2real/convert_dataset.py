import click
import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
from PIL import Image

from bc.dataset import DatasetWriter, Scalars
from bc.dataset import utils

CROP = (34, None, 22, -100)
IMG_SIZE = (240, 240)
NEAR, FAR = 0.5, 2.0

def get_frames_scalars(obs, counter):
    frames, scalars = utils.process_trajectory(
        states=[obs], actions=[{}], steps=[0], seed=counter, compression=True)
    return frames, scalars


def process_depth(depth, near, far):
    depth_proc = depth.copy()
    depth_proc[np.isnan(depth_proc)] = far
    depth_proc[depth_proc <= near] = far
    depth_proc = depth_proc.clip(near, far)
    depth_proc = (255 * (depth_proc - near) / (far - near)).astype(np.uint8)
    depth_proc = Image.fromarray(depth_proc)
    depth_proc = depth_proc.resize(IMG_SIZE)
    return np.asarray(depth_proc)


@click.command(help='convert raw kinect2 images into formatted images')
@click.option('-i', '--input_dir', default='', help='input dataset of raw images')
@click.option('-o', '--output_dir', default='', help='output dataset of raw images')
def main(input_dir, output_dir):
    dataset_real = pkl.load(open(input_dir, 'rb'))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    dataset = DatasetWriter(output_dir, 'RealRobot', rewrite=True)
    dataset.init()
    scalars_dict = {}

    print('Processing real dataset...')
    for i, obs in tqdm(enumerate(dataset_real)):
        depth_raw = obs['depth0'][..., 0] / 1000
        ymin, ymax, xmin, xmax = CROP
        depth_raw = depth_raw[ymin:ymax, xmin:xmax]
        obs['depth0'] = process_depth(depth_raw, NEAR, FAR)
        frames, scalars = get_frames_scalars(obs, i)
        dataset.write_frames(frames)
        scalars_dict.update(scalars)

    scalars = Scalars(scalars_dict)
    pkl.dump(scalars, open(os.path.join(output_dir, 'scalars.pkl'), 'wb'))
    dataset.close()

if __name__ == "__main__":
    main()
