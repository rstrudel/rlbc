import os
import time
import pickle as pkl
from tqdm import tqdm
from termcolor import colored
from sacred import Experiment

from dask.distributed import Client, LocalCluster, Pub, Sub

from configs.bc import collect_ingredient
from bc.dataset import utils
from bc.utils import misc
from bc.dataset import DatasetWriter, Scalars

ex = Experiment('collect', ingredients=[collect_ingredient])


def get_frames_scalars(env, counter):
    obs = env.reset()
    frames, scalars = utils.process_trajectory(
        states=[obs], actions=[{}], steps=[0], seed=counter, compression=True)
    return frames, scalars


def run_env(env_name, seed):
    try:
        import gym
        import mime
        env = gym.make(env_name)
        env.seed(seed)
        pub_obs = Pub('observations')
        sub_reset = Sub('env{}_reset'.format(int(seed)))
        print('seed {} ready'.format(seed))
        for counter in sub_reset:
            frames, scalars = get_frames_scalars(env, counter)
            pub_obs.put((frames, scalars, seed))
    except Exception as e:
        print('Exeception: {}'.format(e))


def create_dataset(path, env_name):
    if not os.path.isdir(path):
        os.mkdir(path)
    dataset = DatasetWriter(path, env_name, rewrite=True)
    dataset.init()
    scalars_dict = {}
    return dataset, scalars_dict


def init_workers(env_name, num_processes):
    cluster = LocalCluster(n_workers=num_processes)
    client = Client(cluster)
    pubs_reset = [Pub('env{}_reset'.format(seed)) for seed in range(num_processes)]
    client.map(run_env, [env_name] * num_processes, range(num_processes))
    sub_obs = Sub('observations')
    # sleep while sub/pub is initialized
    time.sleep(5)
    return client, pubs_reset, sub_obs


def collect_dataset(collect):
    # init workers and create dataset
    client, pubs_reset, sub_obs = init_workers(collect['env'], collect['workers'])
    dataset, scalars_dict = create_dataset(collect['collect_dir'], collect['env'])
    episodes = collect['episodes']

    # init reset submissions
    for counter in range(collect['workers']):
        pubs_reset[counter].put(counter)
    counter = collect['workers']
    print('Submitted resets')

    pbar = tqdm(total=episodes)
    for frames_worker, scalars_worker, seed in sub_obs:
        # reset worker's environment
        pubs_reset[seed].put(counter)
        # add received data to dataset
        dataset.write_frames(frames_worker)
        scalars_dict.update(scalars_worker)
        pbar.update(1)
        if counter - collect['workers'] >= episodes:
            break
        counter += 1
    # write scalars into a pickle file
    scalars = Scalars(scalars_dict)
    pkl.dump(scalars, open(os.path.join(collect['collect_dir'], 'scalars.pkl'), 'wb'))
    # close frames dataset
    dataset.close()
    # close dask client
    client.close()

    return collect['collect_dir'], collect['env'], collect['episodes']


@ex.automain
def main(collect):
    collect = misc.update_arguments(collect=collect)[0]
    dataset_dir, env_name, episodes = collect_dataset(collect)
    print(
        colored('Dataset successfully written to {}'.format(dataset_dir),
                'green'))
    print(
        colored('{} episodes collected from {}'.format(episodes, env_name),
                'green'))
