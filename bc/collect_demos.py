import time
import numpy as np
from tqdm import tqdm
from termcolor import colored
from sacred import Experiment

from dask.distributed import Client, progress
from joblib import Parallel, delayed

from configs.bc import collect_ingredient
from bc.utils import misc
from bc.dataset.collector import Collector
from bc.dataset.utils import create_report, gather_dataset

ex = Experiment('collect', ingredients=[collect_ingredient])


def make_collector(model, collect):
    seed_range = range(collect['seed'], collect['seed'] + collect['episodes'])
    seed_epoch = []
    report, report_path = None, None
    if collect['db_type'] == 'evaluation':
        report, report_path, epochs_range = create_report(collect, model['model_dir'])
        print(colored('Will evaluate epochs {}'.format(epochs_range), 'yellow'))
        for epoch in epochs_range:
            for seed in seed_range:
                if not report.is_entry(epoch, seed):
                    seed_epoch.append((seed, epoch))
    else:
        seed_epoch = [(seed, model['epoch']) for seed in seed_range]
    collector = Collector(**model, **collect, report_path=report_path)

    return collector, seed_epoch, report, report_path, collect['collect_dir']


def run_parallel(collector, seed_epoch, collect):
    if collect['dask']:
        # Launching Dask on cluster
        assert not collect['render'], 'Can not render using dask'
        from daskoia import CPUCluster
        cluster = CPUCluster(mem_req=6000)
        cluster.start_workers(collect['workers'])
        client = Client(cluster)
        print('Scheduler Info {}'.format(client.scheduler_info()))
        futures_traj = client.map(collector, seed_epoch)
        progress(futures_traj)
        results = client.gather(futures_traj)
    else:
        if collect['workers'] > 1:
            assert not collect[
                'render'], 'Can not render using multiple processes'
            results = Parallel(n_jobs=collect['workers'])(
                delayed(collector)(se) for se in tqdm(seed_epoch))
        else:
            results = (collector(se) for se in tqdm(seed_epoch))

    return results


def process_results(results, report, report_path, collect):
    episodes, failed_episodes = collect['episodes'], []
    tot_steps, max_steps = 0, 0
    for success, seed, epoch, num_steps in results:
        if collect['db_type'] == 'evaluation':
            report.add_entry(seed, epoch, success)
        if success or collect['record_failed']:
            tot_steps += num_steps
            if num_steps > max_steps:
                max_steps = num_steps
        else:
            failed_episodes += [seed]
            episodes -= 1

    if collect['db_type'] == 'demos':
        print('Gathering trajectories dataset into one dataset...')
        gather_dataset(collect['collect_dir'])
    elif collect['db_type'] == 'evaluation':
        report.save(report_path)

    return episodes, failed_episodes, tot_steps, max_steps


def print_report(report):
    sr = report.get_success_rate()
    print('Success rate is in average {0:.2f}:'.format(np.mean(list(sr.values()))))
    for epoch, success_rate in sorted(sr.items(), key=lambda kv: -kv[1]):
        print('\tEpoch {0:3d}: success rate = {1:.2f} ({2} seeds)'.format(
            epoch, success_rate, len(report.entries['epochs'][epoch])))


@ex.automain
def main(model, collect):
    t0 = time.time()

    print('Collecting demos in folder {} with parameters:\n collect {}\n model {}'.format(
        collect['folder'], collect, model))
    model, collect = misc.update_arguments(model=model, collect=collect)
    collector, seed_epoch, report, report_path, data_dir = make_collector(model, collect)
    results = run_parallel(collector, seed_epoch, collect)
    episodes, failed_episodes, tot_steps, max_steps = process_results(
        results, report, report_path, collect)

    print(colored('Dataset successfully written to {}'.format(data_dir), 'green'))
    print(colored('Total {} steps in {} trajectories'.format(tot_steps, episodes), 'green'))
    print(colored('Maximum {} steps in one trajectory'.format(max_steps), 'green'))
    print(colored('Data collection took {} seconds'.format(time.time() - t0), 'green'))

    # if len(failed_episodes):
    #     print(colored('Failed {} trajectories: {}'.format(
    #         len(failed_episodes), sorted(failed_episodes)), 'red'))

    if report is not None:
        print_report(report)
