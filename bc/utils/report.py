import numpy as np
import os
import pickle as pkl
from filelock import FileLock


class Report:
    def __init__(self, env_name='', net_path='', path=''):
        if path:
            self.load(path)
        else:
            self.entries = {
                'env_name': env_name,
                'net_path': net_path,
                'epochs': {}
            }

    def add_entry(self, seed, epoch, success):
        if not self.entries['net_path']:
            raise ValueError('Create or load report first.')

        epochs = self.entries['epochs']
        if epoch not in epochs:
            epochs[epoch] = [(seed, success)]
        else:
            epochs[epoch].append((seed, success))

    def is_entry(self, epoch, seed):
        if not self.entries['net_path']:
            raise ValueError('Create or load report first.')

        epochs = self.entries['epochs']
        return epoch in epochs and seed in [seed for seed, _ in epochs[epoch]]

    def get_success_rate(self, seed_range=None):
        if not self.entries['net_path']:
            raise ValueError('Create or load report first.')

        success_rates = {}
        epochs = self.entries['epochs']
        for epoch, results in epochs.items():
            if seed_range is not None:
                results = [(seed, success) for seed, success in results
                           if seed >= seed_range[0] and seed < seed_range[1]]
            if results:
                success_rates[epoch] = sum(
                    [success for seed, success in results]) / len(results)

        return success_rates

    def get_epoch(self, epoch):
        if not self.entries['net_path']:
            raise ValueError('Create or load report first.')

        if epoch not in self.entries['epochs']:
            raise ValueError(
                'Epoch {} is not present in the report entries.'.format(epoch))
        return self.entries['epochs'][epoch]

    def load(self, path):
        self.entries = pkl.load(open(path, 'rb'))

    def save(self, path):
        lock = FileLock(
            os.path.join(
                os.path.dirname(path), '.{}.lock'.format(
                    os.path.basename(path))))
        with lock:
            if os.path.exists(path):
                existing_entries = pkl.load(open(path, 'rb'))
                self.entries['epochs'].update(existing_entries['epochs'])
            pkl.dump(self.entries, open(path, 'wb'))
            print('Report is saved to {}'.format(path))

    @staticmethod
    def add_entry_asynch(path, seed, epoch, success):
        assert os.path.exists(path)
        lock = FileLock(
            os.path.join(
                os.path.dirname(path), '.{}.lock'.format(
                    os.path.basename(path))))
        with lock:
            report_entries = pkl.load(open(path, 'rb'))
            epochs = report_entries['epochs']
            if epoch not in epochs:
                epochs[epoch] = [(seed, success)]
            else:
                epochs[epoch].append((seed, success))
            pkl.dump(report_entries, open(path, 'wb'))
