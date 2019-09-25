import os
import pickle as pkl
import numpy as np
import torch


class Signals:
    def __init__(self, path, keys_signal, normalize=True):
        path_scalars = os.path.join(path, 'scalars.pkl')
        self.scalars = pkl.load(open(path_scalars, 'rb'))
        self.keys = self.scalars.keys
        self.stats = self.scalars.get_statistics()
        self.normalize = normalize

        self.keys_signal = keys_signal

    def __len__(self):
        return len(self.scalars)

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, np.int64):
            scalars = [self.scalars[int(idx)]]
        elif isinstance(idx, slice):
            scalars = self.scalars[idx]
        else:
            raise TypeError('{} is an unvalid index type. {}'.format(
                type(idx), idx))

        stack_signals = [
            self.transform(sc, self.keys_signal, self.stats, self.normalize)
            for sc in scalars
        ]
        return np.vstack(stack_signals)

    @staticmethod
    def transform(scalars, keys_signal, stats=[], normalize=False):
        out_signals = []
        assert len(keys_signal) > 0
        assert len(keys_signal[0]) == 2, \
            'keys signal should be of the form (\'signal\', \'target_position\')'

        for ks0, ks1 in keys_signal:
            signal = np.array(scalars[ks0][ks1])
            if normalize:
                mean, std = stats[ks0][ks1]
                mean, std = np.array(mean), np.array(std)
                signal = (signal - mean) / std
            out_signals.append(signal.flatten())

        out_signals = np.hstack(out_signals)

        return out_signals

    @staticmethod
    def adjust_shape(signals, num_signals):
        missing_rows = num_signals - signals.shape[0]
        a = np.repeat(signals[0][None, :], missing_rows, axis=0)
        if a.shape[0] > 0:
            signals = np.vstack((a, signals))

        return torch.tensor(signals.reshape(-1))
