import numpy as np
import copy

from bc.dataset.keys import Keys


class Scalars:
    def __init__(self, data):
        assert isinstance(data, dict)

        self._scalars = {}
        for key, value in data.items():
            self[key] = value
        scalars = self._scalars
        self.keys = Keys(list(scalars.keys()))
        self._stats = {}
        self._updated_stats = False

        if len(scalars) > 0:
            assert 'state' in scalars[self.keys[0]].keys()
            assert 'action' in scalars[self.keys[0]].keys()
            self._update_statistics()

    def __len__(self):
        return len(self.keys)

    def __getstate__(self):
        print('Computing scalars statistics...')
        if not self._updated_stats:
            self._update_statistics()
        return self._scalars, self._stats

    def __setstate__(self, state):
        self._scalars, self._stats = state
        self.keys = Keys(list(self._scalars.keys()))
        self._updated_stats = True

    def __getitem__(self, idx):
        keys = self.keys
        if isinstance(idx, slice):
            start, end, step = idx.indices(len(self))
            # step bigger than 1 not handled
            assert step == 1
            # make sure all the frames come from the same demo
            idx_min, idx_max = keys.get_idx_min_max(start, end)
            scalars = []
            keys_slice = keys[idx_min:idx_max]
            for key in keys_slice:
                scalars.append(self._scalars[key])
            assert len(scalars) > 0
            return scalars
        elif isinstance(idx, int):
            return self._scalars[keys[idx]]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            seed, step = key
            assert isinstance(seed, int)
            assert isinstance(step, int)
            key = 'S{:06}/T{:06}'.format(seed, step)
        elif not isinstance(key, str):
            raise ValueError('{} ({}) is not a valid key type.'.format(
                key, type(key)))

        self.clean_entry(value)
        self._scalars[key] = copy.deepcopy(value)
        self._updated_stats = False

    def clean_entry(self, entry):
        for k0, d in entry.items():
            for k1, sc in d.items():
                entry[k0][k1] = self.clean_scalar(sc)

    def clean_scalar(self, scalar):
        scalar = np.nan_to_num(scalar)
        return scalar

    def get_statistics(self):
        if not self._stats or not self._updated_stats:
            self._update_statistics()
        return self._stats

    def update(self, data):
        for key, value in data.items():
            self[key] = value

    def _update_statistics(self):
        assert len(self._scalars) > 0
        assert not self._updated_stats

        stack, stats = {}, {}
        spaces = ['action', 'state']
        for space in spaces:
            stack[space] = {}
            stats[space] = {}

        for key, sc in self._scalars.items():
            for space in spaces:
                for key_space in sc[space].keys():
                    if not key_space in stack[space]:
                        stack[space][key_space] = [sc[space][key_space]]
                    else:
                        stack[space][key_space].append(sc[space][key_space])

        for space in spaces:
            for key in sc[space].keys():
                stack[space][key] = np.array(stack[space][key])
                stats[space][key] = (stack[space][key].mean(axis=0),
                                     stack[space][key].std(axis=0))

        self._stats = stats
        self._updated_stats = True
