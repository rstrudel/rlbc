import torch
import time
import numpy as np

from dask.distributed import Client, LocalCluster, Pub, Sub
from gym.spaces import Box
from collections import deque
from tornado import gen

import bc.utils.misc as bc_misc
from ppo.envs.single import SingleEnv
from ppo.parts.misc import RunningMeanStd

SUPPORTED_MIME_ENVS = 'Bowl', 'Breakfast'


class BatchEnv:
    def __init__(self, config):
        assert any([env_prefix in config.env_name for env_prefix in SUPPORTED_MIME_ENVS])
        self._read_config(config)
        self.frames_stack = [deque(maxlen=self.num_frames) for _ in range(self.num_processes)]
        self.action_sent_flags = np.zeros(self.num_processes)
        if 'Cam' not in self.env_name:
            self.obs_running_stats = RunningMeanStd(shape=self.observation_space.shape)
        else:
            self.obs_running_stats = None
        print('Will create BatchEnv with {} processes and batch size of {}'.format(
            self.num_processes, self.batch_size))

        # lazy dask initialization
        self._initialized = False
        self._cluster, self._client, self.pub_out, self.sub_in = None, None, None, None
        self._config = config

    def _read_config(self, config):
        # parsing the environment part of the config
        self.env_name = config.env_name
        self.num_processes = config.num_processes
        self.batch_size = config.dask_batch_size
        assert self.batch_size <= self.num_processes
        self.device = bc_misc.get_device(config.device)
        self.observation_type = config.input_type
        self.num_frames = config.bc_args['num_frames']

    def _client_map(self):
        env_args = []
        for env_idx in range(self.num_processes):
            env_config = dict(env_idx=env_idx)
            env_config.update(vars(self._config))
            env_args.append(env_config)
        return self._client.map(SingleEnv, env_args)

    def _init_dask(self):
        if self._cluster is not None:
            print('WARNING: reinitiailizing dask')
            self._cluster.close()
        if self._client is not None:
            self._client.close()
            del self._dask_futures
        if self.pub_out is not None:
            del self.pub_out
        if self.sub_in is not None:
            del self.sub_in
        self._cluster = LocalCluster(
            n_workers=self.num_processes,
            # silence_logs=0,
            memory_limit=None)
        self._client = Client(self._cluster)
        # always define publishers first then subscribers
        pub_out = [Pub('env{}_input'.format(env_idx)) for env_idx in range(self.num_processes)]
        self._dask_futures = self._client_map()
        sub_in = Sub('observations')
        self.pub_out = pub_out
        self.sub_in = sub_in
        # wait until all the peers are created
        time.sleep(5)

    def step(self, actions):
        for env_idx, action_dict in actions.items():
            assert self.action_sent_flags[env_idx] == 0
            self.action_sent_flags[env_idx] = 1
            self.pub_out[env_idx].put({'function': 'step',
                                       'action': action_dict})
        return self._get_obs_batch(self.batch_size)

    def _get_obs_batch(self, batch_size):
        obs_dict, reward_dict, done_dict, info_dict = {}, {}, {}, {}
        count_envs = 0
        while True:
            try:
                env_dict, env_idx = self.sub_in.get(timeout=30)
            except gen.TimeoutError:
                # recreate all the workers
                self._init_dask()
                # ask them to return observation after the reset with dummy reward, done and info
                for env_idx in range(self.num_processes):
                    self.pub_out[env_idx].put({'function': 'reset_after_crash'})
                # start the _get_obs_batch function from scratch
                self.action_sent_flags[:] = 1
                count_envs = 0
                # make the batch to contain all the observations
                batch_size = self.num_processes
                continue

            assert self.action_sent_flags[env_idx] == 1
            self.action_sent_flags[env_idx] = 0
            obs_dict[env_idx] = env_dict['observation'].to(torch.device(self.device))
            reward_dict[env_idx] = env_dict['reward']
            done_dict[env_idx] = env_dict['done']
            if env_dict['done']:
                self.frames_stack[env_idx].clear()
            info_dict[env_idx] = env_dict['info']
            count_envs += 1
            if count_envs == batch_size:
                break
        return self._stack_obs(obs_dict), reward_dict, done_dict, info_dict

    def reset(self):
        if not self._initialized:
            self._init_dask()
            self._initialized = True
        if sum(self.action_sent_flags) > 0:
            # an early reset was called, need to collect the step observations first
            count_envs = 0
            count_obs = sum(self.action_sent_flags)
            for unused_env_dict, env_idx in self.sub_in:
                assert self.action_sent_flags[env_idx] == 1
                self.action_sent_flags[env_idx] = 0
                count_envs += 1
                if count_envs == count_obs:
                    break
        count_envs = 0
        for env_idx in range(self.num_processes):
            assert self.action_sent_flags[env_idx] == 0
            self.pub_out[env_idx].put({'function': 'reset'})
        obs_dict = {}
        for env_dict, env_idx in self.sub_in:
            obs_dict[env_idx] = env_dict['observation'].to(torch.device(self.device))
            self.frames_stack[env_idx].clear()
            count_envs += 1
            if count_envs == self.num_processes:
                break
        return self._stack_obs(obs_dict)

    def _stack_obs(self, obs_dict):
        for env_idx, obs_tensor in obs_dict.items():
            frames_stack = self.frames_stack[env_idx]
            frames_stack.append(obs_tensor)
            while len(frames_stack) < self.num_frames:
                frames_stack.append(obs_tensor)
            obs_dict[env_idx] = torch.cat(tuple(frames_stack))
        return self._normalize_obs(obs_dict)

    def _normalize_obs(self, obs_dict):
        if self.obs_running_stats:
            obs_numpy_list = []
            for env_idx, obs in sorted(obs_dict.items()):
                obs_numpy_list.append(obs.cpu().numpy())
            self.obs_running_stats.update(np.stack(obs_numpy_list))
            clipob = 10.
            epsilon = 1e-8
            obs_mean = torch.tensor(self.obs_running_stats.mean).type_as(obs)
            obs_var = torch.tensor(self.obs_running_stats.var).type_as(obs)
            for env_idx, obs in sorted(obs_dict.items()):
                obs = torch.clamp((obs - obs_mean) / torch.sqrt(obs_var + epsilon), -clipob, clipob)
                obs_dict[env_idx] = obs
        return obs_dict

    @property
    def observation_space(self):
        if 'Cam' in self.env_name:
            if self.observation_type == 'depth':
                observation_dim = 1 * self.num_frames
            elif self.observation_type == 'rgbd':
                observation_dim = 4 * self.num_frames
            else:
                raise NotImplementedError
            return Box(-np.inf, np.inf, (observation_dim, 224, 224), dtype=np.float)
        elif 'Bowl' in self.env_name:
            return Box(-np.inf, np.inf, (19,), dtype=np.float)
        elif 'Breakfast' in self.env_name:
            num_cups = 2
            num_drops = 0
            num_features = 32 + 10 * num_cups + 3 * num_drops * num_cups
            return Box(-np.inf, np.inf, (num_features,), dtype=np.float)
        elif 'SimplePour' in self.env_name:
            if 'NoDrops' in self.env_name:
                num_drops = 0
            else:
                num_drops = 5
            num_features = 16 + 3 * num_drops
            return Box(-np.inf, np.inf, (num_features,), dtype=np.float)
        else:
            raise NotImplementedError
