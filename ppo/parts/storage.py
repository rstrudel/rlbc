import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self,
                 num_steps,
                 num_processes,
                 obs_shape,
                 action_space,
                 action_memory=0):
        num_steps *= max(int(np.sqrt(num_processes)), 2)
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.action_memory = action_memory
        self.steps = np.zeros(num_processes, dtype=np.int32)

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self,
               obs,
               actions,
               action_log_probs,
               value_preds,
               rewards,
               masks,
               indices=None):
        if indices is None:
            indices = np.range(self.num_processes)
        for index in indices:
            step_value = self.steps[index]
            self.obs[step_value + 1, index].copy_(obs[index])
            self.actions[step_value, index].copy_(actions[index])
            self.action_log_probs[step_value, index].copy_(action_log_probs[index])
            self.value_preds[step_value, index].copy_(value_preds[index])
            self.rewards[step_value, index].copy_(rewards[index])
            self.masks[step_value + 1, index].copy_(masks[index])
            self.steps[index] = (self.steps[index] + 1) % self.num_steps

    def get_last(self, tensor, *args, **kwargs):
        if tensor is self.actions:
            return self._get_last_actions(*args, **kwargs)
        lasts = {}
        for index in range(tensor.shape[1]):
            lasts[index] = tensor[self.steps[index], index]
        return lasts

    def _get_last_actions(self, steps=None, processes=None, as_tensor=False):
        if self.action_memory == 0:
            return None
        if processes is None:
            processes = np.arange(self.num_processes)
        if steps is None:
            steps = self.steps[processes]
        last_actions = -torch.ones(len(processes), self.action_memory).type_as(self.obs)
        for idx, (step, process) in enumerate(zip(steps, processes)):
            process_resets = np.where(self.masks[:step + 1, process, 0] == 0)[0]
            if process_resets.shape[0] > 0:
                last_reset = process_resets.max()
            else:
                last_reset = 0
            actions_available = np.clip(step - last_reset, 0, self.action_memory)
            if actions_available > 0:
                last_actions_ = self.actions[step - actions_available: step, process, 0]
                last_actions[idx, -actions_available:] = last_actions_
        if as_tensor:
            return last_actions
        else:
            last_actions_dict = {}
            for process in processes:
                last_actions_dict[process] = last_actions[process]
            return last_actions_dict

    def after_update(self):
        last_indices = np.stack((self.steps, np.arange(self.steps.shape[0])))
        self.obs[0].copy_(self.obs[last_indices])
        self.masks[0].copy_(self.masks[last_indices])
        self.steps = np.zeros_like(self.steps)

    def compute_returns(self, next_value, gamma):
        for env_idx in next_value.keys():
            self.returns[self.steps[env_idx], env_idx] = next_value[env_idx]
            for step in reversed(range(self.steps[env_idx])):
                self.returns[step, env_idx] = self.returns[step + 1, env_idx] * \
                                        gamma * self.masks[step + 1, env_idx] + \
                                        self.rewards[step, env_idx]

    def feed_forward_generator(self, advantages, num_mini_batch):
        batch_size = int(np.sum(self.steps))
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of env steps ({}) "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(batch_size, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        # get the (i, j) indices of the filled transitions in the right order
        transitions_ordered_indices = np.concatenate(
            [np.stack((range(s), [i] * s)) for i, s in enumerate(self.steps) if s > 0], axis=1)
        for indices in sampler:
            # replace the batch indices i by rollouts indices (i, j)
            indices = transitions_ordered_indices[:, indices]
            obs_batch = self.obs[indices]
            actions_batch = self.actions[indices]
            value_preds_batch = self.value_preds[indices]
            return_batch = self.returns[indices]
            old_action_log_probs_batch = self.action_log_probs[indices]
            adv_targ = advantages[indices]

            timesteps, env_idxs = indices
            last_actions_batch = self._get_last_actions(timesteps, env_idxs, as_tensor=True)
            yield obs_batch, last_actions_batch, actions_batch, \
                value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ
