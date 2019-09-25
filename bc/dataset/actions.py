import os
import pickle as pkl
import numpy as np
import torch


class Actions:
    def __init__(self, path, action_space, normalize=True):
        path_scalars = os.path.join(path, 'scalars.pkl')
        self.scalars = pkl.load(open(path_scalars, 'rb'))
        self.keys = self.scalars.keys
        self.stats = self.scalars.get_statistics()
        self.action_space = action_space
        self.skills = None
        self.normalize = normalize

        self.action_keys, self.dim_action = self.action_space_to_keys(
            action_space)

        if 'skill' in self.scalars[0]['state']:
            self._load_skills()

    def __len__(self):
        return len(self.scalars)

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, np.int64):
            action_scalars = [self.scalars[int(idx)]]
        elif isinstance(idx, slice):
            action_scalars = self.scalars[idx]
        else:
            raise TypeError('{} is an unvalid index type. {}'.format(
                type(idx), idx))
        stats = self.stats if self.normalize else None
        stack_actions = [self.dict_to_tensor(asc, self.dim_action, self.action_keys, stats)
                         for asc in action_scalars]
        actions = np.vstack(stack_actions)

        return actions

    def _load_skills(self):
        scalars = self.scalars
        self.skills = [scalar['state']['skill'] for scalar in scalars]

    @staticmethod
    def action_space_to_keys(action_space):
        dict_conversion = {
            'tool_lin': (('grip_velocity', 'linear_velocity'), 4),
            'tool_lin_ori': (('grip_velocity', 'linear_velocity',
                              'angular_velocity'), 7),
            'joints': (('grip_velocity', 'joint_velocity'), 7),
            'cube_pos': (None, 2),
            '3d': (None, 3),
            '4d': (None, 4),
            '6d': (None, 6),
            '4cubes_pos': (None, 8),
        }
        assert action_space in dict_conversion.keys(
        ), 'Unknown action space : {}'.format(action_space)
        action_keys, dim_action = dict_conversion[action_space]
        return action_keys, dim_action

    @staticmethod
    def get_dict_null_action(action_keys):
        action_dict = {}
        action_name_to_array = {
            'grip_velocity': np.zeros(1),
            'linear_velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'joints_velocity': np.zeros(6)
        }
        for action_key in action_keys:
            assert action_key in action_name_to_array
            action_dict[action_key] = action_name_to_array[action_key]
        return action_dict

    @staticmethod
    def filter_action(action, action_keys):
        # the scripts of the mime env seem to always return both IK and joints velocities
        # we filter the action depending on the action_space of the mime environment
        action_filtered = {}
        for action_key, action_value in action.items():
            if action_key in action_keys:
                action_filtered[action_key] = action_value
        return action_filtered

    @staticmethod
    def dict_to_tensor(action_scalar, dim_action, action_keys, stats=None):
        state = action_scalar['state']
        action = action_scalar['action']

        idx_action = 0
        norm_action = torch.zeros(dim_action, dtype=torch.float)

        # use the grip_velocity of the state which is 1 when open, -1 when closed
        # the grip_velocity of action is -1 only when the gripper is closing, but not when closed
        grip_obs = ['grip_state', 'grip_velocity']
        grip_in_state = False
        for g_obs in grip_obs:
            if g_obs in state:
                grip_in_state = True
                norm_action[idx_action] = float(state[g_obs])
                idx_action += 1

        if grip_in_state:
            action_keys = action_keys[1:]
        else:
            action_keys = action_keys

        for action_key in action_keys:
            raw_action = torch.tensor(action[action_key])
            size = raw_action.shape
            size = (1, ) if not size else size
            if stats is None or 'grip' in action_key:
                unit_action = raw_action
            else:
                mean, std = stats['action'][action_key]
                mean, std = torch.tensor(mean), torch.tensor(std)
                unit_action = (raw_action - mean) / (std + 1e-8)
            norm_action[idx_action:idx_action + size[0]] = unit_action
            idx_action += size[0]

        return norm_action

    @staticmethod
    def tensor_to_dict(action_tensor, action_keys, stats=None):
        action_dict = Actions.get_dict_null_action(action_keys)
        action_key_to_slice = {
            'grip_velocity': slice(0, 2),
            'linear_velocity': slice(2, 5),
            'angular_velocity': slice(5, 8),
            'joint_velocity': slice(2, 8)
        }
        for action_key in action_keys:
            if action_key == 'grip_velocity':
                action_dict[action_key] = -1 + 2 * (action_tensor[0].item() >
                                                    action_tensor[1].item())
            else:
                slice_action = action_key_to_slice[action_key]
                action_dict[action_key] = action_tensor[slice_action]
        action_dict = Actions.denormalize_dict(action_dict, stats)
        return action_dict

    @staticmethod
    def normalize_dict(action_dict, stats):
        if stats is None:
            return action_dict

        for action_key, raw_action in action_dict.keys():
            # raw_action = torch.tensor(raw_action)
            size = raw_action.shape
            size = (1, ) if not size else size
            unit_action = raw_action
            if action_key in ['linear_velocity', 'angular_velocity']:
                mean, std = stats['action'][action_key]
                # mean, std = torch.tensor(mean), torch.tensor(std)
                unit_action = (unit_action - mean) / (std + 1e-8)
            action_dict[action_key] = unit_action
        return action_dict

    @staticmethod
    def denormalize_dict(action_dict, stats):
        if stats is None:
            return action_dict
        for action_key, value in action_dict.items():
            if action_key in ['linear_velocity', 'angular_velocity']:
                mean, std = stats['action'][action_key]
                if isinstance(mean, torch.Tensor):
                    mean = mean.cpu().numpy()
                if isinstance(std, torch.Tensor):
                    std = std.cpu().numpy()
                action_dict[action_key] = mean + value * (std + 1e-8)
            else:
                action_dict[action_key] = 4 * value
        return action_dict

    @staticmethod
    def adjust_shape(actions, steps_action, early_closing=False):
        action = torch.zeros(0).float()
        for step_action in steps_action:
            idx_a = min(step_action - steps_action[0], len(actions) - 1)
            # check if the gripper is closed 3 steps ahead, if it is
            # send the closing signal as gt on current step
            if early_closing:
                idx_forward = min(idx_a + 5, len(actions) - 1)
                # if actions[idx_forward][0] < 0:
                if actions[idx_forward][0] * actions[idx_a][0] < 0:
                    actions[idx_a][0] = actions[idx_forward][0]
            a = torch.tensor(actions[idx_a]).float()
            action = torch.cat((action, a), 0)
        return action

    def get_skill(self, idx, skip_undefined=False):
        if self.skills is None:
            return None

        a = torch.zeros(1).long()
        a = self.skills[idx]
        if a < 0 and skip_undefined:
            a = self.skills[idx + 1]
        return a

    def get_statistics(self):
        return self.stats

    def set_max_demos(self, max_demos):
        self.scalars.set_max_demos(max_demos)
