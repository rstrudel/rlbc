import os
import torch
import itertools
import numpy as np
from collections import OrderedDict

from bc.agent.image import ImageAgent
from bc.agent import utils
from bc.dataset import Actions
from bc.model import utils as bc_utils


class RLAgent(ImageAgent):
    def __init__(self, path, epoch, max_steps, device='cpu', env=None, real_robot_mode=False,
                 timescales_list=None, **kwargs):
        super(RLAgent, self).__init__(path, epoch, max_steps, device)
        self.model, self.rl_args, self.obs_running_stats = self._load_model()
        self.model.args = self.rl_args.bc_args
        self.real_robot_mode = real_robot_mode
        if not self.real_robot_mode:
            self.set_augmentation(self.rl_args.augmentation)
        else:
            self.set_augmentation('')
        self._max_steps = max_steps

        # skills timescales
        if timescales_list is None:
            if isinstance(self.rl_args.timescale, list):
                self._skills_timescales = self.rl_args.timescale
            else:
                assert isinstance(self.rl_args.timescale, int)
                self._skills_timescales = []
                for _ in range(self.rl_args.num_skills):
                    self._skills_timescales.append(self.rl_args.timescale)
        else:
            assert isinstance(timescales_list, list)
            self._skills_timescales = timescales_list

        # memory
        self._action_memory = self.rl_args.action_memory
        if self._action_memory > 0:
            last_skills_tensor = -torch.ones(self._action_memory).float()
            self._last_skills = {0: last_skills_tensor.to(torch.device(device))}
        else:
            self._last_skills = None

        # full state specific stuff
        self._env = env
        self.action_keys = Actions.action_space_to_keys(
            self.rl_args.bc_args['action_space'])[0]

        self.reset()

    def reset(self):
        super(RLAgent, self).reset()
        self._skill = None
        self._need_master_action = True
        self._prev_script, self._prev_action_chain = None, None

    def _load_model(self):
        load_path = os.path.join(
            self._path, 'model_{}.pth'.format(self._epoch))
        device = self._device
        if device == 'cpu':
            loaded_dict = torch.load(
                load_path, map_location=lambda storage, loc: storage)
        else:
            loaded_dict = torch.load(load_path)
        policy = loaded_dict['policy']
        if hasattr(policy.base, 'resnet'):
            # set the batch norm to eval mode
            policy.base.resnet.eval()
        policy.to(torch.device(device))
        return policy, loaded_dict['args'], loaded_dict['obs_running_stats']

    def get_action(self, obs):
        if self._max_steps > -1 and self._count_steps > self._max_steps:
            return None
        self._count_steps += 1

        if 'Cam' in self.rl_args.env_name:
            self._stack_frames, self._stack_signals = self.update_stacks(
                obs, self._stack_frames, self._stack_signals)
            obs_dict = {0: self._stack_frames.float()[None, :].to(self._device)[0]}
        else:
            obs_dict = {0: self._process_full_state_obs(obs)}

        if self._skill is None or self._need_master_action:
            with torch.no_grad():
                _, skill_dict, _ = self.model.act(
                    obs_dict, self._last_skills, deterministic=True)
            self._skill = skill_dict[0][None]
            if self._last_skills is not None:
                self._last_skills = {
                    0: utils.add_to_stack(self._skill, self._last_skills[0])}
            self._step_new_skill = self._count_steps
            if self.real_robot_mode:
                print('New master action (skill) is {}'.format(self._skill.item()))

        if not self.rl_args.use_expert_scripts:
            with torch.no_grad():
                action_dict, _ = self.model.get_worker_action({0: self._skill}, obs_dict)
                action = action_dict[0]
            skill_val = self._skill.item()
            self._need_master_action = (
                (self._count_steps - self._step_new_skill + 1) == self._skills_timescales[skill_val])
        else:
            action = self._get_script_action(self._skill.item())
        return action

    def _get_script_action(self, skill):
        # copy-paste from ppo.envs.mime
        if self._prev_script != skill:
            self._prev_script = skill
            self._prev_action_chain = self._env.unwrapped.scene.script_subtask(skill)
        action_chain = itertools.chain(*self._prev_action_chain)
        action_applied = Actions.get_dict_null_action(self.action_keys)
        action_update = next(action_chain, None)
        if action_update is None:
            self._need_master_action = True
        else:
            self._need_master_action = False
            action_applied.update(
                Actions.filter_action(action_update, self.action_keys))
        return action_applied

    def _process_full_state_obs(self, obs_dict):
        observation = np.array([])
        obs_sorted = OrderedDict(sorted(obs_dict.items(), key=lambda t: t[0]))
        cam_obs_keys = ['rgb0', 'depth0', 'mask0']
        for obs_key, obs_value in obs_sorted.items():
            if obs_key != 'skill' and obs_key not in cam_obs_keys:
                if isinstance(obs_value, (int, float)):
                    obs_value = [obs_value]
                elif isinstance(obs_value, np.ndarray):
                    obs_value = obs_value.flatten()
                elif isinstance(obs_value, list) and isinstance(
                        obs_value[0], np.ndarray):
                    obs_value = np.concatenate(obs_value)
                observation = np.concatenate((observation, obs_value))
        observation = torch.tensor(observation).float()
        obs_mean = torch.tensor(
            self.obs_running_stats.mean).type_as(observation)
        obs_var = torch.tensor(self.obs_running_stats.var).type_as(observation)
        observation = np.clip(
            (observation - obs_mean) / np.sqrt(obs_var + 1e-8), -10., 10.)
        return observation.to(torch.device(self._device))
