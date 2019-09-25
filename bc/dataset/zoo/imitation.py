import torch
from torch.utils.data import Dataset
import numpy as np

from bc.dataset import Frames, Actions, Signals


class ImitationDataset(Dataset):
    def __init__(self, dataset_dir, max_demos, num_cameras, num_frames, channels,
                 action_space, steps_action, num_signals, num_skills,
                 image_augmentation, **unused_kwargs):

        channels = list(channels)
        if image_augmentation:
            channels += ['mask']

        # define frames
        frames = Frames(
            path=dataset_dir,
            channels=channels,
            max_demos=max_demos,
            augmentation=image_augmentation)
        frames.keys.set_query_limit('demo')

        # define actions and signals
        actions = Actions(dataset_dir, action_space)
        signals = Signals(dataset_dir, [('state', 'joint_position'), ('state', 'grip_velocity')])
        self._num_skills = num_skills
        self._get_skills = num_skills > 1
        if self._get_skills:
            actions.keys.set_query_limit('skill')
            actions.keys.set_skill_labels(Signals(dataset_dir, [('state', 'skill')]))
        else:
            actions.keys.set_query_limit('demo')
        actions.keys.set_max_demos(max_demos)
        signals.keys.set_max_demos(max_demos)

        # check datasets length match
        assert len(frames) == len(actions) and len(actions) == len(signals),\
            'Frames length {} Actions length {} Signals length {}'\
            .format(len(frames), len(actions), len(signals))

        self._frames = frames
        self._actions = actions
        self._signals = signals
        self._num_signals = num_signals
        self._channels = channels
        self._num_frames = num_frames
        self._num_cameras = num_cameras
        self._steps_action = steps_action
        self._action_space = action_space
        self._skills_indices = []

        if self._get_skills:
            # enable balanced skill sampling
            if actions.get_skill(0, skip_undefined=True) is None:
                raise ValueError('Skill sampling is on while the dataset contains no skill.')
            self._init_skills_sampling()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        frames = self._frames
        signals = self._signals
        num_signals = self._num_signals
        actions = self._actions
        channels = self._channels
        num_frames = self._num_frames
        steps_action = self._steps_action

        # sample dataset idx balanced with respect to skills
        if self._get_skills:
            idx = self._get_idx_from_skills_sampling(idx)

        # sample camera
        camera_idx = int(torch.randint(high=self._num_cameras, size=(1, )))
        frames.set_camera(camera_idx)

        # images
        frames.keys.set_idx_reference(idx)
        idx_beg, idx_end = self.get_bounded_idxs(idx - (num_frames - 1),
                                                 idx + 1)
        im = torch.zeros(1, 1, 224, 224)
        if num_frames > 0:
            im = frames[idx_beg:idx_end]
            im = frames.adjust_shape(im, num_frames, channels)

        # actions
        actions.keys.set_idx_reference(idx)
        idx_beg, idx_end = self.get_bounded_idxs(idx + steps_action[0],
                                                 idx + steps_action[-1] + 1)
        actions_future = actions[idx_beg:idx_end]
        action = actions.adjust_shape(
            actions_future, steps_action, early_closing=True)

        # signals
        signals.keys.set_idx_reference(idx)
        idx_beg, idx_end = self.get_bounded_idxs(idx - num_signals + 1,
                                                 idx + 1)
        signal = torch.zeros(1, 1)
        if num_signals > 0:
            signal = signals[idx_beg:idx_end]
            signal = signals.adjust_shape(signal, num_signals).float()
            signal += 0.05 * torch.randn((7 * num_signals, ))

        # skills
        skill = actions.get_skill(idx, skip_undefined=True)
        if skill is None:
            skill = []

        return (im.float(), action.float(), signal.float(), skill)

    def _init_skills_sampling(self):
        skills = np.array(self._actions.skills)
        max_skill = skills.max()
        for skill in range(max_skill+1):
            indices_skill = np.where(skills==skill)[0]
            indices_skill = [idx for idx in indices_skill if idx < len(self._actions)]
            self._skills_indices.append(indices_skill)

    def _get_idx_from_skills_sampling(self, idx):
        skill_indices = self._skills_indices
        skill_idx = int(torch.randint(high=len(skill_indices), size=(1, )))
        skill_list_idx = int(torch.randint(high=len(skill_indices[skill_idx]), size=(1, )))
        idx = skill_indices[skill_idx][skill_list_idx]
        return idx

    def get_bounded_idxs(self, idx_min, idx_max):
        return min(max(idx_min, 0),
                   len(self._frames) - 1), max(
                       min(idx_max, len(self._frames)), 0)

    def get_statistics(self):
        return self._actions.scalars.get_statistics()

    def get_num_demos(self):
        return self._frames.get_num_demos()

    def get_demo_frame_idxs(self, demo_idx):
        idx_beg, idx_end = self._frames.get_demo_indices(demo_idx)
        return slice(idx_beg, idx_end)
