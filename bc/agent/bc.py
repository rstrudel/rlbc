import torch

from bc.agent.image import ImageAgent
from bc.model import utils
from bc.dataset import Signals


class BCAgent(ImageAgent):
    def __init__(self, path, epoch, max_steps, device='cuda', augmentation_str='', **kwargs):
        super(BCAgent, self).__init__(path, epoch, max_steps, device)
        self.model = self._load_model(path, epoch)
        self.set_augmentation(augmentation_str)
        super(BCAgent, self).reset()

    def _load_model(self, path, epoch):
        model, optimizer = utils.load_model(path, epoch, self._device)
        model.set_eval()
        return model

    def get_action(self, obs, skill=None):
        if self.max_steps > -1 and self._count_steps > self.max_steps:
            return None
        skip = self.args['skip'] if 'skip' in self.args else 1

        self._stack_frames, self._stack_signals = self.update_stacks(
            obs, self._stack_frames, self._stack_signals)
        net_frames = self._stack_frames.clone()
        net_frames = net_frames[::skip].float()

        mode = self.args['mode']
        if mode != 'signals':
            with torch.no_grad():
                dict_action = self.model.get_dict_action(net_frames[None, :], None, skill)
        else:
            net_signals = self._stack_signals.clone()
            net_signals = Signals.adjust_shape(net_signals, self._num_signals).float()
            with torch.no_grad():
                dict_action = self.model.get_dict_action(
                    net_frames[None, :], net_signals[None, :], skill)

        self._count_steps += 1

        return dict_action
