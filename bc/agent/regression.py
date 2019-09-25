import numpy as np

from bc.model import utils
from bc.agent.image import ImageAgent
from bc.utils.misc import get_device


class RegressionAgent(ImageAgent):
    def __init__(self, path, epoch, device='cpu'):
        super(RegressionAgent, self).__init__(path, epoch, max_steps=None)
        self._device = get_device(device)
        self.model = self._load_model(path, epoch)
        super(RegressionAgent, self).reset()

    def _load_model(self, path, epoch):
        model, optimizer = utils.load_model(path, epoch, self._device)
        model.set_eval()
        return model

    def get_prediction(self, batch_obs):
        ''' Returns regression predictions in meters (denormalized). '''
        if len(batch_obs.shape) == 3:
            # add the batch dimension
            batch_obs = batch_obs[None]
        batch_pred = self.model.get_dict_action(batch_obs)
        batch_output = []
        for pred in batch_pred:
            output_dict = {}
            for key_action in self.model.signal_keys:
                if key_action != 'grip_velocity':
                    assert key_action in pred
                    signal_idx = np.where(np.array(self.model.signal_keys) == key_action)[0][0]
                    signal_length = self.model.signal_lengths[signal_idx]
                    mean, std = self.model.statistics['train']['state'][key_action]
                    pred_denorm = mean[:signal_length] + pred[key_action] * std[:signal_length]
                    output_dict[key_action] = pred_denorm
                else:
                    raise NotImplementedError('predicted key_action is grip_velocity')
            batch_output.append(output_dict)

        return batch_output
