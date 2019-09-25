import torch

from bc.model.base import MetaNetwork, CHANNELS
from bc.model.resnet.config import make_resnet
from bc.dataset import Actions


class Regression(MetaNetwork):
    def __init__(self,
                 archi,
                 mode,
                 action_space,
                 path=None,
                 input_type=('depth', ),
                 signal_keys=None,
                 signal_lengths=None,
                 statistics=None,
                 device='cuda',
                 **kwargs):
        super(Regression, self).__init__()

        # attributes of MetaNetwork
        self.archi = archi
        self.input_type = input_type
        self.action_keys, self.dim_action = Actions.action_space_to_keys(
            action_space)
        self.input_dim = CHANNELS[tuple(self.input_type)]
        self.output_dim = self.dim_action
        self.statistics = {'train': statistics, 'gt': statistics}

        self.signal_keys = []
        for signal_key in signal_keys:
            self.signal_keys.append(signal_key[-1])
        self.signal_lengths = signal_lengths

        self.args = self._get_args()
        self.net = make_resnet(
            archi, mode, self.input_dim, output_dim=self.output_dim)
        self.to(device)

    def _get_args(self):
        args = dict(
            archi=self.archi,
            input_type=self.input_type,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return args

    def __call__(self, obs, signals=None, skill=None, compute_grad=True):
        obs = obs.to(self.device)
        if not compute_grad:
            torch.set_grad_enabled(False)
        pred = self.net(obs)
        if not compute_grad:
            torch.set_grad_enabled(True)
        return pred

    def compute_error(self, pred, gt, signal_key, signal_length):
        with torch.no_grad():
            pred_and_gt = []
            for x, data in zip([pred, gt], ['train', 'gt']):
                mean, std = self.statistics[data]['state'][signal_key]
                mean, std = mean.flatten(), std.flatten()
                mean = torch.tensor(mean).type_as(pred)[:signal_length]
                std = torch.tensor(std).type_as(pred)[:signal_length]
                pred_and_gt.append(mean + x * std)
            pred_meter, gt_meter = pred_and_gt
            error = torch.norm(pred_meter - gt_meter, dim=1).mean()
            if 'position' in signal_key:
                # show position errors in centimeters
                error = error * 100
            return error

    def compute_loss(self,
                     obs,
                     targets_dict,
                     signals,
                     skills,
                     eval=False):
        loss_total, losses_dict = 0, {}
        pred = self.__call__(obs)
        idx_shift = 0
        # pred is given for all the signals, we will loop over them and calculate l2
        for signal_key, signal_length in zip(self.signal_keys,
                                             self.signal_lengths):
            targets = targets_dict[signal_key].to(self.device)
            assert signal_length == targets.shape[1]
            loss_signal = self.l2_loss(
                pred[:, idx_shift:idx_shift + signal_length], targets)
            loss_total += loss_signal
            losses_dict[signal_key] = loss_signal
            if self.statistics['train'] is not None:
                signal_error = self.compute_error(
                    pred[:, idx_shift:idx_shift + signal_length], targets,
                    signal_key, signal_length)
                losses_dict['{}_error'.format(signal_key)] = signal_error
            idx_shift += targets.shape[1]
        assert idx_shift == pred.shape[
            1], 'too many or not enough predictions were made'
        if not eval:
            loss_total.backward()
        return losses_dict

    def get_dict_action(self, batch_obs):
        pred = self.__call__(batch_obs, compute_grad=False)
        action_dicts = []
        for x in pred:
            x_np = x.cpu().numpy()
            action_dict = {}
            idx_shift = 0
            for signal_key, signal_length in zip(self.signal_keys,
                                                 self.signal_lengths):
                action_dict[signal_key] = x_np[idx_shift:idx_shift +
                                               signal_length]
                idx_shift += signal_length
            action_dicts.append(action_dict)
        return action_dicts
