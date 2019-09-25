import torch
import torch.nn as nn


from bc.model.resnet.config import make_resnet
from bc.dataset import Actions
from bc.utils.misc import get_device

CHANNELS = {('depth', ): 1, ('rgb', ): 3, ('depth', 'rgb'): 4}


class MetaNetwork:
    "Base class for network"

    def __init__(self):
        self.net = None
        self.archi = None
        self.mode = None
        self.device = None
        self.args = {}
        self.env_name = ''
        self.statistics = {}

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def to(self, device):
        device_available = get_device(str(device))
        self.device = torch.device(device_available)
        if str(device) != 'cpu':
            self.net = nn.DataParallel(self.net)
            self.net.to(self.device)

    def __call__(self, obs, compute_grad):
        raise NotImplementedError

    def set_eval(self):
        self.net.eval()

    def get_n_param(self):
        n_param = 0
        for parameter in self.net.parameters():
            shape = parameter.size()
            size = 1
            for dim in shape:
                size *= dim
            n_param += size
        return n_param

    def compute_loss(self, **kwargs):
        raise NotImplementedError

    def get_action(self, **kwargs):
        raise NotImplementedError

    def get_dict_action(self, **kwargs):
        raise NotImplementedError


class MetaPolicy(MetaNetwork):
    "Vanilla Architecture"
    "One head prediciting current action and future actions using concatenated Standard Blocks"

    def __init__(self,
                 archi,
                 mode,
                 num_frames,
                 action_space,
                 steps_action,
                 lam_grip=0.1,
                 input_type=('depth', ),
                 env_name='',
                 image_augmentation='',
                 statistics=None,
                 device='cuda',
                 network_extra_args=None,
                 **unused_kwargs):
        super(MetaPolicy, self).__init__()

        self.num_frames = num_frames
        self.action_space = action_space
        self.steps_action = len(steps_action)
        self.action_keys, self.dim_action = Actions.action_space_to_keys(
            action_space)
        self.dim_prediction = (self.dim_action + 1) * len(steps_action)
        self.dim_gt = self.dim_action * len(steps_action)
        self.lam_grip = lam_grip
        input_type = tuple(sorted(input_type))
        assert input_type in CHANNELS
        self.input_dim = CHANNELS[input_type] * num_frames
        self.output_dim = (self.dim_action + 1) * self.steps_action

        # attributes of MetaNetwork
        self.archi = archi
        self.mode = mode
        self.input_type = input_type
        self.env_name = env_name
        self.image_augmentation = image_augmentation
        self.statistics = statistics if statistics is not None else {}
        if network_extra_args is None:
            network_extra_args = {}

        self.net = make_resnet(
            archi,
            mode,
            self.input_dim,
            output_dim=self.output_dim,
            **network_extra_args)
        self.to(device)
        self.args = self._get_args()

    def _get_args(self):
        if hasattr(self.net, 'module'):
            features_dim = self.net.module.features_dim
        else:
            features_dim = self.net.features_dim
        args = dict(
            archi=self.archi,
            mode=self.mode,
            input_type=self.input_type,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_frames=self.num_frames,
            action_space=self.action_space,
            dim_action=self.dim_action,
            steps_action=self.steps_action,
            lam_grip=self.lam_grip,
            features_dim=features_dim,
            env_name=self.env_name,
            image_augmentation=self.image_augmentation)
        return args

    def __call__(self, obs, signals=None, compute_grad=True):
        obs = obs.to(self.device)
        if not compute_grad:
            torch.set_grad_enabled(False)
        if signals is not None:
            signals = signals.to(self.device)
        pred = self.net(obs, signals)
        if not compute_grad:
            torch.set_grad_enabled(True)
        return pred

    def get_dict_action(self, obs, signals=None, skill=None):
        # get the action in the mime format (dictionary)
        action_tensor = self.get_action(obs, signals, skill)
        dict_action = Actions.tensor_to_dict(action_tensor, self.action_keys,
                                             self.statistics)
        return dict_action

    def get_action(self, obs, signals=None, skill=None):
        # get the action as a tensor
        assert obs.shape[0] == 1
        assert skill is None
        pred = self.__call__(obs, signals, compute_grad=False)
        pred = pred[0].cpu().numpy()
        pred = pred[:self.dim_action + 1]
        return pred
