import torch
import torch.nn as nn

from bc.model.resnet.feat import ResNetFeat
from bc.model.resnet.utils import init_weights


class ResNetSignals(ResNetFeat):
    def __init__(self,
                 block=None,
                 layers=None,
                 block_depth=(64, 128, 256, 512),
                 fcs=tuple(),
                 input_dim=3,
                 output_dim=1000,
                 input_signal_dim=0,
                 return_features=False,
                 **kwargs):
        self.inplanes = 64

        super(ResNetSignals, self).__init__(
            block, layers, block_depth, input_dim, return_features, **kwargs)

        # fully connected layers on top of features
        fcs_cnn = [
            nn.Linear(block_depth[3] * block.expansion, 64),
            nn.ReLU(),
        ]
        self.fcs_cnn = nn.Sequential(*fcs_cnn)

        fcs_signal = [
            nn.Linear(input_signal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ]
        self.fcs_signal = nn.Sequential(*fcs_signal)

        fcs = [nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim)]
        self.fcs = nn.Sequential(*fcs)

        init_weights(self.modules())

    def forward(self, obs, signals):
        features = super(ResNetSignals, self).forward(obs)
        x = self.fcs_cnn(features)
        y = self.fcs_signal(signals)
        z = torch.cat((x, y), 1)
        z = self.fcs(z)

        return z if not self.return_features else (z, features)
