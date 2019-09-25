import torch
import torch.nn as nn

from bc.model.resnet.feat import ResNetFeat
from bc.model.resnet.utils import init_weights


class ResNetFlat(ResNetFeat):
    def __init__(self,
                 block=None,
                 layers=None,
                 block_depth=(64, 128, 256, 512),
                 fcs=tuple(),
                 input_dim=3,
                 output_dim=1000,
                 return_features=False,
                 **kwargs):
        self.inplanes = 64
        super(ResNetFlat, self).__init__(block, layers, block_depth, input_dim,
                                         return_features, **kwargs)

        # fully connected layers on top of features
        fc_size = [block_depth[3] * block.expansion] + fcs + [output_dim]
        fcs = []
        for i in range(len(fc_size) - 1):
            fcs.append(nn.Linear(fc_size[i], fc_size[i + 1]))
            if i < len(fc_size) - 2:
                fcs.append(nn.ReLU())
        self.fcs = nn.Sequential(*fcs)

        init_weights(self.modules())

    def forward(self, obs, signals=None):
        features = super(ResNetFlat, self).forward(obs)
        x = self.fcs(features)

        return x if not self.return_features else (x, features)
