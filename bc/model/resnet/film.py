import torch
import torch.nn as nn

from bc.model.resnet.feat import ResNetFeat
from bc.model.resnet.utils import init_weights, make_resnet_layer, make_master_head


class ResNetFilm(ResNetFeat):
    def __init__(self,
                 block=None,
                 layers=None,
                 block_depth=(64, 128, 256, 512),
                 fcs=tuple(),
                 input_dim=3,
                 output_dim=8,
                 num_skills=1,
                 return_features=False,
                 num_conv_layers=4,
                 num_conv_filters=128,
                 master_head_type='fc',
                 size_master_conv_filters=3,
                 **kwargs):
        super(ResNetFilm, self).__init__(block,
                                         layers,
                                         block_depth,
                                         input_dim,
                                         return_features,
                                         filmed=True,
                                         condition_dim=num_skills,
                                         **kwargs)
        self.num_skills = num_skills

        # one fc head to embed the condition in feature space
        self.num_conv_filters = num_conv_filters

        head_fc = []
        head_fc.append(nn.Linear(block_depth[-1], output_dim))
        self.head_fc = nn.Sequential(*head_fc)

        init_weights(self.modules())

    def to_one_hot(self, skill):
        oh_skill = (skill.reshape(-1, 1) == torch.arange(self.num_skills).to(
            skill.device)).float()
        return oh_skill

    def forward(self, obs, skill):
        # visual features shared between skills
        assert skill is not None
        oh_skill = self.to_one_hot(skill)
        x_shared = super(ResNetFilm, self).forward(obs, oh_skill, apply_avg_pool=False)

        x_head_conv = self.avgpool(x_shared)
        x_head_conv = x_head_conv.view(x_head_conv.size(0), -1)
        x_head = self.head_fc(x_head_conv)

        return x_head if not self.return_features else (x_head, x_shared)
