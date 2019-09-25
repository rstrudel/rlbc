import torch
import torch.nn as nn

from bc.model.resnet.feat import ResNetFeat
from bc.model.resnet.utils import init_weights, make_resnet_layer, make_master_head


class ResNetSkills(ResNetFeat):
    def __init__(
            self,
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
        super(ResNetSkills, self).__init__(
            block, layers, block_depth, input_dim, return_features, **kwargs)
        self.num_skills = num_skills

        head_convs, head_fcs = [], []
        # one conv head per skill
        for k in range(num_skills):
            # first create convs
            head_conv, insize, inplanes = make_resnet_layer(
                block=block,
                planes=num_conv_filters,
                num_blocks=num_conv_layers,
                normalization=self.normalization,
                insize=int(self.insize),
                inplanes=self.inplanes,
                stride=1)
            head_convs.append(head_conv)

            # then create fully connected layers (if required)
            head_fc = []
            current_dim = num_conv_filters
            for i, fc_size in enumerate(fcs):
                head_fc.append(nn.Linear(current_dim, fc_size))
                if i < len(fcs) - 1:
                    head_fc.append(nn.ReLU())
                current_dim = fc_size
            head_fc.append(nn.Linear(current_dim, output_dim))
            head_fcs.append(nn.Sequential(*head_fc))

        # master head
        master_conv, master_fc = make_master_head(
            master_head_type=master_head_type,
            num_skills=self.num_skills,
            num_channels=64,
            insize=int(self.insize),
            inplanes=self.inplanes,
            size_conv_filters=size_master_conv_filters)
        head_convs.append(master_conv)
        head_fcs.append(master_fc)

        self.head_convs = nn.Sequential(*head_convs)
        self.head_fcs = nn.Sequential(*head_fcs)

        init_weights(self.modules())

    def forward(self, obs, signals=None):
        x_shared = super(ResNetSkills, self).forward(
            obs, apply_avg_pool=False)
        x_tot = torch.zeros(0).type_as(x_shared)
        for head_conv, head_fc in zip(self.head_convs, self.head_fcs):
            x_head_conv = x_shared
            if head_conv is not None:
                x_head_conv = head_conv(x_head_conv)
            x_head_conv = self.avgpool(x_head_conv)
            x_head_conv = x_head_conv.view(x_head_conv.size(0), -1)
            x_head = head_fc(x_head_conv)
            x_tot = torch.cat((x_tot, x_head), 1)
        return x_tot if not self.return_features else (x_tot, x_shared)
