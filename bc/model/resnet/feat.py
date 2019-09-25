import torch.nn as nn

from bc.model.resnet.utils import make_resnet_layer, init_weights


class ResNetFeat(nn.Module):
    def __init__(self,
                 block=None,
                 layers=None,
                 block_depth=(64, 128, 256, 512),
                 input_dim=3,
                 return_features=False,
                 normalization='batchnorm',
                 filmed=False,
                 condition_dim=0):
        self.inplanes = 64
        self.features_dim = block_depth[-1]
        self.normalization = normalization
        super(ResNetFeat, self).__init__()
        self.conv1 = nn.Conv2d(
            input_dim,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.insize = 224 / 2  # we use stride=2 in self.conv1
        assert normalization in ('batchnorm', 'layernorm', 'instancenorm')
        if normalization == 'batchnorm':
            self.norm1 = nn.BatchNorm2d(self.inplanes)
        elif normalization == 'layernorm':
            self.norm1 = nn.LayerNorm(
                [self.inplanes,
                 int(self.insize),
                 int(self.insize)])
        else:
            self.norm1 = nn.InstanceNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.insize /= 2  # max pooling
        self.layer1, self.insize, self.inplanes = make_resnet_layer(
            block,
            block_depth[0],
            layers[0],
            normalization,
            self.insize,
            self.inplanes,
            stride=1,
            filmed=filmed,
            condition_dim=condition_dim)
        self.layer2, self.insize, self.inplanes = make_resnet_layer(
            block,
            block_depth[1],
            layers[1],
            normalization,
            self.insize,
            self.inplanes,
            stride=2,
            filmed=filmed,
            condition_dim=condition_dim)
        self.layer3, self.insize, self.inplanes = make_resnet_layer(
            block,
            block_depth[2],
            layers[2],
            normalization,
            self.insize,
            self.inplanes,
            stride=2,
            filmed=filmed,
            condition_dim=condition_dim)
        self.layer4, self.insize, self.inplanes = make_resnet_layer(
            block,
            block_depth[3],
            layers[3],
            normalization,
            self.insize,
            self.inplanes,
            stride=2,
            filmed=filmed,
            condition_dim=condition_dim)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        init_weights(self.modules())
        self.return_features = return_features

    def forward(self, x, condition=None, apply_avg_pool=True):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, condition)

        if apply_avg_pool:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        return x
