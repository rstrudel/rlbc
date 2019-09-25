import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def make_resnet_layer(block,
                      planes,
                      num_blocks,
                      normalization,
                      insize,
                      inplanes,
                      stride=1,
                      filmed=False,
                      condition_dim=0):
    downsample = None
    insize /= stride
    if stride != 1 or inplanes != planes * block.expansion:
        if normalization == 'batchnorm':
            normalization_layer = nn.BatchNorm2d(planes * block.expansion)
        elif normalization == 'layernorm':
            normalization_layer = nn.LayerNorm(
                [planes * block.expansion,
                 int(insize),
                 int(insize)])
        else:
            normalization_layer = nn.InstanceNorm2d(planes * block.expansion)
        downsample = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes * block.expansion,
                      kernel_size=1,
                      stride=stride,
                      bias=False),
            normalization_layer,
        )

    layers = []
    layers.append(
        block(inplanes,
              planes,
              stride,
              downsample,
              insize,
              normalization,
              filmed=filmed,
              condition_dim=condition_dim))
    inplanes = planes * block.expansion
    for i in range(1, num_blocks):
        layers.append(
            block(inplanes,
                  planes,
                  insize=insize,
                  normalization=normalization,
                  filmed=filmed,
                  condition_dim=condition_dim))

    return nn.Sequential(*layers), insize, inplanes


def make_master_head(master_head_type,
                     num_skills=1,
                     num_channels=64,
                     inplanes=256,
                     insize=7,
                     size_conv_filters=3):
    # heads constructed on the assumption that input are feature maps of the shape
    # insize x insize x C
    if master_head_type == 'fc':
        head_conv = None
        head_fc = [
            nn.AvgPool2d(insize, stride=1),
            Flatten(),
            nn.Linear(inplanes, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels)
        ]
    elif master_head_type == 'conv':
        head_conv = [
            nn.Conv2d(inplanes,
                      num_channels,
                      size_conv_filters,
                      padding=int((size_conv_filters - 1) / 2)),
            nn.Tanh(),
            nn.Conv2d(num_channels,
                      num_channels,
                      size_conv_filters,
                      padding=int((size_conv_filters - 1) / 2)),
            nn.Tanh()
        ]
        head_conv = nn.Sequential(*head_conv)
        head_fc = [nn.Linear(num_channels, num_skills)]
    else:
        raise ValueError(
            'Unknown master head type: {}'.format(master_head_type))
    head_fc = nn.Sequential(*head_fc)
    return head_conv, head_fc


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear)):
            m.reset_parameters()  # default method used to init the layer
