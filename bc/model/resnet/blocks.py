import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 insize=None,
                 normalization='batchnorm',
                 filmed=False,
                 condition_dim=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.planes = planes
        self.stride = stride
        self.filmed = filmed
        self.condition_dim = condition_dim
        if self.filmed:
            self.head_cond = nn.Linear(condition_dim, 2 * planes)

        self._init_normalization(normalization, planes, insize)

    def _init_normalization(self, normalization, planes, insize):
        if normalization == 'batchnorm':
            self.norm1, self.norm2 = [nn.BatchNorm2d(planes) for _ in range(2)]
        elif normalization == 'layernorm':
            self.norm1, self.norm2 = [
                nn.LayerNorm([planes, int(insize),
                              int(insize)]) for _ in range(2)
            ]
        elif normalization == 'instancenorm':
            self.norm1, self.norm2 = [
                nn.InstanceNorm2d(planes) for _ in range(2)
            ]
        elif normalization == 'identity':
            self.norm1, self.norm2 = [Identity() for _ in range(2)]
        else:
            raise ValueError(
                'Unknown normalization layer: {}'.format(normalization))

    def forward(self, x, condition=None):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        if self.filmed and condition is not None:
            x_cond = self.head_cond(condition)
            gamma, beta = x_cond[:, :self.planes], x_cond[:, self.planes:]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out.shape[-2], out.shape[-1])
            beta = beta.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out.shape[-2], out.shape[-1])
            out = out * (1 + gamma) + beta

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 insize=None,
                 normalization='batchnorm'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
