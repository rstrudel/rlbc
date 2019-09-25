from bc.model.resnet.blocks import BasicBlock, Bottleneck
from bc.model.resnet.flat import ResNetFlat
from bc.model.resnet.feat import ResNetFeat
from bc.model.resnet.signals import ResNetSignals
from bc.model.resnet.skills import ResNetSkills

from bc.model.resnet.film import ResNetFilm


def make_resnet(archi, mode, input_dim, **kwargs):
    model_class, block, layers, block_depth, fcs, extra_args = config_to_params(
        archi, mode)
    kwargs.update(extra_args)
    return model_class(block=block,
                       layers=layers,
                       block_depth=block_depth,
                       fcs=fcs,
                       input_dim=input_dim,
                       **kwargs)

# name of archi resnet_{num_layers}_{suffix}
def config_to_params(archi, mode):
    # define model class
    mode_to_class = {
        'flat': ResNetFlat,
        'features': ResNetFeat,
        'regression': ResNetFlat,
        'signals': ResNetSignals,
        'skills': ResNetSkills,
        'film': ResNetFilm,
    }

    # define network related parameters
    assert archi.startswith('resnet')
    splits = archi.split('_')
    assert len(splits) <= 3, 'Unknown architecture {}'.format(archi)
    if len(splits) == 2:
        splits.append('')
    archi, num_layers, suffix = splits
    num_layers = int(num_layers)
    num_layers_to_block_layers = {
        10: (BasicBlock, [1, 1, 1, 1]),
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3])
    }
    suffix_to_block_depth = {
        '': [64, 128, 256, 512],
        'narrow32': [32, 64, 128, 256],
        'narrow16': [16, 32, 64, 128],
        'narrow8': [8, 16, 32, 64]
    }
    mode_to_fcs = {
        'signals': [64, 64],
    }
    mode_to_extra_args = {
        'skills': {
            'num_conv_layers': 2,
            'num_conv_filters': 64,
            'master_head_type': 'conv',
            'size_master_conv_filters': 1
        },
        'film': {
            'num_conv_layers': 2,
            'num_conv_filters': 256,
            'master_head_type': 'conv',
            'size_master_conv_filters': 1
        },
    }

    model_class = mode_to_class[mode]
    block, layers = num_layers_to_block_layers[num_layers]
    block_depth = suffix_to_block_depth[suffix]
    fcs = []
    extra_args = {}
    if mode in mode_to_fcs:
        fcs = mode_to_fcs[mode]
    if mode in mode_to_extra_args:
        extra_args = mode_to_extra_args[mode]

    return model_class, block, layers, block_depth, fcs, extra_args
