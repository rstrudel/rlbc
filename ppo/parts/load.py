import os
import torch

from bc.dataset import Actions
from ppo.settings import MODEL_LOGDIR


def ppo_model(logdir, device):
    loaded_tuple = None
    model_path = os.path.join(logdir, 'model_current.pth')
    if os.path.exists(model_path):
        if str(device) == 'cpu':
            loaded_tuple = torch.load(
                model_path, map_location=lambda storage, loc: storage)
        else:
            loaded_tuple = torch.load(model_path)
        print('loaded a policy from {}'.format(model_path))
    return loaded_tuple


def bc_model(args, device):
    if args.bc_model_name:
        assert args.bc_model_epoch is not None, 'bc model epoch is not specified'
        bc_model_path = os.path.join(MODEL_LOGDIR, args.bc_model_name, 'model_{}.pth'.format(
            args.bc_model_epoch))
        if device.type == 'cpu':
            loaded_dict = torch.load(bc_model_path, map_location=lambda storage, loc: storage)
        else:
            loaded_dict = torch.load(bc_model_path)
        args.bc_args = loaded_dict['args']
        print('loaded the BC checkpoint from {}'.format(bc_model_path))
        return args, loaded_dict['model'], loaded_dict['statistics']
    else:
        if 'Cam' in args.env_name:
            default_bc_args = dict(
                archi='resnet_18',
                mode='features',
                input_dim=3,
                num_frames=3,
                steps_action=4,
                action_space='tool_lin',
                dim_action=4,
                features_dim=512,
                env_name=args.env_name,
                input_type='depth')
            print('did not load a BC checkpoint, using default BC args: {}'.format(default_bc_args))
        else:
            assert args.mime_action_space is not None
            default_bc_args = dict(
                action_space=args.mime_action_space,
                dim_action=Actions.action_space_to_keys(args.mime_action_space)[1],
                num_frames=1,
                env_name=args.env_name,
                input_type='full_state')
            print('Using a full state env with BC args: {}'.format(default_bc_args))
        args.bc_args = default_bc_args
        return args, None, None


def optimizer(optimizer, optimizer_state_dict, device):
    optimizer.load_state_dict(optimizer_state_dict)
    target_device = 'cpu' if device.type == 'cpu' else 'cuda'
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = getattr(v, target_device)()
