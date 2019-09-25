import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ppo.parts import misc
from ppo.parts.distributions import Categorical, DiagGaussian

from bc.dataset import Actions
from bc.model.resnet import utils as resnet_utils


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MasterPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, bc_model, bc_statistics, **base_kwargs):
        super(MasterPolicy, self).__init__()

        self.action_keys = Actions.action_space_to_keys(base_kwargs['bc_args']['action_space'])[0]
        self.statistics = bc_statistics

        if len(obs_shape) == 3:
            self.base = ResnetBase(bc_model, **base_kwargs)
            # set the eval mode so the behavior of the skills is the same as in BC training
            self.base.resnet.eval()
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    def get_worker_action(self, master_action, obs_dict):
        obs_tensor, env_idxs = misc.dict_to_tensor(obs_dict)
        master_action_filtered = []
        for env_idx in env_idxs:
            master_action_filtered.append(master_action[env_idx])
        master_action_filtered = torch.stack(master_action_filtered)
        action_tensor = self.base(obs_tensor, None, master_action=master_action_filtered)
        action_tensors_dict, env_idxs = misc.tensor_to_dict(action_tensor, env_idxs)
        action_tensors_dict_numpys = {key: value.cpu().numpy()
                                      for key, value in action_tensors_dict.items()}
        action_dicts_dict = {}
        master_action_dict, _ = misc.tensor_to_dict(master_action, env_idxs)
        for env_idx, action_tensor in action_tensors_dict_numpys.items():
            action_dict = Actions.tensor_to_dict(action_tensor, self.action_keys, self.statistics)
            action_dict['skill'] = master_action[env_idx].cpu().numpy()
            action_dicts_dict[env_idx] = action_dict
        return action_dicts_dict, env_idxs

    def act(self, inputs, memory_actions, deterministic=False):
        inputs, env_idxs = misc.dict_to_tensor(inputs)
        value, actor_features = self.base(
            inputs, misc.dict_to_tensor(memory_actions)[0])
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return (misc.tensor_to_dict(value, env_idxs)[0],
                misc.tensor_to_dict(action, env_idxs)[0],
                misc.tensor_to_dict(action_log_probs, env_idxs)[0])

    def get_value_detached(self, inputs, actions):
        inputs, env_idxs = misc.dict_to_tensor(inputs)
        value, _ = self.base(
            inputs, misc.dict_to_tensor(actions)[0])
        return misc.tensor_to_dict(value.detach(), env_idxs)[0]

    def evaluate_actions(self, inputs, actions_memory, action_master):
        ''' This function is called from the PPO update so all the arguments are tensors. '''
        value, actor_features = self.base(inputs, actions_memory)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action_master)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class NNBase(nn.Module):
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class MLPBase(NNBase):
    def __init__(self,
                 num_inputs,
                 hidden_size=64,
                 action_memory=0,
                 **kwargs):
        super(MLPBase, self).__init__(hidden_size)

        init_ = lambda m: misc.init(m,
            misc.init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs + action_memory, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs + action_memory, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, actions_memory):
        if actions_memory is not None:
            # agent has a memory
            x = torch.cat((inputs, actions_memory), dim=1)
        else:
            # the input is the frames stack
            x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor


class ResnetBase(NNBase):
    def __init__(
            self,
            bc_model=None,
            num_skills=None,
            action_memory=0,
            bc_args=None,
            master_type='conv',
            master_num_channels=64,
            master_size_conv_filters=3,
            **unused_kwargs):
        super(ResnetBase, self).__init__(master_num_channels)

        self.dim_action = bc_args['dim_action'] + 1
        self.dim_action_seq = self.dim_action * bc_args['steps_action']
        self.features_dim = bc_args['features_dim']
        assert num_skills is not None
        self.num_skills = num_skills
        self.action_memory = action_memory
        self.resnet = bc_model.net.module
        self.resnet.return_features = True

        self.master_type = master_type

        self.actor, self.critic = [self._create_head(
            master_type=self.master_type,
            num_skills=self.num_skills,
            num_channels=master_num_channels,
            inplanes=self.features_dim,
            size_conv_filters=master_size_conv_filters) for _ in range(2)]

        init_ = lambda m: misc.init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self.output_size, 1))

        self.train()

    def _create_head(self, master_type, num_skills, num_channels, inplanes, size_conv_filters):
        head_conv, head_fc = resnet_utils.make_master_head(
            master_head_type=master_type,
            num_skills=num_skills,
            num_channels=num_channels,
            inplanes=num_skills * inplanes,
            size_conv_filters=size_conv_filters)
        if master_type == 'fc':
            return head_fc
        else:
            return head_conv

    # def _get_skill_actions(self, master_action, skills_actions):
    #     skill_actions = []
    #     for env_idx, skill_id in enumerate(master_action):
    #         skill_action = skills_actions[
    #             env_idx,
    #             self.dim_action_seq * skill_id: self.dim_action_seq * skill_id + self.dim_action]
    #         skill_actions.append(skill_action)
    #     return torch.stack(skill_actions)

    def _get_skill_actions(self, master_action, skills_actions):
        return skills_actions[:, :self.dim_action]

    def forward(self, inputs, actions_memory, master_action=None):
        if master_action is None:
            # we want the master actions
            master_features = torch.zeros(inputs.shape[0],
                                          self.num_skills * self.features_dim, 7, 7).to(inputs.device)
            for i in range(self.num_skills):
                cond = (i * torch.ones(inputs.shape[0])).long().to(inputs.device)
                _, master_features_skill = self.resnet(inputs, cond)
                master_features[:, self.features_dim * i : self.features_dim * (i+1)] = master_features_skill
            hidden_critic = self.critic(master_features.detach())
            hidden_actor = self.actor(master_features.detach())
            if self.master_type != 'fc':
                for tensor in (hidden_actor, hidden_critic):
                    # we assume that both tensors are 7x7 conv activations
                    assert tensor.shape[2] == 7 and tensor.shape[3] == 7
                hidden_critic = F.avg_pool2d(hidden_critic, hidden_critic.shape[-1])[..., 0, 0]
                hidden_actor = F.avg_pool2d(hidden_actor, hidden_actor.shape[-1])[..., 0, 0]
            if actions_memory is not None:
                # normalize the actions memory to be in [-1, 1]
                actions_memory = 2 * actions_memory / (self.num_skills - 1) - 1
                hidden_critic = torch.cat((hidden_critic, actions_memory), dim=1)
                hidden_actor = torch.cat((hidden_actor, actions_memory), dim=1)

            return self.critic_linear(hidden_critic), hidden_actor
        else:
            # we want the skill actions from the BC checkpoint
            skills_actions, master_features = self.resnet(inputs, master_action)
            assert len(master_action) == len(master_features)
            return self._get_skill_actions(master_action, skills_actions)

    @property
    def output_size(self):
        return self._hidden_size + self.action_memory
