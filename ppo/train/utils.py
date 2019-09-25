import os
import torch
import numpy as np

from argparse import Namespace
from gym.spaces import Discrete

from ppo.parts import misc
from ppo.parts import stats
from ppo.parts import load
from ppo.parts import log
from ppo.parts.algo import PPO
from ppo.parts.model import MasterPolicy
from ppo.parts.storage import RolloutStorage
from ppo.envs.batch import BatchEnv

import bc.utils.misc as bc_misc


def init_training(args, logdir):
    # get the device before loading to enable the GPU/CPU transfer
    device = torch.device(bc_misc.get_device(args.device))
    print('Running the experiments on {}'.format(device))

    # try to load from a checkpoint
    loaded_dict = load.ppo_model(logdir, device)
    if loaded_dict:
        args = loaded_dict['args']
    else:
        args, bc_model, bc_statistics = load.bc_model(args, device)
    misc.seed_exp(args)
    log.init_writers(os.path.join(logdir, 'train'), eval_logdir=None)
    if args.write_gifs:
        args.gifdir = os.path.join(logdir, 'gifs')
        print('Gifs will be written to {}'.format(args.gifdir))
        if not os.path.exists(args.gifdir):
            os.mkdir(args.gifdir)
            for env_idx in range(args.num_processes):
                if not os.path.exists(os.path.join(args.gifdir, '{:02d}'.format(env_idx))):
                    os.mkdir(os.path.join(args.gifdir, '{:02d}'.format(env_idx)))

    # create the parallel envs
    envs = BatchEnv(args)

    # create the policy
    action_space = Discrete(args.num_skills)
    if loaded_dict:
        policy = loaded_dict['policy']
        start_step, start_epoch = loaded_dict['start_step'], loaded_dict['start_epoch']
    else:
        policy = create_policy(args, envs, action_space, bc_model, bc_statistics)
        start_step, start_epoch = 0, 0
    policy.to(device)

    # create the PPO algo
    agent = create_agent(args, policy)

    if loaded_dict:
        # load normalization and optimizer statistics
        envs.obs_running_stats = loaded_dict['obs_running_stats']
        load.optimizer(agent.optimizer, loaded_dict['optimizer_state_dict'], device)

    exp_dict = dict(
        device=device,
        envs=envs,
        start_epoch=start_epoch,
        start_step=start_step,
        action_space=action_space,
    )

    # create or load stats
    if loaded_dict:
        stats_global, stats_local = loaded_dict['stats_global'], loaded_dict['stats_local']
    else:
        stats_global, stats_local = stats.init(args.num_processes)
    return args, policy, agent, stats_global, stats_local, Namespace(**exp_dict)


def create_policy(args, envs, action_space, bc_model, bc_statistics):
    policy = MasterPolicy(
        envs.observation_space.shape,
        action_space,
        bc_model,
        bc_statistics,
        **vars(args))
    return policy


def create_agent(args, policy):
    agent = PPO(
        policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
        args.entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
    return agent


def create_rollout_storage(
        args, envs_train, policy, action_space, action_space_skills, device):
    rollouts = RolloutStorage(
        args.num_master_steps_per_update,
        args.num_processes,
        envs_train.observation_space.shape,
        action_space,
        action_memory=args.action_memory)

    obs = envs_train.reset()
    rollouts.obs[0].copy_(misc.dict_to_tensor(obs)[0])
    rollouts.to(device)
    return rollouts, obs


def get_policy_values(
        policy,
        obs,
        memory_actions,
        policy_values_cache,
        need_master_action,
        deterministic=False):
    ''' The function is sampling the policy actions '''
    with torch.no_grad():
        value_new, action_new, log_prob_new = policy.act(
            obs, memory_actions, deterministic)

        if policy_values_cache is None:
            return value_new, action_new, log_prob_new

        value, action, log_prob = policy_values_cache
        for env_idx in np.where(need_master_action)[0]:
            value[env_idx] = value_new[env_idx]
            action[env_idx] = action_new[env_idx]
            log_prob[env_idx] = log_prob_new[env_idx]
        return value, action, log_prob


def do_master_step(action_master, obs, reward_master, policy, envs, args):
    # we expect the action_master to have an action for each env
    # obs contains observations only for the envs that did a step and need a new skill action
    assert len(action_master.keys()) == envs.num_processes
    info_master = np.array([None] * envs.num_processes)
    done_master = np.array([False] * envs.num_processes)
    while True:
        if args.use_expert_scripts:
            # create a dictionary out of master action values
            action_skill_dict = {}
            for env_idx, env_action_master in action_master.items():
                if env_idx in obs.keys():
                    action_skill_dict[env_idx] = {'skill': env_action_master}
        else:
            # get the skill action for env_idx in obs.keys()
            with torch.no_grad():
                action_skill_dict, env_idxs = policy.get_worker_action(action_master, obs)
        obs, reward_envs, done_envs, info_envs = envs.step(action_skill_dict)
        need_master_action = update_master_variables(
            num_envs=envs.num_processes,
            env_idxs=obs.keys(),
            envs_dict={'reward': reward_envs, 'done': done_envs, 'info': info_envs},
            master_dict={'reward': reward_master, 'done': done_master, 'info': info_master})
        if np.any(need_master_action):
            break

    return (obs, reward_master, done_master, info_master, need_master_action)


def update_master_variables(num_envs, env_idxs, envs_dict, master_dict):
    '''' Returns a numpy array of 0/1 with indication which env needs a master action '''
    need_master_action = np.zeros((num_envs,))
    for env_idx in env_idxs:
        if envs_dict['info'][env_idx]['need_master_action']:
            need_master_action[env_idx] = 1
        if envs_dict['done'][env_idx]:
            need_master_action[env_idx] = 1
            master_dict['done'][env_idx] = True
        if need_master_action[env_idx]:
            master_dict['info'][env_idx] = envs_dict['info'][env_idx]
        master_dict['reward'][env_idx] += envs_dict['reward'][env_idx]
    return need_master_action


def update_memory_actions(memory_actions, action, need_master_action, done):
    ''' Updates the actions passed to the policy as a memory. '''
    if memory_actions is None:
        return memory_actions
    for env_idx in np.where(need_master_action)[0]:
        memory_actions[env_idx][:-1] = memory_actions[env_idx][1:]
        memory_actions[env_idx][-1] = action[env_idx][0]
    for env_idx, done_ in enumerate(done):
        if done_:
            memory_actions[env_idx][:] = -1.
    return memory_actions


def perform_skill_sequence(skill_sequence, observation, policy, envs, args):
    reward = torch.zeros((args.num_processes, 1)).type_as(observation[0])
    skill_counters = [0] * args.num_processes
    dones = [False] * args.num_processes
    while True:
        master_action_dict = {env_idx: torch.Tensor([skill_sequence[skill_counter]]).int()
                              for env_idx, skill_counter in enumerate(skill_counters)}
        observation, reward, done, _, need_master_action = do_master_step(
            master_action_dict, observation, reward, policy, envs, args)
        for env_idx, need_master_action_flag in enumerate(need_master_action):
            if need_master_action_flag:
                if skill_counters[env_idx] == len(skill_sequence) - 1:
                    skill_counters[env_idx] = 0
                    dones[env_idx] = True
                else:
                    skill_counters[env_idx] += 1
        print('rewards = {}'.format(reward[:, 0]))
        if all(dones):
            break
    envs.reset()
