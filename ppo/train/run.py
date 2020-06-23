import os
import time
import torch
import numpy as np

from tqdm import tqdm
from sacred import Experiment
from argparse import Namespace
from gym.spaces import Box

from ppo.parts import log
from ppo.parts import stats
from ppo.train import utils
from configs.ppo import train_ingredient


ex = Experiment('train', ingredients=[train_ingredient])


@ex.capture
def parse_args(general, ppo, bc, hierarchy, log):
    args = Namespace(**general, **ppo, **bc, **hierarchy, **log)
    if args.dask_batch_size is None:
        args.dask_batch_size = max(1, int(args.num_processes / 2))
    return args


@ex.automain
def main():
    args = parse_args()
    logdir = os.path.join(os.environ['RLBC_MODELS'], args.folder, 'seed{:02}'.format(args.seed))
    args, policy, agent, stats_global, stats_local, exp_vars = utils.init_training(args, logdir)
    envs = exp_vars.envs
    action_space_skills = Box(-np.inf, np.inf, (args.bc_args['dim_action'],), dtype=np.float)
    rollouts, obs = utils.create_rollout_storage(
        args, envs, policy, exp_vars.action_space, action_space_skills, exp_vars.device)
    start = time.time()

    if args.pudb:
        import pudb; pudb.set_trace()
    env_steps = exp_vars.start_step
    reward = torch.zeros((args.num_processes, 1)).type_as(obs[0])
    need_master_action, policy_values_cache = np.ones((args.num_processes,)), None
    for epoch in range(exp_vars.start_epoch, args.train_epochs):
        print('Starting epoch {}'.format(epoch))
        master_steps_done = 0
        pbar = tqdm(total=args.num_master_steps_per_update * args.num_processes)
        while master_steps_done < args.num_master_steps_per_update * args.num_processes:
            value, action, action_log_prob = utils.get_policy_values(
                policy,
                rollouts.get_last(rollouts.obs),
                rollouts.get_last(rollouts.actions),
                policy_values_cache,
                need_master_action)
            policy_values_cache = value, action, action_log_prob

            # Observe reward and next obs
            obs, reward, done, infos, need_master_action = utils.do_master_step(
                action, obs, reward, policy, envs, args)
            master_steps_done += np.sum(need_master_action)
            pbar.update(np.sum(need_master_action))

            stats_global, stats_local = stats.update(
                stats_global, stats_local, reward, done, infos, args)

            # If done then clean the history of observations.
            masks = {i: torch.FloatTensor([0.0] if done_ else [1.0]) for i, done_ in enumerate(done)}
            # check that the obs dictionary contains obs from all envs that will be stored in rollouts
            # we only store observations from envs which need a master action
            assert len(set(np.where(need_master_action)[0]).difference(obs.keys())) == 0
            rollouts.insert(
                obs,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                indices=np.where(need_master_action)[0])
            reward[np.where(done)] = 0
            env_steps += sum([info['length_after_new_action']
                              for info in np.array(infos)[np.where(need_master_action)[0]]])
        pbar.close()

        # master policy training
        with torch.no_grad():
            next_value = policy.get_value_detached(
                rollouts.get_last(rollouts.obs),
                rollouts.get_last(rollouts.actions))
        rollouts.compute_returns(next_value, args.gamma)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # saving the model
        if epoch % args.save_interval == 0:
            print('Saving the model after epoch {} for offline evaluation'.format(epoch))
            log.save_model(
                logdir, policy, agent.optimizer, epoch, env_steps, exp_vars.device, envs, args,
                stats_global, stats_local)

        # logging
        if epoch % args.log_interval == 0 and len(stats_global['length']) > 1:
            log.log_train(
                env_steps, start, stats_global, action_loss, value_loss, dist_entropy, epoch)
        if env_steps > args.num_train_timesteps:
            print('Number of env steps reached the maximum number of frames')
            break
