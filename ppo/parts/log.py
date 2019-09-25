import torch

import copy
import os
import time
import numpy as np

from tensorboardX import SummaryWriter

train_writer, eval_writer = None, None


def init_writers(train_logdir, eval_logdir):
    global train_writer, eval_writer
    if train_logdir:
        train_writer = SummaryWriter(train_logdir)
    if eval_logdir:
        eval_writer = SummaryWriter(eval_logdir)


def add_summary(tag, value, iter, stage='train'):
    if stage == 'train':
        writer = train_writer
    elif stage == 'eval':
        writer = eval_writer
    else:
        raise NotImplementedError
    writer.add_scalar(tag, value, iter)


def log_train(total_steps, start, stats, action_loss, value_loss, entropy, epoch):
    end = time.time()
    print("Training after {} steps, FPS {}".format(
        total_steps, int(total_steps / (end - start))))
    returns = stats['return']
    print("Last {} training episodes: mean reward {:.1f}, min/max reward {:.1f}/{:.1f}".format(
        len(returns), np.mean(returns),
        np.min(returns), np.max(returns)))
    for stat_key, stat_value in stats.items():
        add_summary('env/{}'.format(stat_key), np.mean(stat_value), total_steps)
        if stat_key == 'return':
            add_summary('env/{}_max'.format(stat_key), np.max(stat_value), total_steps)
    add_summary('loss/action_loss', action_loss, total_steps)
    add_summary('loss/value_loss', value_loss, total_steps)
    add_summary('loss/entropy', entropy, total_steps)
    add_summary('time/epoch', epoch, total_steps)


def log_eval(total_steps, stats):
    if 'return' in stats:
        returns = stats['return']
        print("Evaluation after {} steps using {} episodes: mean reward {:.5f}". format(
            total_steps, len(returns), np.mean(returns)))
    for stat_key, stat_value in stats.items():
        add_summary('env/{}'.format(stat_key), np.mean(stat_value), total_steps, 'eval')
        if stat_key == 'return':
            add_summary('env/{}_max'.format(stat_key), np.max(stat_value), total_steps, 'eval')


def save_model(
        save_path,
        policy,
        optimizer,
        epoch,
        env_steps,
        device,
        envs,
        args,
        stats_global,
        stats_local):
    if save_path == "":
        return
    # A really ugly way to save a model to CPU
    if device.type != 'cpu':
        policy = copy.deepcopy(policy).cpu()
    save_model = dict(
        policy=policy,
        optimizer_state_dict=optimizer.state_dict(),
        obs_running_stats=envs.obs_running_stats,
        start_epoch=epoch+1,
        start_step=env_steps,
        args=args,
        stats_global=stats_global,
        stats_local=stats_local,
    )

    model_name = 'model_{}.pth'.format(epoch)
    model_path = os.path.join(save_path, model_name)
    torch.save(save_model, model_path)
    current_model_symlink = os.path.join(save_path, 'model_current.pth')
    if os.path.islink(current_model_symlink):
        os.unlink(current_model_symlink)
    os.symlink(model_path, current_model_symlink)
