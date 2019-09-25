import os
import numpy as np

from tensorboardX import SummaryWriter

train_writer, eval_writer, test_writer = None, None, None


def init_writers(logdir, init_eval=True, include_test=False):
    global train_writer, eval_writer, test_writer
    train_writer = SummaryWriter(os.path.join(logdir, 'train'))
    if init_eval:
        eval_writer = SummaryWriter(os.path.join(logdir, 'eval'))
    if include_test:
        test_writer = SummaryWriter(os.path.join(logdir, 'test'))


def add_summary(tag, value, iter, stage='train'):
    if stage == 'train':
        writer = train_writer
    elif stage == 'eval':
        writer = eval_writer
    elif stage == 'test':
        writer = test_writer
    else:
        raise NotImplementedError
    writer.add_scalar(tag, value, iter)


def last(loss_dict, timestep, stage='train'):
    for loss_name, loss_list in loss_dict.items():
        if isinstance(loss_list, (tuple, list)):
            loss_value = loss_list[-1]
        else:
            loss_value = loss_list
        summary_name = 'losses/{}'.format(
            loss_name) if '/' not in loss_name else loss_name
        add_summary(summary_name, loss_value, timestep, stage)


def mean(loss_dict, timestep, stage='eval'):
    for loss_name, loss_list in loss_dict.items():
        loss_value = np.mean(loss_list)
        summary_name = 'losses/{}'.format(
            loss_name) if '/' not in loss_name else loss_name
        add_summary(summary_name, loss_value, timestep, stage)
