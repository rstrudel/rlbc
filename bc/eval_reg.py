import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from bc.dataset import Frames, Signals
from bc.agent import RegressionAgent

from bc.settings import MODEL_LOGDIR, DATASET_LOGDIR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-name', '-d', required=True,
        help='name of the dataset to evaluate on')
    parser.add_argument(
        '--net-names', '-n', required=True, nargs='+',
        help='names of network(s) to evaluate')
    parser.add_argument(
        '--net-epochs', '-ne', type=int, nargs=2, default=(2, 101),
        help='network epochs to evaluate')
    parser.add_argument(
        '--demo-indices', '-di', type=int, nargs=2, default=(19500, 20000),
        help='dataset demo indices to use')
    parser.add_argument(
        '--dont-plot', '-dp', action='store_true', default=False,
        help='whether to plot the evaluation')
    parser.add_argument(
        '--signal-keys', '-sk', nargs='+', default=['target_position'],
        help='names of the signal keys of the regression')
    parser.add_argument(
        '--signal-lengths-pred', '-sl', nargs='+', type=int, default=[2],
        help='lengths of the signal keys of the regression (predicted by the network)')
    parser.add_argument(
        '--signal-lengths-real', '-slr', nargs='+', type=int, default=None,
        help='lengths of the signal keys of the regression (in the dataset)')
    args = parser.parse_args()
    args.plot = not args.dont_plot
    return args


def compute_results(images, signals_gts, signal_keys, signal_lengths, agent):
    signals_preds = agent.get_prediction(torch.stack(images))
    signals_preds_list = []
    for signals_pred_dict in signals_preds:
        signals_pred_concat = np.array([])
        for i, signal_key in enumerate(signal_keys):
            signals_pred_concat = np.concatenate((
                signals_pred_concat,
                signals_pred_dict[signal_key[-1]][:signal_lengths[i]]))
        signals_preds_list.append(signals_pred_concat)
    return np.array(list(zip(signals_gts, signals_preds_list)))


def compute_errors(results):
    err = np.linalg.norm(results[:, 0] - results[:, 1], axis=1)
    err_mean, err_std = err.mean(), err.std()
    print('Error {:.2f}cm +/- {:.2f}cm'.format(err_mean * 100, err_std * 100))
    return err_mean * 100, err_std * 100


def plot_errors(errors_nets, args):
    plt.figure(figsize=(15, 5))
    for i, errors_cm in enumerate(errors_nets):
        epoch_min = errors_cm[:, 0].argmin()
        epoch_range = range(*args.net_epochs, 2)
        net_label = 'net {}, error {:.2f}cm +/- {:.2f}cm'.format(
            epoch_range[epoch_min], errors_cm[epoch_min, 0],
            errors_cm[epoch_min, 1])
        plt.plot(epoch_range, errors_cm[:, 0], label=net_label)
        plt.fill_between(
            epoch_range,
            errors_cm[:, 0] - errors_cm[:, 1] / 2,
            errors_cm[:, 0] + errors_cm[:, 1] / 2,
            alpha=0.2)
        print('Best epoch is {}, error {:.2f}cm +/- {:.2f}cm'.format(
            epoch_range[epoch_min], errors_cm[epoch_min, 0],
            errors_cm[epoch_min, 1]))
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('error in cm')
    plt.show()


def read_dataset(args, signal_keys):
    dataset_path = os.path.join(DATASET_LOGDIR, args.dataset_name)
    frames = Frames(dataset_path, channels=['depth'])
    signals = Signals(dataset_path, signal_keys, normalize=False)

    images, signals_gts = [], []
    for demo_num in range(args.demo_indices[0], args.demo_indices[1]):
        demo_idxs = frames.keys.get_demo_indices(demo_num)
        images.append(frames[demo_idxs[0]])
        signals_demo = signals[demo_idxs[0]:demo_idxs[1]]
        if args.signal_lengths_real is None:
            signals_gt = signals_demo[0][:sum(args.signal_lengths_pred)]
        else:
            assert len(args.signal_lengths_pred) == len(args.signal_lengths_real)
            signals_gt = np.array([])
            begin = 0
            for length_pred, length_real in zip(args.signal_lengths_pred, args.signal_lengths_real):
                signals_gt = np.concatenate((signals_gt, signals_demo[0][begin:begin+length_pred]))
                begin += length_real
        signals_gts.append(signals_gt)
    return images, signals_gts


def main():
    args = get_args()
    signal_keys, error_cms = [], []
    for signal_key in args.signal_keys:
        signal_keys.append(('state', signal_key))
    images, signals_gts = read_dataset(args, signal_keys)

    for net_name in args.net_names:
        error_net_cms = []
        net_path = os.path.join(MODEL_LOGDIR, net_name)
        for net_epoch in range(*args.net_epochs, 2):
            agent = RegressionAgent(path=net_path, epoch=net_epoch, device='cuda')
            print('Evaluating epoch {} of network {}'.format(net_epoch, net_path))
            results = compute_results(images, signals_gts, signal_keys, args.signal_lengths_pred, agent)
            error_net_cms.append(compute_errors(results))
        error_cms.append(error_net_cms)

    if args.plot:
        plot_errors(np.array(error_cms), args)


if __name__ == '__main__':
    main()
