import argparse
import os
import numpy as np
import matplotlib.pylab as plt
from shutil import copyfile
from bc.utils.report import Report


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'load_files',
        type=str,
        nargs='*',
        help='files to read the report from')
    parser.add_argument(
        '--plot',
        action='store_true',
        default=False,
        help='whether to plot the report (default only print)')
    parser.add_argument(
        '--copy-to', '-ct',
        type=str,
        default=None,
        help='where to copy seeds to')
    parser.add_argument(
        '--copy-from', '-cf',
        type=str,
        default=None,
        help='where to copy seeds from')
    parser.add_argument(
        '--copy-threshold', '-cth',
        type=float,
        default=None,
        help='copy epochs with the success rate above this number')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    all_sr = {}

    for file_path in args.load_files:
        report = Report(path=file_path)
        sr = report.get_success_rate()
        all_sr[file_path] = sr
        print('Success rate of {0} is in average {1:.2f}:'.format(
            file_path,
            np.mean(list(sr.values()))))
        for epoch, success_rate in sorted(sr.items(), key=lambda kv: -kv[1]):
            print('\tEpoch {0:3d}: success rate = {1:.2f} ({2} seeds)'.format(
                epoch, success_rate, len(report.entries['epochs'][epoch])))

        if args.copy_to and args.copy_from and args.copy_threshold:
            epochs_to_copy = [epoch for epoch, success in sr.items() if success >= args.copy_threshold]
            print('Copying epochs {} from {} to {}'.format(
                epochs_to_copy, args.copy_from, args.copy_to))
            if not os.path.exists(args.copy_to):
                os.mkdir(args.copy_to)
            for epoch_to_copy in epochs_to_copy:
                epoch_path_src = os.path.join(args.copy_from, 'model_{}.pth'.format(epoch_to_copy))
                epoch_path_dist = os.path.join(args.copy_to, 'model_{}.pth'.format(epoch_to_copy))
                copyfile(epoch_path_src, epoch_path_dist)
    if args.plot:
        for file_path, sr in all_sr.items():
            # sorted by key, return a list of tuples
            lists = sorted(sr.items())
            # unpack a list of pairs into two tuples
            x, y = zip(*lists)
            plt.plot(x, y, label=file_path)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
