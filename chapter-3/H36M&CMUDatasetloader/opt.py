import os
import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_dir', type=str, default='./data/h3.6m/dataset', help='path to H36M dataset')
        self.parser.add_argument('--data_dir_cmu', type=str, default='./data/cmu_mocap/', help='path to CMU dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--input_n', type=int, default=10, help='observed seq length')
        self.parser.add_argument('--output_n', type=int, default=10, help='future seq length')
        self.parser.add_argument('--dct_n', type=int, default=15, help='number of DCT coeff. preserved for 3D')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--train_batch', type=int, default=32)
        self.parser.add_argument('--test_batch', type=int, default=128)
        self.parser.add_argument('--job', type=int, default=10, help='subprocesses to use for data loading')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        self._print()
        return self.opt

