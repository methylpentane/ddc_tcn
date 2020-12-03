"""
A script for WaveNet training
"""
import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import wavenet.config as config
from wavenet.model import WaveNet
from wavenet.utils.data import DataLoader_onset


class Trainer:
    def __init__(self, args):
        self.args = args

        self.wavenet = WaveNet(args.layer_size, args.stack_size,
                               args.in_channels, args.res_channels,
                               lr=args.lr)

        self.data_loader = DataLoader_onset(args.data_dir, self.wavenet.receptive_fields, args.in_channels)
        self.summary_writer = None if args.nolog else SummaryWriter(log_dir=args.log_dir)

    def infinite_batch(self):
        while True:
            for dataset in self.data_loader:
                for inputs, targets in dataset:
                    yield inputs, targets

    def log(self, tag, y, x):
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(tag, y, x)

    def run(self):
        total_steps = 0

        for inputs, targets in self.infinite_batch():
            loss = self.wavenet.train(inputs, targets)

            total_steps += 1

            print('[{0}/{1}] loss: {2}'.format(total_steps, args.num_steps, loss))
            self.log('mean_cross_entropy', loss, total_steps)

            if total_steps > self.args.num_steps:
                break

        self.wavenet.save(args.model_dir)


def prepare_output_dir(args):
    log_dirname = 'log_' + datetime.strftime(datetime.now(), '%y%m%d%H%M%S')
    args.log_dir = os.path.join(args.output_dir, log_dirname)
    args.model_dir = os.path.join(args.output_dir, 'model')
    args.test_output_dir = os.path.join(args.output_dir, 'test')

    # os.makedirs(args.log_dir, exist_ok=True) # tensorboard makes log_dir
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)


if __name__ == '__main__':
    args = config.parse_args()

    prepare_output_dir(args)

    trainer = Trainer(args)

    trainer.run()
