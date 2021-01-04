"""
A script for WaveNet training
"""
import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import defaultdict
from textwrap import dedent
import numpy as np

import wavenet.config as config
from wavenet.model import WaveNet_onset, WaveNet_oneshot
from wavenet.utils.data import DataLoader_onset, DataLoader_onset_raw, DataLoader_oneshot


class Trainer:
    def __init__(self, args):
        self.args = args

        # initiation
        if args.mode == 'onset_spectre':
            self.wavenet = WaveNet_onset(args.layer_size, args.stack_size, args.in_channels, args.res_channels, args.out_channels, args.gc_channels, args.input_scale, lr=args.lr)
            self.data_loader = DataLoader_onset(args.data_dir, self.wavenet.receptive_fields, args.ddc_channel_select)
            self.data_loader_valid = DataLoader_onset(args.data_dir, self.wavenet.receptive_fields, args.ddc_channel_select, valid=True, shuffle=False)

        if args.mode == 'onset_raw':
            self.wavenet = WaveNet_onset(args.layer_size, args.stack_size, args.in_channels, args.res_channels, args.out_channels, args.gc_channels, args.input_scale, lr=args.lr)
            self.data_loader = DataLoader_onset_raw(args.data_dir, self.wavenet.receptive_fields, args.sample_size)
            self.data_loader_valid = DataLoader_onset_raw(args.data_dir, self.wavenet.receptive_fields, args.sample_size, valid=True, shuffle=False)

        if args.mode == 'oneshot_spectre':
            self.wavenet = WaveNet_oneshot(args.layer_size, args.stack_size, args.in_channels, args.res_channels, args.out_channels, args.gc_channels, args.input_scale, lr=args.lr)
            self.data_loader = DataLoader_oneshot(args.data_dir, self.wavenet.receptive_fields, args.ddc_channel_select)
            self.data_loader_valid = DataLoader_oneshot(args.data_dir, self.wavenet.receptive_fields, args.ddc_channel_select, valid=True, shuffle=False)

        self.summary_writer = None if args.nolog else SummaryWriter(log_dir=args.log_dir)
        # log hparams
        text = '''\
        #### mode: {mode}
        #### comment: {comment}
        |layer|stack|in|residual|out|global condition|lr|STFT_window_selection|
        |----|----|----|----|----|----|----|----|
        |{layer}|{stack}|{in_}|{res}|{out}|{gc}|{lr}|{stft}|\
        '''.format(
            mode=args.mode,
            comment=' '.join(args.comment),
            layer=str(args.layer_size),
            stack=str(args.stack_size),
            in_=str(args.in_channels),
            res=str(args.res_channels),
            out=str(args.out_channels),
            gc=str(args.gc_channels),
            lr=str(args.lr),
            stft=','.join([str(_) for _ in args.ddc_channel_select]))
        if self.summary_writer is not None:
            self.summary_writer.add_text('condition', dedent(text))

    # def infinite_batch(self): # deprecated by akiba. changed to whileTrue{one_epoch}
    #     while True:
    #         for dataset in self.data_loader:
    #             for inputs, targets in dataset:
    #                 yield inputs, targets

    def one_epoch_batch(self):
        for dataset in self.data_loader:
            for inputs_unrolling, targets_unrolling in dataset:
                yield inputs_unrolling, targets_unrolling

    def one_epoch_batch_valid(self):
        for dataset in self.data_loader_valid:
            for inputs_unrolling, targets_unrolling in dataset:
                yield inputs_unrolling, targets_unrolling

    def log(self, tag, y, x):
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(tag, y, x)

    def run(self):
        total_steps = 0

        while True:
            # training for one epoch
            print('training for one epoch')
            for inputs_unrolling, targets_unrolling in self.one_epoch_batch():
                if self.args.gc_channels:
                    inputs_unrolling, gc_inputs_unrolling = inputs_unrolling
                else:
                    gc_inputs_unrolling = None

                loss = self.wavenet.train(inputs_unrolling, targets_unrolling, gc_inputs_unrolling)

                total_steps += 1

                print('[{0}/{1}] loss: {2}'.format(total_steps, args.num_steps, loss))
                self.log('train/mean_cross_entropy', loss, total_steps)

            # do validation
            print('validating...', end='')
            epoch_metrics = defaultdict(list)
            for inputs_unrolling, targets_unrolling in self.one_epoch_batch_valid():
                if self.args.gc_channels:
                    inputs_unrolling, gc_inputs_unrolling = inputs_unrolling
                else:
                    gc_inputs_unrolling = None
                metrics = self.wavenet.validation(inputs_unrolling, targets_unrolling, gc_inputs_unrolling)
                for metrics_key, metrics_value in metrics.items():
                    epoch_metrics[metrics_key].append(metrics_value)

            epoch_metrics = {k: (np.mean(v), np.var(v)) for k, v in epoch_metrics.items()}

            for key, value in epoch_metrics.items():
                self.log('valid/'+key+'_mean', value[0], total_steps)
                self.log('valid/'+key+'_variance', value[1], total_steps)
            print('done')

            if total_steps > self.args.num_steps:
                break

        # self.wavenet.save(args.model_dir)


def prepare_output_dir(args):
    now = datetime.now()
    log_dirname = datetime.strftime(now, '%mm%dd') + '/' + datetime.strftime(now, '%H:%M:%S')
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
