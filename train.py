# vim:set foldmethod=marker:
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
from wavenet.utils.data import DataLoader_onset, DataLoader_onset_raw, DataLoader_oneshot, DataLoader_oneshot_snap, DataLoader_oneshot_raw, DataLoader_oneshot_raw_snap


class Trainer:
    def __init__(self, args):
        self.args = args

        # init of net & data{{{

        # global conditioning channels
        for d in self.args.data_dir:
            print(d)
            if 'fraxtil' in d:
                self.args.gc_channels += 1
            elif 'itg' in d:
                self.args.gc_channels += 12

        # net & data
        if self.args.mode == 'onset_spectre':
            self.wavenet = WaveNet_onset(self.args.layer_size, self.args.stack_size, self.args.in_channels, self.args.res_channels, self.args.out_channels, self.args.gc_channels, self.args.input_scale, lr=self.args.lr)
            self.data_loader = DataLoader_onset(self.args.data_dir, self.wavenet.receptive_fields, self.args.ddc_channel_select)
            self.data_loader_valid = DataLoader_onset(self.args.data_dir, self.wavenet.receptive_fields, self.args.ddc_channel_select, valid=True, shuffle=False)

        if self.args.mode == 'onset_raw':
            self.wavenet = WaveNet_onset(self.args.layer_size, self.args.stack_size, self.args.in_channels, self.args.res_channels, self.args.out_channels, self.args.gc_channels, self.args.input_scale, lr=self.args.lr)
            self.data_loader = DataLoader_onset_raw(self.args.data_dir, self.wavenet.receptive_fields, self.args.sample_size)
            self.data_loader_valid = DataLoader_onset_raw(self.args.data_dir, self.wavenet.receptive_fields, self.args.sample_size, valid=True, shuffle=False)

        if self.args.mode == 'oneshot_spectre':
            self.wavenet = WaveNet_oneshot(self.args.layer_size, self.args.stack_size, self.args.in_channels, self.args.res_channels, self.args.out_channels, self.args.gc_channels, self.args.input_scale, lr=self.args.lr)
            self.data_loader = DataLoader_oneshot(self.args.data_dir, self.wavenet.receptive_fields, self.args.ddc_channel_select)
            self.data_loader_valid = DataLoader_oneshot(self.args.data_dir, self.wavenet.receptive_fields, self.args.ddc_channel_select, valid=True, shuffle=False)

        if self.args.mode == 'oneshot_spectre_snap':
            self.wavenet = WaveNet_oneshot(self.args.layer_size, self.args.stack_size, self.args.in_channels, self.args.res_channels, self.args.out_channels, self.args.gc_channels, self.args.input_scale, lr=self.args.lr)
            self.data_loader = DataLoader_oneshot_snap(self.args.data_dir, self.wavenet.receptive_fields, self.args.ddc_channel_select, self.args.sample_size)
            self.data_loader_valid = DataLoader_oneshot_snap(self.args.data_dir, self.wavenet.receptive_fields, self.args.ddc_channel_select, self.args.sample_size, valid=True, shuffle=False)

        if self.args.mode == 'oneshot_raw':
            self.wavenet = WaveNet_oneshot(self.args.layer_size, self.args.stack_size, self.args.in_channels, self.args.res_channels, self.args.out_channels, self.args.gc_channels, self.args.input_scale, lr=self.args.lr)
            self.data_loader = DataLoader_oneshot_raw(self.args.data_dir, self.wavenet.receptive_fields, self.args.sample_size)
            self.data_loader_valid = DataLoader_oneshot_raw(self.args.data_dir, self.wavenet.receptive_fields, self.args.sample_size, valid=True, shuffle=False)

        if self.args.mode == 'oneshot_raw_snap':
            self.wavenet = WaveNet_oneshot(self.args.layer_size, self.args.stack_size, self.args.in_channels, self.args.res_channels, self.args.out_channels, self.args.gc_channels, self.args.input_scale, lr=self.args.lr)
            self.data_loader = DataLoader_oneshot_raw_snap(self.args.data_dir, self.wavenet.receptive_fields, self.args.sample_size)
            self.data_loader_valid = DataLoader_oneshot_raw_snap(self.args.data_dir, self.wavenet.receptive_fields, self.args.sample_size, valid=True, shuffle=False)
        # }}}
        print('network & data loader OK')
        # log init {{{
        self.summary_writer = None if args.nolog else SummaryWriter(log_dir=args.log_dir)
        # hparams memo
        text = '''\
        #### mode: {mode}
        #### dataset: {dataset}
        #### comment: {comment}
        |layer|stack|in|residual|out|global condition|lr|sample size|STFT window selection|
        |----|----|----|----|----|----|----|----|----|
        |{layer}|{stack}|{in_}|{res}|{out}|{gc}|{lr}|{sample_size}|{stft}|\
        '''.format(
            mode=args.mode,
            dataset=', '.join(args.data_dir),
            comment=args.comment,
            layer=args.layer_size,
            stack=args.stack_size,
            in_=args.in_channels,
            res=args.res_channels,
            out=args.out_channels,
            gc=args.gc_channels,
            lr=args.lr,
            sample_size=args.sample_size,
            stft=','.join([str(_) for _ in args.ddc_channel_select]))
        if self.summary_writer is not None:
            self.summary_writer.add_text('condition', dedent(text))
        # }}}
        print('tensorboard log OK')

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
        do_calc_mAP_256 = True if args.calc_mAP_256 and 'oneshot' in self.args.mode else False

        while True:
            # training for one epoch
            print('training for one epoch ...')
            loss_sum = 0
            for inputs_unrolling, targets_unrolling in self.one_epoch_batch():
                if self.args.gc_channels:
                    inputs_unrolling, gc_inputs_unrolling = inputs_unrolling
                else:
                    gc_inputs_unrolling = None

                loss = self.wavenet.train(inputs_unrolling, targets_unrolling, gc_inputs_unrolling)
                loss_sum += loss
                total_steps += 1
                print('[{0}/{1}] loss: {2}'.format(total_steps, args.num_steps, loss))

            self.log('train/mean_cross_entropy', loss/len(self.data_loader.dataset), total_steps)

            # do validation
            print('validating...', end='')
            # prediction store initialize (if mAP)
            if do_calc_mAP_256:
                targets_label_onehot_all = np.zeros((0,args.out_channels-1), dtype=np.int)
                y_pred_label_all = np.zeros((0,args.out_channels-1), dtype=np.float32)

            # predict
            epoch_metrics = defaultdict(list)
            for inputs_unrolling, targets_unrolling in self.one_epoch_batch_valid():
                if self.args.gc_channels:
                    inputs_unrolling, gc_inputs_unrolling = inputs_unrolling
                else:
                    gc_inputs_unrolling = None
                metrics = self.wavenet.validation(inputs_unrolling, targets_unrolling, gc_inputs_unrolling)
                if do_calc_mAP_256:
                    targets_label_onehot_all = np.vstack((targets_label_onehot_all, metrics['targets_label_onehot']))
                    y_pred_label_all = np.vstack((y_pred_label_all, metrics['y_pred_label']))
                if 'oneshot' in self.args.mode:
                    del metrics['targets_label_onehot'], metrics['y_pred_label']
                for metrics_key, metrics_value in metrics.items():
                    epoch_metrics[metrics_key].append(metrics_value)

            # orgamize metrics
            epoch_metrics = {k: (np.mean(v), np.var(v)) for k, v in epoch_metrics.items()}
            if do_calc_mAP_256:
                macro_mAP = self.wavenet.macro_mAP(targets_label_onehot_all, y_pred_label_all)

            # write metrics
            for key, value in epoch_metrics.items():
                self.log('valid/'+key+'_mean', value[0], total_steps)
                self.log('valid/'+key+'_variance', value[1], total_steps)
            if do_calc_mAP_256:
                self.log('valid/'+'macro_mAP', macro_mAP, total_steps)
            print('done')

            if total_steps > self.args.num_steps:
                break

        # now, model save is waste of storage
        # self.wavenet.save(args.model_dir)


def prepare_output_dir(args):
    now = datetime.now()
    log_dirname = '/'.join([datetime.strftime(now, '%Y'), datetime.strftime(now, '%m'), datetime.strftime(now, '%d'), datetime.strftime(now, '%H:%M:%S')])
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
