"""
Training Options
"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--layer_size', type=int, default=10,
                    help='layer_size: 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]')
parser.add_argument('--stack_size', type=int, default=5,
                    help='stack_size: 5 = stack[layer1, layer2, layer3, layer4, layer5]')
parser.add_argument('--in_channels', type=int, default=80,
                    help='input channel size. mu-law encode factor, one-hot size')
parser.add_argument('--res_channels', type=int, default=128, help='number of channel for residual network')
parser.add_argument('--out_channels', type=int, default=1, help='number of channel for final output')
parser.add_argument('--gc_channels', type=int, default=5, help='number of channel for global conditioning tensor')
parser.add_argument('--ddc_channel_select', type=int, nargs='+', default=[0], help='channels of ddc datasets melspectrogram')

parser.add_argument('--sample_rate', type=int, default=16000, help='Sampling rates for input sound. this is for original wavenet')
parser.add_argument('--sample_size', type=int, default=100000, help='Sample size for training input this is for original wavenet')
parser.add_argument('--nolog', action='store_true', default=False, help='if True, tensorboard logging wont run')


def parse_args(is_training=True):
    if is_training:
        parser.add_argument('--data_dir', type=str, default='./test/data', help='Training data dir')
        parser.add_argument('--output_dir', type=str, default='./output', help='Output dir for saving model and etc')
        parser.add_argument('--num_steps', type=int, default=50000, help='Total training steps')
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate decay')
    else:
        parser.add_argument('--model_dir', type=str, required=True, help='Pre-trained model dir')
        parser.add_argument('--step', type=int, default=0, help='A specific step of pre-trained model to use')
        parser.add_argument('--seed', type=str, help='A seed file to generate sound')
        parser.add_argument('--out', type=str, help='Output file name which is generated')

    return parser.parse_args()


def print_help():
    parser.print_help()
