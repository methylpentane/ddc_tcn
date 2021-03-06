# delete this line if you want disable fold option in vim.
# vim:set foldmethod=marker:
"""
Neural network modules for WaveNet

References :
    https://arxiv.org/pdf/1609.03499.pdf
    https://github.com/ibab/tensorflow-wavenet
    https://qiita.com/MasaEguchi/items/cd5f7e9735a120f27e2a
    https://github.com/musyoku/wavenet/issues/4
"""
import torch
import numpy as np

from wavenet.exceptions import InputSizeError

# DilatedCausalConv1d {{{
class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet"""
    def __init__(self, channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(channels, channels,
                                    kernel_size=2, stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=0,  # Fixed for WaveNet dilation
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output
# }}}
# CausalConv1d {{{
class CausalConv1d(torch.nn.Module):
    """Causal Convolution for WaveNet"""
    def __init__(self, in_channels, out_channels):
        super(CausalConv1d, self).__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=2, stride=1, padding=1,
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        # remove last value for causal convolution
        return output[:, :, :-1]
# }}}
# ResidualBlock {{{
class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, gc_channels, dilation):
        """
        Residual block
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param gc_channels: number of input channel for global conditioning. 0 for disable gc
        :param dilation:
        """
        super(ResidualBlock, self).__init__()
        self.gc_channels = gc_channels

        self.dilatedconv_t = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.dilatedconv_s = DilatedCausalConv1d(res_channels, dilation=dilation)
        if gc_channels > 0:
            self.gc_conv_t = torch.nn.Conv1d(gc_channels, res_channels, 1)
            self.gc_conv_s = torch.nn.Conv1d(gc_channels, res_channels, 1)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size, gc=None):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :param gc: tensor for global conditioning(time=1)
        :return:
        """
        out_t = self.dilatedconv_t(x)
        out_s = self.dilatedconv_s(x)

        # global conditioning
        if self.gc_channels > 0:
            out_t = out_t + self.gc_conv_t(gc)
            out_s = out_s + self.gc_conv_s(gc)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(out_t)
        gated_sigmoid = self.gate_sigmoid(out_s)
        gated = gated_tanh * gated_sigmoid

        # Residual network
        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2):]
        output += input_cut

        # Skip connection
        skip = gated[:, :, -skip_size:]
        skip = self.conv_skip(skip)

        return output, skip
# }}}
# ResidualStack {{{
class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels, gc_channels):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param gc_channels: number of input channel for global conditioning. 0 for disable gc
        :return:
        """
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size

        self.dilations = self.build_dilations()
        self.res_blocks = torch.nn.ModuleList([ResidualBlock(res_channels, skip_channels, gc_channels, dilation) for dilation in self.dilations])

    def build_dilations(self):
        dilations = []

        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)

        return dilations

    def forward(self, x, skip_size, gc=None):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :param gc: tensor for global conditioning(time=1)
        :return:
        """
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            # output is the next input
            output, skip = res_block(output, skip_size, gc)
            skip_connections.append(skip)

        return torch.stack(skip_connections)
# }}}
# DensNet {{{
class DensNet(torch.nn.Module):
    def __init__(self, channels, out_channels, input_scale):
        """
        The last network of WaveNet
        :param channels: number of channels for input and output
        :return:
        """
        super(DensNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.conv2 = torch.nn.Conv1d(channels, out_channels, kernel_size=input_scale, stride=input_scale)

        self.relu = torch.nn.ReLU()
        if out_channels == 1:
            self.out_nonlin = torch.nn.Sigmoid()
        else:
            self.out_nonlin = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)

        if not self.training:
            output = self.out_nonlin(output)

        return output
# }}}
# WaveNetModule {{{
class WaveNetModule(torch.nn.Module):
    def __init__(self, layer_size, stack_size, in_channels, res_channels, out_channels, gc_channels, input_scale, preconv='none'):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param in_channels: number of channels for input data. skip channel is same as input channel
        :param res_channels: number of residual channel for input, output
        :param out_channels: number of final output channel
        :param gc_channels: number of input channel for global conditioning. 0 for disable gc
        :param input_scale: = input_size / output_size
        :param preconv: = whether do preconv 'none':dont 'raw':preconv for raw 'spectre':preconv for spectre
        :return:
        """
        super(WaveNetModule, self).__init__()

        self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)

        self.preconv = None

        if preconv == 'raw':
            self.preconv = PreConv_raw(in_channels, res_channels)

        if self.preconv:
            self.causal = CausalConv1d(res_channels, res_channels)
        else:
            self.causal = CausalConv1d(in_channels, res_channels)

        self.res_stack = ResidualStack(layer_size, stack_size, res_channels, in_channels, gc_channels)

        self.densnet = DensNet(in_channels, out_channels, input_scale)

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)

        return int(num_receptive_fields)

    def calc_output_size(self, x):
        if self.preconv:
            output_size = int(x.size(2))//160 - self.receptive_fields
        else:
            output_size = int(x.size(2)) - self.receptive_fields

        self.check_input_size(x, output_size)

        return output_size

    def check_input_size(self, x, output_size):
        if output_size < 1:
            raise InputSizeError(int(x.size(2)), self.receptive_fields, output_size)

    def forward(self, x, gc=None):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, timestep, channels]
        :param gc: Tensor[batch, channels]
        :return: Tensor[batch, timestep, channels]
        """
        output = x.transpose(1, 2) # [ntc -> nct]
        if gc is not None:
            gc = gc.unsqueeze(2) # pseudo timestep

        output_size = self.calc_output_size(output)

        if self.preconv:
            output = self.preconv(output)

        output = self.causal(output)

        skip_connections = self.res_stack(output, output_size, gc)

        output = torch.sum(skip_connections, dim=0)

        output = self.densnet(output)

        return output.transpose(1, 2).contiguous()
# }}}

# PreConv_raw {{{
class PreConv_raw(torch.nn.Module):
    """
    Pre-Convolution layer for raw-audio-DDC
    1/160 time compression
    2*2*2*2*2*5(pool)
    achieving receptive field 1280 step (80 ms @ 16000Hz audio)

    objective - get parameters for fourier transform like function
    """
    def __init__(self, in_channels, res_channels):
        super(PreConv_raw, self).__init__()

        self.conv_p = torch.nn.Conv1d(in_channels, res_channels, kernel_size=2, stride=1, padding=4)
        self.dilated_conv_1 = torch.nn.Conv1d(res_channels, res_channels, kernel_size=2, stride=1, dilation=1)
        self.dilated_conv_2 = torch.nn.Conv1d(res_channels, res_channels, kernel_size=2, stride=1, dilation=2)
        self.dilated_conv_3 = torch.nn.Conv1d(res_channels, res_channels, kernel_size=2, stride=1, dilation=4)

        self.BN_1 = torch.nn.BatchNorm1d(res_channels)

        self.conv_1 = torch.nn.Conv1d(res_channels, res_channels, kernel_size=4, stride=4)
        self.conv_2 = torch.nn.Conv1d(res_channels, res_channels, kernel_size=4, stride=4)
        self.conv_3 = torch.nn.Conv1d(res_channels, res_channels, kernel_size=2, stride=2)
        self.maxpool= torch.nn.MaxPool1d(kernel_size=5, stride=5)

        self.BN_2 = torch.nn.BatchNorm1d(res_channels)

    def forward(self, x):
        x = self.conv_p(x)
        x = self.dilated_conv_1(x)
        x = self.dilated_conv_2(x)
        x = self.dilated_conv_3(x)
        x = self.BN_1(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.maxpool(x)
        output = self.BN_2(x)
       #  from IPython import embed
       #  embed()
       #  exit()

        return output
# }}}
