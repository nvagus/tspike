from typing import Sequence
import torch


from .tnn import *


class StackFullColumn(torch.nn.Module):
    def __init__(
        self,
        synapses, channels=1, kernel=2,
        step=16, leak=32,
        fodep=None, w_init=0., theta=None, dense=None
    ):
        super(StackFullColumn, self).__init__()

        self.num_spikes = len(synapses)
        if not isinstance(channels, Sequence):
            channels = [channels] * self.num_spikes
        if not isinstance(step, Sequence):
            step = [step] * self.num_spikes
        if not isinstance(leak, Sequence):
            leak = [leak] * self.num_spikes
        if not isinstance(fodep, Sequence):
            fodep = [fodep] * self.num_spikes
        if not isinstance(w_init, Sequence):
            w_init = [w_init] * self.num_spikes
        if not isinstance(theta, Sequence):
            theta = [theta] * self.num_spikes
        if not isinstance(fodep, Sequence):
            dense = [dense] * self.num_spikes

        self.kernel = kernel
        self.columns = []
        for i in range(self.num_spikes):
            column = FullColumn(
                synapses[i], synapses[i+1], channels[i], channels[i+1],
                step=step[i], leak=leak[i],
                fodep=fodep[i], w_init=w_init[i], theta=theta[i], dense=dense[i]
            )
            self.columns.append(column)
            self.add_module(f'column{i}', column)

    def forward(self, input_spikes, labels=None, bias=0.5):
        for column in self.columns[:-1]:
            output_spikes = column.forward(input_spikes)
            input_spikes = self.pooling(output_spikes)
        output_spikes = self.columns[-1].forward(
            input_spikes, labels=labels, bias=bias)
        return output_spikes

    def pooling(self, input_spikes):
        batch, channel, synapses, time = input_spikes.shape
        input_spikes = input_spikes.reshape(batch, channel * synapses, time)
        output_spikes = torch.max_pool1d(input_spikes, self.kernel)
        return output_spikes.reshape(batch, channel, synapses, -1)


class StackCV(torch.nn.Module):
    def __init__(
        self,
        channels, conv_kernel=3, pooling_kernel=2,
        step=16, leak=32, bias=0.5, winners=0.5,
        fodep=None, w_init=0., theta=None, dense=None
    ):
        super(StackCV, self).__init__()

        self.num_spikes = len(channels) - 1
        if not isinstance(step, Sequence):
            step = [step] * self.num_spikes
        if not isinstance(leak, Sequence):
            leak = [leak] * self.num_spikes
        if not isinstance(bias, Sequence):
            bias = [bias] * self.num_spikes
        if not isinstance(winners, Sequence):
            winners = [winners] * self.num_spikes
        if not isinstance(fodep, Sequence):
            fodep = [fodep] * self.num_spikes
        if not isinstance(w_init, Sequence):
            w_init = [w_init] * self.num_spikes
        if not isinstance(theta, Sequence):
            theta = [theta] * self.num_spikes
        if not isinstance(fodep, Sequence):
            dense = [dense] * self.num_spikes

        self.conv_kernel = conv_kernel
        self.pooling_kernel = pooling_kernel  # time axis

        self.columns = []

        for i in range(self.num_spikes):
            column = ConvColumn(
                channels[i], channels[i+1],
                kernel=conv_kernel, stride=2,
                step=step[i], leak=leak[i], bias=bias[i], winners=winners[i],
                fodep=fodep[i], w_init=w_init[i], theta=theta[i], dense=dense[i]
            )
            self.columns.append(column)
            self.add_module(f'column{i}', column)

    def forward(self, input_spikes, depth=-1, mu_capture=0.2, mu_backoff=-0.2, mu_search=0.0001):
        for column_i, column in enumerate(self.columns):
            output_spikes = column.forward(
                input_spikes, mu_capture=mu_capture, mu_backoff=mu_backoff, mu_search=mu_search)
            input_spikes = self.pooling(output_spikes)

            if column_i == depth:
                return input_spikes

        return input_spikes

    def pooling(self, input_spikes):
        batch, channel, synapses_x, synapses_y, time = input_spikes.shape
        input_spikes = input_spikes.reshape(
            batch, channel * synapses_x * synapses_y, time)
        output_spikes = torch.max_pool1d(input_spikes, self.pooling_kernel)
        return output_spikes.reshape(batch, channel, synapses_x, synapses_y, -1)
