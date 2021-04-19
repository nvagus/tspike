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
            leak = [leak] * (self.num_spikes - 1)
        if not isinstance(fodep, Sequence):
            fodep = [fodep] * (self.num_spikes - 1)
        if not isinstance(w_init, Sequence):
            w_init = [w_init] * (self.num_spikes - 1)
        if not isinstance(theta, Sequence):
            theta = [theta] * (self.num_spikes - 1)
        if not isinstance(fodep, Sequence):
            dense = [dense] * (self.num_spikes - 1)

        self.kernel = kernel
        self.columns = []
        for i in range(self.num_spikes - 1):
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
        output_spikes = self.columns[-1].forward(input_spikes, labels=labels, bias=bias)
        return output_spikes
    
    def pooling(self, input_spikes):
        batch, channel, synapses, time = input_spikes.shape
        input_spikes = input_spikes.reshape(batch, channel * synapses, time)
        output_spikes = torch.max_pool1d(input_spikes, self.kernel)
        return output_spikes.reshape(batch, channel, synapses, -1)
