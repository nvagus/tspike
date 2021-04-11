import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.binomial import Binomial

from .configs import *


class ResponseFunction:
    def __init__(self, w_max):
        self.w_max = w_max
    
    @property
    def w_kernel(self):
        raise NotImplementedError()

    @property
    def padding(self):
        raise NotImplementedError()
    
    @property
    def fodep(self):
        raise NotImplementedError()

    def get_w_kernel(self, weights):
        return self.w_kernel.index_select(0, weights.flatten()).reshape((*weights.shape, -1))


class StepFireLeak(ResponseFunction):
    def __init__(self, w_max, step=1, leak=2, device='cpu'):
        super().__init__(w_max)
        self.step = step
        self.leak = leak
        self.w_kernel_size = w_kernel_size = (w_max - 1) * (step + leak)
        w_zero = torch.zeros(w_max, w_kernel_size)
        w_idx = torch.arange(w_max).repeat(w_kernel_size, 1).transpose(0, 1)
        t_idx = torch.arange(w_kernel_size).repeat(w_max, 1)
        w_step = (1 + t_idx / step).floor().max(w_zero)
        w_leak = (w_idx + ((w_idx - 1) * step - t_idx) / leak).ceil().max(w_zero)
        self._w_kernel = w_step.min(w_leak).int().flip(1).to(device)

    @property
    def w_kernel(self):
        return self._w_kernel

    @property
    def padding(self):
        return self.w_kernel_size

    @property
    def fodep(self):
        return self.w_kernel_size


class FullColumn(torch.nn.Module):
    def __init__(
        self, 
        synapses, neurons, input_channel=1, output_channel=1, 
        w_max=None, theta=None, dense=None, fodep=None,
        mu_capture=64, mu_backoff = -8, mu_search = 2,
        device='cpu'
    ):
        super(FullColumn, self).__init__()
        self.device = device

        self.synapses = synapses
        self.neurons = neurons
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.w_max = w_max = w_max or MAX_WEIGHT
        self.mu_capture = mu_capture or MU_CAPTURE
        self.mu_backoff = mu_backoff or MU_BACKOFF
        self.mu_search = mu_search or MU_SEARCH

        assert theta or dense, "either theta or dense should be specified"
        self.theta = theta = theta or dense * w_max // 2
        self.dense = dense = dense or 2 * theta // w_max
        assert dense < 2 * input_channel * synapses, "invalid theta or density, try setting a smaller value"
        # default response function: StepFireLeak with step=1, leak=2
        self.response_function = StepFireLeak(w_max, device=device)
        self.fodep = fodep or self.response_function.fodep
        # initialize weight based on expected density
        self.initializer = Binomial(w_max - 1, dense / (2 * synapses * input_channel))
        self.weight = self.initializer.sample(
            (self.output_channel * self.neurons, self.input_channel * self.synapses)).type(torch.int64).to(device)
    
    def forward(self, input_spikes):
        potentials = self.get_potentials(input_spikes)
        output_spikes = self.winner_takes_all(potentials)
        return output_spikes

    def get_potentials(self, input_spikes):
        # coalesce input channel and synpases
        batch, channel, synapses, time = input_spikes.shape
        input_spikes = input_spikes.reshape(batch, channel * synapses, time)
        # perform conv1d to compute potentials
        w_kernel = self.response_function.get_w_kernel(self.weight)
        padding = self.response_function.padding
        if self.device == 'cpu':
            torch.nn.functional.conv1d(input_spikes, w_kernel, padding=padding)
        else:
            potentials = torch.nn.functional.conv1d(input_spikes.float(), w_kernel.float(), padding=padding).int()
        # expand output channel and neurons
        potentials = potentials.reshape(batch, self.output_channel, self.neurons, -1)
        return potentials

    def winner_takes_all(self, potentials):
        # move time axis to 0 for better performance
        batch, channel, neurons, time = potentials.shape
        potentials = potentials.permute(3, 0, 1, 2)
        # time to step out of depression, with initial 0 and constrains >= 0
        depression = torch.zeros(batch, channel, dtype=torch.int32, device=self.device)
        min_depression = torch.zeros(batch, channel, dtype=torch.int32, device=self.device)
        # return winners of the same shape as potentials
        winners = torch.zeros(time, batch, channel, neurons, dtype=torch.int32, device=self.device)
        # iterate time axis: get the winner for each batch, channel of neurons, update winners and depression
        for t in range(time):
            winner_t = potentials[t].argmax(axis=-1).unsqueeze(-1)
            cond_t = (potentials[t].gather(-1, winner_t) > self.theta).logical_and(depression == min_depression).int()
            winners[t].scatter_(-1, winner_t, cond_t)
            depression = min_depression.max(depression + winners[t].sum(axis=-1) * (self.fodep + 1) - 1)
        # move time axis to -1 for the convenience of convolutional operations
        return winners.permute(1, 2, 3, 0)

    def stdp(self, input_spikes, output_spikes):
        batch, channel_i, synapses, time_i = input_spikes.shape
        batch, channel_o, neurons, time_o = output_spikes.shape
        input_spikes = input_spikes.reshape(batch, channel_i * synapses, time_i).permute(2, 0, 1)
        output_spikes = output_spikes.reshape(batch, channel_o * neurons, time_o).permute(2, 0, 1)

        output_spike_acc = torch.zeros(time_o + 1, batch, channel_o * neurons, dtype=torch.int32, device=self.device)
        output_spike_acc[0] = output_spikes[0]
        for t in range(time_o):
            output_spike_acc[t+1] = output_spike_acc[t] + output_spikes[t]

        history = torch.zeros_like(self.weight, dtype=torch.int64, device=self.device)
        for t in range(time_i):
            t_max = min(t + self.fodep + 1, time_o)
            has_spike_o = ((output_spike_acc[t_max] - output_spike_acc[t]) > 0).unsqueeze(-1)
            has_spike_i = input_spikes[t].bool().unsqueeze(-2)
            capture = has_spike_i.logical_and(has_spike_o).sum(axis=0) * self.mu_capture
            backoff = has_spike_i.logical_not().logical_and(has_spike_o).sum(axis=0) * self.mu_backoff
            search = has_spike_i.logical_and(has_spike_o.logical_not()).sum(axis=0) * self.mu_search
            history += capture + backoff + search
        
        update = history * (self.weight + 1) * (self.w_max - self.weight) / (self.w_max ** 3)
        sign = torch.where(update > 0, 1, -1)
        prob = update.abs().clip(0, 1)
        delta = Bernoulli(prob).sample().int()
        self.weight = (self.weight + sign * delta).clip(0, self.w_max - 1)
    