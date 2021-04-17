import torch
from torch.distributions.exponential import Exponential
import numpy as np


class StepFireLeak:
    def __init__(self, step=16, leak=32):
        self.step = step
        self.leak = leak
        self.kernel_size = step + leak
        self.padding = int(np.ceil((self.kernel_size + self.step) / 2))
        self.fodep = self.kernel_size

    def get_weight_kernel(self, weight):
        kernel_zero = torch.zeros(*weight.shape, self.kernel_size, device=weight.device)
        t_axis = torch.arange(self.kernel_size, device=weight.device).expand_as(kernel_zero)
        t_spike = t_axis / self.step
        t_leak = -(t_axis - weight.unsqueeze(-1) * self.step) / self.leak + weight.unsqueeze(-1)
        kernel = kernel_zero.max(t_spike.min(t_leak)).flip(-1)
        return kernel


class FullColumn(torch.nn.Module):
    def __init__(
        self, 
        synapses, neurons, input_channel=1, output_channel=1, 
        step=16, leak=32, 
        fodep=None, w_init=0., theta=None, dense=None
    ):
        super(FullColumn, self).__init__()

        self.synapses = synapses
        self.neurons = neurons
        self.input_channel = input_channel
        self.output_channel = output_channel

        assert theta or dense, 'either theta or dense should be specified'
        self.theta = theta = theta or dense * (synapses * input_channel)
        self.dense = dense = dense or theta / (synapses * input_channel)
        assert dense < 2 * input_channel * synapses, 'invalid theta or density, try setting a smaller value'
        # default response function: StepFireLeak with step=1, leak=2
        self.response_function = StepFireLeak(step, leak)
        self.fodep = fodep = fodep or self.response_function.fodep
        assert fodep >= self.response_function.fodep, f'forced depression should be at least {self.response_function.fodep}'
        # initialize weight to w_init
        self.weight = torch.nn.parameter.Parameter(
            Exponential(1 / w_init).sample((self.output_channel * self.neurons, self.input_channel * self.synapses)).clip(0, 1),
            requires_grad=False
        )
        print(f'Building full connected TNN layer with theta={theta:.4f}, dense={dense:.4f}, fodep={fodep}')
    
    def forward(self, input_spikes, labels=None, bias=0.5):
        potentials = self.get_potentials(input_spikes, labels, bias)
        output_spikes = self.winner_takes_all(potentials)
        import code
        code.interact(local=locals())
        if self.training:
            self.stdp(input_spikes, output_spikes)
        return output_spikes

    def get_potentials(self, input_spikes, labels=None, bias=0.5):
        # coalesce input channel and synpases
        batch, channel, synapses, time = input_spikes.shape
        input_spikes = input_spikes.reshape(batch, channel * synapses, time)
        # perform conv1d to compute potentials
        w_kernel = self.response_function.get_weight_kernel(self.weight)
        potentials = torch.nn.functional.conv1d(input_spikes, w_kernel, padding=self.response_function.padding)
        # expand output channel and neurons
        potentials = potentials.reshape(batch, self.output_channel, self.neurons, -1)
        if labels is not None:
            batch, channel, neurons, time = potentials.shape
            supervision = torch.zeros(batch, neurons, dtype=torch.int32, device=labels.device).scatter(
                1, labels.unsqueeze(-1), bias * self.theta
            ).unsqueeze(1).expand(-1, channel, -1).unsqueeze(-1)
            potentials = potentials + supervision
        else:
            potentials = potentials + bias * self.theta
        return potentials

    def winner_takes_all(self, potentials):
        # move time axis to 0 for better performance
        batch, channel, neurons, time = potentials.shape
        potentials = potentials.permute(3, 1, 0, 2)
        # time to step out of depression, with initial 0 and constrains >= 0
        depression = torch.zeros(channel, batch, neurons, dtype=torch.int32, device=potentials.device)
        # return winners of the same shape as potentials
        winners = torch.zeros(time, channel, batch, neurons, dtype=torch.int32, device=potentials.device)
        # iterate time axis: get the winner for each batch, channel of neurons, update winners and depression
        for t in range(time):
            for c in range(channel):
                potential_tc = potentials[t, c] * (depression[c] == 0).int()
                winner_tc = potential_tc.argmax(-1).unsqueeze(-1)
                spike_tc = (potential_tc.gather(-1, winner_tc) > self.theta).int()
                winners[t,c].scatter_(-1, winner_tc, spike_tc)
                depression[c].scatter_add_(-1, winner_tc, -spike_tc * self.fodep)
                depression[c] += spike_tc * self.fodep
                depression += winners[t,c].unsqueeze(0) * self.fodep
            depression = (depression - 1).clip(0, self.fodep - 1)
        # move time axis to -1 for the convenience of convolutional operations
        return winners.permute(2, 1, 3, 0).float()

    def stdp(
        self, 
        input_spikes, output_spikes, 
        mu_capture=0.0200, mu_backoff=-0.0200, mu_search=0.0001
    ):
        batch, channel_i, synapses, time_i = input_spikes.shape
        batch, channel_o, neurons, time_o = output_spikes.shape
        input_spikes = input_spikes.reshape(batch, channel_i * synapses, time_i).permute(2, 0, 1)
        output_spikes = output_spikes.reshape(batch, channel_o * neurons, time_o).permute(2, 0, 1)
        output_spike_acc = torch.cumsum(output_spikes, dim=0)
        output_spike_acc = torch.cat((torch.zeros_like(output_spike_acc[0]).unsqueeze(0), output_spike_acc), 0)
        input_spike_acc = torch.cumsum(input_spikes, dim=0)
        input_spike_acc = torch.cat((torch.zeros_like(input_spike_acc[0]).unsqueeze(0), input_spike_acc), 0)
        history = torch.zeros_like(self.weight)
        # compute capture and search
        for t in range(time_i):
            has_spike_i = input_spikes[t].bool().unsqueeze(-2)
            if has_spike_i.sum() > 0:
                t_max = min(t + self.fodep + 1, time_o)
                has_spike_o = (output_spike_acc[t_max] - output_spike_acc[t] > 0).unsqueeze(-1)
                has_spike_i = input_spikes[t].bool().unsqueeze(-2)
                capture = has_spike_i.logical_and(has_spike_o).sum(0) * mu_capture
                search = has_spike_i.logical_and(has_spike_o.logical_not()).sum(0) * mu_search
                history += capture + search
        # compute backoff
        for t in range(time_o):
            has_spike_o = output_spikes[t].bool().unsqueeze(-1)
            if has_spike_o.sum() > 0:
                t_max = min(t + 1, time_i)
                t_min = max(t - self.fodep, 0)
                if t_min > time_i:
                    break
                has_spike_i = (input_spike_acc[t_max] - input_spike_acc[t_min] > 0).unsqueeze(-2)
                backoff = has_spike_i.logical_not().logical_and(has_spike_o).sum(0) * mu_backoff
                history += backoff
        # update
        update = history * (self.weight * (1 - self.weight) * 3 + 0.25)
        self.weight.add_(update).clip_(0, 1)


class ConvColumn(torch.nn.Module):
    def __init__(
        self, 
        input_channel=1, output_channel=1, 
        kernel=3, stride=2, 
        step=16, leak=32, 
        fodep=None, w_init=0., theta=None, dense=None
    ):
        super(ConvColumn, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel = kernel
        self.stride = stride

        assert theta or dense, 'either theta or dense should be specified'
        self.theta = theta = theta or dense * (kernel * kernel * input_channel)
        self.dense = dense = dense or theta / (kernel * kernel * input_channel)
        assert dense < 2 * input_channel * kernel * kernel, 'invalid theta or density, try setting a smaller value'
        # default response function: StepFireLeak with step=1, leak=2
        self.response_function = StepFireLeak(step, leak)
        self.fodep = fodep = fodep or self.response_function.fodep
        assert fodep >= self.response_function.fodep, f'forced depression should be at least {self.response_function.fodep}'
        # initialize weight to w_init
        self.weight = torch.nn.parameter.Parameter(
            Exponential(1 / w_init).sample((self.output_channel, self.input_channel, self.kernel, self.kernel)).clip(0, 1),
            requires_grad=True
        )
        print(f'Building convolutional TNN layer with theta={theta:.4f}, dense={dense:.4f}, fodep={fodep}')
    
    def forward(self, input_spikes, mu_capture=0.2000, mu_backoff=-0.2000, mu_search=0.0001):
        potentials = self.get_potentials(input_spikes)
        output_spikes = self.winner_takes_all(potentials)
        if self.training:
            self.stdp(potentials, output_spikes, mu_capture, mu_backoff, mu_search)
        return output_spikes
    
    def get_potentials(self, input_spikes):
        # move time axis beween channel and synpases
        input_spikes = input_spikes.permute(0, 1, 4, 2, 3)
        w_kernel = self.response_function.get_weight_kernel(self.weight).permute(0, 1, 4, 2, 3)
        potentials = torch.nn.functional.conv3d(
            input_spikes, w_kernel, 
            stride=(1, self.stride, self.stride),
            padding=(self.response_function.padding, 0, 0)
        )
        return potentials.permute(0, 1, 3, 4, 2)
    
    def winner_takes_all(self, potentials):
        batch, channel, neuron_x, neuron_y, time = potentials.shape
        potentials = potentials.reshape(batch, channel, neuron_x * neuron_y, time).permute(3, 1, 0, 2)
        # time to step out of depression, with initial 0 and constrains >= 0
        depression = torch.zeros(channel, batch, neuron_x * neuron_y, dtype=torch.int32, device=potentials.device)
        # return winners of the same shape as potentials
        winners = torch.zeros(time, channel, batch, neuron_x * neuron_y, dtype=torch.int32, device=potentials.device)
        for t in range(time):
            potential_t = potentials[t] * (depression == 0).int()
            winner_t = potential_t.argmax(-1).unsqueeze(-1)
            spike_t = (potential_t.gather(-1, winner_t) > self.theta).int()
            winners[t].scatter_(-1, winner_t, spike_t)
            depression += winners[t].sum(0).unsqueeze(0) * self.fodep
            depression = (depression - 1).clip(0, self.fodep - 1)
            # TODO k limitation per channel
        return winners.permute(2, 1, 0, 3).reshape(batch, channel, neuron_x, neuron_y, time).float()

    def stdp(self, potentials, output_spikes, mu_capture, mu_backoff, mu_search):
        batch, channel, neuron_x, neuron_y, time = output_spikes.shape
        capture_grad, = torch.autograd.grad((potentials * output_spikes).sum(), self.weight, retain_graph=True)
        search_grad, = torch.autograd.grad(potentials.sum(), self.weight)
        backoff_grad = output_spikes.sum((0, 2, 3, 4)).reshape(-1, 1, 1, 1) - capture_grad
        update = (
            capture_grad * mu_capture + 
            backoff_grad * mu_backoff + 
            search_grad * mu_search
        ) * (
            (self.weight * (1 - self.weight) * 3 + 0.25)
        ) / (
            batch * neuron_x * neuron_y
        )
        update = update - update.mean((1, 2, 3), keepdim=True)
        with torch.no_grad():
            self.weight.add_(update).clip_(0, 1)


class RecurColumn(torch.nn.Module):
    def __init__(
        self, 
        synapses, neurons, input_channel=1, output_channel=1, 
        step=16, leak=32, 
        fodep=None, w_init=0., theta=None, dense=None
    ):
        super(RecurColumn, self).__init__()
        self.synapses = synapses
        self.neurons = neurons
        self.input_channel = input_channel
        self.output_channel = output_channel

        assert theta or dense, 'either theta or dense should be specified'
        self.theta = theta = theta or dense * (synapses * input_channel)
        self.dense = dense = dense or theta / (synapses * input_channel)
        assert dense < 2 * input_channel * synapses, 'invalid theta or density, try setting a smaller value'
        # default response function: StepFireLeak with step=1, leak=2
        self.response_function = StepFireLeak(step, leak)
        self.fodep = fodep = fodep or self.response_function.fodep
        assert fodep >= self.response_function.fodep, f'forced depression should be at least {self.response_function.fodep}'
        # initialize weight to zeros first
        self.weight = torch.zeros(self.output_channel * self.neurons, (self.input_channel + self.output_channel) * self.synapses) + w_init
        self.bias = torch.zeros(self.output_channel * self.neurons) + self.theta
        print(f'Building recurrent TNN layer with theta={theta:.4f}, dense={dense:.4f}, fodep={fodep}')
    
    def forward(self, input_spikes, labels=None):
        
        return
