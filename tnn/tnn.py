import numpy as np
import torch
from torch.distributions.exponential import Exponential


class StepFireLeak:
    def __init__(self, step=16, leak=32):
        self.step = step
        self.leak = leak
        self.kernel_size = step + leak
        self.padding = self.kernel_size
        self.fodep = self.kernel_size

    def get_weight_kernel(self, weight):
        kernel_zero = torch.zeros(
            *weight.shape, self.kernel_size, device=weight.device)
        t_axis = torch.arange(
            self.kernel_size, device=weight.device).expand_as(kernel_zero)
        t_spike = t_axis / self.step
        t_leak = -(t_axis - weight.unsqueeze(-1) * self.step) / \
            self.leak + weight.unsqueeze(-1)
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
        assert dense < 2 * input_channel * \
            synapses, 'invalid theta or density, try setting a smaller value'
        # default response function: StepFireLeak with step=1, leak=2
        self.response_function = StepFireLeak(step, leak)
        self.fodep = fodep = fodep or self.response_function.fodep
        assert fodep >= self.response_function.fodep, f'forced depression should be at least {self.response_function.fodep}'
        # initialize weight to w_init
        self.weight = torch.nn.parameter.Parameter(
            Exponential(1 / w_init).sample((self.output_channel *
                                            self.neurons, self.input_channel * self.synapses)).clip(0, 1),
            requires_grad=True
        )
        print(
            f'Building full connected TNN layer with theta={theta:.4f}, dense={dense:.4f}, fodep={fodep}')

    def forward(self, input_spikes, labels=None, bias=0.5, mu_capture=0.2000, mu_backoff=-0.2000, mu_search=0.0001):
        potentials = self.get_potentials(input_spikes, labels, bias)
        output_spikes = self.winner_takes_all(potentials)

        if self.training:
            self.stdp(potentials, output_spikes,
                      mu_capture=mu_capture, mu_backoff=mu_backoff, mu_search=mu_search)

        return output_spikes

    def get_potentials(self, input_spikes, labels=None, bias=0.5):
        # coalesce input channel and synpases
        batch, channel, synapses, time = input_spikes.shape
        input_spikes = input_spikes.reshape(batch, channel * synapses, time)
        # perform conv1d to compute potentials
        w_kernel = self.response_function.get_weight_kernel(self.weight)
        potentials = torch.nn.functional.conv1d(
            input_spikes, w_kernel, padding=self.response_function.padding)
        # expand output channel and neurons
        potentials = potentials.reshape(
            batch, self.output_channel, self.neurons, -1)
        if labels is not None:
            batch, channel, neurons, time = potentials.shape
            supervision = torch.zeros(batch, channel, dtype=torch.int32, device=labels.device).scatter(
                1, labels.unsqueeze(-1), bias * self.theta
            ).unsqueeze(-1).expand(-1, -1, neurons).unsqueeze(-1)
            # supervision (batch, channel, neurons, 1)
            potentials = potentials + supervision
        else:
            potentials = potentials + bias * self.theta
        return potentials

    def winner_takes_all(self, potentials):
        batch, channel, neurons, time = potentials.shape

        # move time axis to 0 for better performance
        potentials = potentials.permute(3, 0, 2, 1)

        # time to step out of depression, with initial 0 and constrains >= 0
        depression = torch.zeros(
            batch, neurons, channel, dtype=torch.int32, device=potentials.device)
        # return winners of the same shape as potentials
        winners = torch.zeros(time, batch, neurons, channel,
                              dtype=torch.int32, device=potentials.device)

        # iterate time axis: get the winner for each batch, channel of neurons, update winners and depression
        for t in range(time):
            # channel, batch, neurons
            potential_t = potentials[t] * (depression == 0).int()
            winner_t = potential_t.argmax(-1).unsqueeze(-1)

            spike_t = (potential_t.gather(-1, winner_t)
                       > self.theta).int()
            winners[t].scatter_(-1, winner_t, spike_t)
            depression += winners[t].sum(-1).unsqueeze(-1) * self.fodep
            depression = (depression - 1).clip(0, self.fodep - 1)
            # TODO k limitation per channel

        return winners.permute(1, 3, 2, 0).float()

    def stdp(
        self,
        potentials, output_spikes,
        mu_capture, mu_backoff, mu_search
    ):
        batch, _channel, neurons, _time = output_spikes.shape

        capture_grad, = torch.autograd.grad(
            (potentials * output_spikes).sum(), self.weight, retain_graph=True)
        search_grad, = torch.autograd.grad(potentials.sum(), self.weight)
        search_grad /= (self.weight *
                        self.response_function.kernel_size).floor() + 1
        backoff_grad = output_spikes.sum(
            (0, 2, 3)).unsqueeze(-1) - capture_grad

        update = (
            capture_grad * mu_capture +
            backoff_grad * mu_backoff +
            search_grad * mu_search
        ) * (
            (self.weight * (1 - self.weight) * 3 + 0.25)
        ) / (
            batch * neurons
        )

        with torch.no_grad():
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
        assert dense < 2 * input_channel * kernel * \
            kernel, 'invalid theta or density, try setting a smaller value'
        # default response function: StepFireLeak with step=1, leak=2
        self.response_function = StepFireLeak(step, leak)
        self.fodep = fodep = fodep or self.response_function.fodep
        assert fodep >= self.response_function.fodep, f'forced depression should be at least {self.response_function.fodep}'
        # initialize weight to w_init
        self.weight = torch.nn.parameter.Parameter(
            Exponential(1 / w_init).sample((self.output_channel,
                                            self.input_channel, self.kernel, self.kernel)).clip(0, 1),
            requires_grad=True
        )
        print(
            f'Building convolutional TNN layer with theta={theta:.4f}, dense={dense:.4f}, fodep={fodep}')

    def forward(self, input_spikes, mu_capture=0.2000, mu_backoff=-0.4000, mu_search=0.0001):
        potentials = self.get_potentials(input_spikes)
        output_spikes = self.winner_takes_all(potentials)
        if self.training:
            self.stdp(potentials, output_spikes,
                      mu_capture, mu_backoff, mu_search)
        return output_spikes

    def get_potentials(self, input_spikes):
        # move time axis beween channel and synpases
        input_spikes = input_spikes.permute(0, 1, 4, 2, 3)
        w_kernel = self.response_function.get_weight_kernel(
            self.weight).permute(0, 1, 4, 2, 3)
        potentials = torch.nn.functional.conv3d(
            input_spikes, w_kernel,
            stride=(1, self.stride, self.stride),
            padding=(self.response_function.padding, 0, 0)
        )
        return potentials.permute(0, 1, 3, 4, 2)

    def winner_takes_all(self, potentials):
        batch, channel, neuron_x, neuron_y, time = potentials.shape
        potentials = potentials.reshape(
            batch, channel, neuron_x * neuron_y, time).permute(3, 0, 2, 1)
        # time to step out of depression, with initial 0 and constrains >= 0
        depression = torch.zeros(
            batch, neuron_x * neuron_y, channel, dtype=torch.int32, device=potentials.device)
        # return winners of the same shape as potentials
        winners = torch.zeros(time, batch, neuron_x * neuron_y,
                              channel, dtype=torch.int32, device=potentials.device)
        for t in range(time):
            potential_t = potentials[t] * (depression == 0).int()
            winner_t = potential_t.argmax(-1).unsqueeze(-1)
            spike_t = (potential_t.gather(-1, winner_t) > self.theta).int()
            winners[t].scatter_(-1, winner_t, spike_t)
            depression += winners[t].sum(-1).unsqueeze(-1) * self.fodep
            depression = (depression - 1).clip(0, self.fodep - 1)
            # TODO k limitation per channel
        return winners.permute(1, 3, 2, 0).reshape(batch, channel, neuron_x, neuron_y, time).float()

    def stdp(self, potentials, output_spikes, mu_capture, mu_backoff, mu_search):
        batch, _channel, neuron_x, neuron_y, _time = output_spikes.shape
        capture_grad, = torch.autograd.grad(
            (potentials * output_spikes).sum(), self.weight, retain_graph=True)
        search_grad, = torch.autograd.grad(potentials.sum(), self.weight)
        search_grad /= (self.weight *
                        self.response_function.kernel_size).floor() + 1
        backoff_grad = output_spikes.sum(
            (0, 2, 3, 4)).reshape(-1, 1, 1, 1) - capture_grad
        update = (
            capture_grad * mu_capture +
            backoff_grad * mu_backoff +
            search_grad * mu_search
        ) * (
            (self.weight * (1 - self.weight) * 3 + 0.25)
        ) / (
            batch * neuron_x * neuron_y
        )

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
        assert dense < 2 * input_channel * \
            synapses, 'invalid theta or density, try setting a smaller value'
        # default response function: StepFireLeak with step=1, leak=2
        self.response_function = StepFireLeak(step, leak)
        self.fodep = fodep = fodep or self.response_function.fodep
        assert fodep >= self.response_function.fodep, f'forced depression should be at least {self.response_function.fodep}'
        # initialize weight to zeros first
        self.weight = torch.zeros(self.output_channel * self.neurons,
                                  (self.input_channel + self.output_channel) * self.synapses) + w_init
        self.bias = torch.zeros(self.output_channel *
                                self.neurons) + self.theta
        print(
            f'Building recurrent TNN layer with theta={theta:.4f}, dense={dense:.4f}, fodep={fodep}')

    def forward(self, input_spikes, labels=None):

        return
