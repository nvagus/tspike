import numpy as np
import torch
import torch.nn as nn
from torch.distributions.exponential import Exponential

from .dual import SignalDualBackground
from .adam import AdamSTDP


class StepFireLeakKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, step, leak):
        weight = torch.tanh(weight)
        kernel_size = step + leak
        kernel_zero = torch.zeros(
            *weight.shape, kernel_size, device=weight.device)
        t_axis = torch.arange(
            kernel_size, device=weight.device).expand_as(kernel_zero)
        t_spike = t_axis / step
        t_leak = -(t_axis - weight.unsqueeze(-1) * step) / \
            leak + weight.unsqueeze(-1)
        kernel = kernel_zero.max(t_spike.min(t_leak)).flip(-1)
        ctx.save_for_backward(kernel)
        return kernel

    @staticmethod
    def backward(ctx, grad_output):
        kernel, = ctx.saved_tensors
        return (grad_output * (kernel + 1e-8)).sum(-1), None, None


class StepFireLeak(nn.Module):
    def __init__(self, step=16, leak=32):
        super(StepFireLeak, self).__init__()
        self.step = step
        self.leak = leak
        self.kernel_size = step + leak
        self.padding = self.kernel_size
        self.fodep = self.kernel_size

    def forward(self, weight):
        return StepFireLeakKernel.apply(weight, self.step, self.leak)


class TNNColumn(nn.Module):
    def __init__(self):
        super(TNNColumn, self).__init__()

    @staticmethod
    def _describe_weight(weight):
        # param weight: [pattern_0, pattern_1, ..., pattern_k]
        # othogonal
        w_norm = weight / (weight ** 2).sum(1, keepdim=True).sqrt()
        othogonal = (((w_norm @ w_norm.T) ** 2).mean() -
                     1 / weight.shape[0]).sqrt()
        # distribution
        w_mean = weight.mean(-1)
        w_min = w_mean.min()
        w_max = w_mean.max()
        w_avg = w_mean.mean()
        return ', '.join([
            f'othogonal: {othogonal * 100:.2f}',
            f'pattern: {w_min * 100:.2f}-{w_avg * 100:.2f}-{w_max * 100:.2f}'
        ])

    @staticmethod
    def _describe_bias(bias):
        b_min = bias.min()
        b_max = bias.max()
        b_avg = bias.mean()
        return ', '.join([
            f'pattern: {b_min * 100:.2f}-{b_avg * 100:.2f}-{b_max * 100:.2f}'
        ])

    def describe_weight(self):
        raise NotImplementedError()

    def describe_bias(self):
        raise NotImplementedError()

    def describe(self):
        return f'weight<{self.describe_weight()}>; bias<{self.describe_bias()}>'


class FullColumn(TNNColumn):
    def __init__(
        self,
        synapses, neurons, input_channel=1, output_channel=1,
        step=16, leak=32, bias=0.5, winners=None,
        fodep=None, w_init=None, theta=None, dense=None,
        alpha=0.02, beta1=0.99, beta2=0.999
    ):
        super(FullColumn, self).__init__()
        # model skeleton parameters
        self.synapses = synapses
        self.neurons = neurons
        self.input_channel = input_channel
        self.output_channel = output_channel

        # threshold parameters
        assert theta or dense, 'either theta or dense should be specified'
        self.theta = theta = theta or dense * (synapses * input_channel)
        self.dense = dense = dense or theta / (synapses * input_channel)
        w_init = w_init or dense

        # spiking control parameters
        self.response_function = StepFireLeak(step, leak)
        self.fodep = fodep = fodep or self.response_function.fodep
        assert fodep >= self.response_function.fodep, f'forced depression should be at least {self.response_function.fodep}'
        self.winners = winners = winners or neurons

        # initialize weight and bias
        self.bias = nn.parameter.Parameter(
            torch.zeros(self.output_channel * self.neurons) + bias, 
            requires_grad=False
        )
        self.weight = nn.parameter.Parameter(
            Exponential(1 / w_init).sample(
                (self.output_channel * self.neurons, self.input_channel * self.synapses)).clip(min=0),
            requires_grad=True
        )
        self.dual = SignalDualBackground()
        self.optimizer = AdamSTDP(self.weight, alpha, beta1, beta2)
        print(
            'Building full connected TNN layer with '
            f'theta={theta:.4f}, '
            f'dense={dense:.4f}, '
            f'fodep={fodep}, ',
            f'winners={winners}, '
            f'bias={bias}'
        )

    def forward(self, input_spikes, labels=None, bias_decay=0.9999):
        potentials, supervision = self.get_potentials(input_spikes, labels)
        output_spikes = self.winner_takes_all(potentials)
        if self.training:
            self.stdp(potentials, output_spikes, supervision, bias_decay=bias_decay)

        return output_spikes

    def get_potentials(self, input_spikes, labels=None):
        # coalesce input channel and synpases
        batch, channel, synapses, time = input_spikes.shape
        dual_spikes = self.dual(input_spikes).reshape(
            batch, channel * synapses, time)
        input_spikes = input_spikes.reshape(batch, channel * synapses, time)
        # perform conv1d to compute potentials
        w_kernel = self.response_function.forward(self.weight)
        potentials = nn.functional.conv1d(
            input_spikes, w_kernel, padding=self.response_function.padding)
        # apply dual spikes
        with torch.no_grad():
            dual_potentials = nn.functional.conv1d(
                dual_spikes, w_kernel, padding=self.response_function.padding)
        potentials += dual_potentials
        # expand output channel and neurons
        potentials = potentials.reshape(
            batch, self.output_channel, self.neurons, -1)
        # apply bias
        if labels is not None:
            # apply bias to labeled channels
            supervision = torch.zeros(batch, self.output_channel, dtype=torch.int32, device=labels.device).scatter(
                1, labels.unsqueeze(-1), 1
            ).unsqueeze(-1)
            # supervision (batch, channel, 1)
            potentials = potentials + (
                supervision * self.theta *
                self.bias.reshape(1, self.output_channel, self.neurons)
            ).unsqueeze(-1)
            return potentials, supervision
        else:
            # apply bias to all channels
            potentials = potentials + (
                self.theta * self.bias
            ).reshape(1, self.output_channel, self.neurons, 1)
            return potentials, None

    def winner_takes_all(self, potentials):
        batch, channel, neurons, time = potentials.shape

        # move time axis to 0 for better performance
        potentials = potentials.permute(3, 0, 2, 1)

        # time to step out of depression, with initial 0 and constrains >= 0
        depression = torch.zeros(
            batch, neurons, dtype=torch.int32, device=potentials.device)
        
        # return winners of the same shape as potentials
        winners = torch.zeros(
            time, batch, neurons, channel,
            dtype=torch.int32, device=potentials.device)

        # iterate time axis: get the winner for each batch, channel of neurons, update winners and depression
        for t in range(time):
            # apply depression state to the potential
            depress_t = (depression == 0).unsqueeze(-1).int()
            k_depress_t = ((depression != 0).sum(-1) <
                           self.winners).int().reshape(-1, 1, 1)
            potential_t = (potentials[t] * depress_t *
                           k_depress_t).reshape(batch, -1)
            # find channel and neuron winners
            cn_winner_t = potential_t.argmax(-1).unsqueeze(-1)
            p_winner_t = potential_t.gather(-1, cn_winner_t)
            spike_t = (p_winner_t > self.theta).int()
            # set winner@t
            winners[t] = winners[t].reshape(
                batch, -1).scatter(-1, cn_winner_t, spike_t).reshape(batch, neurons, -1)
            # update depression
            depression += winners[t].sum(-1) * self.fodep
            depression = (depression - 1).clip(0, self.fodep - 1)

        return winners.permute(1, 3, 2, 0).float()

    def stdp(
        self,
        potentials, output_spikes, supervision,
        bias_decay, eps=1e-8
    ):
        batch, channel, neurons, time = output_spikes.shape
        
        spiked_potentials = potentials * output_spikes
        with torch.no_grad():
            normed_spiked_potentials = spiked_potentials + eps
        spiked_potentials = spiked_potentials * self.theta / normed_spiked_potentials

        if supervision is not None:
            potentials = potentials * supervision.unsqueeze(-1)
            spiked_potentials = spiked_potentials * (2 * supervision - 1).unsqueeze(-1)

        total_spikes = output_spikes.sum((0, 3))

        capture_grad, = torch.autograd.grad(spiked_potentials.sum(), self.weight, retain_graph=True)
        backoff_grad = - total_spikes.reshape(-1, 1) * self.dense
        search_grad, = torch.autograd.grad(potentials.sum(), self.weight)
        search_grad = search_grad / (self.weight + eps)

        with torch.no_grad():
            bias_update = bias_decay ** total_spikes.reshape(-1)
            self.bias.mul_(bias_update)
            weight_update = self.optimizer.forward(
                total_spikes.reshape(-1, 1),
                self.bias.unsqueeze(-1),
                capture_grad + backoff_grad,
                search_grad
            )
            self.weight.add_(weight_update).clip_(min=0, max=16)

    def describe_weight(self):
        return self._describe_weight(torch.tanh(self.weight).detach())

    def describe_bias(self):
        return self._describe_bias(self.bias)


class ConvColumn(nn.Module):
    def __init__(
        self,
        input_channel=1, output_channel=1,
        kernel=3, stride=2,
        step=16, leak=32, bias=0.5, winners=0.5,
        fodep=None, w_init=None, theta=None, dense=None
    ):
        super(ConvColumn, self).__init__()
        # model skeleton parameters
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel = kernel
        self.stride = stride

        # threshold parameters
        assert theta or dense, 'either theta or dense should be specified'
        self.theta = theta = theta or dense * (kernel * kernel * input_channel)
        self.dense = dense = dense or theta / (kernel * kernel * input_channel)
        assert dense < 2 * input_channel * kernel * \
            kernel, 'invalid theta or density, try setting a smaller value'
        w_init = w_init or dense

        # spiking control parameters
        self.response_function = StepFireLeak(step, leak)
        self.fodep = fodep = fodep or self.response_function.fodep
        assert fodep >= self.response_function.fodep, f'forced depression should be at least {self.response_function.fodep}'
        self.winners = winners

        # initialize weight and bias
        self.bias = nn.parameter.Parameter(torch.zeros(
            self.output_channel) + bias, requires_grad=False)
        self.weight = nn.parameter.Parameter(
            Exponential(1 / w_init).sample((self.output_channel,
                                            self.input_channel, self.kernel, self.kernel)).clip(0, 1),
            requires_grad=True
        )
        print(
            'Building convolutional connected TNN layer with '
            f'theta={theta:.4f}, '
            f'dense={dense:.4f}, '
            f'fodep={fodep}, ',
            f'winners={winners}, '
            f'bias={bias}'
        )

    def forward(self, input_spikes, mu_capture=0.2000, mu_backoff=-0.2000, mu_search=0.0001, beta_decay=0.9):
        potentials = self.get_potentials(input_spikes)
        output_spikes = self.winner_takes_all(potentials)
        if self.training:
            self.stdp(potentials, output_spikes,
                      mu_capture, mu_backoff, mu_search, beta_decay)
        return output_spikes

    def get_potentials(self, input_spikes):
        # move time axis beween channel and synpases
        input_spikes = input_spikes.permute(0, 1, 4, 2, 3)
        w_kernel = self.response_function.forward(
            self.weight).permute(0, 1, 4, 2, 3)
        potentials = nn.functional.conv3d(
            input_spikes, w_kernel,
            bias=self.bias,
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
            batch, neuron_x * neuron_y, dtype=torch.int32, device=potentials.device)
        # return winners of the same shape as potentials
        winners = torch.zeros(time, batch, neuron_x * neuron_y,
                              channel, dtype=torch.int32, device=potentials.device)
        winner_fodep = np.ceil(self.winners * neuron_x * neuron_y)
        for t in range(time):
            depress_t = (depression == 0).unsqueeze(-1).int()
            k_depress_t = ((depression != 0).sum(-1) <
                           winner_fodep).int().reshape(-1, 1, 1)
            potential_t = potentials[t] * depress_t * k_depress_t
            winner_t = potential_t.argmax(-1).unsqueeze(-1)
            spike_t = (potential_t.gather(-1, winner_t) > self.theta).int()
            winners[t].scatter_(-1, winner_t, spike_t)
            depression += winners[t].sum(-1) * self.fodep
            depression = (depression - 1).clip(0, self.fodep - 1)

        return winners.permute(1, 3, 2, 0).reshape(batch, channel, neuron_x, neuron_y, time).float()

    def stdp(
        self,
        potentials, output_spikes,
        mu_capture, mu_backoff, mu_search, beta_decay
    ):
        batch, _channel, neuron_x, neuron_y, _time = output_spikes.shape
        total_spike = output_spikes.sum((0, 2, 3, 4)).reshape(-1, 1, 1, 1)

        capture_grad, = torch.autograd.grad(
            (potentials * output_spikes).sum(), self.weight, retain_graph=True)
        capture_grad = capture_grad.min(total_spike)
        backoff_grad = total_spike - capture_grad
        search_grad, = torch.autograd.grad(
            potentials.sum() / self.response_function.kernel_size, self.weight)
        search_grad = search_grad * (capture_grad == 0).int()

        weight_update = (
            capture_grad * mu_capture * (1 - torch.tanh(self.weight)) +
            backoff_grad * mu_backoff +
            search_grad * mu_search * self.bias.reshape(-1, 1, 1, 1)
        ) / (
            batch * neuron_x * neuron_y
        )

        bias_update = beta_decay ** total_spike.reshape(-1)

        with torch.no_grad():
            self.bias.mul_(bias_update)
            self.weight.add_(weight_update).clip_(0, 1)


class RecurColumn(nn.Module):
    def __init__(
        self,
        synapses, neurons, input_channel=1, output_channel=1,
        step=16, leak=32,
        fodep=None, delay=None, w_init=None, theta=None, dense=None,
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
        w_init = w_init or dense
        # default response function: StepFireLeak
        self.response_function = StepFireLeak(step, leak)
        self.fodep = fodep = fodep or self.response_function.fodep
        assert fodep >= self.response_function.fodep, f'forced depression should be at least {self.response_function.fodep}'
        self.delay = delay or self.fodep
        # initialize weight to w_init
        self.weight_i = nn.parameter.Parameter(
            Exponential(1 / w_init).sample((self.output_channel *
                                            self.neurons, self.input_channel * self.synapses)).clip(0, 1),
            requires_grad=True
        )
        self.weight_s = nn.parameter.Parameter(
            Exponential(1 / w_init).sample((self.output_channel *
                                            self.neurons, self.input_channel * self.synapses)).clip(0, 1),
            requires_grad=True
        )
        print(
            f'Building full connected TNN layer with theta={theta:.4f}, dense={dense:.4f}, fodep={fodep}')

    def forward(self, input_spikes, labels=None, bias=0.5, mu_capture=0.2000, mu_backoff=-0.2000, mu_search=0.0001):
        potentials = self.get_potentials(input_spikes, labels, bias)
        output_spikes = torch.zeros_like(potentials)

        batch, channel, neurons, time_o = output_spikes.shape
        for t in range(0, self.delay, time_o):
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
        w_kernel = self.response_function.forward(self.weight_i)
        potentials = nn.functional.conv1d(
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

        total_spikes = output_spikes.sum((0, 2, 3)).unsqueeze(-1)

        capture_grad, = torch.autograd.grad(
            (potentials * output_spikes).sum(), self.weight, retain_graph=True)
        capture_grad = capture_grad.min(total_spikes)
        search_grad, = torch.autograd.grad(
            potentials.sum() / self.response_function.kernel_size, self.weight)
        backoff_grad = total_spikes - capture_grad

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
