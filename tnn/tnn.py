import torch


class StepFireLeak:
    def __init__(self, step=16, leak=32):
        self.step = step
        self.leak = leak
        self.kernel_size = step + leak

    def get_weight_kernel(self, weight):
        kernel_zero = torch.zeros(*weight.shape, self.kernel_size, device=weight.device)
        t_axis = torch.arange(self.kernel_size, device=weight.device).expand_as(kernel_zero)
        t_spike = t_axis / self.step
        t_leak = -(t_axis - weight.unsqueeze(-1) * self.step) / self.leak + weight.unsqueeze(-1)
        kernel = kernel_zero.max(t_spike.min(t_leak)).flip(-1)
        return kernel

    @property
    def padding(self):
        return self.kernel_size + self.step

    @property
    def fodep(self):
        return self.kernel_size


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
        # initialize weight to zeros first
        self.weight = torch.zeros(self.output_channel * self.neurons, self.input_channel * self.synapses) + w_init

        print(f'Building full connected TNN layer with theta={theta:.4f}, dense={dense:.4f}, fodep={fodep}')
    
    def to(self, device):
        super(FullColumn, self).to(device)
        self.weight = self.weight.to(device)
        return self

    def forward(self, input_spikes, labels=None):
        potentials = self.get_potentials(input_spikes, labels)
        output_spikes = self.winner_takes_all(potentials)
        return output_spikes

    def get_potentials(self, input_spikes, labels=None):
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
                1, labels.unsqueeze(-1), self.theta // 2
            ).unsqueeze(1).expand(-1, channel, -1).unsqueeze(-1)
            potentials = potentials + supervision
        else:
            potentials = potentials + self.theta // 2
        return potentials

    def winner_takes_all(self, potentials):
        # move time axis to 0 for better performance
        batch, channel, neurons, time = potentials.shape
        potentials = potentials.permute(3, 0, 1, 2)
        # time to step out of depression, with initial 0 and constrains >= 0
        depression = torch.zeros(batch, channel, dtype=torch.int32, device=potentials.device)
        min_depression = torch.zeros(batch, channel, dtype=torch.int32, device=potentials.device)
        # return winners of the same shape as potentials
        winners = torch.zeros(time, batch, channel, neurons, dtype=torch.int32, device=potentials.device)
        # iterate time axis: get the winner for each batch, channel of neurons, update winners and depression
        for t in range(time):
            winner_t = potentials[t].argmax(axis=-1).unsqueeze(-1)
            spike_t = (potentials[t].gather(-1, winner_t) > self.theta).squeeze(-1)
            cond_t = spike_t.logical_and(depression == min_depression).int()
            winners[t].scatter_(-1, winner_t, cond_t.unsqueeze(-1))
            depression = min_depression.max(depression + winners[t].sum(axis=-1) * (self.fodep + 1) - 1)
        # move time axis to -1 for the convenience of convolutional operations
        return winners.permute(1, 2, 3, 0)

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

        for t in range(time_i):
            has_spike_i = input_spikes[t].bool().unsqueeze(-2)
            if has_spike_i.sum() > 0:
                t_max = min(t + self.fodep + 1, time_o)
                has_spike_o = (output_spike_acc[t_max] - output_spike_acc[t] > 0).unsqueeze(-1)
                has_spike_i = input_spikes[t].bool().unsqueeze(-2)
                capture = has_spike_i.logical_and(has_spike_o).sum(0) * mu_capture
                search = has_spike_i.logical_and(has_spike_o.logical_not()).sum(0) * mu_search
                history += capture + search
        
        for t in range(time_o):
            has_spike_o = output_spikes[t].bool().unsqueeze(-1)
            if has_spike_o.sum() > 0:
                t_max = min(t + 1, time_i)
                t_min = max(t - self.fodep, 0)
                if t_max > 256 or t_min > 256:
                    import code
                    code.interact(local=locals())
                has_spike_i = (input_spike_acc[t_max] - input_spike_acc[t_min] > 0).unsqueeze(-2)
                backoff = has_spike_i.logical_not().logical_and(has_spike_o).sum(0) * mu_backoff
                history += backoff
        
        update = history * (self.weight * (1 - self.weight) * 3 + 0.25)
        self.weight = (self.weight + update).clip(0, 1)
    