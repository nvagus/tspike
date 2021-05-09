import torch
import torch.nn as nn


class SignalDualBackground(nn.Module):
    def __init__(self, beta=0.99):
        super().__init__()
        self.beta = nn.parameter.Parameter(torch.tensor(beta))
    
    def forward(self, spikes):
        # spikes.shape: batch, channel, synpase, time
        batch, channel, synpase, time = spikes.shape
        stat = spikes.mean(-2).permute(2, 1, 0) # time, channel, batch
        stat[0] = stat[0] * (1 - self.beta)
        for t in range(1, time):
            stat[t] = stat[t-1] * self.beta + stat[t] * (1 - self.beta)
        stat = stat.permute(2, 1, 0).unsqueeze(-2) # batch, channel, 1, time
        dual_spikes = (1 - spikes) * stat
        return dual_spikes
