from typing import Sequence
import torch


from .tnn import *


class StackFullColumn(torch.nn.Module):
    def __init__(
        self, 
        synapses, channels=1, pool=2,
        step=16, leak=32, 
        fodep=None, w_init=0., theta=None, dense=None
    ):
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

        self.pool = pool
        for i in range(self.num_spikes - 1):
            pass