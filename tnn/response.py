import torch
import torch.nn as nn


class StepFireLeakKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, step, leak):
        weight = torch.tanh(weight)
        kernel_size = step + leak
        kernel_zero = torch.zeros(*weight.shape, kernel_size, device=weight.device)
        t_axis = torch.arange(kernel_size, device=weight.device).expand_as(kernel_zero)
        t_spike = t_axis / step
        t_leak = -(t_axis - weight.unsqueeze(-1) * step) / leak + weight.unsqueeze(-1)
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
        self.step = nn.parameter(torch.tensor(step))
        self.leak = nn.parameter(torch.tensor(step))
        self.kernel_size = nn.parameter(step + leak)
        self.padding = nn.parameter(step + leak)
        self.inhib = nn.parameter(step + leak)

    def forward(self, weight):
        return StepFireLeakKernel.apply(weight, self.step, self.leak)

    @property
    def kernel_size(self):
        return self.step + self.leak
    
    @property
    def padding(self):
        return self.step + self.leak
    
    @property
    def inhib(self):
        return self.step + self.leak
