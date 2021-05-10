import torch
import torch.nn as nn


class AdamSTDP(nn.Module):
    def __init__(self, param, alpha=0.1, beta1=0.99, beta2=0.999) -> None:
        super().__init__()
        self.moment1 = nn.parameter.Parameter(torch.zeros_like(param))
        self.moment2 = nn.parameter.Parameter(torch.zeros_like(param))
        self.alpha = nn.parameter.Parameter(torch.tensor(alpha))
        self.beta1 = nn.parameter.Parameter(torch.tensor(beta1))
        self.beta2 = nn.parameter.Parameter(torch.tensor(beta2))

    def forward(self, grad, eps=1e-8):
        self.moment1.set_(self.moment1 * self.beta1 + grad * (1 - self.beta1))
        self.moment2.set_(self.moment2 * self.beta2 + grad ** 2 * (1 - self.beta2))
        snr = self.moment1 / (self.moment2.sqrt() + eps)
        return self.alpha * snr
