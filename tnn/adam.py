import torch
import torch.nn as nn


class AdamSTDP(nn.Module):
    def __init__(self, param, alpha, beta1=0.99, beta2=0.999) -> None:
        super().__init__()
        self.moment1_capture = nn.parameter.Parameter(torch.zeros_like(param))
        self.moment2_capture = nn.parameter.Parameter(torch.zeros_like(param))
        self.moment1_search = nn.parameter.Parameter(torch.zeros_like(param))
        self.moment2_search = nn.parameter.Parameter(torch.zeros_like(param))
        self.alpha = nn.parameter.Parameter(torch.tensor(alpha))
        self.beta1 = nn.parameter.Parameter(torch.tensor(beta1))
        self.beta2 = nn.parameter.Parameter(torch.tensor(beta2))

    def forward(self, spikes, bias, capture, search, eps=1e-8):
        self.moment1_capture.set_(self.moment1_capture * self.beta1 ** spikes + capture * (1 - self.beta1 ** spikes))
        self.moment2_capture.set_(self.moment2_capture * self.beta2 ** spikes + capture ** 2 * (1 - self.beta2 ** spikes))
        self.moment1_search.set_(self.moment1_search * self.beta1 + search * (1 - self.beta1))
        self.moment2_search.set_(self.moment2_search * self.beta2 + search ** 2 * (1 - self.beta2))
        snr_capture = self.moment1_capture / (self.moment2_capture.sqrt() + eps)
        snr_search = self.moment1_search / (self.moment2_search.sqrt() + eps)
        return self.alpha * snr_capture + self.alpha * bias * snr_search
