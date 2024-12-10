import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ParallelLinear(nn.Module):
    def __init__(self, input_dim, output_dim, channels=1):
        super(ParallelLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        k = 1 / math.sqrt(input_dim)
        self.weight = nn.Parameter(torch.empty(channels, input_dim, output_dim).uniform_(-k, k))
        self.bias = nn.Parameter(torch.empty(channels, output_dim).uniform_(-k, k))

    def forward(self, x):
        # x shape: (batch, channels, input_dim)
        # weight shape: (channels, input_dim, output_dim)
        # output shape: (batch, channels, output_dim)
        return torch.einsum('bci,cio->bco', x, self.weight) + self.bias.unsqueeze(0)