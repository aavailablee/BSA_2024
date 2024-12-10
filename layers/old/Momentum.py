import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class Momentum(nn.Module):
    def __init__(self, configs, vector_len):
        super(Momentum, self).__init__()
        self.cfg = configs
        self.vector_len = vector_len
        # self.register_buffer('momentum_matrix', torch.zeros(len(self.cfg.momentum_params) * 2 + 1, vector_len))
        self.momentum_matrix = torch.zeros(len(self.cfg.momentum_params) * 2 + 1, vector_len)
        
        temp = torch.zeros_like(self.momentum_matrix)
        for idx in range(len(self.cfg.momentum_params) * 2 + 1):
            temp[idx:idx + 1, :] = torch.ones(1, vector_len) * (-(((len(self.cfg.momentum_params) - idx) ** 2) / 4))
        self.learnable_matrix = nn.Parameter(temp)

    def forward(self, vector):
        # Ensure that the momentum_matrix is initialized correctly
        N = len(self.cfg.momentum_params)
        if torch.all(self.momentum_matrix == 0):
            with torch.no_grad():
                self.momentum_matrix[len(self.cfg.momentum_params):] = vector.expand_as(self.momentum_matrix[len(self.cfg.momentum_params):])
        else:
            # Update the momentum_matrix based on the current vector and the momentum parameters
            alpha = torch.tensor(self.cfg.momentum_params)[:,None].to(vector.device)
            
            new_momentum_matrix = torch.zeros_like(self.momentum_matrix)
            new_momentum_matrix[N] += 0.5*vector.squeeze()
            new_momentum_matrix[N+1:] = alpha * self.momentum_matrix[N+1:].detach() + (1 - alpha) * vector
            new_momentum_matrix[:N] = torch.flip(vector - new_momentum_matrix[N+1:], dims=(0,))
            self.momentum_matrix = new_momentum_matrix
            
        matrix = torch.softmax(self.learnable_matrix, dim=0)  # bong
        
        # Combine the updated momentum_matrix with the learnable_matrix to produce the final vector
        vector = torch.mul(matrix, self.momentum_matrix).sum(dim=0)
        vector = 2*vector

        return vector       
    
    def reset_momentum(self): 
        self.momentum_matrix = torch.zeros(len(self.cfg.momentum_params) * 2 + 1, self.vector_len).to(self.learnable_matrix.device)