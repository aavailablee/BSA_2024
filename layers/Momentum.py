import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
# from einops import rearrange, repeat


class Momentum(nn.Module): 
    def __init__(self, configs, channels, vector_len):
        super(Momentum, self).__init__()
        self.cfg = configs
        self.vector_len = vector_len
        self.channels = channels
        self.momentum_matrix = torch.zeros(self.channels, len(self.cfg.momentum_params) * 2 + 1, vector_len)
        
        temp = torch.zeros_like(self.momentum_matrix)
        for idx in range(len(self.cfg.momentum_params) * 2 + 1):
            ## 倒反的数值，最开始最大，中间为0，后面为负数，会因为softmax变为0
            temp[:, idx:idx + 1, :] = torch.ones(self.channels, 1, vector_len) * (-(((len(self.cfg.momentum_params) - idx) ** 2) / 4))
        self.learnable_matrix = nn.Parameter(temp)

    def forward(self, vector): #TODO input 이제 channel 다 들어옴. vetor.shape = (1, channels, vector)
        N = len(self.cfg.momentum_params)
        if torch.all(self.momentum_matrix == 0):
            with torch.no_grad():
                vector_reshaped = vector.squeeze(0).unsqueeze(1)
                self.momentum_matrix[:, len(self.cfg.momentum_params):] = vector_reshaped.expand(-1, N+1, -1) # shape = (channels, 7, vector_len)
        else:
            # Update the momentum_matrix based on the current vector and the momentum parameters
            alpha = torch.tensor(self.cfg.momentum_params).to(vector.device).unsqueeze(0).unsqueeze(2)   # alpha.shape = (1, 3, 1)
            
            new_momentum_matrix = torch.zeros_like(self.momentum_matrix) # shape = (channels, 7, vector_len)
            # 0先加0.5倍的输入
            new_momentum_matrix[:, N] += 0.5 * vector.squeeze(0) # shape = (channels, vector_len)
            # self.momentum_matrix[:, N+1:].detach().shape = (channels, 3, vector_len), vector.squeeze(0).unsqueeze(1).shape = (channels, 1, vector_len), # alpha.shape = (1, 3, 1)
            new_momentum_matrix[:, N+1:] = alpha * self.momentum_matrix[:, N+1:].detach() + (1 - alpha) * vector.squeeze(0).unsqueeze(1)
            new_momentum_matrix[:, :N] = torch.flip(vector.squeeze(0).unsqueeze(1) - new_momentum_matrix[:, N+1:], dims=[1]) # (channels, 3, vector_len)
            
            self.momentum_matrix = new_momentum_matrix
            
        matrix = torch.softmax(self.learnable_matrix, dim=1) # self.learnable_matrix.shape = (channels, 7, vector_len)
        
        # Multiply and sum across the second dimension (after channels)
        vector = torch.mul(matrix, self.momentum_matrix).sum(dim=1) # vector.shape = (channels, vector_len)
        vector = 2 * vector  # Scale the output
        
        return vector.unsqueeze(0)  # shape = (1, channels, vector_len)

    def reset_momentum(self):
        self.momentum_matrix = torch.zeros(self.channels, len(self.cfg.momentum_params) * 2 + 1, self.vector_len).to(self.learnable_matrix.device)