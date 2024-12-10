import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class Momentum_batch(nn.Module):
    def __init__(self, configs, channels, vector_len):
        super(Momentum_batch, self).__init__()
        self.cfg = configs
        self.vector_len = vector_len
        self.channels = channels
        self.momentum_matrix = torch.zeros(self.channels, len(self.cfg.momentum_params), vector_len, 1) # channels, 3, vector_len, 1(batch 차원)

        self.batch = configs.batch_size
        # create multiplication tensor
        self.mul_tensor = torch.zeros(self.channels, len(configs.momentum_params), self.batch+1, self.batch) # channels, 3, B+1, B
        for idx, coeff in enumerate(configs.momentum_params):
            for idx_ in range(self.batch+1):
                for idx__ in range(self.batch):
                    if idx__+1>=idx_:
                        self.mul_tensor[:, idx, idx_, idx__] = torch.ones(self.channels) * ((1-coeff)**(1 if idx_>0 else 0)) * (coeff**(idx__+1-idx_ if idx__+1-idx_>=0 else 0))
                        
        #TODO if memory low --> convert minimal value to zero and use sparse matrix (in case very large batch)

        temp = torch.zeros(self.channels, len(self.cfg.momentum_params) * 2 + 1, vector_len) # channels, 7, vector_len
        for idx in range(len(self.cfg.momentum_params) * 2 + 1):
            temp[:, idx:idx + 1, :] = torch.ones(self.channels, 1, vector_len) * (-(((len(self.cfg.momentum_params) - idx) ** 2) / 4))
        self.learnable_matrix = nn.Parameter(temp) # channels, 7, vector_len

    def forward(self, vector): #TODO input 이제 channel 다 들어옴. vector.shape = (B, channels, vector_len)
        #* vector B, vector_len 이였음
        batch = vector.shape[0]
        vector = vector.permute(1,0,2) # channels, B, vector_len

        # TODO if vector batch is not batchsize (last batch)
        # Ensure that the momentum_matrix is initialized correctly
        N = len(self.cfg.momentum_params)
        if torch.all(self.momentum_matrix == 0):
            with torch.no_grad():
                self.momentum_matrix = vector[:,0:1,:,None].expand_as(self.momentum_matrix) # channels, 3(이게 expand), vector_len, 1(batch 차원)

        # lhs.shape = (channels, 3, vector_len, batch+1) 
        lhs = torch.cat((self.momentum_matrix.detach(), vector.permute(0,2,1).unsqueeze(1).expand(self.channels, N, self.vector_len, batch)), dim=3) 
        if batch == self.batch:
            out = torch.matmul(lhs, self.mul_tensor.detach()) # out.shape = channels, 3, vector_len, batch
        else:
            out = torch.matmul(lhs, self.mul_tensor.detach()[:, :, :batch+1, :batch])  # out.shape = channels, 3, vector_len, batch

        out = torch.cat((self.momentum_matrix, out), dim=3)    # channels, 3, vector_len , batch+1
        self.momentum_matrix = out[:,:,:,-1:].clone().detach() # channels, 3, vector_len, 1
        out = out[:,:,:,:-1] # channels, 3, vector_len, batch
        if not self.cfg.bptt:
            out = out.detach()

        matrix = torch.softmax(self.learnable_matrix, dim=1)  # channels, 7, vector_len

        matrix_1 = 2*(matrix[:,N+1:,:] - torch.flip(matrix[:,:N,:], (1,))) # channels, 3, vector_len
        matrix_2 = 2*torch.sum(matrix[:,:N,:], dim=1, keepdim=True) + matrix[:,N:N+1,:] # channels, 1, vector_len

        # Combine the updated momentum_matrix with the learnable_matrix to produce the final vector
        vector = torch.transpose(torch.mul(matrix_1.unsqueeze(3), out).sum(dim=1), 1, 2) + torch.mul(matrix_2, vector) # channels, batch, vector_len

        return torch.transpose(vector, 0, 1) # batch, channels, vector_len

    def reset_momentum(self):
        self.momentum_matrix = torch.zeros(self.channels, len(self.cfg.momentum_params), self.vector_len, 1).to(self.learnable_matrix.device)
        self.mul_tensor = self.mul_tensor.to(self.learnable_matrix.device)