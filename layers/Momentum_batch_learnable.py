import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class Momentum_batch_learnable(nn.Module):
    def __init__(self,  configs, channels, vector_len):
        super(Momentum_batch_learnable, self).__init__()
        self.cfg = configs
        self.vector_len = vector_len #F
        # self.register_buffer('momentum_matrix', torch.zeros(len(self.cfg.momentum_params) * 2 + 1, vector_len))
        self.momentum_matrix = torch.zeros(channels, len(self.cfg.momentum_params), vector_len, 1) #C, 3, F, 1
        self.momentum_len = self.cfg.momentum_params
        self.channels = channels

        self.batch = configs.batch_size
        # create multiplication tensor
        #self.mul_tensor = torch.zeros(len(configs.momentum_params), self.batch+1, self.batch) # 3, B+1, B
        momentum_params = torch.asarray(configs.momentum_params) # 3
        momentum_params = momentum_params.unsqueeze(0).repeat(channels, 1) # C, 3
        momentum_params_learnable = torch.log(torch.div(momentum_params,(1-momentum_params))) # C, 3
        self.momentum_params_learnable = nn.Parameter(momentum_params_learnable.unsqueeze(2).unsqueeze(3)) # C, 3,1,1

        self.exp_mtx_1 = torch.zeros(1,1, self.batch + 1, self.batch) # 1, 1, B+1, B
        self.exp_mtx_2 = torch.zeros(1,1, self.batch + 1, self.batch)
        for idx_ in range(self.batch + 1):
            for idx__ in range(self.batch):
                if idx__ >= idx_:
                    self.exp_mtx_1[:,:, idx_, idx__] = 1+idx__-idx_
                if idx_>0:
                    self.exp_mtx_2[:,:, idx_, idx__] =1

        #TODO if memory low --> convert minimal value to zero and use sparse matrix (in case very large batch)

        temp = torch.zeros(channels, len(self.cfg.momentum_params) * 2 + 1, vector_len)# C, 7, F
        for idx_ in range(channels):
            for idx in range(len(self.cfg.momentum_params) * 2 + 1):
                temp[idx_:idx_+1,idx:idx + 1, :] = torch.ones(1, 1, vector_len) * (-(((len(self.cfg.momentum_params) - idx) ** 2) / 4))
        self.learnable_matrix = nn.Parameter(temp) # C, 7, F

    def gen_mul_tensor(self, momentum_params_learnable): ## C, 3,1,1
        momentum_params_learnable = torch.sigmoid(momentum_params_learnable)
        mul_tensor = (momentum_params_learnable**self.exp_mtx_1)*((1-momentum_params_learnable)**self.exp_mtx_2)
        mul_tensor = torch.triu(mul_tensor, diagonal=-1)
        return mul_tensor # C, 3, B+1, B

    def forward(self, vector): # vector.shape = (B, channels, vector): B,C,F
        self.mul_tensor = self.gen_mul_tensor(self.momentum_params_learnable)
        batch = vector.shape[0]
        vector = torch.transpose(vector, 0, 1) #C,B,F
        # TODO if vector batch is not batchsize (last batch)
        # Ensure that the momentum_matrix is initialized correctly
        N = len(self.cfg.momentum_params) #3
        if torch.all(self.momentum_matrix == 0): #C, 3, F, 1
            with torch.no_grad():
                self.momentum_matrix = vector[:,0:1,:,None].expand_as(self.momentum_matrix) # C, 3, F, 1
        #print('11', self.momentum_matrix.unsqueeze(0).detach().shape) #([1, 7, 96])
        #print(torch.t(vector).expand(N, self.vector_len, batch).shape) #([3, 96, 64])
        lhs = torch.cat((self.momentum_matrix.detach(), torch.transpose(vector, 1,2).unsqueeze(1).repeat(1,N,1,1)), dim=3)
        # C, 3, F, 1 / C,3,F,B --> C,3,F,B+1
        #concat self.momentum_matrix: 3,F,1 and expended vector 3,F,B
        if batch == self.batch:
            out = torch.matmul(lhs, self.mul_tensor) # matmul(C,3,F,B+1, / C, 3, B+1, B) --> out: C,3,F,B
        else:
            out = torch.matmul(lhs, self.mul_tensor[:,:, :batch+1, :batch])  # out: C,3,F,b

        out = torch.cat((self.momentum_matrix, out), dim=3)  # C, 3, F, 1 + C, 3,F,B --> 3,F,B+1
        self.momentum_matrix = out[...,-1:].clone().detach() # C, 3,F,1
        out = out[...,:-1] # C,3,F,B
        if not self.cfg.bptt:
            out = out.detach()

        matrix = torch.softmax(self.learnable_matrix, dim=1)  # C, 7, F

        matrix_1 = 2*(matrix[:,N+1:,:]-torch.flip(matrix[:,:N,:], (1,))) # C, 3,F
        matrix_2 = (2*torch.sum(matrix[:,:N,:], dim=1, keepdim=True)+matrix[:, N:N+1,:]) # C, 1, F

        # Combine the updated momentum_matrix with the learnable_matrix to produce the final vector
        vector = torch.transpose(torch.mul(matrix_1.unsqueeze(3), out).sum(dim=1), 1,2)+torch.mul(matrix_2, vector)
        #torch.t(torch.mul(C, 3,F,1/ C,3,F,B).sum(dim=1))+torch.mul(C, 1, F/ C,B,F)
        # C, B,F
        vector = torch.transpose(vector, 0,1) #B,C,F

        return vector

    def reset_momentum(self):  
        self.momentum_matrix = torch.zeros(self.channels, len(self.cfg.momentum_params), self.vector_len, 1).to(self.learnable_matrix.device)
        self.momentum_params_learnable = self.momentum_params_learnable.to(self.learnable_matrix.device)
        self.exp_mtx_1 = self.exp_mtx_1.to(self.learnable_matrix.device)
        self.exp_mtx_2 = self.exp_mtx_2.to(self.learnable_matrix.device)