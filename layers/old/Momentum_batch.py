import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class Momentum_batch(nn.Module):
    def __init__(self, configs, vector_len):
        super(Momentum_batch, self).__init__()
        self.cfg = configs
        self.vector_len = vector_len
        # self.register_buffer('momentum_matrix', torch.zeros(len(self.cfg.momentum_params) * 2 + 1, vector_len))
        self.momentum_matrix = torch.zeros(len(self.cfg.momentum_params), vector_len, 1) #3, F, 1

        self.batch = configs.batch_size
        # create multiplication tensor
        self.mul_tensor = torch.zeros(len(configs.momentum_params), self.batch+1, self.batch) # 3, B+1, B
        for idx, coeff in enumerate(configs.momentum_params):
            for idx_ in range(self.batch+1):
                for idx__ in range(self.batch):
                    if idx__+1>=idx_:
                        self.mul_tensor[idx, idx_, idx__] = ((1-coeff)**(1 if idx_>0 else 0)) * (coeff**(idx__+1-idx_ if idx__+1-idx_>=0 else 0))
        #print(f'Momentum_batch.py  - self.mul_tensor sum')
        #print(torch.sum(self.mul_tensor, 1))
        #TODO if memory low --> convert minimal value to zero and use sparse matrix (in case very large batch)



        # TODO 이거 수정해야함. 0으로 init 하면 softmax 하면 다르게 나옴. -무한대로 init 해야 softmax하면 0으로 나오는데 이게 맞는 짓인가?
        temp = torch.zeros(len(self.cfg.momentum_params) * 2 + 1, vector_len)# 7, F
        for idx in range(len(self.cfg.momentum_params) * 2 + 1):
            temp[idx:idx + 1, :] = torch.ones(1, vector_len) * (-(((len(self.cfg.momentum_params) - idx) ** 2) / 4))
        self.learnable_matrix = nn.Parameter(temp)

    # TODO 현재 구현은 momentum_params이 hyper parameter로 주어지는 것을 가정하고 있음. learnable parameter로 하려면 다시 코드 짜야 된다
    def forward(self, vector): # B, vector_len | torch.Size([64, 96])
        batch = vector.shape[0]

        # TODO if vector batch is not batchsize (last batch)
        # Ensure that the momentum_matrix is initialized correctly
        N = len(self.cfg.momentum_params)
        if torch.all(self.momentum_matrix == 0):
            with torch.no_grad():
                self.momentum_matrix = vector[0:1,:,None].expand_as(self.momentum_matrix) # 3,F,1
        #print('11', self.momentum_matrix.unsqueeze(0).detach().shape) #([1, 7, 96])
        #print(torch.t(vector).expand(N, self.vector_len, batch).shape) #([3, 96, 64])
        lhs = torch.cat((self.momentum_matrix.detach(), torch.t(vector).expand(N, self.vector_len, batch)), dim=2)
        #concat self.momentum_matrix: 3,F,1 and expended vector 3,F,B
        if batch == self.batch:
            out = torch.matmul(lhs, self.mul_tensor.detach()) # out: 3,F,B
        else:
            out = torch.matmul(lhs, self.mul_tensor.detach()[:, :batch+1, :batch])  # out: 3,F,b

        out = torch.cat((self.momentum_matrix, out), dim=2) # 3,F,B+1
        self.momentum_matrix = out[:,:,-1:].clone().detach() # 3,F,1
        out = out[:,:,:-1] # 3,F,B
        if not self.cfg.bptt:
            out = out.detach()

        matrix = torch.softmax(self.learnable_matrix, dim=0)  # 7, F

        matrix_1 = 2*(matrix[N+1:,:]-torch.flip(matrix[:N,:], (0,))) # 3,F
        matrix_2 = (2*torch.sum(matrix[:N,:], dim=0, keepdim=True)+matrix[N:N+1,:]) # 1, F

        # Combine the updated momentum_matrix with the learnable_matrix to produce the final vector
        vector = torch.t(torch.mul(matrix_1.unsqueeze(2), out).sum(dim=0))+torch.mul(matrix_2, vector)

        return vector

        # def reset_momentum(self): #TODO 이 reset 방법과 아래 방법 중 어떤게 맞는가? 이건 진짜 파라미터 새로 만드는 거고 아래건 있는 파라미터를 0으로 값만 바꾸는 거다

    #     # self.momentum_matrix = torch.zeros_like(self.momentum_matrix)
    #     self.momentum_matrix = nn.Parameter(torch.zeros_like(self.momentum_matrix))

    def reset_momentum(self):  # TODO 이거 수정되었으니 봉균형한테 말하기
        self.momentum_matrix = torch.zeros(len(self.cfg.momentum_params), self.vector_len, 1).to(
            self.learnable_matrix.device)
        self.mul_tensor = self.mul_tensor.to(self.learnable_matrix.device)