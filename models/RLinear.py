import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN
from layers.ParallelLinear import ParallelLinear


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.individual = True #! 체크
        self.rev_enable = True #! 체크
        self.channels = configs.enc_in
        
        self.Linear = ParallelLinear(self.seq_len, self.pred_len, self.channels)
        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(self.channels) if self.rev_enable else None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, finetune):
        # x_enc.shape = (batch, seq_len, channels)
        x_enc = self.rev(x_enc, 'norm') if self.rev_enable else x_enc # x_enc.shape = (batch, seq_len, channels)
        x_enc = self.dropout(x_enc) # x_enc.shape = (batch, seq_len, channels)
        
        pred = self.Linear(x_enc.permute(0, 2, 1)).permute(0,2,1) # pred.shape = (batch, pred_len, channels)
        
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred#, self.forward_loss(pred, y)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, finetune=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, finetune=finetune)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise NotImplementedError
        