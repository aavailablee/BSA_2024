import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.ParallelLinear import ParallelLinear
    
    
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=True):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        self.Linear_Seasonal = ParallelLinear(self.seq_len, self.pred_len, self.channels)
        self.Linear_Trend = ParallelLinear(self.seq_len, self.pred_len, self.channels)
        
        # Re-initialize the weights
        self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len) * torch.ones([self.channels, self.seq_len, self.pred_len]))
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len) * torch.ones([self.channels, self.seq_len, self.pred_len]))
        
    def encoder(self, x): # x.shape = (batch, seq_len, channels)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1) # (batch, channels, seq_len)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, finetune=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else: raise NotImplementedError
