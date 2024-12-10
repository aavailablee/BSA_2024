import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

from layers.Momentum import Momentum
from layers.Momentum_batch import Momentum_batch
from layers.Momentum_batch_learnable import Momentum_batch_learnable


class Model(nn.Module):
    """
    The momentum1 model applies momentum before embedding the normalized input, and the rest of the process remains the same.
    The iTransformer uses x_enc & x_mark in the embedding stage, but in this model, the momentum module passes through x_enc.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        else:
            raise NotImplementedError

    ##############################################################################
    #                             MOMENTUM CODE                                  #
        self.channels = configs.enc_in  #! Check
        self.train_components_to_disable = configs.TRAIN_COMPONENTS_TO_DISABLE
        self.finetune_components_to_disable = configs.FINETUNE_COMPONENTS_TO_DISABLE

        # Momentum module
        self.momentum = nn.ModuleList()
        for i in range(1): #! Check 
            if configs.batch_size ==1:
                self.momentum.append(Momentum(configs, self.channels, self.seq_len)) #! Check specify as a hyperparameter to determine the length of the vector
            elif configs.LEARN_MOMENTUM == False:
                self.momentum.append(Momentum_batch(configs, self.channels, self.seq_len))
            else:
                self.momentum.append(Momentum_batch_learnable(configs, self.channels, self.seq_len))

    def reset_momentum(self):
        for i in range(len(self.momentum)):
            self.momentum[i].reset_momentum() 
    #                              END OF CODE                                   #
    ##############################################################################

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, finetune):  # x_enc.shape = (batch, seq_len, num_variables)
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape # x_enc.shape = (batch, seq_len, num_variables)

        ##############################################################################
        #                             MOMENTUM CODE                                  #
        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                x_enc = self.momentum[0](x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                x_enc = self.momentum[0](x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        #                              END OF CODE                                   #
        ##############################################################################

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, attns = self.encoder(enc_out, finetune=finetune, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, finetune=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, finetune=finetune)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise NotImplementedError