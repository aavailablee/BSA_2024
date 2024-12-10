import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, finetune = False): 
        # 현재 q k v 다 time_in 들어옴 (input embedding이 들어온거다)
        B, L, _ = queries.shape # (batch, number of segments, d_model) 
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1) #(batch, number of segments, n_heads, d_keys) d_keys = d_model // n_heads
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn 
    
class AttentionLayer_momentum_Q(nn.Module):
    def __init__(self, attention, d_model, n_heads, configs, momentum, d_keys=None, d_values=None):
        super(AttentionLayer_momentum_Q, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        self.momentum = momentum
        self.train_components_to_disable = configs.TRAIN_COMPONENTS_TO_DISABLE
        self.finetune_components_to_disable = configs.FINETUNE_COMPONENTS_TO_DISABLE
    #                              END OF CODE                                   #
    ##############################################################################
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, finetune=False): 
        # 현재 q k v 다 time_in 들어옴 (input embedding이 들어온거다)
        B, L, _ = queries.shape # (batch, number of segments, d_model) 
        _, S, _ = keys.shape
        H = self.n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        queries = self.query_projection(queries) # (batch, number of segments, d_model) 
    
        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                queries = self.momentum(queries)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                queries = self.momentum(queries)
                
        queries = queries.view(B, L, H, -1) #(batch, number of segments, n_heads, d_keys) d_keys = d_model // n_heads
    #                              END OF CODE                                   #
    ##############################################################################
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
    
class AttentionLayer_momentum_K(nn.Module):
    def __init__(self, attention, d_model, n_heads, configs, momentum, d_keys=None, d_values=None):
        super(AttentionLayer_momentum_K, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        self.momentum = momentum
        self.train_components_to_disable = configs.TRAIN_COMPONENTS_TO_DISABLE
        self.finetune_components_to_disable = configs.FINETUNE_COMPONENTS_TO_DISABLE
    #                              END OF CODE                                   #
    ##############################################################################
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, finetune=False): 
        # 현재 q k v 다 time_in 들어옴 (input embedding이 들어온거다)
        B, L, _ = queries.shape # (batch, number of segments, d_model) 
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1) #(batch, number of segments, n_heads, d_keys) d_keys = d_model // n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        keys = self.key_projection(keys)
        
        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                keys = self.momentum(keys)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                keys = self.momentum(keys)
                
        keys = keys.view(B, S, H, -1)
    #                              END OF CODE                                   #
    ##############################################################################
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class AttentionLayer_momentum_V(nn.Module):
    def __init__(self, attention, d_model, n_heads, configs, momentum, d_keys=None, d_values=None):
        super(AttentionLayer_momentum_V, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        self.momentum = momentum
        self.train_components_to_disable = configs.TRAIN_COMPONENTS_TO_DISABLE
        self.finetune_components_to_disable = configs.FINETUNE_COMPONENTS_TO_DISABLE
    #                              END OF CODE                                   #
    ##############################################################################
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, finetune=False): 
        # 현재 q k v 다 time_in 들어옴 (input embedding이 들어온거다)
        B, L, _ = queries.shape # (batch, number of segments, d_model) 
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1) #(batch, number of segments, n_heads, d_keys) d_keys = d_model // n_heads
        keys = self.key_projection(keys).view(B, S, H, -1)
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        values = self.value_projection(values)
        
        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                values = self.momentum(values)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                values = self.momentum(values)
                
        values = values.view(B, S, H, -1)
    #                              END OF CODE                                   #
    ##############################################################################
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class AttentionLayer_momentum_All(nn.Module):
    def __init__(self, attention, d_model, n_heads, configs, momentum, d_keys=None, d_values=None):
        super(AttentionLayer_momentum_All, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        self.momentum = momentum
        self.train_components_to_disable = configs.TRAIN_COMPONENTS_TO_DISABLE
        self.finetune_components_to_disable = configs.FINETUNE_COMPONENTS_TO_DISABLE
    #                              END OF CODE                                   #
    ##############################################################################
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, finetune=False): 
        # 현재 q k v 다 time_in 들어옴 (input embedding이 들어온거다)
        B, L, _ = queries.shape # (batch, number of segments, d_model) 
        _, S, _ = keys.shape
        H = self.n_heads

    ##############################################################################
    #                             MOMENTUM CODE                                  #
        queries = self.query_projection(queries) # (batch, number of segments, d_model) 
    
        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                queries = self.momentum[0](queries)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                queries = self.momentum[0](queries)
                
        queries = queries.view(B, L, H, -1) #(batch, number of segments, n_heads, d_keys) d_keys = d_model // n_heads
    
        keys = self.key_projection(keys)
        
        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                keys = self.momentum[1](keys)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                keys = self.momentum[1](keys)
                
        keys = keys.view(B, S, H, -1)

        values = self.value_projection(values)
        
        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                values = self.momentum[2](values)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                values = self.momentum[2](values)
                
        values = values.view(B, S, H, -1)
    #                              END OF CODE                                   #
    ##############################################################################
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class AttentionLayer_momentum_Q_reshape(nn.Module):
    def __init__(self, attention, d_model, n_heads, configs, momentum, d_keys=None, d_values=None):
        super(AttentionLayer_momentum_Q_reshape, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        self.momentum = momentum
        self.train_components_to_disable = configs.TRAIN_COMPONENTS_TO_DISABLE
        self.finetune_components_to_disable = configs.FINETUNE_COMPONENTS_TO_DISABLE
        # self.batch = configs.batch_size
        self.n_vars = configs.enc_in
    #                              END OF CODE                                   #
    ##############################################################################
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, finetune=False): 
        # 현재 q k v 다 time_in 들어옴 (input embedding이 들어온거다)
        B, L, D = queries.shape # (batch * n_vars, number of segments, d_model) 
        _, S, _ = keys.shape
        H = self.n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        queries = self.query_projection(queries) # (batch * n_vars, seg_num, d_model) 
        
        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                # queries.shape = (batch * n_vars, seg_num, d_model) <-> (batch, seg_num, d_model * n_vars) 
                queries = self.momentum(queries.view(B//self.n_vars, L, -1)).reshape(B, L, D)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                # queries.shape = (batch * n_vars, seg_num, d_model) <-> (batch, seg_num, d_model * n_vars) 
                queries = self.momentum(queries.view(B//self.n_vars, L, -1)).reshape(B, L, D)
                
        queries = queries.view(B, L, H, -1) #(batch, number of segments, n_heads, d_keys) d_keys = d_model // n_heads
    #                              END OF CODE                                   #
    ##############################################################################
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
    
class AttentionLayer_momentum_K_reshape(nn.Module):
    def __init__(self, attention, d_model, n_heads, configs, momentum, d_keys=None, d_values=None):
        super(AttentionLayer_momentum_K_reshape, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        self.momentum = momentum
        self.train_components_to_disable = configs.TRAIN_COMPONENTS_TO_DISABLE
        self.finetune_components_to_disable = configs.FINETUNE_COMPONENTS_TO_DISABLE
        # self.batch = configs.batch_size
        self.n_vars = configs.enc_in
    #                              END OF CODE                                   #
    ##############################################################################
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, finetune=False): 
        # 현재 q k v 다 time_in 들어옴 (input embedding이 들어온거다)
        B, L, _ = queries.shape # (batch * n_vars, number of segments, d_model) 
        _, S, D = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1) #(batch * n_vars, number of segments, n_heads, d_keys) d_keys = d_model // n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        keys = self.key_projection(keys)

        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                # keys.shape = (batch, seg_num, d_model * n_vars) <-> (batch * n_vars, seg_num, d_model)
                keys = self.momentum(keys.reshape(B//self.n_vars, S, -1)).reshape(B, S, D)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                # keys.shape = (batch, seg_num, d_model * n_vars) <-> (batch * n_vars, seg_num, d_model)
                keys = self.momentum(keys.reshape(B//self.n_vars, S, -1)).reshape(B, S, D)
                
        keys = keys.view(B, S, H, -1)
    #                              END OF CODE                                   #
    ##############################################################################
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
    
class AttentionLayer_momentum_V_reshape(nn.Module):
    def __init__(self, attention, d_model, n_heads, configs, momentum, d_keys=None, d_values=None):
        super(AttentionLayer_momentum_V_reshape, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        self.momentum = momentum
        self.train_components_to_disable = configs.TRAIN_COMPONENTS_TO_DISABLE
        self.finetune_components_to_disable = configs.FINETUNE_COMPONENTS_TO_DISABLE
        # self.batch = configs.batch_size
        self.n_vars = configs.enc_in
    #                              END OF CODE                                   #
    ##############################################################################
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, finetune=False): 
        # 현재 q k v 다 time_in 들어옴 (input embedding이 들어온거다)
        B, L, _ = queries.shape # (batch * n_vars, number of segments, d_model) 
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1) #(batch, number of segments, n_heads, d_keys) d_keys = d_model // n_heads
        keys = self.key_projection(keys).view(B, S, H, -1)
    ##############################################################################
    #                             MOMENTUM CODE                                  #
        values = self.value_projection(values)
        _, seg_num, d_model = values.shape
        
        if not finetune:
            if 'momentum' not in self.train_components_to_disable:
                # values.shape = (batch, seg_num, d_model * n_vars) <-> (batch * n_vars, seg_num, d_model)
                values = self.momentum(values.reshape(B//self.n_vars, seg_num, -1)).reshape(B, seg_num, d_model)
        else:
            if 'momentum' not in self.finetune_components_to_disable:
                # values.shape = (batch, seg_num, d_model * n_vars) <-> (batch * n_vars, seg_num, d_model)
                values = self.momentum(values.reshape(B//self.n_vars, seg_num, -1)).reshape(B, seg_num, d_model)
                
        values = values.view(B, S, H, -1)
    #                              END OF CODE                                   #
    ##############################################################################
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
    
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 여기엔 linear mapping 된 q k v 값이 들어온다
        B, L, H, E = queries.shape # (batch * n_var, seg_num, n_heads, d_keys)
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # (batch * n_var, n_heads, seg_num, seg_num) 확인요망

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
        
        
class FullAttention_custom(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention_custom, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, queries_distance=None): 
        # 여기엔 linear mapping 된 q k v 값이 들어온다
        B, L, H, E = queries.shape  #(batch * n_var, seg_num, n_heads, d_keys)
        _, S, _, D = values.shape
        
        #! 이 논리 맞는가
        queries_distance = (torch.tanh(queries_distance) + 1) / 2 # queries_distance.shape = (batch * n_var, seg_num, n_heads, seg_num)
        keys_distance = torch.eye(L, device=keys.device).unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1).permute(0, 2, 1, 3) # (batch * n_var, seg_num, n_heads, seg_num)
        # keys_distance[0][5][0] = tensor([0., 0., 0., 0., 0., 1., 0., 0.], device='cuda:0')
        
        scale = self.scale or 1. / sqrt(E)

        #! 이 논리 맞는가
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # scores.shape = (batch * n_var, n_heads, seg_num, seg_num)
        distance_scores = torch.einsum("blhe,bshe->bhls", queries_distance, keys_distance) # distance_scores.shape = (batch * n_var, n_heads, seg_num, seg_num)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        # A = self.dropout(torch.softmax(scale * scores, dim=-1) * distance_scores) # A.shape = (batch * n_var, n_heads, seg_num, seg_num)
        scaled_scores = scale * scores
        e_x = torch.exp(scaled_scores - torch.max(scaled_scores)) * distance_scores 
        A = self.dropout(e_x / e_x.sum(dim=-1, keepdim = True))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer_custom(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, seg_num=None):
        super(AttentionLayer_custom, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        seg_num = seg_num or ValueError("segment number must be given")

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads) # (input, output)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        
        self.query_distance_projection = nn.Linear(d_model, seg_num * n_heads) # layer 별로 head 별로 다르다
        
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 현재 q k v 다 time_in 들어옴 (input embedding이 들어온거다)
        B, L, _ = queries.shape # (batch * n_var, seq_len/patch_len = number of segments, d_model) 
        _, S, _ = keys.shape #! 왜 굳이 이렇게 한건지 모르겠네
        H = self.n_heads

        # 여기서 q k v linear mapping
        queries_distance = self.query_distance_projection(queries).view(B, L, H, -1) # (batch * n_var, seq_len/patch_len = number of segments, n_heads, seg_num)
        queries = self.query_projection(queries).view(B, L, H, -1) #(batch * n_var, seq_len/patch_len = number of segments, n_heads, d_keys)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
            queries_distance=queries_distance
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model # if d_ff is None, set d_ff = 4 * d_model
        
        # AttentionLayer : attention, d_model, n_heads, d_keys=None, d_values=None
        # FullAttention : mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out

class TwoStageAttentionLayer_custom(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer_custom, self).__init__()
        d_ff = d_ff or 4 * d_model # if d_ff is None, set d_ff = 4 * d_model
        
        # AttentionLayer : attention, d_model, n_heads, d_keys=None, d_values=None
        # FullAttention : mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False
        self.time_attention = AttentionLayer_custom(FullAttention_custom(False, configs.factor, attention_dropout=configs.dropout, #! 체크 mask!
                                                           output_attention=configs.output_attention), d_model, n_heads, seg_num=seg_num) #! d_distance = segement 개수 넣어야 함
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0] # (batch, n_var, seq_len/patch_len = number of segments, d_model)
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model') # (batch * n_var, seq_len/patch_len = number of segments, d_model)
        time_enc, attn = self.time_attention( # queries, keys, values, attn_mask, tau=None, delta=None
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out