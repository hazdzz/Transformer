import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, attn_type, feat_dim, num_head, value_drop_prob) -> None:
        super(MultiHeadAttention, self).__init__()
        assert attn_type in ['self_attn', 'cross_attn', 'mask_attn']
        assert num_head >= 1
        assert feat_dim % num_head == 0, 'feat_dim should be divisible by num_head'

        self.attn_type = attn_type
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.tau = math.sqrt(self.head_dim)
        self.value_dropout = nn.Dropout(p=value_drop_prob)
        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
    
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_linear.weight)
        init.xavier_uniform_(self.key_linear.weight)
        init.xavier_uniform_(self.value_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)
        init.zeros_(self.query_linear.bias)
        init.zeros_(self.key_linear.bias)
        init.zeros_(self.value_linear.bias)
        init.zeros_(self.output_linear.bias)

    def split_head(self, input: Tensor) -> Tensor:
        batch_size, seq_len, _ = input.size()
        return input.contiguous().view(batch_size, seq_len, self.num_head, self.head_dim).permute(0, 2, 1, 3)
    
    def concat_head(self, input: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = input.size()
        return input.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.feat_dim)

    def forward(self, input: Tensor, option_input: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if self.attn_type == 'self_attn' or self.attn_type == 'mask_attn':
            # self-attention or mask attention
            query = self.query_linear(input)
            key = self.key_linear(input)
            value = self.value_linear(input)
        elif self.attn_type == 'cross_attn':
            # cross attention
            assert option_input is not None
            query = self.query_linear(input)
            key = self.key_linear(option_input)
            value = self.value_linear(option_input)
        
        multihead_query = self.split_head(query)
        multihead_key = self.split_head(key)
        multihead_value = self.split_head(value)

        # Value dropout
        multihead_value = self.value_dropout(multihead_value)
        
        multihead_attn = torch.einsum('bhnd,bhmd->bhnm', multihead_query, multihead_key)

        if mask is not None:
            multihead_attn = multihead_attn.masked_fill(mask == 0, -1e9)

        multihead_attn = multihead_attn / self.tau
        multihead_attn_score = self.softmax(multihead_attn)

        multihead_attn_score = torch.einsum('bhnm,bhmd->bhnd', multihead_attn_score, multihead_value)
        multihead_attn_score_concat = self.concat_head(multihead_attn_score)
        multihead_attn_output = self.output_linear(multihead_attn_score_concat)

        return multihead_attn_output