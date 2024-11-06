import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .embedding import Embedding
from .attention import MultiHeadAttention
from .norm import PostLayerNorm
from .ffn import FeedForward
from torch import Tensor
from typing import Optional


class Encoder(nn.Module):
    def __init__(self, feat_dim, num_head, hid_dim, value_drop_prob, ffn_drop_prob, num_enc_block) -> None:
        super(Encoder, self).__init__()
        attn_type = 'self_attn'
        self.encoder = nn.ModuleList([])
        for _ in range(num_enc_block):
            self.encoder.append(nn.ModuleList([
                PostLayerNorm(feat_dim, MultiHeadAttention(attn_type, feat_dim, num_head, value_drop_prob)),
                PostLayerNorm(feat_dim, FeedForward(feat_dim, hid_dim, ffn_drop_prob))
            ]))

    def forward(self, enc_input: Tensor, mhsa_mask: Optional[Tensor] = None) -> Tensor:
        for mhsa, ffn in self.encoder:
            enc_input = mhsa(enc_input, None, mhsa_mask)
            enc_input = ffn(enc_input)

        return enc_input


class Decoder(nn.Module):
    def __init__(self, feat_dim, num_head, hid_dim, value_drop_prob, ffn_drop_prob, num_dec_block, tgt_vocab_size) -> None:
        super(Decoder, self).__init__()
        attn_type1 = 'mask_attn'
        attn_type2 = 'cross_attn'
        self.decoder = nn.ModuleList([])
        for _ in range(num_dec_block):
            self.decoder.append(nn.ModuleList([
                PostLayerNorm(feat_dim, MultiHeadAttention(attn_type1, feat_dim, num_head, value_drop_prob)),
                PostLayerNorm(feat_dim, MultiHeadAttention(attn_type2, feat_dim, num_head, value_drop_prob)),
                PostLayerNorm(feat_dim, FeedForward(feat_dim, hid_dim, ffn_drop_prob))
            ]))
        self.linear = nn.Linear(feat_dim, tgt_vocab_size, bias=True)
    
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, dec_input: Tensor, enc_output: Tensor, mhma_mask: Optional[Tensor] = None, mhca_mask: Optional[Tensor] = None) -> Tensor:
        for mhma, mhca, ffn in self.decoder:
            dec_input = mhma(dec_input, None, mhma_mask)
            dec_input = mhca(dec_input, enc_output, mhca_mask)
            dec_input = ffn(dec_input)
        
        dec_output = self.linear(dec_input)

        return dec_output
    

class Transformer(nn.Module):
    def __init__(self, args, src_pad_idx, tgt_pad_idx, tgt_sos_idx, src_vocab_size, tgt_vocab_size) -> None:
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.encoder_embedding = Embedding(args.pe_type, src_vocab_size, args.max_seq_len, args.embed_dim, args.embed_drop_prob, None)
        self.decoder_embedding = Embedding(args.pe_type, tgt_vocab_size, args.max_seq_len, args.embed_dim, args.embed_drop_prob, None)
        self.encoder = Encoder(args.embed_dim, args.num_head, args.hid_dim, args.value_drop_prob, args.ffn_drop_prob, args.num_enc_block)
        self.decoder = Decoder(args.embed_dim, args.num_head, args.hid_dim, args.value_drop_prob, args.ffn_drop_prob, args.num_dec_block, tgt_vocab_size)

    def make_src_mask(self, src) -> Tensor:
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_tgt_mask(self, tgt) -> Tensor:
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool)).to(tgt.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(1)

        return tgt_mask
    
    def make_cross_attn_mask(self, src, tgt) -> Tensor:
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        cross_attn_mask = src_mask.expand(-1, -1, tgt.size(1), -1)
    
        return cross_attn_mask
    
    def forward(self, enc_input: Tensor, dec_input: Tensor) -> Tensor:
        src_mask = self.make_src_mask(enc_input)
        tgt_mask = self.make_tgt_mask(dec_input)
        cross_attn_mask = self.make_cross_attn_mask(enc_input, dec_input)
        enc_embed = self.encoder_embedding(enc_input)
        dec_embed = self.decoder_embedding(dec_input)
        enc_output = self.encoder(enc_embed, src_mask)
        output = self.decoder(dec_embed, enc_output, tgt_mask, cross_attn_mask)

        return output