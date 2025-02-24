

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="gelu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, tgt, memory,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None,
                return_attn=False):
        
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        out = tgt + self.dropout2(tgt2)
        out = self.norm2(out)
        
        if return_attn:
            return out, attn
        else:
            return out

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, num_layers=1, dropout=0.1):
        super().__init__()
        crossattn = CrossAttentionLayer(d_model, nhead, dropout)
        self.layers = _get_clones(crossattn, num_layers)

    def forward(self, tgt, memory,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None,
                return_attn=False):
        output = tgt

        for layer in self.layers:
            if return_attn:
                output, attn = layer(output, memory, memory_key_padding_mask, pos, query_pos, return_attn)
            else:
                output = layer(output, memory, memory_key_padding_mask, pos, query_pos, return_attn)
        
        if return_attn:
            return output, attn 
        else: 
            return output



class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="gelu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, src, memory_key_padding_mask=None,
                pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.multihead_attn(query=q, key=k,
                                   value=src, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        out = src + self.dropout2(src2)
        out = self.norm2(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, num_layers=1, dropout=0.1):
        super().__init__()
        selfattn = SelfAttentionLayer(d_model, nhead, dropout)
        self.layers = _get_clones(selfattn, num_layers)

    def forward(self, src, memory_key_padding_mask=None,
                pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, memory_key_padding_mask, pos)
        
        return output


class AttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048, activation="relu"):
        super().__init__()
        self.self = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, memory_key_padding_mask=None):

        tgt2 = self.self(tgt, tgt, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross(tgt, memory, value=memory, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Attention(nn.Module):
    def __init__(self, d_model, nhead, num_layers=1, dropout=0.1):
        super().__init__()
        att = AttentionLayer(d_model, nhead,dropout)
        self.layers = _get_clones(att, num_layers)

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        
        output = tgt
        for layer in self.layers:
            tgt = layer(output, memory, memory_key_padding_mask)
        
        return output