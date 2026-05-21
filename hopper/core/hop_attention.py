"""
Hop Attention.
Computes attention over M hop slots instead of T tokens.
This is the pre-attention reasoning — hops interact before
token-level attention runs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hopper.core.hop_representation import HopBatch


class HopAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        assert d_model % num_heads == 0

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(0.1)

    def forward(self, hops: HopBatch) -> HopBatch:
        """
        Runs multi-head attention over hop slots.
        Input/output: HopBatch (B, M, d)
        """
        B, M, d = hops.e_head.shape

        # Concatenate e_head, r, e_tail as the hop representation
        # hop_repr: (B, M, d) — combine the three components
        hop_repr = hops.e_head + hops.r + hops.e_tail   # additive fusion

        # Weight by confidence
        hop_repr = hop_repr * hops.w.unsqueeze(-1)       # (B, M, d)

        # Multi-head attention
        Q = self.W_q(hop_repr)   # (B, M, d)
        K = self.W_k(hop_repr)
        V = self.W_v(hop_repr)

        # Reshape for multi-head: (B, heads, M, head_dim)
        Q = Q.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        scores  = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn    = F.softmax(scores, dim=-1)
        attn    = self.dropout(attn)
        out     = torch.matmul(attn, V)                  # (B, heads, M, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, M, d)
        out = self.W_o(out)

        # Residual + norm on each component
        new_e_head = self.layer_norm(hops.e_head + out)
        new_e_tail = self.layer_norm(hops.e_tail + out)
        new_r      = self.layer_norm(hops.r      + out)

        return HopBatch(
            e_head  = new_e_head,
            r       = new_r,
            e_tail  = new_e_tail,
            w       = hops.w,
            routing = hops.routing,
            layer   = hops.layer,
        )