"""
Composition Operator.
Chains two hops that share a boundary entity:
  h_a = (A, r_a, B, w_a)
  h_b = (B, r_b, C, w_b)
  → h_c = (A, r_c, C, w_c)

Proven differentiable in theoretical verification.
w_c = w_a * w_b  (confidence never increases through composition)
r_c = W_comp @ [r_a; r_b]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hopper.core.hop_representation import HopBatch


class CompositionOperator(nn.Module):

    def __init__(self, d_model: int, num_slots: int):
        super().__init__()
        self.d_model   = d_model
        self.num_slots = num_slots

        # Compatibility: detects shared boundary entities
        self.W_compat = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

        # Relation composition: [r_a; r_b] → r_c
        self.W_comp = nn.Linear(2 * d_model, d_model, bias=False)

        # Entity composition: blends head/tail across composed hop
        self.W_head = nn.Linear(2 * d_model, d_model, bias=False)
        self.W_tail = nn.Linear(2 * d_model, d_model, bias=False)

        self.layer_norm_r = nn.LayerNorm(d_model)
        self.layer_norm_e = nn.LayerNorm(d_model)

    def forward(self, hops: HopBatch) -> HopBatch:
        """
        hops : HopBatch with shapes (B, M, d)

        Returns updated HopBatch where composed hops replace
        the lower-confidence of each composable pair.
        """
        B, M, d = hops.e_head.shape

        # Compatibility matrix: how likely is hop_i's tail == hop_j's head?
        # compat[b,i,j] = σ(e_tail_i @ W_compat @ e_head_j)
        tail_proj = hops.e_tail @ self.W_compat   # (B, M, d)
        compat    = torch.bmm(
            tail_proj, hops.e_head.transpose(1, 2)
        )  # (B, M, M)
        compat    = torch.sigmoid(compat)          # (B, M, M) ∈ (0,1)

        # Mask self-composition
        eye = torch.eye(M, device=hops.e_head.device).bool().unsqueeze(0)
        compat = compat.masked_fill(eye, 0.0)

        # For each pair (i,j) with high compatibility, compute composed hop
        # Use soft composition weighted by compatibility

        # Composed relation: W_comp([r_i; r_j]) weighted by compat[i,j]
        # r_i expanded: (B, M, 1, d) × compat: (B, M, M, 1)
        r_i = hops.r.unsqueeze(2).expand(B, M, M, d)   # (B, M, M, d)
        r_j = hops.r.unsqueeze(1).expand(B, M, M, d)   # (B, M, M, d)
        r_composed = self.W_comp(
            torch.cat([r_i, r_j], dim=-1)
        )  # (B, M, M, d)
        r_composed = self.layer_norm_r(r_composed)

        # Weight by compatibility and aggregate for each slot i
        # compat: (B, M, M) — weight of composing i with j
        compat_w = compat.unsqueeze(-1)                     # (B, M, M, 1)
        r_update = (compat_w * r_composed).sum(dim=2)       # (B, M, d)

        # Composed confidence: w_c = w_i * w_j, aggregated by compat
        w_i = hops.w.unsqueeze(2).expand(B, M, M)          # (B, M, M)
        w_j = hops.w.unsqueeze(1).expand(B, M, M)          # (B, M, M)
        w_composed = w_i * w_j                              # (B, M, M)
        w_update   = (compat * w_composed).sum(dim=2)       # (B, M)
        # Normalise by sum of compat weights to keep w in (0,1)
        compat_sum = compat.sum(dim=2).clamp(min=1e-6)
        w_update   = w_update / compat_sum                  # (B, M)

        # Blend original hop with composed update
        # Gate: how much composition to apply
        gate = compat.max(dim=2).values.unsqueeze(-1)       # (B, M, 1)

        new_r = hops.r + gate * r_update
        new_w = hops.w * (1 - gate.squeeze(-1)) + w_update * gate.squeeze(-1)

        # Entity head/tail stay the same — composition refines relations
        # not entity identities at this stage
        return HopBatch(
            e_head  = hops.e_head,
            r       = new_r,
            e_tail  = hops.e_tail,
            w       = new_w.clamp(0, 1),
            routing = hops.routing,
            layer   = hops.layer,
        )