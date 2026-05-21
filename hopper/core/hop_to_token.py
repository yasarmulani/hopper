"""
Hop to Token Bridge.
Propagates hop reasoning results back into token space.
X' = X + routing @ H'
"""

import torch
import torch.nn as nn
from hopper.core.hop_representation import HopBatch


class HopToTokenBridge(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.gate       = nn.Linear(d_model, 1, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.W_proj     = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,     # (B, T, d) — original token embeddings
        hops: HopBatch,      # updated hop batch
    ) -> torch.Tensor:
        """
        Returns updated token embeddings (B, T, d).
        """
        # Hop summary: combine components weighted by confidence
        hop_repr = (
            hops.e_head + hops.r + hops.e_tail
        ) * hops.w.unsqueeze(-1)   # (B, M, d)

        hop_repr = self.W_proj(hop_repr)   # (B, M, d)

        # routing: (B, T, M) — tells us how much each token
        # contributed to each slot → use transpose to distribute back
        # x_update: (B, T, d)
        x_update = torch.bmm(hops.routing, hop_repr)   # (B, T, d)

        # Learned gate: how much hop info to inject per token
        gate   = torch.sigmoid(self.gate(x))   # (B, T, 1)
        x_out  = x + gate * x_update
        x_out  = self.layer_norm(x_out)

        return x_out