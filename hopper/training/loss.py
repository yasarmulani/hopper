"""
HOPPER Loss.
total_loss = span_extraction_loss + lambda * transitivity_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hopper.core.hop_representation import HopBatch
from typing import List


class HOPPERLoss(nn.Module):

    def __init__(self, lambda_trans: float = 0.1):
        """
        lambda_trans : weight for transitivity regulariser
        """
        super().__init__()
        self.lambda_trans = lambda_trans

    def span_loss(
        self,
        start_logits:  torch.Tensor,   # (B, T)
        end_logits:    torch.Tensor,   # (B, T)
        start_targets: torch.Tensor,   # (B,)
        end_targets:   torch.Tensor,   # (B,)
        ignored_index: int = -1,
    ) -> torch.Tensor:
        start_logits = start_logits.clamp(min=-1e4, max=1e4)
        end_logits   = end_logits.clamp(min=-1e4, max=1e4)
        loss_start   = F.cross_entropy(
            start_logits, start_targets, ignore_index=ignored_index
        )
        loss_end     = F.cross_entropy(
            end_logits, end_targets, ignore_index=ignored_index
        )
        return (loss_start + loss_end) / 2

    def transitivity_loss(self, hop_batches: List[HopBatch]) -> torch.Tensor:
        """
        For each HOPPER layer's hop batch, penalise violated transitivity.
        L_trans = sum over composable pairs of max(0, w_a*w_b - w_c)^2
        """
        if not hop_batches:
            return torch.tensor(0.0)

        total = torch.tensor(0.0, device=hop_batches[0].w.device)

        for hops in hop_batches:
            w = hops.w   # (B, M)
            B, M = w.shape

            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    # Implied composed confidence
                    implied = w[:, i] * w[:, j]   # (B,)
                    # Actual confidence of implied third hop
                    k       = (i + j) % M
                    w_c     = w[:, k]
                    # Hinge loss: penalise if implied > actual
                    violation = F.relu(implied - w_c)
                    total    += (violation ** 2).mean()

        return total / max(len(hop_batches), 1)

    def forward(
        self,
        start_logits:  torch.Tensor,
        end_logits:    torch.Tensor,
        start_targets: torch.Tensor,
        end_targets:   torch.Tensor,
        hop_batches:   List[HopBatch],
    ) -> dict:
        """
        Returns dict with total loss and components for logging.
        """
        l_span  = self.span_loss(
            start_logits, end_logits, start_targets, end_targets
        )
        l_trans = self.transitivity_loss(hop_batches)
        total   = l_span + self.lambda_trans * l_trans

        return {
            "loss":       total,
            "span_loss":  l_span,
            "trans_loss": l_trans,
        }