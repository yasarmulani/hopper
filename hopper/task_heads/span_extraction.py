"""
Span Extraction Head.
Two linear projections → start and end logits over token positions.
Standard for HotpotQA, MuSiQue.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SpanExtractionHead(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.start_proj = nn.Linear(d_model, 1, bias=True)
        self.end_proj   = nn.Linear(d_model, 1, bias=True)

        nn.init.zeros_(self.start_proj.bias)
        nn.init.zeros_(self.end_proj.bias)

    def forward(
        self,
        sequence_output: torch.Tensor,   # (B, T, d)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            start_logits : (B, T)
            end_logits   : (B, T)
        """
        start_logits = self.start_proj(sequence_output).squeeze(-1)
        end_logits   = self.end_proj(sequence_output).squeeze(-1)
        return start_logits, end_logits