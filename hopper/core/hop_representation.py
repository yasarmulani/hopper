"""
Hop object definition.
h = (e_head, r, e_tail, w)
This is the core data contract. Every module speaks this language.
"""

from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class HopObject:
    """
    Single hop: a typed directed edge in the latent reasoning graph.

    e_head : (d,)     source entity vector
    r      : (d,)     relation vector
    e_tail : (d,)     target entity vector
    w      : scalar   confidence weight in (0,1)
    """
    e_head: torch.Tensor   # (d,)
    r:      torch.Tensor   # (d,)
    e_tail: torch.Tensor   # (d,)
    w:      torch.Tensor   # scalar


@dataclass
class HopBatch:
    """
    Batched hop representations — what flows between modules.

    e_head : (B, M, d)   head entity vectors for all hops
    r      : (B, M, d)   relation vectors for all hops
    e_tail : (B, M, d)   tail entity vectors for all hops
    w      : (B, M)      confidence weights for all hops
    routing: (B, T, M)   routing weights — token to slot assignments
    """
    e_head:  torch.Tensor          # (B, M, d)
    r:       torch.Tensor          # (B, M, d)
    e_tail:  torch.Tensor          # (B, M, d)
    w:       torch.Tensor          # (B, M)
    routing: torch.Tensor          # (B, T, M)
    layer:   Optional[int] = None  # which layer produced this

    def detach_for_logging(self):
        """Returns a detached copy for logging/interpretability — no grad."""
        return HopBatch(
            e_head  = self.e_head.detach(),
            r       = self.r.detach(),
            e_tail  = self.e_tail.detach(),
            w       = self.w.detach(),
            routing = self.routing.detach(),
            layer   = self.layer,
        )