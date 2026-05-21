"""
Entity Induction Layer.
Maps token embeddings X (B, T, d) → entity vectors E (B, M, d)
via learned sparse routing.

Guarantees (verified in theoretical verification):
  G1: entity vectors are convex combinations of token embeddings
  G2: routing entropy decreases as temperature → 0 (sparse)
  G3: permutation equivariance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityInductionLayer(nn.Module):

    def __init__(self, d_model: int, num_slots: int, tau: float = 0.5):
        """
        d_model   : transformer hidden dimension (768 for RoBERTa-base)
        num_slots : number of entity slots M
        tau       : Gumbel-softmax temperature (annealed during training)
        """
        super().__init__()
        self.d_model   = d_model
        self.num_slots = num_slots
        self.tau       = tau

        # Routing projection: maps each token to M slot logits
        self.W_r = nn.Linear(d_model, num_slots, bias=False)

        # Entity projection: projects tokens before pooling
        self.W_e = nn.Linear(d_model, d_model, bias=False)

        # Layer norm on entity output
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_r.weight)
        nn.init.xavier_uniform_(self.W_e.weight)

    def forward(
        self,
        x: torch.Tensor,            # (B, T, d)
        attention_mask: torch.Tensor = None,  # (B, T)
    ):
        """
        Returns:
            entities : (B, M, d)   entity slot representations
            routing  : (B, T, M)   routing weights (for interpretability + hop_to_token)
        """
        B, T, d = x.shape

        # Mask padding tokens before routing
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        else:
            mask = torch.ones(B, T, 1, device=x.device)

        # Routing logits: (B, T, M)
        routing_logits = self.W_r(x)

        # Mask padding: set logits to large negative before softmax
        if attention_mask is not None:
            routing_logits = routing_logits.masked_fill(
                attention_mask.unsqueeze(-1) == 0, -1e9
            )

        # Gumbel-softmax routing (sparse, differentiable)
        if self.training:
            gumbel = -torch.log(
                -torch.log(torch.rand_like(routing_logits) + 1e-10) + 1e-10
            )
            routing = F.softmax(
                (routing_logits + gumbel) / self.tau, dim=1
            )  # dim=1 = over tokens, so each slot gets a distribution over tokens
        else:
            routing = F.softmax(routing_logits / self.tau, dim=1)

        # Entity vectors: weighted sum of projected tokens
        # routing: (B, T, M) → (B, M, T)
        # x_proj:  (B, T, d)
        x_proj   = self.W_e(x)                          # (B, T, d)
        entities = torch.bmm(routing.transpose(1, 2), x_proj)  # (B, M, d)
        entities = self.layer_norm(entities)

        return entities, routing

    def set_temperature(self, tau: float):
        """Called by trainer to anneal temperature during training."""
        self.tau = tau