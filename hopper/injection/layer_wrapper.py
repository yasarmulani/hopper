"""
Single layer wrapper.
Wraps one transformer layer with the full HOPPER block:
  token embeddings
  → entity induction
  → relation induction
  → hop attention
  → composition
  → hop to token
  → original transformer layer
"""

import torch
import torch.nn as nn
from hopper.core.hop_representation import HopBatch
from hopper.core.entity_induction import EntityInductionLayer
from hopper.core.relation_induction import RelationInductionLayer
from hopper.core.composition import CompositionOperator
from hopper.core.hop_attention import HopAttention
from hopper.core.hop_to_token import HopToTokenBridge


class HOPPERLayerWrapper(nn.Module):

    def __init__(
        self,
        transformer_layer,       # original RoBERTa layer
        d_model:    int,
        num_slots:  int,
        num_heads:  int,
        num_relation_types: int,
        tau:        float = 0.5,
    ):
        super().__init__()
        self.transformer_layer = transformer_layer

        self.entity_induction   = EntityInductionLayer(d_model, num_slots, tau)
        self.relation_induction = RelationInductionLayer(d_model, num_relation_types)
        self.hop_attention      = HopAttention(d_model, num_heads)
        self.composition        = CompositionOperator(d_model, num_slots)
        self.hop_to_token       = HopToTokenBridge(d_model)

        # Confidence initialiser: raw weights before sigmoid
        self.w_raw = nn.Parameter(torch.zeros(num_slots))

    def forward(
        self,
        hidden_states:   torch.Tensor,
        attention_mask:  torch.Tensor = None,
        **kwargs,
    ):
        """
        hidden_states : (B, T, d)
        Returns (hidden_states, hop_batch) — same interface as transformer layer
        plus hop_batch for loss computation and interpretability.
        """
        B, T, d = hidden_states.shape

        # ── HOPPER reasoning block ────────────────────────────
        # 1. Entity induction
        entities, routing = self.entity_induction(
            hidden_states, attention_mask
        )  # (B, M, d), (B, T, M)

        # 2. Relation induction
        relations, type_dist = self.relation_induction(entities)

        # 3. Initial confidence weights
        w = torch.sigmoid(self.w_raw).unsqueeze(0).expand(B, -1)  # (B, M)

        # 4. Assemble hop batch
        hops = HopBatch(
            e_head  = entities,
            r       = relations,
            e_tail  = entities,   # initially head == tail; composition refines
            w       = w,
            routing = routing,
        )

        # 5. Hop attention (hops interact)
        hops = self.hop_attention(hops)

        # 6. Composition (chain hops)
        hops = self.composition(hops)

        # 7. Hop to token (inject reasoning back)
        hidden_states = self.hop_to_token(hidden_states, hops)

        # ── Original transformer layer ────────────────────────
        layer_outputs = self.transformer_layer(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = layer_outputs[0]

        return (hidden_states, hops) + layer_outputs[1:]