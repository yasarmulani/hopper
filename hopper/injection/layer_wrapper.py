"""
Single layer wrapper.
Must match RoBERTa's exact calling signature:
  layer_module(hidden_states, attention_mask, head_mask, 
               encoder_hidden_states, encoder_attention_mask,
               past_key_value, output_attentions)
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
        transformer_layer,
        d_model:            int,
        num_slots:          int,
        num_heads:          int,
        num_relation_types: int,
        tau:                float = 0.5,
    ):
        super().__init__()
        self.transformer_layer  = transformer_layer
        self.entity_induction   = EntityInductionLayer(d_model, num_slots, tau)
        self.relation_induction = RelationInductionLayer(d_model, num_relation_types)
        self.hop_attention      = HopAttention(d_model, num_heads)
        self.composition        = CompositionOperator(d_model, num_slots)
        self.hop_to_token       = HopToTokenBridge(d_model)
        self.w_raw              = nn.Parameter(torch.zeros(num_slots))

    def forward(
        self,
        hidden_states:              torch.Tensor,        # (B, T, d)
        attention_mask:             torch.Tensor = None, # (B, 1, 1, T) — RoBERTa extended mask
        head_mask:                  torch.Tensor = None,
        encoder_hidden_states:      torch.Tensor = None,
        encoder_attention_mask:     torch.Tensor = None,
        past_key_value                           = None,
        output_attentions:          bool         = False,
        **kwargs,
    ):
        B, T, d = hidden_states.shape

        # ── Unpack attention mask for entity induction ──────────
        # RoBERTa passes extended mask of shape (B, 1, 1, T)
        # with 0.0 for real tokens and -10000.0 for padding.
        # Convert to binary (B, T) for our routing.
        if attention_mask is not None and attention_mask.dim() == 4:
            # (B, 1, 1, T) → (B, T): real tokens are 0.0, padding is large negative
            binary_mask = (attention_mask.squeeze(1).squeeze(1) > -1).long()
        elif attention_mask is not None and attention_mask.dim() == 2:
            binary_mask = attention_mask
        else:
            binary_mask = torch.ones(B, T, device=hidden_states.device, dtype=torch.long)

        # ── HOPPER reasoning block ───────────────────────────────
        entities, routing = self.entity_induction(hidden_states, binary_mask)
        relations, _      = self.relation_induction(entities)

        w    = torch.sigmoid(self.w_raw).unsqueeze(0).expand(B, -1)
        hops = HopBatch(
            e_head  = entities,
            r       = relations,
            e_tail  = entities,
            w       = w,
            routing = routing,
        )
        hops           = self.hop_attention(hops)
        hops           = self.composition(hops)
        hidden_states  = self.hop_to_token(hidden_states, hops)

        # ── Original RoBERTa layer ───────────────────────────────
        layer_outputs = self.transformer_layer(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        hidden_states = layer_outputs[0]

        # Return (hidden_states, hop_batch, ...rest of layer outputs)
        return (hidden_states, hops) + layer_outputs[1:]