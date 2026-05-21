"""
RoBERTa-base + HOPPER for span extraction.
Primary Tier 1 model.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from hopper.injection.injector import HOPPERInjector
from hopper.task_heads.span_extraction import SpanExtractionHead
from hopper.core.hop_representation import HopBatch
from typing import List, Optional


class RoBERTaHOPPER(nn.Module):

    def __init__(self, config: dict):
        """
        config keys:
          model_name          : "roberta-base"
          hopper_layers       : [8, 9, 10, 11]
          num_slots           : 8
          num_heads           : 4
          num_relation_types  : 16
          tau                 : 0.5
          freeze_layers       : 8  (freeze first N layers)
        """
        super().__init__()
        self.config = config

        # Load RoBERTa-base
        self.roberta = RobertaModel.from_pretrained(
            config["model_name"],
            add_pooling_layer=False,
        )
        d_model = self.roberta.config.hidden_size  # 768

        # Freeze early layers
        freeze_up_to = config.get("freeze_layers", 8)
        self._freeze_layers(freeze_up_to)

        # Inject HOPPER into specified layers
        self.roberta, self.hopper_layer_indices = HOPPERInjector.inject(
            model               = self.roberta,
            layer_indices       = config["hopper_layers"],
            d_model             = d_model,
            num_slots           = config.get("num_slots", 8),
            num_heads           = config.get("num_heads", 4),
            num_relation_types  = config.get("num_relation_types", 16),
            tau                 = config.get("tau", 0.5),
        )

        # Span extraction head
        self.span_head = SpanExtractionHead(d_model)

        # Store hop batches from forward pass (for loss + interpretability)
        self._last_hop_batches: List[HopBatch] = []

    def _freeze_layers(self, n: int):
        """Freeze embeddings and first n transformer layers."""
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Returns:
            start_logits : (B, T)
            end_logits   : (B, T)
            hop_batches  : list of HopBatch from each HOPPER layer
        """
        self._last_hop_batches = []

        # We need to intercept hop outputs from HOPPER layers.
        # Use a forward hook approach.
        hop_batches = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output[1] is the HopBatch from HOPPERLayerWrapper
                if isinstance(output, tuple) and len(output) > 1:
                    if isinstance(output[1], HopBatch):
                        hop_batches.append(output[1].detach_for_logging())
            return hook

        hooks = []
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i in self.hopper_layer_indices:
                h = layer.register_forward_hook(make_hook(i))
                hooks.append(h)

        # Forward pass
        outputs = self.roberta(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        # Remove hooks
        for h in hooks:
            h.remove()

        sequence_output = outputs.last_hidden_state   # (B, T, d)
        self._last_hop_batches = hop_batches

        start_logits, end_logits = self.span_head(sequence_output)

        return start_logits, end_logits, hop_batches

    def set_temperature(self, tau: float):
        """Anneal Gumbel temperature — called by trainer."""
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i in self.hopper_layer_indices:
                layer.entity_induction.set_temperature(tau)

    def get_hop_batches(self) -> List[HopBatch]:
        return self._last_hop_batches