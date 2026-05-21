import torch
import torch.nn as nn
from transformers import RobertaModel
from hopper.injection.injector import HOPPERInjector
from hopper.task_heads.span_extraction import SpanExtractionHead
from hopper.core.hop_representation import HopBatch
from typing import List, Optional


class RoBERTaHOPPER(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Load RobertaModel directly
        self.roberta = RobertaModel.from_pretrained(
            config["model_name"],
            add_pooling_layer=False,
        )
        d_model = self.roberta.config.hidden_size  # 768

        # Freeze early layers
        self._freeze_layers(config.get("freeze_layers", 8))

        # Inject HOPPER — pass self.roberta which IS the RobertaModel
        self.roberta, self.hopper_layer_indices = HOPPERInjector.inject(
            model               = self.roberta,
            layer_indices       = config["hopper_layers"],
            d_model             = d_model,
            num_slots           = config.get("num_slots", 8),
            num_heads           = config.get("num_heads", 4),
            num_relation_types  = config.get("num_relation_types", 16),
            tau                 = config.get("tau", 0.5),
        )

        self.span_head = SpanExtractionHead(d_model)
        self._last_hop_batches: List[HopBatch] = []

    def _freeze_layers(self, n: int):
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

    def _get_encoder_layers(self):
        """Single place to get encoder layers — no attribute confusion."""
        return self.roberta.encoder.layer

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        self._last_hop_batches = []
        hop_batches = []

        def make_hook():
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    if isinstance(output[1], HopBatch):
                        hop_batches.append(output[1].detach_for_logging())
            return hook

        # Attach hooks to HOPPER-wrapped layers
        hooks = []
        for i, layer in enumerate(self._get_encoder_layers()):
            if i in self.hopper_layer_indices:
                h = layer.register_forward_hook(make_hook())
                hooks.append(h)

        outputs = self.roberta(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        for h in hooks:
            h.remove()

        sequence_output       = outputs.last_hidden_state
        self._last_hop_batches = hop_batches

        start_logits, end_logits = self.span_head(sequence_output)

        return start_logits, end_logits, hop_batches

    def set_temperature(self, tau: float):
        for i, layer in enumerate(self._get_encoder_layers()):
            if i in self.hopper_layer_indices:
                if hasattr(layer, "entity_induction"):
                    layer.entity_induction.set_temperature(tau)

    def get_hop_batches(self) -> List[HopBatch]:
        return self._last_hop_batches