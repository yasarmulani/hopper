"""
ModelFactory — one config change to swap models.
"""

from hopper.models.roberta_hopper import RoBERTaHOPPER


class ModelFactory:

    @staticmethod
    def create(config: dict):
        """
        config["model_family"] controls which model is built.
        Currently: "encoder_only" → RoBERTaHOPPER
        """
        family = config.get("model_family", "encoder_only")

        if family == "encoder_only":
            return RoBERTaHOPPER(config)
        else:
            raise ValueError(
                f"Unknown model_family: {family}. "
                f"Supported: encoder_only"
            )

    @staticmethod
    def create_baseline(config: dict):
        """
        Creates vanilla RoBERTa-base without HOPPER.
        Used for baseline comparison.
        """
        from transformers import RobertaModel
        import torch.nn as nn
        from hopper.task_heads.span_extraction import SpanExtractionHead
        import torch

        class RoBERTaBaseline(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.roberta = RobertaModel.from_pretrained(
                    config["model_name"],
                    add_pooling_layer=False,
                )
                d_model = self.roberta.config.hidden_size
                self.span_head = SpanExtractionHead(d_model)

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                outputs = self.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                start_logits, end_logits = self.span_head(
                    outputs.last_hidden_state
                )
                return start_logits, end_logits, []

        return RoBERTaBaseline(config)