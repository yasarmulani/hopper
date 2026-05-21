import torch.nn as nn
from hopper.injection.layer_wrapper import HOPPERLayerWrapper


class HOPPERInjector:

    @staticmethod
    def inject(
        model,
        layer_indices:      list,
        d_model:            int,
        num_slots:          int   = 8,
        num_heads:          int   = 4,
        num_relation_types: int   = 16,
        tau:                float = 0.5,
    ):
        # RobertaModel loaded directly → encoder is at model.encoder.layer
        # RobertaForQuestionAnswering → model.roberta.encoder.layer
        # Try all three patterns
        layers = None

        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            # RobertaModel loaded directly (our case)
            layers = model.encoder.layer

        elif hasattr(model, "roberta"):
            layers = model.roberta.encoder.layer

        elif hasattr(model, "bert"):
            layers = model.bert.encoder.layer

        if layers is None:
            raise ValueError(
                f"Cannot find transformer layers in {type(model).__name__}. "
                f"Expected model.encoder.layer, model.roberta.encoder.layer, "
                f"or model.bert.encoder.layer."
            )

        hopper_layers = []
        for i, layer in enumerate(layers):
            if i in layer_indices:
                wrapped = HOPPERLayerWrapper(
                    transformer_layer   = layer,
                    d_model             = d_model,
                    num_slots           = num_slots,
                    num_heads           = num_heads,
                    num_relation_types  = num_relation_types,
                    tau                 = tau,
                )
                layers[i] = wrapped
                hopper_layers.append(i)

        print(f"HOPPER injected into layers: {hopper_layers}")
        return model, hopper_layers