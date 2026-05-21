"""
HOPPER Injector.
Wraps specified layers of any transformer with HOPPERLayerWrapper.
One-line config change to switch which layers get HOPPER.
"""

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
        """
        Wraps transformer layers at layer_indices with HOPPER blocks.

        model         : HuggingFace transformer model
        layer_indices : e.g. [8, 9, 10, 11] for last 4 layers of RoBERTa-base
        """
        # Get the list of transformer layers
        # Works for RoBERTa, BERT, and most HuggingFace encoder models
        try:
            layers = model.roberta.encoder.layer
        except AttributeError:
            try:
                layers = model.bert.encoder.layer
            except AttributeError:
                raise ValueError(
                    "Cannot find transformer layers. "
                    "Supported: RoBERTa, BERT. "
                    "For T5 use T5HOPPERInjector."
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