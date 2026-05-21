"""
Interpretability Probe.
Attaches hooks to HOPPER layers to extract routing weights
and hop triplets during inference.
"""

import torch
from hopper.core.hop_representation import HopBatch
from typing import List, Dict


class InterpretabilityProbe:

    def __init__(self):
        self._hooks         = []
        self._routing_store: List[torch.Tensor] = []
        self._hop_store:     List[HopBatch]     = []

    def attach(self, model):
        """Attach forward hooks to all HOPPER layers."""
        self._routing_store = []
        self._hop_store     = []

        def make_hook():
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    if isinstance(output[1], HopBatch):
                        hops = output[1].detach_for_logging()
                        self._routing_store.append(hops.routing)
                        self._hop_store.append(hops)
            return hook

        for name, module in model.named_modules():
            if "HOPPERLayerWrapper" in type(module).__name__:
                h = module.register_forward_hook(make_hook())
                self._hooks.append(h)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_routing(self) -> List[torch.Tensor]:
        return self._routing_store

    def get_hops(self) -> List[HopBatch]:
        return self._hop_store

    def get_routing_entropy(self) -> List[float]:
        """Per-layer routing entropy — lower = more specialised."""
        entropies = []
        for R in self._routing_store:
            H = -(R * (R + 1e-10).log()).sum(dim=1).mean().item()
            entropies.append(H)
        return entropies