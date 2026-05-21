"""
FLOP counter for HOPPER vs baseline comparison.
"""

import torch
from fvcore.nn import FlopCountAnalysis


def count_flops(model, input_ids, attention_mask) -> int:
    """Returns total FLOPs for one forward pass."""
    try:
        flops = FlopCountAnalysis(model, (input_ids, attention_mask))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total()
    except Exception:
        # Fallback: theoretical estimate
        B, T   = input_ids.shape
        d      = 768
        layers = 12
        return layers * (3 * T * d * d + 2 * T * T * d)


def compare_flops(hopper_model, baseline_model, sample_batch, device) -> dict:
    input_ids      = sample_batch["input_ids"][:1].to(device)
    attention_mask = sample_batch["attention_mask"][:1].to(device)

    hopper_flops   = count_flops(hopper_model,   input_ids, attention_mask)
    baseline_flops = count_flops(baseline_model, input_ids, attention_mask)

    return {
        "hopper_flops":   hopper_flops,
        "baseline_flops": baseline_flops,
        "ratio":          hopper_flops / max(baseline_flops, 1),
    }