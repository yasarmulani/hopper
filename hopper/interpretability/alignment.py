"""
Hop-Gold Alignment Scorer.
Measures whether HOPPER's hop structures correspond
to gold supporting facts in HotpotQA.
This is the key interpretability metric for the paper.
"""

import torch
from hopper.core.hop_representation import HopBatch
from hopper.evaluation.metrics import normalize_answer
from typing import List


class HopAlignmentScorer:

    def score_batch(
        self,
        hop_batches:         List[HopBatch],
        gold_supporting_facts: List[List[str]],
        tokenizer,
        input_ids: torch.Tensor,
    ) -> List[float]:
        """
        For each example, measure overlap between hop slot
        token assignments and gold supporting fact tokens.

        Returns per-example alignment scores in [0,1].
        """
        scores = []
        B      = input_ids.shape[0]

        if not hop_batches:
            return [0.0] * B

        # Use the last HOPPER layer's routing (most refined)
        routing = hop_batches[-1].routing   # (B, T, M)

        for b in range(B):
            # Get top-routing tokens for each slot
            top_tokens_per_slot = []
            for m in range(routing.shape[2]):
                slot_weights = routing[b, :, m]   # (T,)
                top_k        = slot_weights.topk(5).indices.tolist()
                tokens        = tokenizer.convert_ids_to_tokens(
                    input_ids[b][top_k].tolist()
                )
                top_tokens_per_slot.append(
                    set(normalize_answer(t) for t in tokens)
                )

            # Gold supporting fact tokens
            gold_tokens = set()
            for fact in gold_supporting_facts[b]:
                for tok in normalize_answer(fact).split():
                    gold_tokens.add(tok)

            # Hop tokens: union of top tokens across all slots
            hop_tokens = set()
            for slot_tokens in top_tokens_per_slot:
                hop_tokens |= slot_tokens

            # Overlap
            if not gold_tokens:
                scores.append(0.0)
                continue

            overlap = len(hop_tokens & gold_tokens)
            score   = overlap / len(gold_tokens)
            scores.append(min(score, 1.0))

        return scores