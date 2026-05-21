"""
HOPPER Evaluator.
Runs model on validation set, extracts predictions, computes metrics.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hopper.evaluation.metrics import compute_metrics
from typing import List


class HOPPEREvaluator:

    def __init__(self, device: str = "cuda"):
        self.device = device

    def evaluate(
        self,
        model,
        dataloader: DataLoader,
        dataset_name: str,
    ) -> dict:
        model.eval()
        predictions = []
        references  = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                start_logits, end_logits, _ = model(input_ids, attention_mask)

                # Decode span predictions
                start_preds = start_logits.argmax(dim=-1)
                end_preds   = end_logits.argmax(dim=-1)

                # Enforce start ≤ end
                for i in range(len(start_preds)):
                    if start_preds[i] > end_preds[i]:
                        end_preds[i] = start_preds[i]

                for i in range(len(start_preds)):
                    tokens = batch["tokens"][i] if "tokens" in batch else []
                    s, e   = start_preds[i].item(), end_preds[i].item()
                    answer = " ".join(tokens[s:e+1]) if tokens else ""

                    predictions.append({
                        "answer":           answer,
                        "supporting_facts": batch.get(
                            "pred_supporting_facts", [[]]
                        )[i],
                    })
                    references.append({
                        "answer":           batch["answer_text"][i],
                        "supporting_facts": batch.get(
                            "gold_supporting_facts", [[]]
                        )[i],
                    })

        metrics = compute_metrics(predictions, references)
        return metrics