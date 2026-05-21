"""
Evaluation metrics for span extraction.
EM, F1, Supporting Fact F1.
"""

import re
import string
from collections import Counter
from typing import List, Dict


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens   = normalize_answer(prediction).split()
    gt_tokens     = normalize_answer(ground_truth).split()
    common        = Counter(pred_tokens) & Counter(gt_tokens)
    num_same      = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def supporting_fact_f1(
    predicted_hops: List[str],
    gold_facts:     List[str],
) -> float:
    """
    Measures overlap between predicted supporting sentences
    and gold supporting fact sentences.
    This is the key metric that validates HOPPER's reasoning structure.
    """
    if not gold_facts:
        return 0.0
    if not predicted_hops:
        return 0.0

    pred_set = set(normalize_answer(h) for h in predicted_hops)
    gold_set = set(normalize_answer(g) for g in gold_facts)

    tp        = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gold_set) if gold_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    predictions: List[Dict],
    references:  List[Dict],
) -> Dict:
    """
    predictions: list of {"answer": str, "supporting_facts": [str]}
    references:  list of {"answer": str, "supporting_facts": [str]}
    """
    em_scores      = []
    f1_scores      = []
    supp_f1_scores = []

    for pred, ref in zip(predictions, references):
        em_scores.append(exact_match(pred["answer"], ref["answer"]))
        f1_scores.append(token_f1(pred["answer"],   ref["answer"]))
        supp_f1_scores.append(
            supporting_fact_f1(
                pred.get("supporting_facts", []),
                ref.get("supporting_facts",  []),
            )
        )

    return {
        "em":                  sum(em_scores)      / len(em_scores),
        "f1":                  sum(f1_scores)      / len(f1_scores),
        "supporting_fact_f1":  sum(supp_f1_scores) / len(supp_f1_scores),
        "num_examples":        len(em_scores),
    }