"""
HotpotQA dataset for span extraction.
Distractor setting — standard for multi-hop QA benchmarks.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import RobertaTokenizerFast
from typing import List, Dict


class HotpotQADataset(Dataset):

    MAX_LENGTH = 512

    def __init__(
        self,
        split:          str,
        tokenizer_name: str  = "roberta-base",
        max_length:     int  = 512,
        cache_dir:      str  = "/kaggle/working/cache",
    ):
        self.tokenizer  = RobertaTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length

        print(f"Loading HotpotQA ({split})...")
        raw = load_dataset(
            "hotpot_qa", "distractor",
            split    = split,
            cache_dir= cache_dir,
        )
        self.examples = self._process(raw)
        print(f"Loaded {len(self.examples)} examples.")

    def _process(self, raw) -> List[Dict]:
        examples = []
        for item in raw:
            question = item["question"]

            # Flatten supporting paragraphs into one context
            context_parts = []
            for title, sentences in zip(
                item["context"]["title"],
                item["context"]["sentences"],
            ):
                context_parts.append(" ".join(sentences))
            context = " ".join(context_parts)

            answer  = item["answer"]

            # Supporting facts
            gold_facts = []
            for title, sent_id in zip(
                item["supporting_facts"]["title"],
                item["supporting_facts"]["sent_id"],
            ):
                idx = item["context"]["title"].index(title)
                sentences = item["context"]["sentences"][idx]
                if sent_id < len(sentences):
                    gold_facts.append(sentences[sent_id])

            # Tokenize
            encoding = self.tokenizer(
                question,
                context,
                max_length     = self.max_length,
                truncation     = True,
                padding        = "max_length",
                return_offsets_mapping = True,
                return_tensors = "pt",
            )

            # Find answer span in tokenized input
            start_pos, end_pos = self._find_answer_span(
                encoding, context, answer
            )

            examples.append({
                "input_ids":             encoding["input_ids"].squeeze(0),
                "attention_mask":        encoding["attention_mask"].squeeze(0),
                "start_positions":       torch.tensor(start_pos),
                "end_positions":         torch.tensor(end_pos),
                "answer_text":           answer,
                "gold_supporting_facts": gold_facts,
                "tokens":                self.tokenizer.convert_ids_to_tokens(
                    encoding["input_ids"].squeeze(0).tolist()
                ),
            })

        return examples

    def _find_answer_span(self, encoding, context, answer) -> tuple:
        """Find start/end token positions of answer in encoded input."""
        try:
            # Find answer character offset in context
            answer_start_char = context.find(answer)
            if answer_start_char == -1:
                return 0, 0

            answer_end_char = answer_start_char + len(answer) - 1

            offsets = encoding["offset_mapping"].squeeze(0)

            start_pos = end_pos = 0
            for idx, (start, end) in enumerate(offsets.tolist()):
                if start <= answer_start_char < end:
                    start_pos = idx
                if start <= answer_end_char < end:
                    end_pos = idx
                    break

            return start_pos, end_pos
        except Exception:
            return 0, 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]