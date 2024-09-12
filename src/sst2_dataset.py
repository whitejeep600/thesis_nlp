from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.constants import ID, INPUT_IDS, LABEL, ORIGINAL_SENTENCE, SENTENCE


class SST2Dataset(Dataset):
    def __init__(
        self,
        dataset_csv_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ):
        super().__init__()

        source_df = pd.read_csv(dataset_csv_path)

        sentences: list[str] = source_df[SENTENCE].values.tolist()

        tokenized_sentences = tokenizer(
            sentences,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = tokenized_sentences[INPUT_IDS]

        # Beside the tokenized sentences, during training we also want access to the original
        # sentences in text format. This is to run control models (semantic similarity model, attack
        # victim model and others). At the same time, due to truncation, we should not consider
        # those sentences in full length.
        original_sentences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        self.input_ids = input_ids
        self.original_sentences = original_sentences

        self.labels: list[int] = source_df[LABEL].values.tolist()
        self.ids: list[str] = source_df[ID].values.tolist()

    def __len__(self):
        return len(self.original_sentences)

    def __getitem__(self, i):
        input_ids = self.input_ids[i].clone()
        original_sentence = self.original_sentences[i]
        label = self.labels[i]
        id_ = self.ids[i]

        return {
            INPUT_IDS: input_ids,
            ORIGINAL_SENTENCE: original_sentence,
            LABEL: torch.tensor(label),
            ID: torch.tensor(id_),
        }
