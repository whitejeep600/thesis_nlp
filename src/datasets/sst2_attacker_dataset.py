from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from src.constants import ID, INPUT_IDS, LABEL, ORIGINAL_SENTENCE, SENTENCE
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SST2AttackerDataset(Dataset):
    def __init__(
        self,
        dataset_csv_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        min_length: int | None = None,
        label_to_keep: int | None = None,
    ):
        """
        Using the "label_to_keep" argument, sentences with a specific label only can be
        kept in the dataset. This is done because for targeted attacks (making the victim
        model erroneously output a specific target label), we are not very interested in
        sentences that already have that label.
        """

        super().__init__()

        source_df = pd.read_csv(dataset_csv_path)

        if label_to_keep is not None:
            source_df = source_df[source_df[LABEL] == label_to_keep].reset_index(drop=True)

        sentences: list[str] = source_df[SENTENCE].values.tolist()

        # We want to reject samples that were truncated to max_length, as this may result in
        # nonsensical sentences. We leave an extra token beyond max_length, and later reject
        # samples for which this token is not a padding token. Finally we remove that token.
        tokenized_sentences = tokenizer(
            sentences,
            return_tensors="pt",
            max_length=max_length + 1,  # See comment above about the +1
            truncation=True,
            padding="max_length",
        )
        input_ids = tokenized_sentences[INPUT_IDS]

        if min_length is None:
            appropriate_length_sample_indices = list(range(len(input_ids)))
        else:
            lengths = [
                sum(input_ids[i] != tokenizer.pad_token_id).item() for i in range(len(input_ids))
            ]
            appropriate_length_sample_indices = [
                i for i in range(len(input_ids)) if min_length <= lengths[i] <= max_length
            ]

        # See comment above about the -1
        input_ids = [input_ids[i][:-1] for i in appropriate_length_sample_indices]

        # Beside the tokenized sentences, during training we also want access to the original
        # sentences in text format. This is to run control models (semantic similarity model, attack
        # victim model and others). At the same time, due to truncation, we should not consider
        # those sentences in full length.
        original_sentences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        self.input_ids = input_ids
        self.original_sentences = original_sentences

        self.labels: list[int] = source_df[LABEL][appropriate_length_sample_indices].values.tolist()
        self.ids: list[str] = source_df[ID][appropriate_length_sample_indices].values.tolist()

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
