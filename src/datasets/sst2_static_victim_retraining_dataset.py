from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.constants import (
    ID,
    INPUT_IDS,
    LABEL,
    MODEL_RESPONSE,
    ORIGIN,
    ORIGIN_GENERATED,
    ORIGIN_SAMPLED,
    PROMPT_ORIGINAL_TARGET_LABEL_PROB,
    SENTENCE,
    SIMILARITY,
    TARGET_LABEL_PROB,
    TrainMode,
)
from textattack.models.tokenizers import GloveTokenizer
from torch.utils.data import Dataset

MAX_PROMPT_ORIGINAL_TARGET_LABEL_PROB = 0.5
MIN_GENERATION_TARGET_LABEL_PROB = 0.8
MIN_SEMSIM = 0.8
MAX_EXAMPLES_PER_SAMPLE_ID = 5


class SST2VictimRetrainingDataset(Dataset):
    def __init__(
        self,
        input_ids: list[list[int]],
        sentences: list[str],
        labels: list[int],
        ids: list[int],
        origins: list[str],
    ):
        super().__init__()
        self.input_ids = input_ids
        self.original_sentences = sentences
        self.labels = labels
        self.ids = ids
        self.origins = origins

    def __len__(self):
        return len(self.original_sentences)

    def __getitem__(self, i):
        input_ids = self.input_ids[i]
        sentence = self.original_sentences[i]
        label = self.labels[i]
        id_ = self.ids[i]
        origin = self.origins[i]

        return {
            INPUT_IDS: torch.IntTensor(input_ids),
            SENTENCE: sentence,
            LABEL: torch.tensor(label),
            ID: torch.tensor(id_),
            ORIGIN: origin,
        }

    @classmethod
    def from_original_dataset_csv_path(
        cls,
        dataset_csv_path: Path,
        tokenizer: GloveTokenizer,
        max_length: int,
        min_length: int | None = None,
        label_to_keep: int | None = None,
    ) -> "SST2VictimRetrainingDataset":

        source_df = pd.read_csv(dataset_csv_path)

        if label_to_keep is not None:
            source_df = source_df[source_df[LABEL] == label_to_keep].reset_index(drop=True)

        sentences: list[str] = source_df[SENTENCE].values.tolist()

        input_ids = tokenizer(
            sentences,
        )

        if min_length is None:
            appropriate_length_sample_indices = list(range(len(input_ids)))
        else:
            lengths = [
                (np.array(input_ids[i]) != tokenizer.pad_token_id).sum()
                for i in range(len(input_ids))
            ]
            appropriate_length_sample_indices = [
                i for i in range(len(input_ids)) if min_length <= lengths[i] <= max_length
            ]

        input_ids = [input_ids[i] for i in appropriate_length_sample_indices]
        original_sentences = [sentences[i] for i in appropriate_length_sample_indices]
        origins = [ORIGIN_SAMPLED for _ in range(len(sentences))]

        return cls(
            input_ids=input_ids,
            sentences=original_sentences,
            labels=source_df[LABEL][appropriate_length_sample_indices].values.tolist(),
            ids=source_df[ID][appropriate_length_sample_indices].values.tolist(),
            origins=origins,
        )

    @classmethod
    def from_attacker_training_save_paths(
        cls,
        attacker_training_save_paths: list[Path],
        tokenizer: GloveTokenizer,
        mode: TrainMode,
        attacker_target_label_code: int,
    ) -> "SST2VictimRetrainingDataset":
        epoch_csv_paths = [
            epoch_csv_path
            for attacker_training_save_path in attacker_training_save_paths
            for epoch_csv_path in (
                attacker_training_save_path / "generated_sentences" / mode.value
            ).iterdir()
        ]
        epoch_dfs = [pd.read_csv(path) for path in epoch_csv_paths]
        successful_attack_epoch_dfs = [
            df[
                (df[PROMPT_ORIGINAL_TARGET_LABEL_PROB] < MAX_PROMPT_ORIGINAL_TARGET_LABEL_PROB)
                & (df[TARGET_LABEL_PROB] > MIN_GENERATION_TARGET_LABEL_PROB)
                & (df[SIMILARITY] > MIN_SEMSIM)
            ]
            for df in epoch_dfs
        ]
        successful_attack_df = pd.concat(successful_attack_epoch_dfs, axis=0)

        successful_attack_df = successful_attack_df.groupby(ID).head(n=MAX_EXAMPLES_PER_SAMPLE_ID)

        sentences = successful_attack_df[MODEL_RESPONSE].values.tolist()
        input_ids = tokenizer(
            sentences,
        )
        labels = [1 - attacker_target_label_code for _ in range(len(sentences))]
        ids = successful_attack_df[ID].values.tolist()
        origins = [ORIGIN_GENERATED for _ in range(len(sentences))]

        return cls(
            input_ids=input_ids,
            sentences=sentences,
            labels=labels,
            ids=ids,
            origins=origins,
        )
