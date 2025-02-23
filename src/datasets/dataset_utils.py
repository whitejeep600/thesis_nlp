from __future__ import annotations

from pathlib import Path

from src.constants import TrainMode
from src.datasets.sst2_attacker_dataset import SST2AttackerDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


def prepare_dataloaders(
    dataset_paths: dict[TrainMode, Path],
    tokenizer: PreTrainedTokenizer,
    max_len: int,
    min_len: int,
    batch_size: int,
    label_to_keep: int | None = None,
) -> dict[TrainMode, DataLoader]:
    datasets = {
        mode: SST2AttackerDataset(
            dataset_paths[mode],
            tokenizer,
            max_len,
            min_len,
            label_to_keep=label_to_keep,
        )
        for mode in dataset_paths.keys()
    }
    dataloaders = {
        mode: DataLoader(datasets[mode], batch_size=batch_size, shuffle=True)
        for mode in datasets.keys()
    }
    return dataloaders
