from pathlib import Path

import pandas as pd
from transformers import BartTokenizer
from transformers.utils import logging

from src.constants import LABEL
from src.datasets.sst2_attacker_dataset import SST2AttackerDataset


def _print_stats_of_set(set_path: Path) -> None:
    df = pd.read_csv(set_path)

    tokenizer = BartTokenizer.from_pretrained("eugenesiow/bart-paraphrase")

    # short_sentences_dataset = SST2AttackerDataset(
    #     set_path, tokenizer=tokenizer, min_length=0, max_length=11
    # )
    # print("Example short sentences:\n")
    # for i in range(20):
    #     print(short_sentences_dataset[i]["original_sentence"])
    # print("\n")

    length_over_12_dataset = SST2AttackerDataset(
        set_path, tokenizer=tokenizer, label_to_keep=1, min_length=12, max_length=256
    )

    length_over_12_under_32_dataset = SST2AttackerDataset(
        set_path, tokenizer=tokenizer, label_to_keep=1, min_length=12, max_length=32
    )

    print(
        f"the original length is {len(df)}. {len(df[df[LABEL] == 1])} samples are left"
        f" after label filtering (removing samples with negative ground truth sentiment)."
        f" Out of those, {len(length_over_12_dataset)} samples have length of 12 or more,"
        f" and in turn out of those, {len(length_over_12_under_32_dataset)} samples have length"
        f" 32 or less."
    )

    print("\n\n")


def _print_stats_of_sets() -> None:
    eval_path = Path("data/sst2/validation.csv")
    train_path = Path("data/sst2/train.csv")

    print("eval stats:\n")
    _print_stats_of_set(eval_path)

    print("test stats:\n")
    _print_stats_of_set(train_path)


def main() -> None:
    logging.set_verbosity_error()
    _print_stats_of_sets()
    pass


if __name__ == "__main__":
    main()
