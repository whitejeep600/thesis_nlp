from pathlib import Path
from typing import Literal

import torch
import yaml
from textattack.models.helpers import WordCNNForClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import INPUT_IDS, LABEL_NAME_TO_CODE, TrainMode
from src.control_models.grammaticality_evaluator import GrammaticalityEvaluator
from src.datasets.sst2_static_victim_retraining_dataset import SST2VictimRetrainingDataset
from src.utils import get_available_torch_devices


class VictimStaticRetrainer:
    def __init__(self):
        # trained model, dataloaders, trained_model_optimizer, run_save_dir, n_epochs, lr

        # iteration
        # save model checkpoint
        # save for each epoch: predictions, label origins, ground truth labels, ids in each epoch
        # stats: accuracy on  adv examples, and for the original eval set, general accuracy,
        #   and accuracy on samples from both classes separately
        # save plots
        pass

    def run_training(self):
        pass


def main(
    static_victim_retraining_runs_save_dir: Path,
    attacker_run_dir: Path,
    original_train_split_path: Path,
    original_eval_split_path: Path,
    min_len: int,
    max_len: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    attacker_target_label: Literal["positive", "negative"],
) -> None:
    target_label_code = LABEL_NAME_TO_CODE[attacker_target_label]
    dataset_paths = {
        TrainMode.train: original_train_split_path,
        TrainMode.eval: original_eval_split_path,
    }
    victim = WordCNNForClassification.from_pretrained("cnn-sst2")

    original_datasets = {
        TrainMode.train: SST2VictimRetrainingDataset.from_dataset_csv_path(
            dataset_paths[TrainMode.train],
            victim.tokenizer,
            max_len,
            min_len,
            label_to_keep=target_label_code,
        ),
        TrainMode.eval: SST2VictimRetrainingDataset.from_dataset_csv_path(
            dataset_paths[TrainMode.eval], victim.tokenizer, max_len, min_len
        ),
    }
    original_dataloaders = {
        mode: DataLoader(original_datasets[mode], batch_size=batch_size, shuffle=True)
        for mode in original_datasets.keys()
    }

    attacker_target_label_code = LABEL_NAME_TO_CODE[attacker_target_label]
    successful_attack_datasets = {
        mode: SST2VictimRetrainingDataset.from_attacker_training_save_path(
            attacker_run_dir, victim.tokenizer, mode, attacker_target_label_code
        )
        for mode in dataset_paths.keys()
    }
    successful_attack_dataloaders = {
        mode: DataLoader(successful_attack_datasets[mode], batch_size=batch_size, shuffle=True)
        for mode in successful_attack_datasets.keys()
    }

    training_device = get_available_torch_devices()[0]
    victim_optimizer = AdamW(
        victim.parameters(), lr=lr, fused=training_device == torch.device("cuda"), foreach=False
    )

    for _ in tqdm(range(n_epochs), desc="training...", position=0):
        for batch_no, batch in tqdm(
            enumerate(original_dataloaders[TrainMode.train]),
            total=len(original_dataloaders[TrainMode.train]),
            desc="train iteration",
            leave=False,
            position=1,
        ):
            print(victim(batch[INPUT_IDS]))
            # composition of train/test sets?
            # train: k samples each epoch (chosen randomly) with gt=negative, from the original set
            #        k samples -----------------;-------------------positive, from the attacks
            #            alternate the batches each epoch
            # k is selected as the minimum of (512, available samples from the sets)
            # so at the beginning, the model perceives everything as negative, and also it won't
            #   see any unaltered positive examples. But that should be fine
            # eval: two separate sets, one is whole original eval, the other is whole successful attacks
            # support gpu
            # find max possible batch size
            # limit successful attacks per id? Like, not to have 2137 paraphrases of the same sentence

    pass


if __name__ == "__main__":
    script_path = "src.training_scripts.retrain_victim_statically"
    params = yaml.safe_load(open("params.yaml"))[script_path]

    main(
        static_victim_retraining_runs_save_dir=Path(
            params["static_victim_retraining_runs_save_dir"]
        ),
        attacker_run_dir=Path(params["attacker_run_dir"]),
        original_train_split_path=Path(params["original_train_split_path"]),
        original_eval_split_path=Path(params["original_eval_split_path"]),
        max_len=int(params["max_len"]),
        min_len=int(params["min_len"]),
        batch_size=int(params["batch_size"]),
        n_epochs=int(params["n_epochs"]),
        lr=float(params["lr"]),
        attacker_target_label=params["target_label"],
    )
