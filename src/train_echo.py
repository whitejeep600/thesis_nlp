from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.constants import TrainMode
from src.dpo_trainer import DPORewardAndMetricCalculator, DPOTrainer, RewardAndMetrics
from src.generative_bart import GenerativeBart
from src.sst2_dataset import SST2Dataset
from src.text_evaluation_models.semantic_similarity_evaluators import (
    EmbeddingBasedSemanticSimilarityEvaluator,
)
from src.utils import get_available_torch_devices, get_next_run_subdir_name


def _assign_model_devices() -> tuple[torch.device, torch.device]:
    """
    In the setting the experiments were developed in, training could only be run
    on the author's machine locally, or on a university server. Locally,
    there were no GPU devices available, or one MPS device, depending on the
    equipment used. On the server, there were two CUDA devices. One of them
    was assigned for the trained model exclusively (higher memory requirements
    due to backpropagation), while the other for all the remaining models
    used in the training (referred to as the "control models").
    """
    torch_devices = get_available_torch_devices()
    trained_model_device = torch_devices[0]
    if len(torch_devices) > 1:
        control_models_device = torch_devices[1]
    else:
        control_models_device = torch_devices[0]
    return trained_model_device, control_models_device


def _prepare_dataloaders(
    dataset_paths: dict[TrainMode, Path],
    tokenizer: PreTrainedTokenizer,
    max_len: int,
    batch_size: int,
) -> dict[TrainMode, DataLoader]:
    datasets = {
        mode: SST2Dataset(
            dataset_paths[mode],
            tokenizer,
            max_len,
        )
        for mode in dataset_paths.keys()
    }
    dataloaders = {
        mode: DataLoader(datasets[mode], batch_size=batch_size, shuffle=True)
        for mode in datasets.keys()
    }
    return dataloaders


def _prepare_run_save_dir_and_log_file(
    all_runs_save_dir: Path, training_log_filename: str
) -> tuple[Path, Path]:
    all_runs_save_dir.mkdir(exist_ok=True, parents=True)
    run_save_dir = all_runs_save_dir / get_next_run_subdir_name(all_runs_save_dir)
    run_save_dir.mkdir(parents=True)
    all_runs_log_path = all_runs_save_dir / training_log_filename
    all_runs_log_path.touch()
    return run_save_dir, all_runs_log_path


class EchoDPORewardAndMetricCalculator(DPORewardAndMetricCalculator):
    def __init__(self, sentence_transformer_similarity_evaluator_name: str, device: torch.device):
        super().__init__()
        self.similarity_evaluator = EmbeddingBasedSemanticSimilarityEvaluator(
            sentence_transformer_similarity_evaluator_name, device
        )

    def get_rewards_and_metrics(
        self,
        prompt: str,
        generations: tuple[str, str],
    ) -> tuple[RewardAndMetrics, RewardAndMetrics]:
        similarity_score_0, similarity_score_1 = self.similarity_evaluator.evaluate_many_to_one(
            many=list(generations), one=prompt
        )
        return RewardAndMetrics(reward=similarity_score_0, metrics={}), RewardAndMetrics(
            reward=similarity_score_1, metrics={}
        )


def main(
    source_bart_model_name: str,
    sentence_transformer_similarity_evaluator_name: str,
    train_split_path: Path,
    eval_split_path: Path,
    max_len: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    dpo_beta: float,
    temperature: float,
    n_max_train_samples_per_epoch: int | None,
    echo_runs_save_dir: Path,
    training_log_filename: str,
) -> None:
    """
    The sst2 training set is relatively large, and takes a lot of time for the trainer to process,
    which may also result in big changes to the model. We might want to evaluate its performance on
    the validation set more often than once every time the whole training set is processed.
    By passing n_max_train_samples different than None, the number of train samples per epoch
    can be restricted.
    """

    trained_model_device, control_models_device = _assign_model_devices()

    echo = GenerativeBart(source_bart_model_name, trained_model_device)
    echo_optimizer = AdamW(echo.parameters(), lr=lr)
    reference_model = GenerativeBart(source_bart_model_name, control_models_device)

    dataset_paths = {
        TrainMode.train: train_split_path,
        TrainMode.eval: eval_split_path,
    }
    dataloaders = _prepare_dataloaders(dataset_paths, echo.tokenizer, max_len, batch_size)

    run_save_dir, all_runs_log_path = _prepare_run_save_dir_and_log_file(
        echo_runs_save_dir, training_log_filename
    )

    n_max_train_batches_per_epoch = (
        n_max_train_samples_per_epoch // batch_size if n_max_train_samples_per_epoch else None
    )

    metric_calculator = EchoDPORewardAndMetricCalculator(
        sentence_transformer_similarity_evaluator_name,
        device=control_models_device,
    )

    trainer = DPOTrainer(
        trained_model=echo,
        reference_model=reference_model,
        dataloaders=dataloaders,
        metric_calculator=metric_calculator,
        trained_model_optimizer=echo_optimizer,
        run_save_dir=run_save_dir,
        general_training_log_path=echo_runs_save_dir / training_log_filename,
        n_epochs=n_epochs,
        max_len=max_len,
        beta=dpo_beta,
        temperature=temperature,
        lr=lr,
        n_max_train_batches_per_epoch=n_max_train_batches_per_epoch,
    )
    trainer.run_training()


if __name__ == "__main__":
    script_path = "src.train_echo"
    echo_params = yaml.safe_load(open("params.yaml"))[script_path]

    main(
        source_bart_model_name=echo_params["source_bart_model_name"],
        sentence_transformer_similarity_evaluator_name=echo_params[
            "sentence_transformer_similarity_evaluator_name"
        ],
        train_split_path=Path(echo_params["train_split_path"]),
        eval_split_path=Path(echo_params["eval_split_path"]),
        max_len=int(echo_params["max_len"]),
        batch_size=int(echo_params["batch_size"]),
        n_epochs=int(echo_params["n_epochs"]),
        lr=float(echo_params["lr"]),
        dpo_beta=float(echo_params["dpo_beta"]),
        temperature=float(echo_params["temperature"]),
        n_max_train_samples_per_epoch=(
            int(echo_params["n_max_train_samples_per_epoch"])
            if echo_params["n_max_train_samples_per_epoch"]
            else None
        ),
        echo_runs_save_dir=Path(echo_params["echo_runs_save_dir"]),
        training_log_filename=echo_params["training_log_filename"],
    )
