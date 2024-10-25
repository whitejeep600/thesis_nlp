from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW

from src.constants import SIMILARITY, TrainMode
from src.control_models.semantic_similarity_evaluators import (
    EmbeddingBasedSemanticSimilarityEvaluator,
)
from src.datasets.dataset_utils import prepare_dataloaders
from src.dpo_trainer import DPORewardAndMetricCalculator, DPOTrainer, RewardAndMetrics
from src.generative_bart import GenerativeBart
from src.utils import (
    assign_model_devices,
    get_current_git_commit_id,
    prepare_run_save_dir_and_log_file,
)


class EchoDPORewardAndMetricCalculator(DPORewardAndMetricCalculator):
    def __init__(self, sentence_transformer_similarity_evaluator_name: str, device: torch.device):
        super().__init__()
        self.similarity_evaluator = EmbeddingBasedSemanticSimilarityEvaluator(
            sentence_transformer_similarity_evaluator_name, device
        )

    def get_rewards_and_metrics_for_single_generation(
        self, prompt: str, generation: str
    ) -> RewardAndMetrics:
        similarity_score = self.similarity_evaluator.evaluate_one_to_one(generation, prompt)
        rewards_and_metrics = RewardAndMetrics(
            reward=similarity_score, metrics={SIMILARITY: similarity_score}
        )
        return rewards_and_metrics

    def get_rewards_and_metrics(
        self,
        prompt: str,
        generations: tuple[str, str],
    ) -> tuple[RewardAndMetrics, RewardAndMetrics]:
        return (
            self.get_rewards_and_metrics_for_single_generation(prompt, generations[0]),
            self.get_rewards_and_metrics_for_single_generation(prompt, generations[1]),
        )


def main(
    source_bart_model_name: str,
    sentence_transformer_similarity_evaluator_name: str,
    train_split_path: Path,
    eval_split_path: Path,
    max_len: int,
    min_len: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    dpo_beta: float,
    temperature: float,
    n_max_train_samples_per_epoch: int | None,
    echo_runs_save_dir: Path,
    training_log_filename: str,
    params_to_save: dict[str, Any],
) -> None:
    """
    The sst2 training set is relatively large, and takes a lot of time for the trainer to process,
    which may also result in big changes to the model. We might want to evaluate its performance on
    the validation set more often than once every time the whole training set is processed.
    By passing n_max_train_samples different than None, the number of train samples per epoch
    can be restricted.
    """

    trained_model_device, control_models_device = assign_model_devices()

    echo = GenerativeBart(source_bart_model_name, trained_model_device)
    echo_optimizer = AdamW(echo.parameters(), lr=lr)
    reference_model = GenerativeBart(source_bart_model_name, control_models_device)

    dataset_paths = {
        TrainMode.train: train_split_path,
        TrainMode.eval: eval_split_path,
    }
    dataloaders = prepare_dataloaders(dataset_paths, echo.tokenizer, max_len, min_len, batch_size)

    run_save_dir, all_runs_log_path = prepare_run_save_dir_and_log_file(
        echo_runs_save_dir, training_log_filename
    )

    n_max_train_batches_per_epoch = (
        n_max_train_samples_per_epoch // batch_size if n_max_train_samples_per_epoch else None
    )

    metric_calculator = EchoDPORewardAndMetricCalculator(
        sentence_transformer_similarity_evaluator_name,
        device=control_models_device,
    )

    params_to_save.update(
        {"commit_id": get_current_git_commit_id(), "run_save_dir": str(run_save_dir)}
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
        params_to_save=params_to_save,
        n_max_train_batches_per_epoch=n_max_train_batches_per_epoch,
    )
    trainer.run_training()


if __name__ == "__main__":
    script_path = "src.training_scripts.train_echo"
    params = yaml.safe_load(open("params.yaml"))[script_path]

    main(
        source_bart_model_name=params["source_bart_model_name"],
        sentence_transformer_similarity_evaluator_name=params[
            "sentence_transformer_similarity_evaluator_name"
        ],
        train_split_path=Path(params["train_split_path"]),
        eval_split_path=Path(params["eval_split_path"]),
        max_len=int(params["max_len"]),
        min_len=int(params["min_len"]),
        batch_size=int(params["batch_size"]),
        n_epochs=int(params["n_epochs"]),
        lr=float(params["lr"]),
        dpo_beta=float(params["dpo_beta"]),
        temperature=float(params["temperature"]),
        n_max_train_samples_per_epoch=(
            int(params["n_max_train_samples_per_epoch"])
            if params["n_max_train_samples_per_epoch"]
            else None
        ),
        echo_runs_save_dir=Path(params["echo_runs_save_dir"]),
        training_log_filename=params["training_log_filename"],
        params_to_save=params,
    )
