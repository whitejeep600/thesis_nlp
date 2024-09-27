from __future__ import annotations

import copy
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch.optim import AdamW

from src.constants import LABEL_NAME_TO_CODE, SIMILARITY, TARGET_LABEL_PROB, TrainMode
from src.control_models.semantic_similarity_evaluators import (
    EmbeddingBasedSemanticSimilarityEvaluator,
)
from src.control_models.sentiment_classifier import CNN_SST2_SentimentClassifier
from src.dpo_trainer import DPORewardAndMetricCalculator, DPOTrainer, RewardAndMetrics
from src.generative_bart import GenerativeBart
from src.utils import (
    assign_model_devices,
    get_current_git_commit_id,
    harmonic_mean,
    prepare_dataloaders,
    prepare_run_save_dir_and_log_file,
)


class AttackerDPORewardAndMetricCalculator(DPORewardAndMetricCalculator):
    def __init__(
        self,
        sentence_transformer_similarity_evaluator_name: str,
        device: torch.device,
        target_label: int,
    ):
        super().__init__()
        self.similarity_evaluator = EmbeddingBasedSemanticSimilarityEvaluator(
            sentence_transformer_similarity_evaluator_name, device
        )
        self.sentiment_classifier = CNN_SST2_SentimentClassifier(device)
        self.target_label = target_label

    def get_similarity_scores_for_generations(
        self, prompt: str, generations: list[str]
    ) -> list[float]:
        return self.similarity_evaluator.evaluate_many_to_one(many=generations, one=prompt)

    def get_target_label_probs_for_generations(self, generations: list[str]) -> list[float]:
        classification_probabilities = self.sentiment_classifier.evaluate_texts(
            generations, return_probs=True
        )
        return classification_probabilities[:, self.target_label].tolist()

    def get_rewards_and_metrics(
        self,
        prompt: str,
        generations: tuple[str, str],
    ) -> tuple[RewardAndMetrics, RewardAndMetrics]:

        with ThreadPoolExecutor(max_workers=2) as executor:
            similarity_calculation = executor.submit(
                partial(
                    self.get_similarity_scores_for_generations,
                    prompt=prompt,
                    generations=list(generations),
                )
            )
            target_label_probs_calculation = executor.submit(
                partial(self.get_target_label_probs_for_generations, generations=list(generations))
            )

        similarity_scores = similarity_calculation.result()
        target_label_probabilities = target_label_probs_calculation.result()

        rewards = [
            harmonic_mean(numbers=[similarity_score, target_label_probability], weights=[1, 3])
            for (similarity_score, target_label_probability) in zip(
                similarity_scores, target_label_probabilities
            )
        ]
        rewards_and_metrics = tuple(
            [
                RewardAndMetrics(
                    reward=reward,
                    metrics={
                        SIMILARITY: similarity_score,
                        TARGET_LABEL_PROB: target_label_probability,
                    },
                )
                for (reward, similarity_score, target_label_probability) in zip(
                    rewards, similarity_scores, target_label_probabilities
                )
            ]
        )
        assert len(rewards_and_metrics) == 2
        return rewards_and_metrics


def main(
    source_bart_model_name: str,
    sentence_transformer_similarity_evaluator_name: str,
    source_bart_weights_path: Path | None,
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
    attacker_runs_save_dir: Path,
    training_log_filename: str,
    params_to_save: dict[str, Any],
    target_label: Literal["positive", "negative"],
) -> None:

    target_label_code = LABEL_NAME_TO_CODE[target_label]
    trained_model_device, control_models_device = assign_model_devices()

    attacker = GenerativeBart(
        source_bart_model_name, trained_model_device, weights_path=source_bart_weights_path
    )
    attacker_optimizer = AdamW(attacker.parameters(), lr=lr)
    reference_model = copy.deepcopy(attacker)

    dataset_paths = {
        TrainMode.train: train_split_path,
        TrainMode.eval: eval_split_path,
    }
    dataloaders = prepare_dataloaders(
        dataset_paths,
        attacker.tokenizer,
        max_len,
        min_len,
        batch_size,
        label_to_keep=1 - target_label_code,
    )

    run_save_dir, all_runs_log_path = prepare_run_save_dir_and_log_file(
        attacker_runs_save_dir, training_log_filename
    )

    n_max_train_batches_per_epoch = (
        n_max_train_samples_per_epoch // batch_size if n_max_train_samples_per_epoch else None
    )

    metric_calculator = AttackerDPORewardAndMetricCalculator(
        sentence_transformer_similarity_evaluator_name,
        device=control_models_device,
        target_label=target_label_code,
    )

    params_to_save.update(
        {"commit_id": get_current_git_commit_id(), "run_save_dir": str(run_save_dir)}
    )

    trainer = DPOTrainer(
        trained_model=attacker,
        reference_model=reference_model,
        dataloaders=dataloaders,
        metric_calculator=metric_calculator,
        trained_model_optimizer=attacker_optimizer,
        run_save_dir=run_save_dir,
        general_training_log_path=attacker_runs_save_dir / training_log_filename,
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
    script_path = "src.train_attacker"
    attacker_params = yaml.safe_load(open("params.yaml"))[script_path]

    main(
        source_bart_model_name=attacker_params["source_bart_model_name"],
        source_bart_weights_path=(
            Path(attacker_params["source_bart_weights_path"])
            if attacker_params["source_bart_weights_path"]
            else None
        ),
        sentence_transformer_similarity_evaluator_name=attacker_params[
            "sentence_transformer_similarity_evaluator_name"
        ],
        train_split_path=Path(attacker_params["train_split_path"]),
        eval_split_path=Path(attacker_params["eval_split_path"]),
        max_len=int(attacker_params["max_len"]),
        min_len=int(attacker_params["min_len"]),
        batch_size=int(attacker_params["batch_size"]),
        n_epochs=int(attacker_params["n_epochs"]),
        lr=float(attacker_params["lr"]),
        dpo_beta=float(attacker_params["dpo_beta"]),
        temperature=float(attacker_params["temperature"]),
        n_max_train_samples_per_epoch=(
            int(attacker_params["n_max_train_samples_per_epoch"])
            if attacker_params["n_max_train_samples_per_epoch"]
            else None
        ),
        attacker_runs_save_dir=Path(attacker_params["attacker_runs_save_dir"]),
        training_log_filename=attacker_params["training_log_filename"],
        params_to_save=attacker_params,
        target_label=attacker_params["target_label"],
    )
