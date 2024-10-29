from __future__ import annotations

import copy
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch.optim import AdamW

from src.constants import (
    LABEL_NAME_TO_CODE,
    PROMPT_ORIGINAL_TARGET_LABEL_PROB,
    SIMILARITY,
    SUCCESSFUL_ATTACK,
    TARGET_LABEL_PROB,
    TrainMode,
)
from src.control_models.grammaticality_evaluator import GrammaticalityEvaluator
from src.control_models.semantic_similarity_evaluators import (
    EmbeddingBasedSemanticSimilarityEvaluator,
    T5HardLabelEntailmentEvaluator,
)
from src.control_models.sentiment_classifier import CNN_SST2_SentimentClassifier
from src.datasets.dataset_utils import prepare_dataloaders
from src.dpo_trainer import DPORewardAndMetricCalculator, DPOTrainer, RewardAndMetrics
from src.generative_bart import GenerativeBart
from src.utils import (
    assign_model_devices,
    get_current_git_commit_id,
    get_length_difference_scores,
    harmonic_mean,
    prepare_run_save_dir_and_log_file,
)


class AttackerDPORewardAndMetricCalculator(DPORewardAndMetricCalculator):
    def __init__(
        self,
        device: torch.device,
        target_label: int,
        similarity_evaluator_name: str,
    ):
        super().__init__()
        self.entailment_evaluator = T5HardLabelEntailmentEvaluator(device)
        self.sentiment_classifier = CNN_SST2_SentimentClassifier(device)
        self.similarity_evaluator = EmbeddingBasedSemanticSimilarityEvaluator(
            similarity_evaluator_name, device
        )
        self.grammaticality_evaluator = GrammaticalityEvaluator(device)
        self.target_label = target_label

    def get_similarity_scores_for_generations(
        self, prompt: str, generations: list[str]
    ) -> list[float]:
        similarity_scores = self.similarity_evaluator.evaluate_many_to_one(
            many=generations, one=prompt
        )
        entailment_scores = self.entailment_evaluator.get_hard_labels_many_to_one(
            one=prompt, many=generations
        )
        grammaticality_scores = self.grammaticality_evaluator.evaluate_texts(
            generations, return_probs=True
        )

        length_difference_scores = get_length_difference_scores(prompt, generations)

        return [
            similarity_score * length_score * entailment_score * grammaticality_score
            for (similarity_score, entailment_score, length_score, grammaticality_score) in zip(
                similarity_scores,
                entailment_scores,
                length_difference_scores,
                grammaticality_scores,
            )
        ]

    @staticmethod
    def attack_is_successful(
        prompt_original_target_label_prob: float,
        generation_target_label_prob: float,
        semantic_similarity: float,
    ) -> bool:
        return (
            prompt_original_target_label_prob < 0.5
            and generation_target_label_prob > 0.5
            and semantic_similarity > 0.7
        )

    def get_target_label_probs(self, sequences: list[str]) -> list[float]:
        classification_probabilities = self.sentiment_classifier.evaluate_texts(
            sequences, return_probs=True
        )
        return classification_probabilities[:, self.target_label].tolist()

    def get_rewards_and_metrics(
        self,
        prompt: str,
        generations: tuple[str, str],
    ) -> tuple[RewardAndMetrics, RewardAndMetrics]:

        with ThreadPoolExecutor(max_workers=3) as executor:
            similarity_calculation = executor.submit(
                partial(
                    self.get_similarity_scores_for_generations,
                    prompt=prompt,
                    generations=list(generations),
                )
            )
            generations_target_label_probs_calculation = executor.submit(
                partial(self.get_target_label_probs, sequences=list(generations))
            )
            prompt_target_label_prob_calculation = executor.submit(
                partial(self.get_target_label_probs, sequences=[prompt])
            )

        similarity_scores = similarity_calculation.result()
        generation_target_label_probabilities = generations_target_label_probs_calculation.result()
        prompt_target_label_probability = prompt_target_label_prob_calculation.result()[0]

        rewards = [
            harmonic_mean(numbers=[similarity_score, target_label_probability], weights=[1, 3])
            for (similarity_score, target_label_probability) in zip(
                similarity_scores, generation_target_label_probabilities
            )
        ]

        successful_or_not_scores = [
            self.attack_is_successful(
                prompt_target_label_probability,
                generation_target_label_probability,
                similarity_score,
            )
            for (similarity_score, generation_target_label_probability) in zip(
                similarity_scores, generation_target_label_probabilities
            )
        ]

        rewards_and_metrics = tuple(
            [
                RewardAndMetrics(
                    reward=reward,
                    metrics={
                        SIMILARITY: similarity_score,
                        TARGET_LABEL_PROB: target_label_probability,
                        PROMPT_ORIGINAL_TARGET_LABEL_PROB: prompt_target_label_probability,
                        SUCCESSFUL_ATTACK: successful_or_not_score,
                    },
                )
                for (
                    reward,
                    similarity_score,
                    target_label_probability,
                    successful_or_not_score,
                ) in zip(
                    rewards,
                    similarity_scores,
                    generation_target_label_probabilities,
                    successful_or_not_scores,
                )
            ]
        )
        assert len(rewards_and_metrics) == 2
        return rewards_and_metrics


def main(
    source_bart_model_name: str,
    source_bart_weights_path: Path | None,
    reference_bart_weights_path: Path | None,
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
    attacker_runs_save_dir: Path,
    training_log_filename: str,
    params_to_save: dict[str, Any],
    target_label: Literal["positive", "negative"],
) -> None:

    target_label_code = LABEL_NAME_TO_CODE[target_label]
    attacker_device, control_models_device = assign_model_devices()

    attacker = GenerativeBart(
        source_bart_model_name, attacker_device, weights_path=source_bart_weights_path
    )
    attacker_optimizer = AdamW(
        attacker.parameters(), lr=lr, fused=attacker_device == torch.device("cuda"), foreach=False
    )

    # It is allowed to explicitly pass the weights to a reference model (used for reference
    # probabilities in the DPO algorithm - intuitively, the model that we want not to deviate
    # too far from). Otherwise, the initial state of the tuned model is taken as the reference
    # model. This default behavior is common practice. We may want to diverge from this behavior,
    # for example, if resuming a previous training. Then the initial weights are passed
    # as the last-epoch weights of the previous training, and the reference weights are the same
    # as they were in that previous training.
    if reference_bart_weights_path:
        reference_model = GenerativeBart(
            source_bart_model_name, attacker_device, weights_path=reference_bart_weights_path
        )
    else:
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
        device=control_models_device,
        target_label=target_label_code,
        similarity_evaluator_name=sentence_transformer_similarity_evaluator_name,
    )

    params_to_save.update(
        {
            "commit_id": get_current_git_commit_id(),
            "run_save_dir": str(run_save_dir),
            "run_start_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        }
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
        params_to_save=params_to_save,
        n_max_train_batches_per_epoch=n_max_train_batches_per_epoch,
        metrics_excluded_from_plotting=[PROMPT_ORIGINAL_TARGET_LABEL_PROB],
    )
    trainer.run_training()


if __name__ == "__main__":
    script_path = "src.training_scripts.train_attacker"
    params = yaml.safe_load(open("params.yaml"))[script_path]

    main(
        source_bart_model_name=params["source_bart_model_name"],
        source_bart_weights_path=(
            Path(params["source_bart_weights_path"])
            if params.get("source_bart_weights_path")
            else None
        ),
        reference_bart_weights_path=(
            Path(params["reference_bart_weights_path"])
            if params.get("reference_bart_weights_path")
            else None
        ),
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
            if params.get("n_max_train_samples_per_epoch")
            else None
        ),
        attacker_runs_save_dir=Path(params["attacker_runs_save_dir"]),
        training_log_filename=params["training_log_filename"],
        params_to_save=params,
        target_label=params["target_label"],
    )
