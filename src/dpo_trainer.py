from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy import mean
from torch.nn.functional import logsigmoid
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import (
    ID,
    INPUT_IDS,
    MODEL_RESPONSE,
    ORIGINAL_SENTENCE,
    PLOTTING_MOVING_AVERAGE_WINDOW_LENGTH,
    REWARD,
    TrainMode,
)
from src.generative_bart import GenerativeBart
from src.utils import (
    _moving_average_with_left_side_padding,
    get_generation_length_until_first_stop_token,
    sequence_log_prob,
    undo_batch_torch_repeat_interleave_2,
)


class RewardAndMetrics:
    def __init__(self, reward: float, metrics: dict[str, float]):
        self.reward = reward
        self.metrics = metrics

    def to_str(self) -> str:
        return f"reward: {self.reward}, metrics: {self.metrics}\n"


class DPORewardAndMetricCalculator:
    def __init__(self):
        pass

    def get_rewards_and_metrics(
        self,
        prompt: str,
        generations: tuple[str, str],
    ) -> tuple[RewardAndMetrics, RewardAndMetrics]:
        """
        Abstract method whose implementation depends on the goals of the DPO training.
        It accepts a prompt and two responses to the prompt. It returns the
        RewardAndMetrics for the responses.
        """
        raise NotImplementedError


class RawGeneration:
    def __init__(
        self,
        generated_token_ids: torch.Tensor,
        generation_probs: torch.Tensor,
        reference_probs: torch.Tensor,
        stop_token_id: int,
    ):
        real_length = get_generation_length_until_first_stop_token(
            generated_token_ids, stop_token_id
        )
        self.generated_token_ids = generated_token_ids[:real_length]
        self.generation_probs = generation_probs[:real_length]
        self.reference_probs = reference_probs[:real_length]


class Generation:
    def __init__(
        self,
        generation_text: str,
        reward_and_metrics: RewardAndMetrics,
        generation_probs: torch.Tensor,
        reference_probs: torch.Tensor,
    ):
        self.generation_text = generation_text
        self.reward_and_metrics = reward_and_metrics
        self.generation_probs = generation_probs
        self.reference_probs = reference_probs

        log_prob = sequence_log_prob(generation_probs)
        reference_log_prob = sequence_log_prob(reference_probs)

        log_ratio = log_prob - reference_log_prob

        self.log_ratio = log_ratio


class SampleProcessingResults:
    def __init__(
        self,
        prompt: str,
        sample_id: int,
        sample_generation_texts: tuple[str, str],
        sample_rewards_and_metrics: tuple[RewardAndMetrics, RewardAndMetrics],
        sample_raw_generations: tuple[RawGeneration, RawGeneration],
        dpo_beta: float,
    ):
        if sample_rewards_and_metrics[0].reward > sample_rewards_and_metrics[1].reward:
            preferred_index, dispreferred_index = 0, 1
        else:
            preferred_index, dispreferred_index = 1, 0

        self.preferred = Generation(
            generation_text=sample_generation_texts[preferred_index],
            reward_and_metrics=sample_rewards_and_metrics[preferred_index],
            generation_probs=sample_raw_generations[preferred_index].generation_probs,
            reference_probs=sample_raw_generations[preferred_index].reference_probs,
        )
        self.dispreferred = Generation(
            generation_text=sample_generation_texts[dispreferred_index],
            reward_and_metrics=sample_rewards_and_metrics[dispreferred_index],
            generation_probs=sample_raw_generations[dispreferred_index].generation_probs,
            reference_probs=sample_raw_generations[dispreferred_index].reference_probs,
        )

        self.prompt = prompt
        self.sample_id = sample_id

        self.loss = -1 * logsigmoid(
            dpo_beta * (self.preferred.log_ratio - self.dispreferred.log_ratio)
        )


class BatchProcessingResults:
    def __init__(self, batch_sample_processing_results: list[SampleProcessingResults]):
        self.batch_sample_processing_results = batch_sample_processing_results
        self.loss = torch.stack(
            [processing_results.loss for processing_results in batch_sample_processing_results]
        ).mean()


def _calculate_mean_metrics(metrics: list[RewardAndMetrics]) -> RewardAndMetrics:
    return RewardAndMetrics(
        reward=mean([m.reward for m in metrics]),
        metrics={key: mean([m.metrics[key] for m in metrics]) for key in metrics[0].metrics.keys()},
    )


def _calculate_moving_average_metrics(metrics: list[RewardAndMetrics]) -> list[RewardAndMetrics]:
    metric_names = list(metrics[0].metrics.keys())
    reward_moving_averages = _moving_average_with_left_side_padding(
        [m.reward for m in metrics], window_length=PLOTTING_MOVING_AVERAGE_WINDOW_LENGTH
    )
    metrics_moving_averages = {
        key: _moving_average_with_left_side_padding(
            [m.metrics[key] for m in metrics], window_length=PLOTTING_MOVING_AVERAGE_WINDOW_LENGTH
        )
        for key in metric_names
    }
    return [
        RewardAndMetrics(
            reward=reward_moving_averages[i],
            metrics={
                key: metrics_moving_averages[key][i] for key in metrics_moving_averages.keys()
            },
        )
        for i in range(len(reward_moving_averages))
    ]


def _get_all_metrics_from_epoch_processing_results(
    epoch_processing_results: list[BatchProcessingResults],
) -> list[RewardAndMetrics]:
    return [
        generation.reward_and_metrics
        for batch_results in epoch_processing_results
        for sample_results in batch_results.batch_sample_processing_results
        for generation in [sample_results.preferred, sample_results.dispreferred]
    ]


def _epoch_processing_results_to_dataframe(
    all_batch_results: list[BatchProcessingResults],
) -> pd.DataFrame:
    return [
        {
            ID: sample_results.sample_id,
            ORIGINAL_SENTENCE: sample_results.prompt,
            MODEL_RESPONSE: generation.generation_text,
            REWARD: generation.reward_and_metrics.reward,
            **generation.reward_and_metrics.metrics,
        }
        for single_batch_results in all_batch_results
        for sample_results in single_batch_results.batch_sample_processing_results
        for generation in [sample_results.preferred, sample_results.dispreferred]
    ]


class DPOTrainer:
    def __init__(
        self,
        trained_model: GenerativeBart,
        reference_model: GenerativeBart,
        dataloaders: dict[TrainMode, DataLoader],
        metric_calculator: DPORewardAndMetricCalculator,
        trained_model_optimizer: Optimizer,
        run_save_dir: Path,
        general_training_log_path: Path,
        n_epochs: int,
        max_len: int,
        beta: float,
        temperature: float,
        lr: float,
        params_to_save: dict[str, Any],
        n_max_train_batches_per_epoch: int | None = None,
    ):
        """
        The term "reference model" is taken from the original DPO paper ("Direct Preference
        Optimization: Your Language Model is Secretly a Reward Model", Rafailov et al. 2023).
        It is the model meant to control that the trained model's policy does not deviate too
        much from a model that we know generates coherent text. Thus the training is stabilized.
        In this repository, and following common practice, this is just snapshot of the model
        being tuned, taken before the beginning of the DPO training.

        During the training, various results are saved. This includes:
            - epoch model checkpoints (the last one, and also the best one in terms of mean
                eval metrics),
            - the responses generated by the model, both at train and eval stages, and the
                corresponding metrics,
            - plots of the rewards and metrics over the epochs. For the eval stage, that is mean
                metrics for every epoch. For the train stage, to track changes over the epoch
                while preserving readability of the plot, that is the moving average.
            - training parameters and latest git commit id at training time, for reproducibility,
            - a short summary of the training.

        """
        self.trained_model = trained_model
        self.reference_model = reference_model
        self.dataloaders = dataloaders
        self.metric_calculator = metric_calculator
        self.trained_model_optimizer = trained_model_optimizer
        self.run_save_dir = run_save_dir
        self.general_training_log_path = general_training_log_path
        self.n_epochs = n_epochs
        self.max_len = max_len
        self.beta = beta
        self.temperature = temperature
        self.lr = lr
        self.n_max_train_batches_per_epoch = n_max_train_batches_per_epoch

        self.all_epoch_processing_results: dict[TrainMode, list[list[SampleProcessingResults]]] = {
            TrainMode.train: [],
            TrainMode.eval: [],
        }
        self.mean_epoch_eval_metrics: list[RewardAndMetrics] = []
        self.moving_average_epoch_train_metrics: list[list[RewardAndMetrics]] = []
        self.params_to_save = params_to_save
        self.train_start_time = time.time()

        self.checkpoints_dir = self.run_save_dir / "checkpoints"
        self.generated_sentences_dirs = {
            TrainMode.train: run_save_dir / "generated_sentences" / TrainMode.train.value,
            TrainMode.eval: run_save_dir / "generated_sentences" / TrainMode.eval.value,
        }
        self.plots_dir = run_save_dir / "plots"
        for dir_ in [
            self.run_save_dir,
            self.checkpoints_dir,
            self.generated_sentences_dirs[TrainMode.train],
            self.generated_sentences_dirs[TrainMode.eval],
            self.plots_dir,
        ]:
            dir_.mkdir(exist_ok=True, parents=True)

        self.reference_model.set_mode(TrainMode.eval)

    def save_model_checkpoint(self, save_filename: str) -> None:
        torch.save(self.trained_model.bert.state_dict(), self.checkpoints_dir / save_filename)

    def save_model_responses_and_metrics(
        self, epoch_processing_results: list[BatchProcessingResults], mode: TrainMode, epoch_no: int
    ) -> None:
        df = pd.DataFrame(_epoch_processing_results_to_dataframe(epoch_processing_results))
        save_path = self.generated_sentences_dirs[mode] / f"epoch_{epoch_no}.csv"
        df.to_csv(save_path, index=False)

    def save_plots(self) -> None:
        eval_metrics_and_rewards = {
            REWARD: [epoch_metrics.reward for epoch_metrics in self.mean_epoch_eval_metrics],
            **{
                metric_name: [
                    epoch_metrics.metrics[metric_name]
                    for epoch_metrics in self.mean_epoch_eval_metrics
                ]
                for metric_name in self.mean_epoch_eval_metrics[0].metrics.keys()
            },
        }
        eval_xs = list(range(self.n_epochs))

        train_metrics_and_rewards = {
            REWARD: [
                batch_metrics.reward
                for epoch_metrics in self.moving_average_epoch_train_metrics
                for batch_metrics in epoch_metrics
            ],
            **{
                metric_name: [
                    batch_metrics.metrics[metric_name]
                    for epoch_metrics in self.moving_average_epoch_train_metrics
                    for batch_metrics in epoch_metrics
                ]
                for metric_name in self.moving_average_epoch_train_metrics[0][0].metrics.keys()
            },
        }
        train_xs = np.linspace(0, self.n_epochs - 1, len(train_metrics_and_rewards[REWARD]))

        for reward_or_metric_name in eval_metrics_and_rewards.keys():
            plt.plot(
                train_xs,
                train_metrics_and_rewards[reward_or_metric_name],
                label="train",
                color="blue",
            )
            plt.plot(
                eval_xs,
                eval_metrics_and_rewards[reward_or_metric_name],
                label="eval",
                color="green",
            )
            plt.savefig(self.plots_dir / f"{reward_or_metric_name}.png", dpi=420)
            plt.clf()

    def sample_two_sequences_per_sample(
        self,
        batch_input_ids: torch.Tensor,
    ) -> list[tuple[RawGeneration, RawGeneration]]:
        """
        batch_input_ids has the shape (batch_size, sequence_length), with
        sequence_length being the length that the dataset tokenizer pads
        or truncates to.

        """

        # The input tensor is repeated with repeat_interleave so that all generations
        # can be processed in parallel.
        batch_input_ids = torch.repeat_interleave(batch_input_ids, 2, dim=0).to(
            self.trained_model.device
        )

        # Decoded sequences are initialized to a [2 * batch_size, 1] tensor of start tokens.
        all_decoded_ids = (
            torch.Tensor([[self.trained_model.start_token_id] for _ in range(len(batch_input_ids))])
            .int()
            .to(self.trained_model.device)
        )

        all_generation_probabilities: list[torch.Tensor] = []
        all_reference_probabilities: list[torch.Tensor] = []

        for _ in range(self.max_len - 1):  # -1 because of the initialization above
            new_logits = self.trained_model.bert(
                input_ids=batch_input_ids,
                decoder_input_ids=all_decoded_ids,
            ).logits[:, -1, :]

            with torch.no_grad():
                new_reference_logits = (
                    self.reference_model.bert(
                        input_ids=batch_input_ids.to(self.reference_model.device),
                        decoder_input_ids=all_decoded_ids.to(self.reference_model.device),
                    )
                    .logits[:, -1, :]
                    .to(self.trained_model.device)
                )

            new_probabilities = torch.softmax(new_logits / self.temperature, dim=-1)
            new_reference_probabilities = torch.softmax(
                new_reference_logits / self.temperature, dim=-1
            )

            next_ids = torch.multinomial(new_probabilities, 1)

            next_ids_generation_probabilities = new_probabilities[
                range(len(batch_input_ids)),
                next_ids.flatten(),
            ]

            next_ids_reference_probabilities = new_reference_probabilities[
                range(len(batch_input_ids)),
                next_ids.flatten(),
            ]

            all_decoded_ids = torch.cat((all_decoded_ids, next_ids), dim=-1)
            all_generation_probabilities.append(next_ids_generation_probabilities)
            all_reference_probabilities.append(next_ids_reference_probabilities)

            if (next_ids == self.trained_model.stop_token_id).all():
                break

        generation_probs_tensor = torch.stack(all_generation_probabilities, dim=-1)
        reference_probs_tensor = torch.stack(all_reference_probabilities, dim=-1)

        return [
            (
                RawGeneration(
                    sample_decoded_ids[0],
                    sample_generation_probs[0],
                    sample_reference_probs[0],
                    self.reference_model.stop_token_id,
                ),
                RawGeneration(
                    sample_decoded_ids[1],
                    sample_generation_probs[1],
                    sample_reference_probs[1],
                    self.reference_model.stop_token_id,
                ),
            )
            for (sample_decoded_ids, sample_generation_probs, sample_reference_probs) in zip(
                undo_batch_torch_repeat_interleave_2(all_decoded_ids),
                undo_batch_torch_repeat_interleave_2(generation_probs_tensor),
                undo_batch_torch_repeat_interleave_2(reference_probs_tensor),
            )
        ]

    def process_batch(self, batch: dict, mode: TrainMode) -> BatchProcessingResults:
        batch_raw_generations = self.sample_two_sequences_per_sample(batch[INPUT_IDS])
        batch_decoded_sequences = [
            (
                self.trained_model.decode_single_sequence(
                    sample_generations[0].generated_token_ids
                ),
                self.trained_model.decode_single_sequence(
                    sample_generations[1].generated_token_ids
                ),
            )
            for sample_generations in batch_raw_generations
        ]
        batch_original_sequences = batch[ORIGINAL_SENTENCE]
        batch_ids = batch[ID]

        batch_rewards_and_metrics = [
            self.metric_calculator.get_rewards_and_metrics(prompt, generations)
            for (prompt, generations) in zip(batch_original_sequences, batch_decoded_sequences)
        ]
        batch_sample_processing_results = [
            SampleProcessingResults(
                prompt=sample_original_sequence,
                sample_id=sample_id,
                sample_generation_texts=sample_generation_texts,
                sample_rewards_and_metrics=sample_rewards,
                sample_raw_generations=sample_raw_generations,
                dpo_beta=self.beta,
            )
            for (
                sample_original_sequence,
                sample_id,
                sample_generation_texts,
                sample_rewards,
                sample_raw_generations,
            ) in zip(
                batch_original_sequences,
                batch_ids,
                batch_decoded_sequences,
                batch_rewards_and_metrics,
                batch_raw_generations,
            )
        ]

        batch_results = BatchProcessingResults(batch_sample_processing_results)
        if mode == TrainMode.train:
            self.trained_model_optimizer.zero_grad()
            batch_results.loss.backward()
            self.trained_model_optimizer.step()

        return batch_results

    def iteration(self, mode: TrainMode) -> list[BatchProcessingResults]:
        dataloader = self.dataloaders[mode]
        self.trained_model.set_mode(mode)
        torch.set_grad_enabled(mode == TrainMode.train)

        n_batches_to_process = (
            self.n_max_train_batches_per_epoch
            if self.n_max_train_batches_per_epoch and mode == TrainMode.train
            else len(dataloader)
        )

        epoch_processing_results: list[BatchProcessingResults] = []

        for batch_no, batch in tqdm(
            enumerate(dataloader),
            total=n_batches_to_process,
            desc=f"{mode} iteration",
            leave=False,
            position=1,
        ):
            batch_processing_results = self.process_batch(batch, mode)
            epoch_processing_results.append(batch_processing_results)
            if mode == TrainMode.train and batch_no + 1 == n_batches_to_process:
                break

        return epoch_processing_results

    def get_training_summary(self):
        time_now = time.time()
        time_elapsed = time.gmtime(time_now - self.train_start_time)

        n_epochs_elapsed = len(self.mean_epoch_eval_metrics)
        best_epoch_no = np.argmax([metrics.reward for metrics in self.mean_epoch_eval_metrics])
        best_epoch_stats = self.mean_epoch_eval_metrics[best_epoch_no]

        return (
            f"Training time: {time.strftime('%H:%M:%S', time_elapsed)},"
            f" number of epochs elapsed: {n_epochs_elapsed}, best stats"
            f" for epoch {best_epoch_no}, as follows: {best_epoch_stats}"
        )

    def save_call_parameters_and_summary_to_global_log(self, training_summary: str) -> None:
        info_to_save = self.params_to_save
        info_to_save.update({"summary": training_summary})

        with open(self.general_training_log_path, "a") as f:
            f.write(f"{json.dumps(self.params_to_save, indent=2)}\n\n")

    def run_training(self) -> None:
        best_mean_eval_reward = -np.inf
        for epoch_no in tqdm(range(self.n_epochs), desc="training...", position=0):
            train_epoch_results = self.iteration(TrainMode.train)
            eval_epoch_results = self.iteration(TrainMode.eval)

            all_eval_metrics = _get_all_metrics_from_epoch_processing_results(eval_epoch_results)
            mean_eval_metrics = _calculate_mean_metrics(all_eval_metrics)

            print(f"Epoch no {epoch_no}, mean eval metrics: {mean_eval_metrics.to_str()}")
            self.mean_epoch_eval_metrics.append(mean_eval_metrics)

            if mean_eval_metrics.reward > best_mean_eval_reward:
                best_mean_eval_reward = mean_eval_metrics.reward
                self.save_model_checkpoint(save_filename="best.pt")

            all_train_metrics = _get_all_metrics_from_epoch_processing_results(train_epoch_results)
            moving_average_train_metrics = _calculate_moving_average_metrics(all_train_metrics)
            self.moving_average_epoch_train_metrics.append(moving_average_train_metrics)

            self.save_model_responses_and_metrics(train_epoch_results, TrainMode.train, epoch_no)
            self.save_model_responses_and_metrics(eval_epoch_results, TrainMode.eval, epoch_no)

        self.save_model_checkpoint(save_filename="last.pt")
        training_summary = self.get_training_summary()
        print(f"{training_summary}\n")
        self.save_call_parameters_and_summary_to_global_log(training_summary)
        self.save_plots()
