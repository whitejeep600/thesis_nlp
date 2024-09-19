from __future__ import annotations

from pathlib import Path

import torch
from torch.nn.functional import logsigmoid
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import INPUT_IDS, ORIGINAL_SENTENCE, TrainMode
from src.generative_bart import GenerativeBart
from src.utils import (
    get_generation_length_until_first_stop_token,
    sequence_logprob,
    undo_batch_torch_repeat_interleave_2,
)


class RewardAndMetrics:
    def __init__(self, reward: float, metrics: dict[str, float]):
        self.reward = reward
        self.metrics = metrics


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

        log_prob = sequence_logprob(generation_probs)
        reference_log_prob = sequence_logprob(reference_probs)

        log_ratio = log_prob - reference_log_prob

        self.log_ratio = log_ratio


class SampleProcessingResults:
    def __init__(
        self,
        prompt: str,
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

        self.loss = -1 * logsigmoid(
            dpo_beta * (self.preferred.log_ratio - self.dispreferred.log_ratio)
        )


class BatchProcessingResults:
    def __init__(self, batch_sample_processing_results: list[SampleProcessingResults]):
        self.batch_sample_processing_results = batch_sample_processing_results
        self.loss = torch.stack(
            [processing_results.loss for processing_results in batch_sample_processing_results]
        ).mean()


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
        n_max_train_batches_per_epoch: int | None = None,
    ):
        """
        The term "reference model" is taken from the original DPO paper ("Direct Preference
        Optimization: Your Language Model is Secretly a Reward Model", Rafailov et al. 2023).
        It is the model meant to control that the trained model's policy does not deviate too
        much from a model that we know generates coherent text. Thus the training is stabilized.
        In this repository, and following common practice, this is just snapshot of the model
        being tuned, taken before the beginning of the DPO training.

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

        self.reference_model.set_mode(TrainMode.eval)

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
                str(
                    self.trained_model.decode(
                        torch.unflatten(
                            sample_generations[0].generated_token_ids, dim=0, sizes=[1, -1]
                        )
                    )[0]
                ),
                str(
                    self.trained_model.decode(
                        torch.unflatten(
                            sample_generations[1].generated_token_ids, dim=0, sizes=[1, -1]
                        )
                    )[0]
                ),
            )
            for sample_generations in batch_raw_generations
        ]
        batch_original_sequences = batch[ORIGINAL_SENTENCE]

        batch_sample_rewards_and_metrics = [
            self.metric_calculator.get_rewards_and_metrics(prompt, generations)
            for (prompt, generations) in zip(batch_original_sequences, batch_decoded_sequences)
        ]
        batch_sample_processing_results = [
            SampleProcessingResults(
                prompt=prompt,
                sample_generation_texts=sample_generation_texts,
                sample_rewards_and_metrics=sample_rewards,
                sample_raw_generations=sample_raw_generations,
                dpo_beta=self.beta,
            )
            for (
                prompt,
                sample_generation_texts,
                sample_rewards,
                sample_raw_generations,
            ) in zip(
                batch_original_sequences,
                batch_decoded_sequences,
                batch_sample_rewards_and_metrics,
                batch_raw_generations,
            )
        ]

        batch_results = BatchProcessingResults(batch_sample_processing_results)
        if mode == TrainMode.train:
            self.trained_model_optimizer.zero_grad()
            batch_results.loss.backward()
            self.trained_model_optimizer.step()

        return batch_results

    def iteration(self, mode: TrainMode) -> None:
        dataloader = self.dataloaders[mode]
        self.trained_model.set_mode(mode)
        torch.set_grad_enabled(mode == mode.train)

        n_batches_to_process = (
            self.n_max_train_batches_per_epoch
            if self.n_max_train_batches_per_epoch and mode == TrainMode.train
            else len(dataloader)
        )

        epoch_all_batch_results: list[BatchProcessingResults] = []

        for batch_no, batch in tqdm(
            enumerate(dataloader),
            total=n_batches_to_process,
            desc=f"{mode} iteration",
            leave=False,
            position=1,
        ):
            batch_processing_results = self.process_batch(batch, mode)
            epoch_all_batch_results.append(batch_processing_results)
            if mode == TrainMode.train and batch_no + 1 == n_batches_to_process:
                break

    def run_training(self) -> None:
        for _ in tqdm(range(self.n_epochs), desc="training...", position=0):
            self.iteration(TrainMode.train)
            with torch.no_grad():
                self.iteration(TrainMode.eval)
