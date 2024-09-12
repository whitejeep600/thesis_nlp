from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import INPUT_IDS, ORIGINAL_SENTENCE, TrainMode
from src.generative_bart import GenerativeBart
from src.utils import undo_batch_torch_repeat_interleave_2


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


class BatchProcessingResults:
    def __init__(self):
        pass


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        batch_input_ids has the shape (batch_size, sequence_length), with
        sequence_length being the length that the dataset tokenizer pads
        or truncates to.
        Three tensors of shapes (batch_size, 2, max_generation_length)
        and twice (batch_size, 2, max_generation_length), respectively, are returned.
        The first of these are the generated output ids, and the rest are the
        corresponding generation and reference probabilities (shorter because the first generated
        token is the start token, without generation probability). The first dimension
        corresponds to the index in the batch. Size 2 in dimension 2 comes
        from the fact that two generations are sampled per input sample.
        max_generation_length does not exceed the Trainer's max_len, but it can
        also be smaller if the generation was terminated before reaching max_len.
        This is done to save computation, if end tokens have been generated for each
        input sequence.

        """

        # The input tensor is repeated with repeat_interleave so that all generations
        # can be processed in parallel.
        batch_input_ids = torch.repeat_interleave(batch_input_ids, 2, dim=0).to(
            self.trained_model.device
        )

        # Decoded sequences are initialized to a [2 * batch_size, 1] tensor of start tokens.
        all_decoded_ids = (
            torch.Tensor([[self.trained_model.start_token] for _ in range(len(batch_input_ids))])
            .int()
            .to(self.trained_model.device)
        )

        all_generation_probabilities: list[torch.Tensor] = []
        all_reference_probabilities: list[torch.Tensor] = []

        for _ in range(self.max_len - 1):  # -1 from the initialization above
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

            if (next_ids == self.trained_model.stop_token).all():
                break

        generation_probs_tensor = torch.stack(all_generation_probabilities, dim=-1)
        reference_probs_tensor = torch.stack(all_reference_probabilities, dim=-1)

        return (
            undo_batch_torch_repeat_interleave_2(all_decoded_ids),
            undo_batch_torch_repeat_interleave_2(generation_probs_tensor),
            undo_batch_torch_repeat_interleave_2(reference_probs_tensor),
        )

    def process_batch(self, batch: dict) -> BatchProcessingResults:
        # calculate batch loss and backpropagate if needed
        decoded_ids, generation_probs, reference_probs = self.sample_two_sequences_per_sample(
            batch[INPUT_IDS]
        )
        batch_decoded_sequences = [
            self.trained_model.decode(decoded_ids[i]) for i in range(len(decoded_ids))
        ]
        batch_original_sequences = batch[ORIGINAL_SENTENCE]

        # sample_rewards_and_metrics = [
        #     self.metric_calculator.get_rewards_and_metrics(prompt, generations)
        #     for (prompt, generations) in zip(batch_original_sequences, batch_decoded_sequences)
        # ]

        # for each sample in the batch get a SampleProcessingResults object which
        # stores the prompt, and two Generation objects, named preferred and dispreferred.
        # each Generation has a MetricsAndRewards, also its text, generation probabilities
        # and reference probabilities.
        # during initialization (I think?) each SampleProcessingResult can calculate its loss.
        # Then BatchProcessingResults can do that too. It should be possible to backpropagate
        # and also store the batch loss for future purposes (plotting and stuff).
        # but also careful so that the Generations don't unnecessarily store info
        # on the gpu, god forbid any computation graphs
        return BatchProcessingResults()

    def iteration(self, mode: TrainMode) -> None:
        dataloader = self.dataloaders[mode]
        self.trained_model.set_mode(mode)

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
            batch_processing_results = self.process_batch(batch)
            epoch_all_batch_results.append(batch_processing_results)
            if mode == TrainMode.train and batch_no + 1 == n_batches_to_process:
                break

    def run_training(self) -> None:
        for _ in tqdm(range(self.n_epochs), desc="training...", position=0):
            self.iteration(TrainMode.train)
            with torch.no_grad():
                self.iteration(TrainMode.eval)
