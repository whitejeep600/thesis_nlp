from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from textattack.models.helpers import WordCNNForClassification
from textattack.models.tokenizers import GloveTokenizer
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import (
    INPUT_IDS,
    LABEL,
    LABEL_CODE_TO_NAME,
    LABEL_NAME_TO_CODE,
    ORIGIN,
    ORIGIN_GENERATED,
    ORIGIN_SAMPLED,
    SENTENCE,
    TrainMode,
)
from src.datasets.sst2_static_victim_retraining_dataset import SST2VictimRetrainingDataset
from src.utils import get_available_torch_devices, get_next_run_subdir_name

MAX_LABEL_SAMPLES_PER_TRAIN_EPOCH = 256

PREDICTED_LABEL = "predicted_label"


class VictimRetrainerSampleProcessingResult:
    def __init__(
        self,
        prediction_logits: torch.Tensor,
        true_label: int,
        sentence: str,
        origin: Literal["sampled", "generated"],
    ):
        self.prediction_logits = prediction_logits.detach().cpu()
        self.true_label = true_label
        self.sentence = sentence
        self.predicted_label = prediction_logits.argmax().item()
        self.origin = origin

    def to_dict(self):
        return {
            SENTENCE: self.sentence,
            LABEL: self.true_label,
            PREDICTED_LABEL: self.predicted_label,
            ORIGIN: self.origin,
        }


class VictimRetrainerBatchProcessingResult:
    def __init__(
        self,
        loss: torch.Tensor,
        sample_processing_results: list[VictimRetrainerSampleProcessingResult],
    ):
        self.loss = loss
        self.sample_processing_results = sample_processing_results

    def remove_tensors_required_for_loss_only(self) -> None:
        self.loss = self.loss.detach().cpu()


def _epoch_processing_results_to_dataframe(
    epoch_processing_results: list[VictimRetrainerBatchProcessingResult],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            sample_processing_result.to_dict()
            for batch_processing_result in epoch_processing_results
            for sample_processing_result in batch_processing_result.sample_processing_results
        ]
    )


def get_label_recall_for_all_epochs_and_origin(
    all_epoch_predictions: list[pd.DataFrame], label_code: int, origin: str | None = None
) -> list[float]:
    return [
        get_label_recall_for_single_epoch_and_origin(single_epoch_predictions, label_code, origin)
        for single_epoch_predictions in all_epoch_predictions
    ]


class VictimStaticRetrainer:
    def __init__(
        self,
        victim_model: WordCNNForClassification,
        original_dataset_dataloaders: dict[TrainMode, DataLoader],
        adversarial_example_dataloaders: dict[TrainMode, DataLoader],
        victim_optimizer: torch.optim.Optimizer,
        training_device: torch.device,
        n_label_batches_per_source_per_train_epoch: int,
        n_epochs: int,
        run_save_dir: Path,
        attacker_target_label_code: int,
    ):
        self.victim_model = victim_model
        self.original_dataset_dataloaders = original_dataset_dataloaders
        self.adversarial_example_dataloaders = adversarial_example_dataloaders
        self.victim_optimizer = victim_optimizer
        self.training_device = training_device
        self.n_label_batches_per_source_per_train_epoch = n_label_batches_per_source_per_train_epoch
        self.n_epochs = n_epochs
        self.run_save_dir = run_save_dir
        self.attacker_target_label_code = attacker_target_label_code

        self.loss_function = CrossEntropyLoss()

        self.original_eval_epoch_results: list[pd.DataFrame] = []
        self.adversarial_eval_epoch_results: list[pd.DataFrame] = []
        self.train_epoch_results: list[pd.DataFrame] = []

        self.summary_file_path = run_save_dir / "summary.txt"

        self.plots_dir = run_save_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True, parents=True)

        predictions_dir = run_save_dir / "predictions"
        self.train_predictions_dir = predictions_dir / "train"
        self.original_eval_predictions_dir = predictions_dir / "eval_original"
        self.adversarial_eval_predictions_dir = predictions_dir / "adversarial_original"

        for dir_ in [
            self.train_predictions_dir,
            self.original_eval_predictions_dir,
            self.adversarial_eval_predictions_dir,
            self.plots_dir,
        ]:
            dir_.mkdir(exist_ok=True, parents=True)

        self.attacker_target_label_name = LABEL_CODE_TO_NAME[attacker_target_label_code]
        self.attacker_non_target_label_name = LABEL_CODE_TO_NAME[1 - attacker_target_label_code]

    def process_batch(self, batch: dict[str, Any]) -> VictimRetrainerBatchProcessingResult:
        predictions = self.victim_model(batch[INPUT_IDS].to(self.training_device))
        labels = batch[LABEL].to(self.training_device)
        sentences = batch[SENTENCE]
        loss = self.loss_function(predictions, labels)
        origins = batch[ORIGIN]
        sample_processing_results = [
            VictimRetrainerSampleProcessingResult(
                prediction_logits=prediction,
                true_label=label.item(),
                sentence=sentence,
                origin=origin,
            )
            for (prediction, label, sentence, origin) in zip(
                predictions, labels, sentences, origins
            )
        ]
        return VictimRetrainerBatchProcessingResult(loss, sample_processing_results)

    def train_iteration(self) -> pd.DataFrame:
        self.victim_model.train()
        n_total_batches = 2 * self.n_label_batches_per_source_per_train_epoch
        alternating_batch_generator = chain.from_iterable(
            zip(
                self.adversarial_example_dataloaders[TrainMode.train],
                self.original_dataset_dataloaders[TrainMode.train],
            )
        )
        all_batch_processing_results: list[VictimRetrainerBatchProcessingResult] = []

        for batch_no, batch in tqdm(
            enumerate(alternating_batch_generator),
            total=n_total_batches,
            desc="train iteration",
            leave=False,
            position=1,
        ):
            processing_result = self.process_batch(batch)
            self.victim_optimizer.zero_grad()
            processing_result.loss.backward()
            self.victim_optimizer.step()
            processing_result.remove_tensors_required_for_loss_only()
            all_batch_processing_results.append(processing_result)
            if batch_no + 1 == n_total_batches:
                break

        return _epoch_processing_results_to_dataframe(all_batch_processing_results)

    def eval_iteration(self, dataloader: DataLoader, pbar_description: str) -> pd.DataFrame:
        self.victim_model.eval()
        all_batch_processing_results: list[VictimRetrainerBatchProcessingResult] = []
        with torch.no_grad():
            for batch_no, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=pbar_description,
                leave=False,
                position=1,
            ):
                processing_result = self.process_batch(batch)
                processing_result.remove_tensors_required_for_loss_only()
                all_batch_processing_results.append(processing_result)

        return _epoch_processing_results_to_dataframe(all_batch_processing_results)

    def original_dataset_eval_iteration(self) -> pd.DataFrame:
        return self.eval_iteration(
            dataloader=self.original_dataset_dataloaders[TrainMode.eval],
            pbar_description="eval iteration 0 out of 2, original dataset samples",
        )

    def adversarial_eval_iteration(self) -> pd.DataFrame:
        return self.eval_iteration(
            dataloader=self.adversarial_example_dataloaders[TrainMode.eval],
            pbar_description="eval iteration 1 out of 2, adversarial examples",
        )

    @staticmethod
    def save_mode_predictions(
        mode_predictions_all_epochs: list[pd.DataFrame], mode_predictions_save_dir: Path
    ) -> None:
        for i, epoch_predictions in enumerate(mode_predictions_all_epochs):
            path = mode_predictions_save_dir / f"epoch_{i}.csv"
            epoch_predictions.to_csv(path, index=False)

    def save_predictions(self) -> None:
        self.save_mode_predictions(self.train_epoch_results, self.train_predictions_dir)
        self.save_mode_predictions(
            self.original_eval_epoch_results, self.original_eval_predictions_dir
        )
        self.save_mode_predictions(
            self.adversarial_eval_epoch_results, self.adversarial_eval_predictions_dir
        )

    def save_victim_checkpoint(self):
        self.victim_model.save_pretrained(self.run_save_dir)

    @staticmethod
    def save_recall_plot(target_path: Path, include_legend: bool = True) -> None:
        if include_legend:
            plt.legend(loc="best")
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Recall", fontsize=14)
        plt.ylim(0, 1)
        plt.savefig(target_path, dpi=1000)
        plt.clf()

    def write_to_summary_file(self, text: str) -> None:
        with open(self.summary_file_path, "a") as f:
            f.write(f"{text}\n")

    def save_train_plot(self) -> None:
        target_path = self.plots_dir / "train.png"
        train_adversarial_non_target_label_recall = get_label_recall_for_all_epochs_and_origin(
            self.train_epoch_results,
            label_code=1 - self.attacker_target_label_code,
            origin=ORIGIN_GENERATED,
        )
        train_original_target_label_recall = get_label_recall_for_all_epochs_and_origin(
            self.train_epoch_results,
            label_code=self.attacker_target_label_code,
            origin=ORIGIN_SAMPLED,
        )

        plot_xs = list(range(self.n_epochs))
        plt.plot(
            plot_xs,
            train_adversarial_non_target_label_recall,
            label=f"Adversarial samples ({self.attacker_non_target_label_name})",
            color="orange",
        )
        plt.plot(
            plot_xs,
            train_original_target_label_recall,
            label=f"Original samples ({self.attacker_target_label_name})",
            color="blue",
        )

        self.save_recall_plot(target_path)

        first_train_adversarial_non_target_label_recall = train_adversarial_non_target_label_recall[
            1
        ]
        last_train_adversarial_non_target_label_recall = train_adversarial_non_target_label_recall[
            -1
        ]
        first_train_original_target_label_recall = train_original_target_label_recall[1]
        last_train_original_target_label_recall = train_original_target_label_recall[-1]
        self.write_to_summary_file(
            f"First train {self.attacker_non_target_label_name} label recall (generated samples):"
            f" {first_train_adversarial_non_target_label_recall}, last:"
            f" {last_train_original_target_label_recall};"
            f" first {self.attacker_target_label_name} recall (sampled from the original dataset):"
            f" {first_train_original_target_label_recall}, last:"
            f" {last_train_adversarial_non_target_label_recall}"
        )

    def save_adversarial_eval_plot(self) -> None:
        target_path = self.plots_dir / "adversarial_eval.png"
        eval_adversarial_non_target_label_recall = get_label_recall_for_all_epochs_and_origin(
            self.adversarial_eval_epoch_results,
            label_code=1 - self.attacker_target_label_code,
            origin=ORIGIN_GENERATED,
        )
        plot_xs = list(range(self.n_epochs))
        plt.plot(
            plot_xs,
            eval_adversarial_non_target_label_recall,
            color="orange",
        )

        self.save_recall_plot(target_path, include_legend=False)

        first_eval_adversarial_non_target_label_recall = eval_adversarial_non_target_label_recall[1]
        last_eval_adversarial_non_target_label_recall = eval_adversarial_non_target_label_recall[-1]
        self.write_to_summary_file(
            f"First eval {self.attacker_non_target_label_name} label recall (generated samples):"
            f" {first_eval_adversarial_non_target_label_recall}, last:"
            f" {last_eval_adversarial_non_target_label_recall},"
        )

    def save_original_eval_plot(self) -> None:
        target_path = self.plots_dir / "original_eval.png"
        eval_original_target_label_recall = get_label_recall_for_all_epochs_and_origin(
            self.original_eval_epoch_results,
            label_code=self.attacker_target_label_code,
            origin=ORIGIN_SAMPLED,
        )
        eval_original_non_target_label_recall = get_label_recall_for_all_epochs_and_origin(
            self.original_eval_epoch_results,
            label_code=1 - self.attacker_target_label_code,
            origin=ORIGIN_SAMPLED,
        )

        plot_xs = list(range(self.n_epochs))
        plt.plot(
            plot_xs,
            eval_original_target_label_recall,
            label=f"{self.attacker_target_label_name.capitalize()} samples",
            color="blue",
        )
        plt.plot(
            plot_xs,
            eval_original_non_target_label_recall,
            label=f"{self.attacker_non_target_label_name.capitalize()} samples",
            color="orange",
        )

        self.save_recall_plot(target_path)

        first_eval_original_non_target_label_recall = eval_original_non_target_label_recall[1]
        last_eval_original_non_target_label_recall = eval_original_non_target_label_recall[-1]
        first_eval_original_target_label_recall = eval_original_target_label_recall[1]
        last_eval_original_target_label_recall = eval_original_target_label_recall[-1]

        self.write_to_summary_file(
            f"First eval {self.attacker_non_target_label_name} label recall"
            f" (sampled from the original dataset): {first_eval_original_non_target_label_recall},"
            f" last: {last_eval_original_non_target_label_recall};"
            f" first eval {self.attacker_target_label_name} recall"
            f" (sampled from the original dataset): {first_eval_original_target_label_recall},"
            f" last: {last_eval_original_target_label_recall}"
        )

    def save_plots(self) -> None:
        self.save_train_plot()
        self.save_adversarial_eval_plot()
        self.save_original_eval_plot()

    def run_training(self):
        for _ in tqdm(range(self.n_epochs), desc="training...", position=0):
            train_epoch_result = self.train_iteration()
            original_eval_epoch_result = self.original_dataset_eval_iteration()
            adversarial_eval_epoch_result = self.adversarial_eval_iteration()

            self.train_epoch_results.append(train_epoch_result)
            self.original_eval_epoch_results.append(original_eval_epoch_result)
            self.adversarial_eval_epoch_results.append(adversarial_eval_epoch_result)

        self.save_predictions()
        self.save_victim_checkpoint()
        self.save_plots()


def get_label_recall_for_single_epoch_and_origin(
    epoch_predictions: pd.DataFrame, label_code: int, origin: str | None = None
) -> float:
    if origin is not None:
        epoch_predictions = epoch_predictions[epoch_predictions[ORIGIN] == origin]
    all_samples_with_the_label = epoch_predictions[epoch_predictions[LABEL] == label_code]
    correctly_predicted_samples_with_the_label = all_samples_with_the_label[
        all_samples_with_the_label[PREDICTED_LABEL] == label_code
    ]
    return round(
        len(correctly_predicted_samples_with_the_label) / len(all_samples_with_the_label), 2
    )


def _prepare_original_dataset_dataloaders(
    original_dataset_paths: dict[TrainMode, Path],
    victim_tokenizer: GloveTokenizer,
    min_len: int,
    max_len: int,
    attacker_target_label_code: int,
    batch_size: int,
) -> dict[TrainMode, DataLoader]:
    original_datasets = {
        TrainMode.train: SST2VictimRetrainingDataset.from_original_dataset_csv_path(
            original_dataset_paths[TrainMode.train],
            victim_tokenizer,
            max_len,
            min_len,
            label_to_keep=attacker_target_label_code,
        ),
        TrainMode.eval: SST2VictimRetrainingDataset.from_original_dataset_csv_path(
            original_dataset_paths[TrainMode.eval], victim_tokenizer, max_len, min_len
        ),
    }
    original_dataloaders = {
        mode: DataLoader(original_datasets[mode], batch_size=batch_size, shuffle=True)
        for mode in original_datasets.keys()
    }
    return original_dataloaders


def _prepare_adversarial_example_dataloaders(
    attacker_run_dirs: list[Path],
    victim_tokenizer: GloveTokenizer,
    attacker_target_label_code: int,
    batch_size: int,
) -> dict[TrainMode, DataLoader]:
    adversarial_example_datasets = {
        mode: SST2VictimRetrainingDataset.from_attacker_training_save_paths(
            attacker_run_dirs, victim_tokenizer, mode, attacker_target_label_code
        )
        for mode in [TrainMode.train, TrainMode.eval]
    }
    adversarial_example_dataloaders = {
        mode: DataLoader(adversarial_example_datasets[mode], batch_size=batch_size, shuffle=True)
        for mode in adversarial_example_datasets.keys()
    }
    return adversarial_example_dataloaders


def main(
    static_victim_retraining_runs_save_dir: Path,
    attacker_run_dirs: list[Path],
    original_train_split_path: Path,
    original_eval_split_path: Path,
    min_len: int,
    max_len: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    attacker_target_label: Literal["positive", "negative"],
) -> None:
    """
    It is not immediately obvious what data to retrain (and evaluate) the model on.
    We want the model to gain robustness to the successful adversarial examples,
    but not at the expense of performance on the data it was originally trained on.
    Additionally, balance between the ground truth labels must be maintained.

    It was decided to take the same number k of samples of each label during each
    training epoch. All the samples of the target label (the label the attacker was
    trained to fake) are taken from the _original_ training dataset (sst2 in this case).
    All the samples of the opposite label are successful adversarial examples produced
    by the attacker.
    k is chosen as the minimum of:
      - a constant "max_samples" set to 256 (we want to run the evaluation stage fairly often),
      - available_original_target_label_samples,
      - available_adversarial_examples.
    There should be more available samples than max_samples; thus, k samples are randomly
    selected from each source each epoch, and the selected samples may be different between
    the epochs.
    Batches from both sources are taken alternately during each epoch.
    We don't want to have too many too similar adversarial examples in the dataset, so
    we limit the number of paraphrases of a single sample from the original dataset
    to MAX_EXAMPLES_PER_SAMPLE_ID, which is set to 5.

    As for the evaluation, metrics are reported separately for both sources - the original
    dataset and the adversarial examples. No balance really needs to be maintained.
    Additionally, separate metrics are reported for both classes on the original evaluation
    set (not on the adversarial examples set, because there is only one ground truth label
    there). The original evaluation set is filtered to only include samples of appropriate
    length - this should the same length limitations as were set for the attacker training.
    """

    training_device = get_available_torch_devices()[0]
    attacker_target_label_code = LABEL_NAME_TO_CODE[attacker_target_label]
    original_dataset_paths = {
        TrainMode.train: original_train_split_path,
        TrainMode.eval: original_eval_split_path,
    }
    victim = WordCNNForClassification.from_pretrained("cnn-sst2")
    victim.to(training_device)

    original_dataset_dataloaders = _prepare_original_dataset_dataloaders(
        original_dataset_paths=original_dataset_paths,
        victim_tokenizer=victim.tokenizer,
        min_len=min_len,
        max_len=max_len,
        attacker_target_label_code=attacker_target_label_code,
        batch_size=batch_size,
    )
    adversarial_example_dataloaders = _prepare_adversarial_example_dataloaders(
        attacker_run_dirs=attacker_run_dirs,
        victim_tokenizer=victim.tokenizer,
        attacker_target_label_code=attacker_target_label_code,
        batch_size=batch_size,
    )

    victim_optimizer = SGD(victim.parameters(), lr=lr)

    n_label_batches_per_source_per_train_epoch = min(
        len(original_dataset_dataloaders[TrainMode.train]),
        len(adversarial_example_dataloaders[TrainMode.train]),
        MAX_LABEL_SAMPLES_PER_TRAIN_EPOCH // batch_size,
    )

    static_victim_retraining_runs_save_dir.mkdir(exist_ok=True, parents=True)
    run_save_dir = static_victim_retraining_runs_save_dir / get_next_run_subdir_name(
        static_victim_retraining_runs_save_dir
    )

    retrainer = VictimStaticRetrainer(
        victim_model=victim,
        original_dataset_dataloaders=original_dataset_dataloaders,
        adversarial_example_dataloaders=adversarial_example_dataloaders,
        victim_optimizer=victim_optimizer,
        training_device=training_device,
        n_label_batches_per_source_per_train_epoch=n_label_batches_per_source_per_train_epoch,
        n_epochs=n_epochs,
        run_save_dir=run_save_dir,
        attacker_target_label_code=attacker_target_label_code,
    )
    retrainer.run_training()


if __name__ == "__main__":
    script_path = "src.training_scripts.retrain_victim_statically"
    params = yaml.safe_load(open("params.yaml"))[script_path]

    main(
        static_victim_retraining_runs_save_dir=Path(
            params["static_victim_retraining_runs_save_dir"]
        ),
        attacker_run_dirs=[
            Path(attacker_run_dir) for attacker_run_dir in (params["attacker_run_dirs"])
        ],
        original_train_split_path=Path(params["original_train_split_path"]),
        original_eval_split_path=Path(params["original_eval_split_path"]),
        max_len=int(params["max_len"]),
        min_len=int(params["min_len"]),
        batch_size=int(params["batch_size"]),
        n_epochs=int(params["n_epochs"]),
        lr=float(params["lr"]),
        attacker_target_label=params["target_label"],
    )
