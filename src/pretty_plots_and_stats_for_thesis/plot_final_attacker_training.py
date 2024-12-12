from pathlib import Path

import pandas as pd

from src.constants import (
    ID,
    PROMPT_ORIGINAL_TARGET_LABEL_PROB,
    REWARD,
    SIMILARITY,
    TARGET_LABEL_PROB,
    TrainMode,
)
from src.datasets.sst2_static_victim_retraining_dataset import (
    MAX_PROMPT_ORIGINAL_TARGET_LABEL_PROB,
    MIN_GENERATION_TARGET_LABEL_PROB,
    MIN_SEMSIM,
)
from src.pretty_plots_and_stats_for_thesis.thesis_utils import (
    dump_dataframe_to_latex,
    get_all_generations_dfs_for_experiment,
    plot_train_and_eval_metrics_together,
    reformat_examples_for_thesis_tables,
)


def _save_examples_for_comparison_to_run_13(target_tables_path: Path) -> None:
    random_examples_run_13_path = Path(
        "tables_for_thesis/run_13_first_exploits/random_example_ids.csv"
    )
    random_examples_idx = pd.read_csv(random_examples_run_13_path)["idx"].values

    run_15_final_path = Path("runs/attacker/run_15/generated_sentences/eval/epoch_39.csv")
    run_15_df = pd.read_csv(run_15_final_path)
    random_examples = (
        run_15_df[run_15_df["idx"].isin(random_examples_idx)]
        .groupby(ID)
        .sample(n=1, random_state=0)
    )

    NATURAL = "Natural"
    UNNATURAL = "Grammatical, not natural"
    UNGRAMMATICAL = "Ungrammatical"

    random_examples = random_examples.sort_values(
        by="idx", key=lambda idx: [random_examples_idx.tolist().index(x) for x in idx]
    )

    random_examples["Naturality"] = [
        UNGRAMMATICAL,
        NATURAL,
        NATURAL,
        NATURAL,
        NATURAL,
        UNNATURAL,
        NATURAL,
        UNNATURAL,
        NATURAL,
        NATURAL,
        UNGRAMMATICAL,
        UNGRAMMATICAL,
    ]

    random_examples = reformat_examples_for_thesis_tables(random_examples)

    column_format = "|p{5cm}|p{5cm}|P{1.6cm}|P{1.6cm}|P{1.8cm}|p{5cm}|"
    dump_dataframe_to_latex(
        random_examples.iloc[:, 1:],
        target_tables_path / "random_examples_compare_to_run_13.tex",
        column_format=column_format,
        resize_points=450,
        label="training_with_grammaticality_random_examples",
        caption="Training with grammaticality evaluation: paraphrases to compare.",
    )


def _plot_metrics_across_epochs(
    train_dfs: list[pd.DataFrame], eval_dfs: list[pd.DataFrame], plots_path: Path
) -> None:
    metrics = [REWARD, TARGET_LABEL_PROB, SIMILARITY]
    y_labels = ["Reward", "Fooling", "Semsim"]

    for metric, y_label in zip(metrics, y_labels):
        plot_train_and_eval_metrics_together(
            train_dfs=train_dfs,
            eval_dfs=eval_dfs,
            metric_name=metric,
            save_path=plots_path / f"{metric}.png",
            y_label=y_label,
            scatter_eval_values=False,
        )


def save_random_high_quality_examples(
    target_tables_path: Path,
    n_examples: int,
    eval_dfs: list[pd.DataFrame],
) -> None:
    successful_attack_epoch_dfs = [
        df[
            (df[PROMPT_ORIGINAL_TARGET_LABEL_PROB] < MAX_PROMPT_ORIGINAL_TARGET_LABEL_PROB)
            & (df[TARGET_LABEL_PROB] > MIN_GENERATION_TARGET_LABEL_PROB)
            & (df[SIMILARITY] > MIN_SEMSIM)
        ]
        for df in eval_dfs
    ]
    successful_attack_df = pd.concat(successful_attack_epoch_dfs, axis=0)

    successful_attack_df = successful_attack_df.groupby(ID).head(n=1)
    selection = successful_attack_df.sample(n=n_examples, random_state=0)

    selection = reformat_examples_for_thesis_tables(selection)

    column_format = "|p{5cm}|p{5cm}|P{1.6cm}|P{1.6cm}|P{1.8cm}|"
    dump_dataframe_to_latex(
        selection.iloc[:, 1:],
        target_tables_path / "random_successful_attacks.tex",
        column_format=column_format,
        resize_points=400,
        label="random_successful_attacks",
        caption="A random selection of successful adversarial examples.",
    )


def main() -> None:
    target_tables_path = Path("tables_for_thesis/run_14_15_after_adding_grammaticality")
    target_tables_path.mkdir(exist_ok=True, parents=True)

    plots_path = Path("plots/run_14_15_after_adding_grammaticality")
    plots_path.mkdir(exist_ok=True, parents=True)

    _save_examples_for_comparison_to_run_13(target_tables_path)

    run_paths = [
        Path("runs/attacker/run_14"),
        Path("runs/attacker/run_15"),
    ]

    train_dfs = get_all_generations_dfs_for_experiment(run_paths, TrainMode.train)
    eval_dfs = get_all_generations_dfs_for_experiment(run_paths, TrainMode.eval)

    save_random_high_quality_examples(target_tables_path, n_examples=12, eval_dfs=eval_dfs)
    _plot_metrics_across_epochs(train_dfs, eval_dfs, plots_path)


if __name__ == "__main__":
    main()
