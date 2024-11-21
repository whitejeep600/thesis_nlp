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
    _get_all_generations_dfs_for_experiment,
    plot_train_and_eval_metrics_together,
    reformat_examples,
)


def _save_examples_for_comparison_to_run_13(target_tables_path: Path) -> None:
    random_examples_run_13_path = Path(
        "tables_for_thesis/run_13_first_exploits/random_examples.csv"
    )
    random_examples_idx = pd.read_csv(random_examples_run_13_path)["idx"].values

    run_15_final_path = Path("runs/attacker/run_15/generated_sentences/eval/epoch_39.csv")
    run_15_df = pd.read_csv(run_15_final_path)
    random_examples = (
        run_15_df[run_15_df["idx"].isin(random_examples_idx)]
        .groupby(ID)
        .sample(n=1, random_state=0)
    )

    random_examples = reformat_examples(random_examples)
    random_examples.to_csv(
        target_tables_path / "random_examples_compare_to_run_13.csv", index=False
    )


def _plot_stuff(
    train_dfs: list[pd.DataFrame], eval_dfs: list[pd.DataFrame], plots_path: Path
) -> None:
    metrics = [REWARD, TARGET_LABEL_PROB, SIMILARITY]
    y_labels = ["Reward", "Fooling", "Semsim"]
    plot_titles = [
        "Rewards in the training after adding grammaticality evaluation",
        "Fooling in the training after adding grammaticality evaluation",
        "Semsim in the training after adding grammaticality evaluation",
    ]

    for metric, y_label, plot_title in zip(metrics, y_labels, plot_titles):
        plot_train_and_eval_metrics_together(
            train_dfs=train_dfs,
            eval_dfs=eval_dfs,
            metric_name=metric,
            save_path=plots_path / f"{metric}.png",
            y_label=y_label,
            plot_title=plot_title,
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

    selection = reformat_examples(selection)

    selection.to_csv(target_tables_path / "random_successful_attacks.csv", index=False)


def main() -> None:
    target_tables_path = Path("tables_for_thesis/run_14_15_after_adding_grammaticality")
    target_tables_path.mkdir(exist_ok=True, parents=True)

    plots_path = Path("plots/run_14_15_after_adding_grammaticality")
    plots_path.mkdir(exist_ok=True, parents=True)

    # _save_examples_for_comparison_to_run_13(target_tables_path)

    run_14_path = Path("runs/attacker/run_14")
    run_15_path = Path("runs/attacker/run_15")
    run_paths = [
        run_14_path,
        run_15_path,
    ]

    _get_all_generations_dfs_for_experiment(run_paths, TrainMode.train)
    eval_dfs = _get_all_generations_dfs_for_experiment(run_paths, TrainMode.eval)

    # _plot_stuff(train_dfs, eval_dfs, plots_path)

    save_random_high_quality_examples(target_tables_path, n_examples=12, eval_dfs=eval_dfs)

    # run for longer?


if __name__ == "__main__":
    main()
