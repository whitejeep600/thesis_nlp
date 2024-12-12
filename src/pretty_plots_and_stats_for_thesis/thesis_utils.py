from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from src.constants import (
    MODEL_RESPONSE,
    ORIGINAL_SENTENCE,
    PROMPT_ORIGINAL_TARGET_LABEL_PROB,
    REWARD,
    SIMILARITY,
    TARGET_LABEL_PROB,
    TrainMode,
)


def epoch_df_sorting_key(path: Path):
    return int(path.stem.split("_")[-1])


def get_all_generations_dfs_for_experiment(
    run_paths: list[Path], mode: TrainMode
) -> list[pd.DataFrame]:
    """
    A single experiment may be contained in multiple runs, each being a continuation
    of the previous one.
    """
    run_generated_sentences_paths = [
        run_path / "generated_sentences" / mode.value for run_path in run_paths
    ]
    epoch_df_paths = [
        epoch_df_path
        for run_generated_sentences_path in run_generated_sentences_paths
        for epoch_df_path in sorted(
            list(run_generated_sentences_path.iterdir()), key=epoch_df_sorting_key
        )
    ]

    return [pd.read_csv(epoch_df_path) for epoch_df_path in epoch_df_paths]


def plot_train_and_eval_metrics_together(
    train_dfs: list[pd.DataFrame],
    eval_dfs: list[pd.DataFrame],
    metric_name: str,
    save_path: Path,
    y_lim: tuple[int, int] = (0, 1),
    x_label: str = "Epoch number",
    y_label: str = "",
    plot_title: str = "",
    smoothing_window_length: int = 1024,
    scatter_eval_values: bool = True,
) -> None:
    train_values = np.concatenate(
        [train_df[metric_name] for train_df in train_dfs],
        axis=0,
    )
    smoothed_train_values = savgol_filter(
        train_values, window_length=smoothing_window_length, polyorder=2, mode="mirror"
    )

    for i, eval_df in enumerate(eval_dfs):
        eval_df[metric_name].mean()
    averaged_eval_values = np.array([eval_df[metric_name].mean() for eval_df in eval_dfs])

    n_epochs = len(averaged_eval_values)

    eval_average_value_xs = list(range(1, n_epochs + 1))
    smoothed_train_value_xs = np.linspace(0, n_epochs, len(smoothed_train_values))

    plt.plot(
        smoothed_train_value_xs,
        smoothed_train_values,
        label="Training",
        color="blue",
    )
    plt.plot(
        eval_average_value_xs,
        averaged_eval_values,
        label="Validation",
        color="orange",
    )

    if scatter_eval_values:
        plt.scatter(range(1, n_epochs + 1), averaged_eval_values, color="orange", s=12)

    plt.ylim(y_lim)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.legend(fontsize=12, loc="lower right")
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.clf()


def plot_ratio_of_generations_containing_word_across_epochs(
    word: str,
    all_epoch_eval_dfs: list[pd.DataFrame],
    plots_path: Path,
) -> None:
    percentages = [
        df[MODEL_RESPONSE].str.contains(word).sum() / len(df) for df in all_epoch_eval_dfs
    ]

    xs = list(range(len(percentages)))

    plt.plot(
        xs,
        percentages,
        color="blue",
    )
    plt.ylim(0, 1)
    plt.xlabel("Epoch number", fontsize=12)
    plt.ylabel("Ratio", fontsize=12)
    plt.savefig(plots_path / f"{word}_percentages.png", bbox_inches="tight")


def reformat_examples_for_thesis_tables(examples: pd.DataFrame) -> pd.DataFrame:
    examples = examples.round(
        {SIMILARITY: 2, TARGET_LABEL_PROB: 2, REWARD: 2, PROMPT_ORIGINAL_TARGET_LABEL_PROB: 2}
    )
    examples = examples.rename(
        columns={
            SIMILARITY: "Semsim",
            ORIGINAL_SENTENCE: "Prompt",
            MODEL_RESPONSE: "Answer",
            TARGET_LABEL_PROB: "Fooling",
            REWARD: "Reward",
            PROMPT_ORIGINAL_TARGET_LABEL_PROB: "Victim's original prediction",
        }
    )
    columns = ["idx", "Prompt", "Answer", "Semsim", "Fooling", "Victim's original prediction"]
    for optional_column in "Exploit", "Naturality":
        if optional_column in examples.columns:
            columns.append(optional_column)
    examples = examples[columns]
    return examples


def dump_dataframe_to_latex(
    df: pd.DataFrame,
    save_path: Path,
    column_format: str,
    caption: str = "",
    resize_points: int | None = None,
    label: str | None = None,
) -> None:
    """
    Sample column format: "|p{6cm}|p{6cm}|P{1.6cm}|"
    """
    output = df.to_latex(
        None,
        index=False,
        escape=True,
        caption=caption,
        float_format="%.2f",
        bold_rows=True,
        column_format=column_format,
    )

    output = output.replace("\\\\", "\\\\ \\noalign{\\hrule height 1pt}")
    output = output.replace("\\toprule", "\\noalign{\\hrule height 1pt}")
    for rule in ["\\bottomrule", "\\midrule"]:
        output = output.replace(rule, "")
    for column in df.columns:
        output = output.replace(column, "\\textbf{" + column + "}")

    if label is not None:
        text_after_tabular_end = f"\\label{{table:{label}}}"
    else:
        text_after_tabular_end = ""

    if resize_points is not None:
        text_before_tabular_start = f"\\resizebox{{{resize_points}pt}}{{!}}{{"
        text_after_tabular_end += "}"
    else:
        text_before_tabular_start = ""

    output = output.replace("\\begin{tabular}", text_before_tabular_start + "\\begin{tabular}")
    output = output.replace("\\end{tabular}", "\\end{tabular}" + text_after_tabular_end)

    output = output.replace(
        "\\begin{table}", "\\begin{table} \\centering \\captionsetup{justification=centering}"
    )
    # output = output.replace("``", "\\textasciigrave\\textasciigrave")

    with open(save_path, "w") as f:
        f.write(output)
