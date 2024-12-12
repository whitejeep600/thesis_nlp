from pathlib import Path

import pandas as pd

from src.constants import MODEL_RESPONSE, ORIGINAL_SENTENCE, REWARD, SIMILARITY, TARGET_LABEL_PROB
from src.pretty_plots_and_stats_for_thesis.thesis_utils import (
    dump_dataframe_to_latex,
    epoch_df_sorting_key,
    plot_ratio_of_generations_containing_word_across_epochs,
)


def _reformat_df_for_thesis_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.round({SIMILARITY: 2, TARGET_LABEL_PROB: 2, REWARD: 2})
    df = df.rename(
        columns={
            SIMILARITY: "Semsim",
            ORIGINAL_SENTENCE: "Prompt",
            MODEL_RESPONSE: "Answer",
            TARGET_LABEL_PROB: "Fooling",
            REWARD: "Reward",
        }
    )

    return df


def main() -> None:
    all_run_eval_dfs_path = Path("runs/attacker/run_3/generated_sentences/eval/")
    eval_df_path = all_run_eval_dfs_path / "epoch_31.csv"
    tables_save_dir = Path("tables_for_thesis/run_3_label_flip")
    plots_path = Path("plots/run_3_label_flip")
    plots_path.mkdir(exist_ok=True, parents=True)

    tables_save_dir.mkdir(exist_ok=True, parents=True)
    eval_df = pd.read_csv(eval_df_path)

    manual_not_examples = (
        eval_df[
            eval_df[MODEL_RESPONSE].str.contains("soderbergh")
            | eval_df[MODEL_RESPONSE].str.contains("does not quite")
            | eval_df[MODEL_RESPONSE].str.contains("about following your dreams")
            | eval_df[MODEL_RESPONSE].str.contains("mull over")
        ]
        .groupby("idx")
        .sample(n=1)
    )
    manual_not_examples = _reformat_df_for_thesis_table(manual_not_examples)

    column_format = "|P{1cm}|p{5cm}|p{5cm}|P{1.6cm}|P{1.6cm}|P{1.6cm}|"

    dump_dataframe_to_latex(
        manual_not_examples,
        tables_save_dir / "manual_not_examples.tex",
        column_format=column_format,
        caption="Label flipping: manually selected examples of negations.",
        label="label_flip_manual_not",
        resize_points=350,
    )

    random_lacking_examples = eval_df[eval_df[MODEL_RESPONSE].str.contains("lacking")].sample(
        n=5, random_state=0
    )

    all_epoch_dfs_paths = sorted(
        [epoch_df_path for epoch_df_path in all_run_eval_dfs_path.iterdir()],
        key=epoch_df_sorting_key,
    )

    all_epoch_dfs = [pd.read_csv(epoch_df_path) for epoch_df_path in all_epoch_dfs_paths]
    plot_ratio_of_generations_containing_word_across_epochs(
        word="lacking", all_epoch_eval_dfs=all_epoch_dfs, plots_path=plots_path
    )

    random_lacking_examples = _reformat_df_for_thesis_table(random_lacking_examples)

    dump_dataframe_to_latex(
        random_lacking_examples,
        tables_save_dir / "random_lacking_examples.tex",
        column_format=column_format,
        caption="Label flipping: random examples with the word ''lacking''.",
        label="label_flip_random_lacking",
        resize_points=350,
    )


if __name__ == "__main__":
    main()
