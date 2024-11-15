from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from src.constants import MODEL_RESPONSE, ORIGINAL_SENTENCE, REWARD, SIMILARITY, TARGET_LABEL_PROB
from src.pretty_plots_and_stats_for_thesis.thesis_utils import epoch_df_sorting_key


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
    manual_not_examples = manual_not_examples.round(
        {SIMILARITY: 2, TARGET_LABEL_PROB: 2, REWARD: 2}
    )
    manual_not_examples = manual_not_examples.rename(
        columns={
            SIMILARITY: "Semsim",
            ORIGINAL_SENTENCE: "Prompt",
            MODEL_RESPONSE: "Answer",
            TARGET_LABEL_PROB: "Fooling",
            REWARD: "Reward",
        }
    )

    manual_not_examples.to_csv(tables_save_dir / "manual_not_examples.csv", index=False)

    random_lacking_examples = eval_df[eval_df[MODEL_RESPONSE].str.contains("lacking")].sample(
        n=5, random_state=0
    )

    all_epoch_dfs_paths = sorted(
        [epoch_df_path for epoch_df_path in all_run_eval_dfs_path.iterdir()],
        key=epoch_df_sorting_key,
    )

    all_epoch_dfs = [pd.read_csv(epoch_df_path) for epoch_df_path in all_epoch_dfs_paths]

    lacking_percentages = [
        df[MODEL_RESPONSE].str.contains("lacking").sum() / len(df) for df in all_epoch_dfs
    ]

    xs = list(range(len(lacking_percentages)))

    plt.plot(
        xs,
        lacking_percentages,
        color="blue",
    )
    plt.ylim(0, 1)
    plt.title('Ratio of generations containing the word "lacking"', fontsize=14)
    plt.xlabel("Epoch number", fontsize=12)
    plt.ylabel("Ratio", fontsize=12)
    plt.savefig(plots_path / "lacking_percentages.png")

    random_lacking_examples = random_lacking_examples.round(
        {SIMILARITY: 2, TARGET_LABEL_PROB: 2, REWARD: 2}
    )
    random_lacking_examples = random_lacking_examples.rename(
        columns={
            SIMILARITY: "Semsim",
            ORIGINAL_SENTENCE: "Prompt",
            MODEL_RESPONSE: "Answer",
            TARGET_LABEL_PROB: "Fooling",
            REWARD: "Reward",
        }
    )
    random_lacking_examples.to_csv(tables_save_dir / "random_lacking_examples.csv", index=False)


if __name__ == "__main__":
    main()
