from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from src.constants import MODEL_RESPONSE, ORIGINAL_SENTENCE, SIMILARITY
from src.pretty_plots_and_stats_for_thesis.thesis_utils import dump_dataframe_to_latex


def _save_plot(
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    plots_path: Path,
) -> None:
    eval_similarity = eval_df[SIMILARITY].mean()

    train_similarities = train_df[SIMILARITY].values.tolist()
    train_similarities = savgol_filter(
        train_similarities, window_length=128, polyorder=3, mode="mirror"
    )

    n_epochs = 1
    train_xs = np.linspace(0, 1, len(train_similarities))

    plt.tight_layout()
    plt.plot(
        train_xs,
        train_similarities,
        label="Semsim over the training phase",
        color="blue",
    )
    plt.plot(
        n_epochs,
        eval_similarity,
        "o",
        label="Final average semsim on the validation set",
        color="orange",
    )
    plt.legend(fontsize=12, loc="lower center")
    plt.ylim(0, 1)
    plt.xlabel("Completion of the training phase", fontsize=14)
    plt.ylabel("Semsim", fontsize=14)
    # plt.title("Semsim in echo training", fontsize=16)
    plt.savefig(plots_path / "semsim.png", dpi=1000, bbox_inches="tight")


def _reformat_df_for_thesis_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["idx", "reward"])
    df = df.round({SIMILARITY: 2})
    df = df.rename(
        columns={
            SIMILARITY: "Semsim",
            ORIGINAL_SENTENCE: "Prompt",
            MODEL_RESPONSE: "Answer",
        }
    )
    return df


def main() -> None:
    echo_training_path = Path("runs/echo/run_0")
    generated_sentences_path = echo_training_path / "generated_sentences"
    train_path = generated_sentences_path / "train" / "epoch_0.csv"
    eval_path = generated_sentences_path / "eval" / "epoch_0.csv"

    plots_path = Path("plots/echo")
    plots_path.mkdir(exist_ok=True, parents=True)

    tables_path = Path("tables_for_thesis") / "echo"
    tables_path.mkdir(exist_ok=True, parents=True)

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    _save_plot(eval_df, train_df, plots_path)

    train_to_save = train_df.iloc[list(range(0, 10, 2)), :]
    eval_to_save = eval_df.sample(n=5, random_state=0)
    train_to_save = _reformat_df_for_thesis_table(train_to_save)
    eval_to_save = _reformat_df_for_thesis_table(eval_to_save)

    target_train_tex_path = tables_path / "train_beginning.tex"
    dump_dataframe_to_latex(
        train_to_save,
        target_train_tex_path,
        column_format="|p{6cm}|p{6cm}|P{1.6cm}|",
        caption="Echo training: one answer each for the first 5 train samples"
        " in the training phrase.",
        resize_points=250,
        label="echo_train_beginning_samples",
    )

    target_eval_tex_path = tables_path / "eval_random.tex"
    dump_dataframe_to_latex(
        eval_to_save,
        target_eval_tex_path,
        column_format="|p{6cm}|p{6cm}|P{1.6cm}|",
        caption="Echo training: 5 randomly selected validation samples (after the training).",
        resize_points=250,
        label="echo_eval_random_samples",
    )


if __name__ == "__main__":
    main()
