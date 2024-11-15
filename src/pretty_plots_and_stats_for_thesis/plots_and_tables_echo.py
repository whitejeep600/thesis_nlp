from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from src.constants import MODEL_RESPONSE, ORIGINAL_SENTENCE, SIMILARITY


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
    train_xs = np.linspace(0, n_epochs, len(train_similarities))

    plt.plot(
        train_xs,
        train_similarities,
        label="Semsim over the training phase",
        color="blue",
    )
    plt.plot(
        n_epochs,
        eval_similarity,
        "go",
        label="Final average semsim on the validation set",
        color="orange",
    )
    plt.legend(fontsize=12, loc="lower center")
    plt.ylim(0, 1)
    plt.xlabel("Completion of the training phase", fontsize=14)
    plt.ylabel("Semsim", fontsize=14)
    plt.title("Semsim in echo training", fontsize=16)
    plt.savefig(plots_path / "semsim.png", dpi=300)


def _reformat_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["idx", "reward"])
    df = df.round({SIMILARITY: 2})
    df = df.rename(
        columns={SIMILARITY: "Semsim", ORIGINAL_SENTENCE: "Prompt", MODEL_RESPONSE: "Answer"}
    )
    return df


def main() -> None:
    echo_training_path = Path("runs/echo/run_0")
    generated_sentences_path = echo_training_path / "generated_sentences"
    train_path = generated_sentences_path / "train" / "epoch_0.csv"
    eval_path = generated_sentences_path / "eval" / "epoch_0.csv"

    plots_path = Path("plots/echo")
    plots_path.mkdir(exist_ok=True, parents=True)

    tables_path = Path("tables_for_thesis")

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    # _save_plot(eval_df, train_df, plots_path)

    train_to_save = train_df.iloc[list(range(0, 10, 2)), :]
    eval_to_save = eval_df.sample(n=5, random_state=0)
    train_to_save = _reformat_df(train_to_save)
    eval_to_save = _reformat_df(eval_to_save)

    train_to_save.to_csv(tables_path / "train_beginning.csv", index=False)
    eval_to_save.to_csv(tables_path / "eval_random.csv", index=False)


if __name__ == "__main__":
    main()
