from pathlib import Path

from src.constants import REWARD, SIMILARITY, TARGET_LABEL_PROB, TrainMode
from src.pretty_plots_and_stats_for_thesis.thesis_utils import (
    get_all_generations_dfs_for_experiment,
    plot_train_and_eval_metrics_together,
)


def main() -> None:
    run_path = Path("runs/attacker/run_3")
    plots_path = Path("plots/harmonic")

    plots_path.mkdir(exist_ok=True, parents=True)

    train_dfs = get_all_generations_dfs_for_experiment([run_path], TrainMode.train)
    eval_dfs = get_all_generations_dfs_for_experiment([run_path], TrainMode.eval)

    metrics = [REWARD, TARGET_LABEL_PROB, SIMILARITY]
    y_labels = ["Reward", "Fooling", "Semsim"]

    for metric, y_label in zip(metrics, y_labels):
        plot_train_and_eval_metrics_together(
            train_dfs=train_dfs,
            eval_dfs=eval_dfs,
            metric_name=metric,
            save_path=plots_path / f"{metric}.png",
            y_label=y_label,
        )


if __name__ == "__main__":
    main()
