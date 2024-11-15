from pathlib import Path

from src.constants import REWARD, SIMILARITY, TARGET_LABEL_PROB, TrainMode
from src.pretty_plots_and_stats_for_thesis.thesis_utils import (
    _get_all_generations_dfs_for_experiment,
    plot_train_and_eval_metrics_together,
)


def main() -> None:
    run_path = Path("runs/attacker/run_1")
    plots_path = Path("plots/base_training")

    plots_path.mkdir(exist_ok=True, parents=True)

    train_dfs = _get_all_generations_dfs_for_experiment([run_path], TrainMode.train)
    eval_dfs = _get_all_generations_dfs_for_experiment([run_path], TrainMode.eval)

    metrics = [REWARD, TARGET_LABEL_PROB, SIMILARITY]
    y_labels = ["Reward", "Fooling", "Semsim"]
    plot_titles = [
        "Rewards in the base training",
        "Fooling in the base training",
        "Semsim in the base training",
    ]

    for metric, y_label, plot_title in zip(metrics, y_labels, plot_titles):
        plot_train_and_eval_metrics_together(
            train_dfs=train_dfs,
            eval_dfs=eval_dfs,
            metric_name=metric,
            save_path=plots_path / f"{metric}.png",
            y_label=y_label,
            plot_title=plot_title,
            smoothing_window_length=512,
        )


if __name__ == "__main__":
    main()