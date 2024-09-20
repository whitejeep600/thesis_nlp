from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import torch


def get_next_run_subdir_name(run_save_dir: Path) -> str:
    """
    run_save_dir is a directory which keeps the training history for a given
    experiment. Subsequent trainings are saved in run_0, run_1, ... subdirectories.
    This util finds the next available subdirectory name.
    """
    subdir_regex = re.compile("run_([0-9]+)")

    existing_run_numbers: set[int] = set()
    for run_subdir in run_save_dir.iterdir():
        subdir_name = run_subdir.name
        match = subdir_regex.match(subdir_name)
        if match is not None:
            run_no = match.groups()[0]
            existing_run_numbers.add(int(run_no))

    next_no = max(existing_run_numbers) + 1 if existing_run_numbers else 0
    return f"run_{next_no}"


def get_available_torch_devices() -> list[torch.device]:
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        return [torch.device("cpu")]


def undo_batch_torch_repeat_interleave_2(t: torch.Tensor) -> torch.Tensor:
    dim_target_size = t.shape[0] // 2
    repetition_0 = t[[2 * i for i in range(dim_target_size)], :]
    repetition_1 = t[[2 * i + 1 for i in range(dim_target_size)], :]
    return torch.stack((repetition_0, repetition_1), dim=1)


def sequence_log_prob(token_probabilities: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.log(token_probabilities)).reshape(1)


def get_generation_length_until_first_stop_token(
    token_ids: torch.Tensor, stop_token_id: int
) -> int:
    for i in range(len(token_ids)):
        if token_ids[i] == stop_token_id:
            return i
    return len(token_ids)


def _moving_average_with_left_side_padding(xs: list[float], window_length: int) -> list[float]:
    """
    For the i-th index, calculate the average of the previous window_length elements
    including the i-th one. These elements are replaced with zeros at the beginning of the
    array, where they don't exist (that's what's meant by "left-side" padding), which means
    the resulting array has the same length as the input.
    """
    padding = [0 for _ in range(window_length)]
    xs_array = np.array(padding + xs)
    x_cumsum = np.cumsum(xs_array)
    moving_sums = x_cumsum[window_length:] - x_cumsum[:-window_length]
    denominators = np.array(
        list(range(1, window_length + 1)) + [window_length for _ in range(len(xs) - window_length)]
    )
    return (moving_sums / denominators).tolist()


def get_command_output(command: str, arguments: list[str]) -> str:
    return (
        subprocess.run([command, *arguments], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()[1:-1]  # removing the quotes around the output
    )


def get_current_git_commit_id() -> str:
    return get_command_output("git", ["log", '--format="%H"', "-n 1"])
