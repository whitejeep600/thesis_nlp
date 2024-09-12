import re
from pathlib import Path

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
    # elif torch.backends.mps.is_available():
    #    return [torch.device("mps")]
    else:
        return [torch.device("cpu")]


def undo_batch_torch_repeat_interleave_2(t: torch.Tensor) -> torch.Tensor:
    dim_target_size = t.shape[0] // 2
    repetition_0 = t[[2 * i for i in range(dim_target_size)], :]
    repetition_1 = t[[2 * i + 1 for i in range(dim_target_size)], :]
    return torch.stack((repetition_0, repetition_1), dim=1)
