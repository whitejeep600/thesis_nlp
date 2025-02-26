# Requires conda, poetry. This will set up a Python environment with all the required packages
# (except GPU support for torch, see below).
conda env create -f environment.yaml
conda activate thesis_nlp
poetry install

# Note that after running poetry install, torch may installed in the cpu-only version.
# To get GPU support too, an installation command appropriate for a given machine might have
# to be run, for example:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# See https://pytorch.org/get-started/locally/ for further information.

# Enable static typechecking with mypy
mypy --install-types

# Local copy for adjustable parameters like file paths. For this reason the
# params.yaml file is ignored by Git.
cp default_params.yaml params.yaml

# The main dataset used for the experiment. It is a publicly available dataset and has not
# been included directly in the repository to save memory.
python -m src.download_sst2
