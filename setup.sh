# Requires conda, poetry

conda env create -f environment.yaml
conda activate thesis_nlp
poetry install

# Note that after running poetry install, torch is installed in the cpu-only version.
# To get GPU support too, an installation command appropriate for a given machine needs
# to be run, for example:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# See https://pytorch.org/get-started/locally/ for further information.

mypy --install-types

cp default_params.yaml params.yaml  # Local copy for adjustable parameters like file paths
