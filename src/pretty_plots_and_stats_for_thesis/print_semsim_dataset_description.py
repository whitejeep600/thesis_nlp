import json
from pathlib import Path


def main() -> None:
    pairs_path = Path("data/semsim_experiments.json")
    plots_path = Path("plots/semsim_experiments")
    plots_path.mkdir(exist_ok=True, parents=True)

    with open(pairs_path, "r") as f:
        pairs_json = json.load(f)
    annotations = [pair["annotation"] for pair in pairs_json]
    annotation_to_count = {
        annotation: len([pair for pair in pairs_json if pair["annotation"] == annotation])
        for annotation in annotations
    }
    for annotation, count in annotation_to_count.items():
        print(f'{count} samples of type "{annotation}";')
    print(f"{len(pairs_json)} samples in total.")


if __name__ == "__main__":
    main()
