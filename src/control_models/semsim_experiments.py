import json
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt
from seaborn import lineplot, scatterplot
from sentence_transformers import SentenceTransformer


class SentencePair:
    def __init__(self, sentence_0: str, sentence_1: str, score: float, annotation: str = ""):
        self.sentences = sentence_0, sentence_1
        self.human_score = score
        self.annotation = annotation


class SemsimEvaluator:
    def __init__(self, name):
        self.name = name

    def evaluate_pair(self, pair: SentencePair) -> float:
        raise NotImplementedError


class PureEmbeddingEvaluator(SemsimEvaluator):
    def __init__(self, model: SentenceTransformer):
        super().__init__("pure_embedding")
        self.model = model

    def evaluate_pair(self, pair: SentencePair) -> float:
        encoding_0 = self.model.encode(pair.sentences[0], convert_to_tensor=True)
        encoding_1 = self.model.encode(pair.sentences[1], convert_to_tensor=True)
        return torch.cosine_similarity(encoding_0, encoding_1, dim=0).item()


def main() -> None:
    pairs_path = Path("data/semsim_experiments.json")
    with open(pairs_path, "r") as f:
        pairs_json = json.load(f)
    pairs = [
        SentencePair(
            pair["sentence_0"], pair["sentence_1"], pair["score"], pair.get("annotation", "")
        )
        for pair in pairs_json
    ]

    embedder = SentenceTransformer("flax-sentence-embeddings/all_datasets_v4_MiniLM-L6")
    embedder.eval()

    for evaluator in [
        PureEmbeddingEvaluator(embedder),
    ]:
        model_scores = [evaluator.evaluate_pair(pair) for pair in pairs]
        human_scores = [pair.human_score for pair in pairs]
        annotations = [pair.annotation for pair in pairs]
        data_to_plot = pd.DataFrame(
            {"model_scores": model_scores, "human_scores": human_scores, "annotations": annotations}
        )
        scatterplot(
            data=data_to_plot,
            x="human_scores",
            y="model_scores",
            hue="annotations",
            palette="turbo",
            s=100,
        )
        lineplot(x=[0, 1], y=[0, 1], color="red")
        plt.gca().set_aspect("equal")
        plt.xlabel("Human scores", fontsize=30)
        plt.ylabel("Model scores", fontsize=30)
        plt.title(evaluator.name, fontsize=30)
        plt.show()
        plt.clf()


if __name__ == "__main__":
    main()
