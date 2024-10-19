import json
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from seaborn import lineplot, scatterplot
from sentence_transformers import SentenceTransformer
from sklearn.metrics import r2_score

from src.control_models.semantic_similarity_evaluators import DistilbertEntailmentEvaluator
from src.utils import get_length_difference_scores


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
        super().__init__("Pure embedding")
        self.model = model

    def evaluate_pair(self, pair: SentencePair) -> float:
        encoding_0 = self.model.encode(pair.sentences[0], convert_to_tensor=True)
        encoding_1 = self.model.encode(pair.sentences[1], convert_to_tensor=True)
        return torch.cosine_similarity(encoding_0, encoding_1, dim=0).item()


class EmbeddingAndHardEntailmentEvaluator(SemsimEvaluator):
    def __init__(
        self, model: SentenceTransformer, entailment_evaluator: DistilbertEntailmentEvaluator
    ):
        super().__init__("Embedding and hard entailment")
        self.embedder = model
        self.entailment_model = entailment_evaluator

    def evaluate_pair(self, pair: SentencePair) -> float:
        entailment_label = self.entailment_model.get_binary_entailment_for_text_pairs(
            [(pair.sentences[0], pair.sentences[1])]
        )[0]
        if not entailment_label:
            return 0
        encoding_0 = self.embedder.encode(pair.sentences[0], convert_to_tensor=True)
        encoding_1 = self.embedder.encode(pair.sentences[1], convert_to_tensor=True)
        return torch.cosine_similarity(encoding_0, encoding_1, dim=0).item()


class EmbeddingAndLengthAndHardEntailmentEvaluator(SemsimEvaluator):
    def __init__(
        self, model: SentenceTransformer, entailment_evaluator: DistilbertEntailmentEvaluator
    ):
        super().__init__("Embedding and length and hard entailment")
        self.embedder = model
        self.entailment_model = entailment_evaluator

    def evaluate_pair(self, pair: SentencePair) -> float:
        entailment_label = self.entailment_model.get_binary_entailment_for_text_pairs(
            [(pair.sentences[0], pair.sentences[1])]
        )[0]
        if not entailment_label:
            return 0
        encoding_0 = self.embedder.encode(pair.sentences[0], convert_to_tensor=True)
        encoding_1 = self.embedder.encode(pair.sentences[1], convert_to_tensor=True)
        length_difference_score = get_length_difference_scores(
            pair.sentences[0], [pair.sentences[1]]
        )[0]
        return (
            torch.cosine_similarity(encoding_0, encoding_1, dim=0).item() * length_difference_score
        )


def main() -> None:
    pairs_path = Path("data/semsim_experiments.json")
    plots_path = Path("plots")
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

    entailment_evaluator = DistilbertEntailmentEvaluator(torch.device("cpu"))

    for evaluator in [
        EmbeddingAndLengthAndHardEntailmentEvaluator(embedder, entailment_evaluator),
        EmbeddingAndHardEntailmentEvaluator(embedder, entailment_evaluator),
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
            s=50,
        )
        lineplot(x=[0, 1], y=[0, 1], color="red")
        spearman = round(spearmanr(human_scores, model_scores).statistic, 2)
        pearson = round(pearsonr(human_scores, model_scores).statistic, 2)
        r_2 = round(r2_score(human_scores, model_scores), 2)
        text = f"$N={len(human_scores)}$\n$spearman={spearman}$\n$pearson={pearson}$\n$R^2={r_2}$"
        plt.text(
            1.1,
            1,
            text,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="none", edgecolor="black", pad=5),
        )
        plt.gca().set_aspect("equal")
        plt.xlabel("Human scores", fontsize=15)
        plt.ylabel("Model scores", fontsize=15)
        plt.title(evaluator.name, fontsize=15)
        plt.legend(fontsize=5)
        plt.savefig(plots_path / f"{evaluator.name}.png", dpi=400)
        plt.clf()


if __name__ == "__main__":
    main()
