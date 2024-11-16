import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bert_score import score
from dotenv import load_dotenv
from groq import Groq
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from seaborn import lineplot, scatterplot
from sentence_transformers import SentenceTransformer
from sklearn.metrics import r2_score
from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from src.constants import INPUT_IDS
from src.control_models.semantic_similarity_evaluators import DistilbertEntailmentEvaluator
from src.utils import get_length_difference_scores


class SentencePair:
    def __init__(
        self, sentence_0: str, sentence_1: str, score: float, id_: int, annotation: str = ""
    ):
        self.sentences = sentence_0, sentence_1
        self.human_score = score
        self.annotation = annotation
        self.id = id_


class SemsimEvaluator:
    def __init__(self, name):
        self.name = name

    def evaluate_pair(self, pair: SentencePair) -> float:
        raise NotImplementedError


class PureEmbeddingEvaluator(SemsimEvaluator):
    def __init__(self, model: SentenceTransformer):
        super().__init__("embedding similarity")
        self.model = model

    def evaluate_pair(self, pair: SentencePair) -> float:
        encoding_0 = self.model.encode(pair.sentences[0], convert_to_tensor=True)
        encoding_1 = self.model.encode(pair.sentences[1], convert_to_tensor=True)
        return max(torch.cosine_similarity(encoding_0, encoding_1, dim=0).item(), 0)


class DistilbertEntailmentModelEvaluator(SemsimEvaluator):
    """
    Has to be this one because t5 won't return the probs
    """

    def __init__(
        self, model: SentenceTransformer, entailment_evaluator: DistilbertEntailmentEvaluator
    ):
        super().__init__("entailment probability")
        self.embedder = model
        self.entailment_model = entailment_evaluator

    def evaluate_pair(self, pair: SentencePair) -> float:
        return self.entailment_model.get_entailment_probs_for_text_pairs(
            [(pair.sentences[0], pair.sentences[1])]
        )[0]


class T5EntailmentHardLabelAndSentenceEmbeddingAndLengthEvaluator(SemsimEvaluator):
    def __init__(self, embedder: SentenceTransformer):
        super().__init__("\n heuristic (embedding, entailment label and length)")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.entailment_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.embedder = embedder

    def evaluate_pair(self, pair: SentencePair) -> float:
        input_ids = self.tokenizer(
            f"mnli premise: {pair.sentences[0]}. hypothesis: {pair.sentences[1]}",
            return_tensors="pt",
        ).input_ids

        outputs = self.entailment_model.generate(input_ids=input_ids)
        decoded = self.tokenizer.decode(outputs[0])
        entailment_score = 1 if "entailment" in decoded else 0.5 if "neutral" in decoded else 0

        encoding_0 = self.embedder.encode(pair.sentences[0], convert_to_tensor=True)
        encoding_1 = self.embedder.encode(pair.sentences[1], convert_to_tensor=True)
        embedder_score = torch.cosine_similarity(encoding_0, encoding_1, dim=0).item()
        length_difference_score = get_length_difference_scores(
            pair.sentences[0], [pair.sentences[1]]
        )[0]
        return embedder_score * entailment_score * length_difference_score


class BertScoreEvaluator(SemsimEvaluator):
    def __init__(self):
        super().__init__("BertScore")

    def evaluate_pair(self, pair: SentencePair) -> float:
        _, _, bert_score = score([pair.sentences[0]], [pair.sentences[1]], lang="en")
        return bert_score.item()


class ReconstructionLossEvaluator(SemsimEvaluator):
    def __init__(self):
        super().__init__("reconstruction loss analogue")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        return self.tokenizer(sentence, truncation=True, return_tensors="pt")[INPUT_IDS]

    def evaluate_pair(self, pair: SentencePair) -> float:
        tokenized_input = self.encode_sentence(pair.sentences[0])
        tokenized_output = self.encode_sentence(pair.sentences[1])

        # BART tokenizer tokenizes each sequence to start with a start token and end with an
        # end token, but for correct calculation of output probabilities, it requires the
        # output to start with a sequence of: a stop token in position 0 and a start token in
        # position 1. This can be verified by running self.model.generate(tokenized_input) -
        # the output should be the same as the tokenized input, except for starting with the
        # [stop_token, start_token] sequence.

        stop_token = self.tokenizer.encode("")[1]  # empty string is encoded as [start, stop]
        tokenized_output = torch.cat([torch.IntTensor([[stop_token]]), tokenized_output], dim=1)

        generation_probs: list[float] = []

        for output_token_number in range(2, len(tokenized_output[0])):
            new_logits = self.model(
                input_ids=tokenized_input,
                decoder_input_ids=tokenized_output[:, :output_token_number],
            ).logits[0, -1, :]
            new_probabilities = torch.softmax(new_logits, dim=0)
            output_token_probability = new_probabilities[
                tokenized_output[0][output_token_number]
            ].item()
            generation_probs.append(output_token_probability)

        sequence_generation_likelihood = np.log(np.array(generation_probs).prod())
        return sequence_generation_likelihood


class LLMEvaluator(SemsimEvaluator):
    def __init__(self):
        super().__init__("LLM")
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    @staticmethod
    def get_user_role_content(pair: SentencePair) -> str:
        return (
            f"Assign a numerical score to the following sequence pair, based on their semantic"
            f" similarity:\n\n"
            f"Sequence 1: {pair.sentences[0]}\nSequence 2: {pair.sentences[1]}\n\n"
            f"The score is between 0 and 1."
            f" All pairs with some degree of contradiction are evaluated between 0 and 0.2."
            f" The more contradictory the pair is, the lower the score. Direct contradiction is"
            f" evaluated at 0. If the sequences in the pair are unrelated or non-contradictory,"
            f" or at least one of them is nonsensical, then they are evaluated at about 0.2."
            f" All pairs with some degree of agreement are evaluated between 0.2 and 1."
            f" The higher the agreement, the higher the score. A pair of identical sequences is"
            f" evaluated at 1. A pair representing a very distant paraphrase"
            f" is evaluated at about 0.4."
            f" Pay attention to the fact that contradiction may also take the form of"
            f" negation or the usage of word antonyms.\n"
            f"Return the numerical score only, without additional text."
        )


    def evaluate_pair(self, pair: SentencePair) -> float:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": self.get_user_role_content(pair),
                }
            ],
            model="llama3-8b-8192",
        )
        return float(chat_completion.choices[0].message.content)


def main() -> None:
    load_dotenv()
    pairs_path = Path("data/semsim_experiments.json")
    plots_path = Path("plots/semsim_experiments")
    plots_path.mkdir(exist_ok=True, parents=True)

    with open(pairs_path, "r") as f:
        pairs_json = json.load(f)
    pairs = [
        SentencePair(
            pair["sentence_0"],
            pair["sentence_1"],
            pair["score"],
            pair["id"],
            pair.get("annotation", ""),
        )
        for pair in pairs_json
    ]

    embedder = SentenceTransformer("flax-sentence-embeddings/all_datasets_v4_MiniLM-L6")
    embedder.eval()

    entailment_evaluator = DistilbertEntailmentEvaluator(torch.device("cpu"))

    for evaluator in tqdm(
        [
            LLMEvaluator(),
            ReconstructionLossEvaluator(),
            BertScoreEvaluator(),
            DistilbertEntailmentModelEvaluator(embedder, entailment_evaluator),
            T5EntailmentHardLabelAndSentenceEmbeddingAndLengthEvaluator(embedder),
            PureEmbeddingEvaluator(embedder),
        ]
    ):
        model_scores = [evaluator.evaluate_pair(pair) for pair in pairs]
        human_scores = [pair.human_score for pair in pairs]
        annotations = [pair.annotation for pair in pairs]
        ids = [pair.id for pair in pairs]
        data_to_plot = pd.DataFrame(
            {
                "id": ids,
                "model_score": model_scores,
                "human_score": human_scores,
                "annotation": annotations,
            }
        )
        figure = plt.figure()
        scatterplot(
            data=data_to_plot,
            x="human_score",
            y="model_score",
            hue="annotation",
            palette="turbo",
            s=50,
        )
        for _, row_data in data_to_plot.iterrows():
            plt.text(
                x=row_data["human_score"] + 0.02,
                y=row_data["model_score"],
                s=row_data["id"],
                fontsize=5,
            )
        lineplot(x=[0, 1], y=[0, 1], color="red")
        spearman = round(spearmanr(human_scores, model_scores).statistic, 2)
        r_2 = round(r2_score(human_scores, model_scores), 2)
        text = (f"$N={len(human_scores)}$\n"
                f"$spearman={spearman}$\n"
                f"$R^2={r_2}$")
        plt.text(
            1.1,
            1.03,
            text,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="none", edgecolor="black", pad=5),
        )
        plt.gca().set_aspect("equal")
        plt.xlabel("Human score", fontsize=15)
        plt.ylabel("Model score", fontsize=15)
        plt.title(f"Semsim scores with the method: {evaluator.name}", fontsize=15)
        plt.legend().remove()
        legend = figure.legend(fontsize=6, loc="outside lower right", bbox_to_anchor=(1.1, 0.1))
        plt.savefig(
            plots_path / f"{evaluator.name}.png",
            dpi=400,
            bbox_extra_artists=[legend],
            bbox_inches="tight",
        )
        plt.clf()


if __name__ == "__main__":
    main()
