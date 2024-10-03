import textattack
import torch
import transformers
from sentence_transformers import SentenceTransformer
from textattack.model_args import HUGGINGFACE_MODELS


class EmbeddingBasedSemanticSimilarityEvaluator:
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.model.eval()

    def evaluate_many_to_one(self, many: list[str], one: str) -> list[float]:
        one_encoding = self.model.encode(one, convert_to_tensor=True)
        many_encodings = self.model.encode(many, convert_to_tensor=True)

        return [
            torch.cosine_similarity(one_encoding, one_of_many_encodings, dim=0).item()
            for one_of_many_encodings in many_encodings
        ]

    def evaluate_one_to_one(self, one_0: str, one_1: str) -> float:
        encoding_0 = self.model.encode(one_0, convert_to_tensor=True)
        encoding_1 = self.model.encode(one_1, convert_to_tensor=True)
        return torch.cosine_similarity(encoding_0, encoding_1, dim=0).item()


class AlbertEntailmentEvaluator:
    def __init__(self, device: torch.device):
        super().__init__()

        textattack_model_name = HUGGINGFACE_MODELS["albert-base-v2-snli"]
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            textattack_model_name
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(textattack_model_name, use_fast=True)
        model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        self.model = model

        self.device = device
        self.model.to(device)
        self.model.model.eval()

        self.entailment_code = 0
        self.neutral_code = 1
        self.contradiction_code = 2

    def get_logits_for_text_pairs(self, texts: list[tuple[str, str]]) -> torch.Tensor:
        prepared_inputs = [
            f"Premise: {premise} \nHypothesis: {hypothesis}" for (premise, hypothesis) in texts
        ]
        with torch.no_grad():
            logits = self.model(prepared_inputs)
        return logits

    def get_probs_for_text_pairs(self, texts: list[tuple[str, str]]) -> list[float]:
        logits = self.get_logits_for_text_pairs(texts)
        probs = torch.softmax(logits, dim=1)
        return probs[:, self.entailment_code].tolist()

    def get_probs_many_to_one(self, many: list[str], one: str) -> list[float]:
        return self.get_probs_for_text_pairs([(one, one_of_many) for one_of_many in many])

    def get_binary_entailment_for_text_pairs(self, texts: list[tuple[str, str]]) -> list[bool]:
        logits = self.get_logits_for_text_pairs(texts)
        labels = logits.argmax(dim=1).tolist()
        return [label == self.entailment_code for label in labels]

    def get_binary_entailment_many_to_one(self, many: list[str], one: str) -> list[bool]:
        return self.get_binary_entailment_for_text_pairs(
            [(one, one_of_many) for one_of_many in many]
        )
