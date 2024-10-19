import textattack
import torch
import transformers
from sentence_transformers import SentenceTransformer
from textattack.model_args import HUGGINGFACE_MODELS
from transformers import T5ForConditionalGeneration, T5Tokenizer


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


class DistilbertEntailmentEvaluator:
    def __init__(self, device: torch.device):
        super().__init__()

        textattack_model_name = HUGGINGFACE_MODELS["distilbert-base-cased-snli"]
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

    def get_all_label_logits_for_text_pairs(self, texts: list[tuple[str, str]]) -> torch.Tensor:
        prepared_inputs = [
            f"Premise: {premise} \nHypothesis: {hypothesis}" for (premise, hypothesis) in texts
        ]
        with torch.no_grad():
            logits = self.model(prepared_inputs)
        return logits

    def get_entailment_probs_for_text_pairs(self, texts: list[tuple[str, str]]) -> list[float]:
        logits = self.get_all_label_logits_for_text_pairs(texts)
        probs = torch.softmax(logits, dim=1)
        return probs[:, self.entailment_code].tolist()

    def get_entailment_probs_many_to_one(self, many: list[str], one: str) -> list[float]:
        return self.get_entailment_probs_for_text_pairs(
            [(one, one_of_many) for one_of_many in many]
        )

    def get_binary_entailment_for_text_pairs(self, texts: list[tuple[str, str]]) -> list[bool]:
        logits = self.get_all_label_logits_for_text_pairs(texts)
        labels = logits.argmax(dim=1).tolist()
        return [label == self.entailment_code for label in labels]

    def get_binary_entailment_many_to_one(self, many: list[str], one: str) -> list[bool]:
        return self.get_binary_entailment_for_text_pairs(
            [(one, one_of_many) for one_of_many in many]
        )


class T5HardLabelEntailmentEvaluator:
    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.entailment_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.entailment_model.to(device)
        self.entailment_model.eval()

    @staticmethod
    def model_hard_label_to_float_score(output: str) -> float:
        if "entailment" in output:
            return 1
        elif "neutral" in output:
            return 0.5
        elif "contradiction" in output:
            return 0
        else:
            raise ValueError(f"Unexpected T5 entailment model output {output}")

    def get_hard_labels_for_text_pairs(self, texts: list[tuple[str, str]]) -> list[float]:
        inputs_to_tokenize = [
            f"mnli premise: {premise}. hypothesis: {hypothesis}" for (premise, hypothesis) in texts
        ]
        input_ids = self.tokenizer(
            inputs_to_tokenize, return_tensors="pt", padding=True
        ).input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.entailment_model.generate(input_ids=input_ids)
        decoded = self.tokenizer.batch_decode(outputs)
        float_scores = [self.model_hard_label_to_float_score(label) for label in decoded]
        return float_scores

    def get_hard_labels_many_to_one(self, many: list[str], one: str) -> list[float]:
        return self.get_hard_labels_for_text_pairs([(one, one_of_many) for one_of_many in many])
