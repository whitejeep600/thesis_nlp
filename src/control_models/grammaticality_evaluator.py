import textattack
import torch
import transformers
from textattack.model_args import HUGGINGFACE_MODELS


class GrammaticalityEvaluator:
    def __init__(self, device: torch.device):
        super().__init__()

        raw_name = "bert-base-uncased-cola"
        textattack_model_name = HUGGINGFACE_MODELS[raw_name]
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            textattack_model_name
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(textattack_model_name, use_fast=True)
        model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        self.model = model

        self.device = device
        self.model.to(device)
        self.model.model.eval()
        self.grammaticality_label_code = 1

    def evaluate_texts(self, texts: list[str], return_probs=False) -> list[float]:
        with torch.no_grad():
            logits = self.model(texts)
        if return_probs:
            probs = torch.softmax(logits, dim=1)
            result_tensor = probs
        else:
            result_tensor = logits

        return result_tensor[:, self.grammaticality_label_code].tolist()