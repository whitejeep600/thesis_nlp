import textattack
import torch
import transformers
from textattack.model_args import HUGGINGFACE_MODELS

RAW_MODEL_NAME = "bert-base-uncased-sst2"


class SentimentClassifier:
    def __init__(self, device: torch.device):
        super().__init__()

        textattack_model_name = HUGGINGFACE_MODELS[RAW_MODEL_NAME]
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            textattack_model_name
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(textattack_model_name, use_fast=True)
        model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.model.eval()

    def evaluate_texts(self, texts: list[str], return_probs=False) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(texts)
        return torch.softmax(logits, dim=1) if return_probs else logits
