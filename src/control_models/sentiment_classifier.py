import torch
from textattack.models.helpers import WordCNNForClassification


class CNN_SST2_SentimentClassifier:
    def __init__(self, device: torch.device):
        super().__init__()

        # Only the cnn-sst2 model is supported for now, idk if it can be done more elegantly
        # with the interface provided by the TextAttack library.
        self.model = WordCNNForClassification.from_pretrained("cnn-sst2")
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_texts(self, texts: list[str], return_probs=False) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(
                torch.IntTensor(self.model.tokenizer.batch_encode(texts)).to(self.device)
            )
        return torch.softmax(logits, dim=1) if return_probs else logits
