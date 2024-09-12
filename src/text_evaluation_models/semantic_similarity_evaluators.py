import torch
from sentence_transformers import SentenceTransformer


class BaseSemanticSimilarityEvaluator:
    def __init__(self, model_name: str, device: torch.device):
        pass

    def evaluate_many_to_one(self, many: list[str], one: str) -> list[float]:
        raise NotImplementedError


class EmbeddingBasedSemanticSimilarityEvaluator(BaseSemanticSimilarityEvaluator):
    def __init__(self, model_name: str, device: torch.device):
        super().__init__(model_name, device)
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.model.eval()

    def evaluate_many_to_one(self, many: list[str], one: str) -> list[float]:
        one_encoding = self.model.encode(one, convert_to_tensor=True)
        many_encodings = self.model.encode(many, convert_to_tensor=True)

        return [
            torch.cosine_similarity(one_encoding, prefix_encoding, dim=0).item()
            for prefix_encoding in many_encodings
        ]
