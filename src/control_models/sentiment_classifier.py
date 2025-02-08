import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RAW_MODEL_NAME = "bert-base-uncased-sst2"

GEMMA_KEY = os.environ.get("GEMMA_KEY")


class SentimentClassifier:
    def __init__(self, device: torch.device):
        super().__init__()

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=GEMMA_KEY)
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=GEMMA_KEY
        )

        input_text = "Tell me in two sentences what coffee is."
        input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

        outputs = model.generate(**input_ids, max_new_tokens=32)
        print(tokenizer.decode(outputs[0]))

    def evaluate_texts(self, texts: list[str], return_probs=False) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(texts)
        return torch.softmax(logits, dim=1) if return_probs else logits


if __name__ == "__main__":
    sc = SentimentClassifier("cuda:0")

