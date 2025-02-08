import os
from pathlib import Path

import torch

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer, BitsAndBytesConfig,
)


class LLMSimilarityEvaluator:

    def __init__(self, device: torch.device):

        PROMPT_PATH = Path("data/llm_semsim_prompt.txt")
        with open(PROMPT_PATH, "r") as f:
            self.prompt = f.read()

        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            device_map=device,
            trust_remote_code=True,
            token=os.environ.get("GEMMA_KEY"),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-2b-it", trust_remote_code=True, token=os.environ.get("GEMMA_KEY")
        )

        self.device = device

    def evaluate_one_to_one(self, one: list[str], two: str) -> float:
        input_text = self.prompt.replace("<SEQUENCE_1>", one).replace("<SEQUENCE_2>", two)

        inputs = self.tokenizer(input_text, return_tensors="pt", return_attention_mask=False).to(self.device)

        outputs = self.model.generate(**inputs, max_length=200)
        text = self.tokenizer.batch_decode(outputs)[0]
        print(text)
        return text

    def evaluate_many_to_one(self, one: list[str], many: list[str]) -> list[float]:
        return [self.evaluate_one_to_one(one, two) for two in many]


if __name__ == "__main__":
    se = LLMSimilarityEvaluator(device="cuda:0")
    one = "I didn't like this movie at all, it was a waste of money."
    many = [
        "I liked this movie a lot, it's money well spent.",
        "I didn't enjoy the film altogether, I regret buying the ticket.",
        "Mitochondrium is the powerhouse of the cell.",
    ]
    print(se.evaluate_many_to_one(one=one, many=many))
