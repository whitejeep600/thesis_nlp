import os
import re
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
            load_in_4b=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-2b-it",
            trust_remote_code=True,
            token=os.environ.get("GEMMA_KEY"),
        )

        self.device = device

    def evaluate_one_to_one(self, one: str, two: str) -> float:
        input_text = self.prompt.replace("<SEQUENCE_1>", one).replace("<SEQUENCE_2>", two)

        inputs = self.tokenizer(input_text, return_tensors="pt", return_attention_mask=False).to(
            self.device
        )

        outputs = self.model.generate(**inputs, max_new_tokens=10)
        text = self.tokenizer.batch_decode(outputs)[0]

        score = re.search(r"0\.[0-9]+|0\n|1\n", text)
        if not score:
            warnings.warn(f"Couldn't parse the LLM's output: {text}")
            return 0

        return float(score.group())

    def evaluate_many_to_one(self, one: str, many: list[str]) -> list[float]:
        return [self.evaluate_one_to_one(one, two) for two in many]
