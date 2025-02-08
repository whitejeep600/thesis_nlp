from pathlib import Path

import torch

from transformers import (
    pipeline,
)


class LLMSimilarityEvaluator:

    def __init__(self, device: torch.device):

        PROMPT_PATH = Path("data/llm_semsim_prompt.txt")
        with open(PROMPT_PATH, "r") as f:
            self.prompt = f.read()

        CONTEXT_PATH = Path("data/llm_context.txt")
        with open(CONTEXT_PATH, "r") as f:
            self.context = f.read()

        self.pipeline = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )



    def evaluate_one_to_one(self, one: list[str], two: str) -> float:
        input_text = f"First sentence: {one}\nSecond sentence: {two}\n"

        messages = [
            {
                "role": "system",
                "content": self.context,
            },
            {"role": "user", "content": input_text},
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95
        )

        result = outputs[0]["generated_text"]
        print(result)
        return float(result)

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
