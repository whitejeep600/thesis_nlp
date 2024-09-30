from __future__ import annotations

from pathlib import Path

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from src.constants import TrainMode


class GenerativeBart:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        weights_path: Path | None = None,
    ):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained(model_name)
        if weights_path:
            self.bert.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        self.bert.to(device)
        self.bert.gradient_checkpointing_enable()
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.stop_token_id = self.token_to_tokenizer_id("</s>")
        self.start_token_id = self.token_to_tokenizer_id("<s>")

    def token_to_tokenizer_id(self, word: str) -> int:
        return self.tokenizer.encode(word)[1]

    def parameters(self):
        return self.bert.parameters()

    def set_mode(self, mode: TrainMode) -> None:
        if mode == TrainMode.train:
            self.bert.train()
        else:
            self.bert.eval()

    def decode_single_sequence(self, generated_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
