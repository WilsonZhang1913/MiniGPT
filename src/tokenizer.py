from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass(frozen=True)
class TokenizerConfig:
    name: str = "gpt2"


def load_tokenizer(name: str = "gpt2") -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

