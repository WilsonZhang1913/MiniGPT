from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .io import exists_path, load_yaml, torch_load, torch_save
from .tokenizer import load_tokenizer


FALLBACK_TEXTS = [
    "MiniGPT is a compact language model used for learning how transformers are trained.",
    "A transformer predicts the next token by attending to previous tokens in a sequence.",
    "Cloud training pipelines should save checkpoints frequently and support resume.",
    "Instruction tuning teaches a model to answer prompts with helpful responses.",
    "The tokenizer converts text into integer token IDs before those IDs are passed into the model.",
    "During pretraining the model receives many sequences and learns to predict each next token.",
    "Validation loss is measured on held out examples so training progress can be checked.",
    "A small debug run is useful because it verifies the full pipeline before spending cloud budget.",
    "Gradient accumulation lets several small batches behave like one larger optimization batch.",
    "Generation starts with a prompt and repeatedly samples one new token from the model output.",
]


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def extract_text(example: dict, text_field: str | None = None) -> str:
    if text_field and text_field in example:
        return str(example[text_field])
    for field in ("text", "content", "response", "output", "chosen"):
        if field in example and example[field]:
            return str(example[field])
    return " ".join(str(v) for v in example.values() if isinstance(v, str))


def iter_texts(data_cfg: dict, fallback: Iterable[str] = FALLBACK_TEXTS) -> Iterable[str]:
    dataset_name = data_cfg.get("dataset_name")
    if not dataset_name:
        yield from fallback
        return
    from datasets import load_dataset

    dataset_config = data_cfg.get("dataset_config")
    split = data_cfg.get("split", "train")
    streaming = bool(data_cfg.get("streaming", False))
    limit = data_cfg.get("limit")
    text_field = data_cfg.get("text_field")
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    seen = set()
    for i, example in enumerate(ds):
        if limit is not None and i >= int(limit):
            break
        text = normalize_text(extract_text(example, text_field))
        if len(text) < int(data_cfg.get("min_chars", 32)):
            continue
        if text in seen:
            continue
        seen.add(text)
        yield text


def tokenize_blocks(texts: Iterable[str], tokenizer_name: str, block_size: int) -> list[list[int]]:
    tokenizer = load_tokenizer(tokenizer_name)
    eos = tokenizer.eos_token_id
    token_stream: list[int] = []
    for text in tqdm(texts, desc="tokenizing"):
        token_stream.extend(tokenizer.encode(text))
        token_stream.append(eos)
    blocks = [token_stream[i : i + block_size] for i in range(0, len(token_stream), block_size)]
    if blocks and len(blocks[-1]) < block_size:
        blocks[-1].extend([eos] * (block_size - len(blocks[-1])))
    return blocks


class TokenBlockDataset(Dataset):
    def __init__(self, blocks: list[list[int]]) -> None:
        if not blocks:
            raise ValueError("TokenBlockDataset requires at least one block")
        self.blocks = torch.tensor(blocks, dtype=torch.long)

    def __len__(self) -> int:
        return self.blocks.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.blocks[index]
        x = block[:-1].contiguous()
        y = block[1:].contiguous()
        return x, y


class SFTDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer_name: str, block_size: int) -> None:
        if not examples:
            raise ValueError("SFTDataset requires at least one example")
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.block_size = block_size
        self.samples = [self._encode(example) for example in examples]

    def _encode(self, example: dict) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = format_prompt(example)
        response = str(example.get("response") or example.get("output") or example.get("answer") or "")
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response + self.tokenizer.eos_token)
        ids = (prompt_ids + response_ids)[: self.block_size]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.block_size]
        pad_id = self.tokenizer.pad_token_id
        if len(ids) < self.block_size:
            pad_len = self.block_size - len(ids)
            ids.extend([pad_id] * pad_len)
            labels.extend([-100] * pad_len)
        return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(labels[1:], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


def format_prompt(example: dict) -> str:
    instruction = str(example.get("instruction") or example.get("prompt") or example.get("question") or "").strip()
    context = str(example.get("input") or example.get("context") or "").strip()
    if context:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def load_sft_examples(data_cfg: dict) -> list[dict]:
    dataset_name = data_cfg.get("dataset_name")
    limit = int(data_cfg.get("limit", 256))
    if not dataset_name:
        return [
            {"instruction": "Explain what a transformer does.", "response": "A transformer predicts tokens by using attention over context."},
            {"instruction": "Give one GCP training tip.", "response": "Save checkpoints to Cloud Storage so jobs can resume safely."},
        ]
    from datasets import load_dataset

    ds = load_dataset(dataset_name, data_cfg.get("dataset_config"), split=data_cfg.get("split", "train"))
    examples = []
    for example in ds:
        response = example.get("response") or example.get("output") or example.get("answer")
        if not response and "messages" in example:
            continue
        instruction = example.get("instruction") or example.get("prompt") or example.get("question")
        if instruction and response:
            examples.append({"instruction": instruction, "input": example.get("input", ""), "response": response})
        if len(examples) >= limit:
            break
    return examples


def prepare_pretrain(config_path: str) -> None:
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    blocks = tokenize_blocks(iter_texts(data_cfg), cfg.get("tokenizer", "gpt2"), int(data_cfg["block_size"]) + 1)
    output_path = data_cfg.get("tokenized_path", "data/pretrain_blocks.pt")
    torch_save({"blocks": blocks}, output_path)
    print(f"saved {len(blocks)} pretraining blocks to {output_path}")


def load_pretrain_dataset(config: dict) -> TokenBlockDataset:
    data_cfg = config["data"]
    tokenized_path = data_cfg.get("tokenized_path")
    if tokenized_path and exists_path(tokenized_path):
        print(f"loading tokenized pretraining blocks from {tokenized_path}")
        return TokenBlockDataset(torch_load(tokenized_path)["blocks"])
    blocks = tokenize_blocks(iter_texts(data_cfg), config.get("tokenizer", "gpt2"), int(data_cfg["block_size"]) + 1)
    if tokenized_path:
        print(f"saving tokenized pretraining blocks to {tokenized_path}")
        torch_save({"blocks": blocks}, tokenized_path)
    return TokenBlockDataset(blocks)


def load_sft_dataset(config: dict) -> SFTDataset:
    data_cfg = config["data"]
    jsonl_path = data_cfg.get("jsonl_path")
    if jsonl_path:
        examples = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        examples = load_sft_examples(data_cfg)
    return SFTDataset(examples, config.get("tokenizer", "gpt2"), int(data_cfg["block_size"]) + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--kind", choices=["pretrain"], default="pretrain")
    args = parser.parse_args()
    if args.kind == "pretrain":
        prepare_pretrain(args.config)


if __name__ == "__main__":
    main()
