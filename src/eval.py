from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .tokenizer import load_tokenizer
from .train_utils import device_for_training, load_checkpoint


def read_prompts(path: str) -> list[str]:
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        prompts.append(str(obj.get("prompt", "")))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    device = device_for_training()
    tokenizer = load_tokenizer(args.tokenizer)
    model, _, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    for prompt in read_prompts(args.prompts):
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(idx, args.max_new_tokens, eos_token_id=tokenizer.eos_token_id)
        print(
            json.dumps(
                {"prompt": prompt, "completion": tokenizer.decode(out[0].tolist(), skip_special_tokens=True)},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
