from __future__ import annotations

import argparse

import torch

from .tokenizer import load_tokenizer
from .train_utils import device_for_training, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = device_for_training()
    tokenizer = load_tokenizer(args.tokenizer)
    model, _, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()

