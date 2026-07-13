from __future__ import annotations

import argparse

import torch

from .tokenizer import load_tokenizer
from .train_utils import DEFAULT_SEED, device_for_training, load_checkpoint, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = device_for_training()
    tokenizer = load_tokenizer(args.tokenizer)
    model, _, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(out[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
