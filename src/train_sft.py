from __future__ import annotations

import argparse

from .data import load_sft_dataset
from .io import load_yaml
from .model import build_model
from .train_utils import train_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset = load_sft_dataset(cfg)
    output_dir = cfg["train"].get("output_dir", "outputs/sft")
    last = train_loop(
        config=cfg,
        dataset=dataset,
        output_dir=output_dir,
        model_factory=lambda: build_model(cfg["model"]),
        resume_checkpoint=args.checkpoint,
        reset_step=True,
        reset_optimizer=True,
    )
    print(f"saved final checkpoint to {last}")


if __name__ == "__main__":
    main()
