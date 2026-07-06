from __future__ import annotations

import argparse

from .data import load_pretrain_dataset
from .io import load_yaml
from .model import build_model
from .train_utils import train_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset = load_pretrain_dataset(cfg)
    output_dir = cfg["train"].get("output_dir", "outputs/pretrain")
    last = train_loop(
        config=cfg,
        dataset=dataset,
        output_dir=output_dir,
        model_factory=lambda: build_model(cfg["model"]),
        resume_checkpoint=args.resume_checkpoint,
    )
    print(f"saved final checkpoint to {last}")


if __name__ == "__main__":
    main()

