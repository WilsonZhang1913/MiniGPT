from __future__ import annotations

import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .io import torch_load, torch_save
from .model import GPT, GPTConfig

DEFAULT_SEED = 1337


def device_for_training() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_seed(config: dict | None = None) -> int:
    if not config:
        return DEFAULT_SEED
    if "seed" in config:
        return int(config["seed"])
    train_cfg = config.get("train", {})
    return int(train_cfg.get("seed", DEFAULT_SEED))


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def split_dataset(dataset, val_fraction: float = 0.05, seed: int = DEFAULT_SEED):
    val_size = max(1, int(len(dataset) * val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size == 0:
        return dataset, dataset
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def learning_rate(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: GPT, loader: DataLoader, device: torch.device, eval_iters: int) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def reproducibility_metadata(seed: int, device: torch.device) -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        "seed": seed,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
        },
        "hardware": {
            "device": str(device),
            "cuda_available": cuda_available,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        },
    }


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    path: str,
) -> None:
    torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "model_config": asdict(model.config),
            "config": config,
            "reproducibility": reproducibility_metadata(get_seed(config), next(model.parameters()).device),
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> tuple[GPT, dict, int, dict]:
    ckpt = torch_load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt.get("optimizer_state", {}), int(ckpt.get("step", 0)), ckpt


def train_loop(
    *,
    config: dict,
    dataset,
    output_dir: str,
    model_factory: Callable[[], GPT],
    resume_checkpoint: str | None = None,
    reset_step: bool = False,
    reset_optimizer: bool = False,
) -> str:
    train_cfg = config["train"]
    seed = get_seed(config)
    set_seed(seed)
    device = device_for_training()
    train_ds, val_ds = split_dataset(dataset, float(train_cfg.get("val_fraction", 0.05)), seed=seed)
    batch_size = int(train_cfg["batch_size"])
    drop_last = len(train_ds) >= batch_size
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if resume_checkpoint:
        model, optimizer_state, start_step, _ = load_checkpoint(resume_checkpoint, device)
        if reset_step:
            start_step = 0
        if reset_optimizer:
            optimizer_state = {}
    else:
        model = model_factory().to(device)
        optimizer_state = {}
        start_step = 0

    optimizer = make_optimizer(model, float(train_cfg["learning_rate"]), float(train_cfg.get("weight_decay", 0.1)))
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    use_amp = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    max_steps = int(train_cfg["max_steps"])
    eval_interval = int(train_cfg.get("eval_interval", 100))
    save_interval = int(train_cfg.get("save_interval", eval_interval))
    eval_iters = int(train_cfg.get("eval_iters", 10))
    output_path = output_dir.rstrip("/")

    step = start_step
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=max_steps, initial=start_step, desc="training")
    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break
            lr = learning_rate(step, float(train_cfg["learning_rate"]), int(train_cfg.get("warmup_steps", 100)), max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            xb, yb = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(xb, yb)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss.item())
            micro_step += 1
            if micro_step < grad_accum:
                continue
            if float(train_cfg.get("grad_clip", 1.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            micro_step = 0 
            pbar.update(1)
            pbar.set_postfix(loss=f"{running_loss:.4f}", lr=f"{lr:.2e}")
            running_loss = 0.0

            if step % eval_interval == 0 or step == max_steps:
                val_loss = estimate_loss(model, val_loader, device, eval_iters)
                print(f"step={step} val_loss={val_loss:.4f}")
            if step % save_interval == 0 or step == max_steps:
                ckpt_path = f"{output_path}/checkpoint_{step}.pt"
                save_checkpoint(model, optimizer, step, config, ckpt_path)
                save_checkpoint(model, optimizer, step, config, f"{output_path}/checkpoint_last.pt")
        if len(train_loader) == 0:
            raise ValueError("Training loader is empty; reduce batch size or provide more data")
    pbar.close()
    return f"{output_path}/checkpoint_last.pt"
