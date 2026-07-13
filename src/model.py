from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    n_expert: int = 4
    n_expert_active: int = 2
    expert_hidden_mult: int = 4
    moe_aux_loss_coef: float = 0.01


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class Expert(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_size = config.expert_hidden_mult * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_expert < 1:
            raise ValueError("n_expert must be at least 1")
        if config.n_expert_active < 1 or config.n_expert_active > config.n_expert:
            raise ValueError("n_expert_active must be in [1, n_expert]")
        self.n_expert = config.n_expert
        self.n_expert_active = config.n_expert_active
        self.router = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_expert)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, channels = x.shape
        flat_x = x.reshape(batch * seq_len, channels)
        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_idx = torch.topk(router_probs, self.n_expert_active, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        flat_out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_pos, choice_pos = torch.where(top_idx == expert_id)
            if token_pos.numel() == 0:
                continue
            expert_out = expert(flat_x[token_pos])
            weights = top_probs[token_pos, choice_pos].unsqueeze(-1)
            flat_out.index_add_(0, token_pos, expert_out * weights)

        importance = router_probs.mean(dim=0)
        selected = F.one_hot(top_idx, num_classes=self.n_expert).float().sum(dim=1)
        load = selected.mean(dim=0) / self.n_expert_active
        aux_loss = self.n_expert * torch.sum(importance * load)
        return flat_out.view(batch, seq_len, channels), aux_loss


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MixtureOfExperts(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln_1(x))
        moe_out, aux_loss = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, aux_loss


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        aux_loss = torch.zeros((), device=idx.device)
        for block in self.transformer["h"]:
            x, block_aux_loss = block(x)
            aux_loss = aux_loss + block_aux_loss
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            loss = loss + self.config.moe_aux_loss_coef * aux_loss / max(1, self.config.n_layer)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and torch.all(idx_next == eos_token_id):
                break
        return idx


def build_model(config_dict: dict) -> GPT:
    return GPT(GPTConfig(**config_dict))
