"""A compact GPT-style causal LM in plain PyTorch.

Self-contained so the example has no model-hub dependency; swap in any
HuggingFace model if you prefer — the data pipeline does not care.

Presets (parameters exclude embeddings-tying savings):

- tiny   :   4L /  128d /  4h  — CPU smoke tests (~1M params w/ byte vocab)
- small  :  12L /  768d / 12h  — ~124M params, single-GPU sanity runs
- medium :  24L / 1024d / 16h  — ~350M params
- large  :  24L / 2048d / 16h  — ~1.3B params, multi-node H200 territory
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768

    @classmethod
    def preset(cls, name: str, vocab_size: int, seq_len: int) -> "GPTConfig":
        shapes = {
            "tiny": (4, 4, 128),
            "small": (12, 12, 768),
            "medium": (24, 16, 1024),
            "large": (24, 16, 2048),
        }
        n_layer, n_head, d_model = shapes[name]
        return cls(vocab_size, seq_len, n_layer, n_head, d_model)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        mask = torch.full((cfg.seq_len, cfg.seq_len), float("-inf")).triu(1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, input_ids: torch.Tensor, loss_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return mean next-token loss over positions where loss_mask is True."""
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        mask = self.causal_mask[:t, :t]
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))

        targets = input_ids[:, 1:]
        logits = logits[:, :-1]
        loss = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).view(b, t - 1)
        if loss_mask is not None:
            m = loss_mask[:, 1:].float()
            return (loss * m).sum() / m.sum().clamp(min=1.0)
        return loss.mean()


def make_model(preset: str, vocab_size: int, seq_len: int, seed: int = 1234) -> GPT:
    torch.manual_seed(seed)
    model = GPT(GPTConfig.preset(preset, vocab_size, seq_len))
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, std=0.02 / math.sqrt(2 * model.cfg.n_layer))
    return model
