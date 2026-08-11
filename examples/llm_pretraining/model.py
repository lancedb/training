"""A compact GPT-style causal LM in plain PyTorch.

nanoGPT-flavored: fused QKV, F.scaled_dot_product_attention (flash kernels
on H100), pre-norm, weight tying, no biases.  Self-contained so the example
has no model-hub dependency.

Presets (params at GPT-2 vocab, excluding tied head):

- tiny   :   4L /  4H /  128d — CPU smoke tests
- small  :  12L / 12H /  768d — GPT-2 124M, the nanoGPT classic
- medium :  24L / 16H / 1024d — ~350M
- large  :  24L / 16H / 2048d — ~1.3B
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
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp_up = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.mlp_down = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (
            y.view(b, t, self.n_head, d // self.n_head).transpose(1, 2)
            for y in (q, k, v)
        )
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp_down(F.gelu(self.mlp_up(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, input_ids: torch.Tensor, loss_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return mean next-token loss over positions where loss_mask is True."""
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))

        targets = input_ids[:, 1:]
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.cfg.vocab_size),
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
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            std = 0.02
            if name.endswith(("proj.weight", "mlp_down.weight")):
                std = 0.02 / math.sqrt(2 * model.cfg.n_layer)
            nn.init.normal_(p, std=std)
    return model
