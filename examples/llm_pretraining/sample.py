"""Sample text from a trained checkpoint.

Usage
-----
python sample.py --ckpt checkpoints/step_00004800.pt --tokenizer hf:gpt2 \
    --model small --prompt "The history of mathematics" --tokens 120
"""

from __future__ import annotations

import argparse

import torch

from common import load_tokenizer
from model import make_model


@torch.no_grad()
def generate(model, tok, prompt: str, n_tokens: int, temperature: float, top_k: int):
    device = next(model.parameters()).device
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(n_tokens):
        ctx = ids[:, -model.cfg.seq_len :]
        x = model.tok_emb(ctx) + model.pos_emb(
            torch.arange(ctx.shape[1], device=device)
        )
        for block in model.blocks:
            x = block(x)
        logits = model.head(model.ln_f(x))[:, -1] / max(temperature, 1e-5)
        logits[:, tok.pad_token_id] = float("-inf")  # never sample the pad id
        if top_k:
            kth = torch.topk(logits, top_k).values[:, -1, None]
            logits[logits < kth] = float("-inf")
        next_id = torch.multinomial(torch.softmax(logits, -1), 1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0].tolist()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="hf:gpt2")
    p.add_argument("--model", default="small")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--prompt", default="The")
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    tok = load_tokenizer(args.tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.model, tok.vocab_size, args.seq_len).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    print(f"loaded {args.ckpt} (step {ckpt['opt_step']})\n")

    torch.manual_seed(args.seed)
    ids = generate(model, tok, args.prompt, args.tokens, args.temperature, args.top_k)
    if args.tokenizer.startswith("hf:"):
        from transformers import AutoTokenizer

        print(AutoTokenizer.from_pretrained(args.tokenizer[3:]).decode(ids))
    else:
        print(bytes(i for i in ids if i < 256).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
