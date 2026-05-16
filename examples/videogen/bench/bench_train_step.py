"""
Training-step throughput: cached vs raw.

The companion to ``bench_dataloader.py``.  Where that one measures
**forward-only** throughput (no_grad, no backward, no optimizer),
this one measures the **full training step** for both paths:

  forward → flow-matching loss → backward → optimizer step

Both paths train the same Wan2.2-TI2V-5B + LoRA model.  The only
difference is what the dataloader hands to the step:

  cached → t5_hidden_states  +  vae_latent       (pre-computed columns)
  raw    → video_bytes  +  caption               (decode + VAE + UMT5 in-loop)

We report samples/s, training-step wall-clock, and a forward+backward
FLOP-derived MFU estimate (3× the bench_dataloader proxy).

Usage
-----
python -m bench.bench_train_step --db <db> --view <view> \\
    --warmup 2 --steps 10 --batch-size 1
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

import lancedb
import torch
import torch.nn.functional as F

from videogen.dataloader import make_cached_loader, make_raw_loader
from videogen.schema import (
    T5_SEQ_LEN, VAE_INPUT_FRAMES, VAE_INPUT_H, VAE_INPUT_W,
    VAE_LATENT_C, VAE_LATENT_H, VAE_LATENT_T, VAE_LATENT_W,
)
from videogen.train_wan22_lora import (
    attach_lora,
    build_transformer,
    WAN_MODEL_ID,
)
from bench.bench_dataloader import (
    LATENT_TOKENS, PEAK_FLOPS, TEXT_TOKENS,
    _flops_per_sample_train,
)


def _iter_forever(loader: Iterable):
    while True:
        for b in loader:
            yield b


def _flow_matching_loss(model, z0, ctx, *, dtype):
    bsz = z0.shape[0]
    t = torch.rand(bsz, device=z0.device, dtype=dtype).clamp(1e-3, 1.0 - 1e-3)
    z1 = torch.randn_like(z0)
    t_b = t.view(bsz, 1, 1, 1, 1)
    z_t = (1.0 - t_b) * z0 + t_b * z1
    target = z1 - z0
    pred = model(hidden_states=z_t,
                 timestep=(t * 1000.0).long(),
                 encoder_hidden_states=ctx).sample
    return F.mse_loss(pred.float(), target.float())


# ---------------------------------------------------------------------------
# Cached path
# ---------------------------------------------------------------------------

def time_cached_train(model, optimizer, loader, *, device, dtype,
                      warmup: int, steps: int) -> dict:
    print(f"\n[B5+train/cached] warmup={warmup}, steps={steps}")
    it = _iter_forever(loader)

    def _step(batch):
        z = batch["vae_latent"].to(device=device, dtype=dtype, non_blocking=True)
        ctx = batch["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = _flow_matching_loss(model, z, ctx, dtype=dtype)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        _step(next(it))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    samples = 0
    for _ in range(steps):
        batch = next(it)
        _step(batch)
        samples += batch["vae_latent"].shape[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return _report("cached", samples, dt, peak_gb)


# ---------------------------------------------------------------------------
# Raw path
# ---------------------------------------------------------------------------

def _build_vae(device, dtype_for_cast):
    from diffusers import AutoencoderKLWan
    print("  Loading AutoencoderKLWan (fp32) …", flush=True)
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        WAN_MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"    loaded in {time.time() - t0:.1f}s "
          f"({sum(p.numel() for p in vae.parameters()) / 1e6:.0f}M params)")
    return vae


def _build_umt5(device, dtype):
    from transformers import T5TokenizerFast, UMT5EncoderModel
    print("  Loading UMT5-XXL (fp16) …", flush=True)
    t0 = time.time()
    tok = T5TokenizerFast.from_pretrained(WAN_MODEL_ID, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(
        WAN_MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16,
    ).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f"    loaded in {time.time() - t0:.1f}s "
          f"({sum(p.numel() for p in enc.parameters()) / 1e9:.2f}B params)")
    return tok, enc


def time_raw_train(model, optimizer, vae, tok, enc, loader,
                   *, device, dtype, warmup: int, steps: int) -> dict:
    print(f"\n[B5+train/raw]    warmup={warmup}, steps={steps}")
    import io
    import numpy as np
    from PIL import Image
    import av

    it = _iter_forever(loader)

    def _decode_to_clip(b: bytes) -> torch.Tensor:
        try:
            container = av.open(io.BytesIO(b))
        except Exception:
            return torch.zeros(3, VAE_INPUT_FRAMES, VAE_INPUT_H, VAE_INPUT_W,
                               device=device, dtype=torch.float32)
        total = container.streams.video[0].frames or 0
        if total <= 0:
            frames = [f.to_image() for f in container.decode(video=0)]
            container.close()
            if not frames:
                return torch.zeros(3, VAE_INPUT_FRAMES, VAE_INPUT_H, VAE_INPUT_W,
                                   device=device, dtype=torch.float32)
            n = VAE_INPUT_FRAMES
            idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
            frames = [frames[i] for i in idx]
        else:
            n = VAE_INPUT_FRAMES
            tgt = sorted({round(i * (total - 1) / max(n - 1, 1)) for i in range(n)})
            frames = []
            tgt_set = set(tgt)
            for j, f in enumerate(container.decode(video=0)):
                if j in tgt_set:
                    frames.append(f.to_image())
            container.close()
        while len(frames) < VAE_INPUT_FRAMES:
            frames.append(frames[-1])
        arr = np.stack([
            np.asarray(im.resize((VAE_INPUT_W, VAE_INPUT_H), Image.BICUBIC))
            for im in frames[:VAE_INPUT_FRAMES]
        ], axis=0)
        t = torch.from_numpy(arr).to(device).float().div(127.5).sub(1.0)
        return t.permute(3, 0, 1, 2).contiguous()

    def _step(batch):
        captions = batch["captions"]
        bytes_list = batch["video_bytes"]
        bs = len(captions)

        clips = torch.stack([_decode_to_clip(b) for b in bytes_list]).to(device)
        with torch.no_grad():
            z = vae.encode(clips).latent_dist.sample().to(dtype)
            toks = tok(list(captions), padding="max_length", truncation=True,
                       max_length=T5_SEQ_LEN, return_tensors="pt").to(device)
            ctx = enc(**toks).last_hidden_state.to(dtype)

        optimizer.zero_grad(set_to_none=True)
        loss = _flow_matching_loss(model, z, ctx, dtype=dtype)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        _step(next(it))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    samples = 0
    for _ in range(steps):
        batch = next(it)
        _step(batch)
        samples += len(batch["captions"])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return _report("raw", samples, dt, peak_gb)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report(label: str, samples: int, dt: float, peak_gb: float) -> dict:
    sps = samples / dt
    flops = sps * _flops_per_sample_train()
    mfu = flops / PEAK_FLOPS * 100
    print(
        f"  {label:>6s}  samples={samples:<5d}  wall={dt:.2f}s  "
        f"train-throughput={sps:.2f} samples/s  "
        f"train≈{flops / 1e12:5.0f} TFLOPS  MFU≈{mfu:5.2f}%  "
        f"VRAM peak={peak_gb:.1f} GB"
    )
    return {"label": label, "samples": samples, "wall_s": dt,
            "samples_per_s": sps,
            "tflops": flops / 1e12, "mfu_pct": mfu,
            "vram_peak_gb": peak_gb}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_train_model(device, dtype, rank: int, alpha: int):
    model = build_transformer(dtype=dtype, device=device)
    model = attach_lora(model, rank=rank, alpha=alpha,
                        targets=["to_q", "to_k", "to_v", "to_out.0"])
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.95),
                                  weight_decay=1e-2, fused=True)
    return model, optimizer


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db",         default="data/videos/lancedb")
    p.add_argument("--view",       default="videos_raw")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--warmup",     type=int, default=2)
    p.add_argument("--steps",      type=int, default=10)
    p.add_argument("--rank",       type=int, default=16)
    p.add_argument("--alpha",      type=int, default=16)
    p.add_argument("--both",       action="store_true", default=True,
                   help="Bench both cached and raw (default).")
    p.add_argument("--cached-only", action="store_true")
    p.add_argument("--raw-only",    action="store_true")
    args = p.parse_args(argv)

    device = torch.device("cuda")
    dtype  = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    print(f"Device: {device}  dtype: {dtype}")
    print(f"FLOPs/sample (train ≈ 3× fwd): {_flops_per_sample_train() / 1e12:.2f} TFLOPs")
    print(f"Latent tokens={LATENT_TOKENS}  text tokens={TEXT_TOKENS}")
    print(f"Peak (H100 bf16) = {PEAK_FLOPS / 1e12:.0f} TFLOPS\n")

    db = lancedb.connect(args.db)
    n_rows = len(db.open_table(args.view))
    print(f"View '{args.view}'  rows={n_rows}\n")

    results = []

    if not args.raw_only:
        # Cached run — fresh model so we don't share weights between runs.
        model_c, opt_c = _build_train_model(device, dtype, args.rank, args.alpha)
        cached_loader = make_cached_loader(args.db, args.view,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=False)
        r = time_cached_train(model_c, opt_c, cached_loader,
                              device=device, dtype=dtype,
                              warmup=args.warmup, steps=args.steps)
        results.append(r)
        del model_c, opt_c, cached_loader
        torch.cuda.empty_cache()

    if not args.cached_only:
        # Raw run — need its own model + VAE + UMT5.
        model_r, opt_r = _build_train_model(device, dtype, args.rank, args.alpha)
        vae = _build_vae(device, dtype)
        tok, enc = _build_umt5(device, dtype)
        raw_loader = make_raw_loader(args.db, args.view,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=False)
        r = time_raw_train(model_r, opt_r, vae, tok, enc, raw_loader,
                           device=device, dtype=dtype,
                           warmup=args.warmup, steps=args.steps)
        results.append(r)

    if len(results) == 2:
        a, b = results
        print()
        print(f"  {a['label']:>6s}/{b['label']:<6s}  "
              f"throughput × {a['samples_per_s'] / b['samples_per_s']:.2f}   "
              f"MFU lift +{a['mfu_pct'] - b['mfu_pct']:.2f} pts   "
              f"VRAM saved {b['vram_peak_gb'] - a['vram_peak_gb']:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
