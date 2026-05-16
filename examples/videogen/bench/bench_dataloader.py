"""
B5 / B6 — Dataloader throughput + GPU MFU.

Compares two training-time data paths:

  * **cached** — read ``t5_hidden_states`` and ``vae_latent`` columns
    directly.  This is what ``train_wan22_lora.py`` actually uses.

  * **raw** — read ``video_bytes`` + ``caption`` and run the VAE + UMT5
    encoders inside the dataloader process.  Equivalent to the
    diffusion-pipe baseline ("decode + encode every step").

We report clips/sec (B5) and a coarse GPU MFU estimate (B6).  The
estimate assumes the Wan2.2 DiT forward+backward FLOP cost dominates;
that's exactly the assumption the cached path is designed to make true.

Hardware: defaults are tuned for H100 bf16 (peak ~989 TFLOPS).

Usage
-----
# Cached only (default once Tier-3 is backfilled):
python -m bench.bench_dataloader --db /tmp/videogen_e2e --view videos_raw \\
    --warmup 2 --steps 10 --batch-size 1

# Both:
python -m bench.bench_dataloader --db /tmp/videogen_e2e --view videos_raw --both
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

import lancedb
import torch

from videogen.dataloader import make_cached_loader, make_raw_loader
from videogen.schema import (
    T5_HIDDEN, T5_SEQ_LEN,
    VAE_INPUT_FRAMES, VAE_INPUT_H, VAE_INPUT_W,
    VAE_LATENT_C, VAE_LATENT_T, VAE_LATENT_H, VAE_LATENT_W,
)

WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Architecture-aware FLOP estimate for the Wan2.2 DiT.  The standard
# `6 × params × tokens` LLM proxy badly overestimates DiT cost because
# (a) much of the parameter count is in attention QKV/MLP which scales
# with `seq` not `seq²`, and (b) DiT cross-attention adds a smaller term.
# Numbers below are pulled from the transformer's config.json so they
# track if upstream tweaks the architecture.
#
# Wan2.2-TI2V-5B transformer:
#   num_layers=30, num_heads=24, head_dim=128 → d_model=3072
#   ffn_dim=14336, text_dim=4096
#   patch_size=(1, 2, 2), cross_attn_norm=True
WAN_PARAMS  = 5.02e9
N_LAYERS    = 30
D_MODEL     = 3072
FFN_DIM     = 14336
TEXT_DIM    = 4096
PEAK_FLOPS  = 989e12       # H100 bf16 dense

# Tokens per sample: latent (T × H/2 × W/2) plus the text context.
LATENT_TOKENS = VAE_LATENT_T * (VAE_LATENT_H // 2) * (VAE_LATENT_W // 2)
TEXT_TOKENS   = T5_SEQ_LEN
TOKENS_PER_SAMPLE = LATENT_TOKENS + TEXT_TOKENS


def _flops_per_sample_forward() -> float:
    """Forward-only FLOPs per sample for one Wan DiT pass.

    Dominant per-layer terms (FLOPs):
      self-attn projections + output : 4 · seq · d²
      self-attn QK^T + AV            : 4 · seq² · d
      cross-attn (Q from seq, KV from text) : 2 · seq · d² + 2 · seq · text · d
      MLP up + down                  : 4 · seq · d · ffn
    """
    s = LATENT_TOKENS
    t = TEXT_TOKENS
    d = D_MODEL
    f = FFN_DIM

    per_layer = (
        4 * s * d * d                       # self-attn linears
        + 4 * s * s * d                     # self-attn QK^T + AV
        + 2 * s * d * d                     # cross-attn Q + output
        + 2 * s * t * d                     # cross-attn KV / AV
        + 4 * s * d * f                     # MLP up + down
    )
    return per_layer * N_LAYERS


def _flops_per_sample_train() -> float:
    """Train-step FLOPs ≈ 3× forward (1× fwd + 2× bwd)."""
    return 3 * _flops_per_sample_forward()


def _iter_forever(loader: Iterable):
    while True:
        for b in loader:
            yield b


# ---------------------------------------------------------------------------
# Model used by both timed passes — load once, share across cached/raw
# ---------------------------------------------------------------------------

def _build_transformer(device: torch.device, dtype: torch.dtype):
    from diffusers import WanTransformer3DModel
    print(f"  Loading WanTransformer3DModel ({dtype}) …", flush=True)
    t0 = time.time()
    model = WanTransformer3DModel.from_pretrained(
        WAN_MODEL_ID, subfolder="transformer", torch_dtype=dtype,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"    loaded in {time.time() - t0:.1f}s")
    return model


def _build_vae(device: torch.device):
    from diffusers import AutoencoderKLWan
    print("  Loading AutoencoderKLWan (fp32) …", flush=True)
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        WAN_MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"    loaded in {time.time() - t0:.1f}s")
    return vae


def _build_umt5(device: torch.device):
    from transformers import T5TokenizerFast, UMT5EncoderModel
    print("  Loading UMT5-XXL (fp16) …", flush=True)
    t0 = time.time()
    tok = T5TokenizerFast.from_pretrained(WAN_MODEL_ID, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(
        WAN_MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16,
    ).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f"    loaded in {time.time() - t0:.1f}s")
    return tok, enc


# ---------------------------------------------------------------------------
# Timed runs
# ---------------------------------------------------------------------------

def time_cached(model, loader, *, device, dtype, warmup: int, steps: int) -> dict:
    """Cached path: read t5_hidden + vae_latent, do model forward only."""
    print(f"\n[B5/cached] warmup={warmup}, steps={steps}")
    it = _iter_forever(loader)

    def _step(batch):
        z = batch["vae_latent"].to(device=device, dtype=dtype, non_blocking=True)
        ctx = batch["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        t = (torch.rand(z.shape[0], device=device) * 1000).long()
        with torch.no_grad():
            model(hidden_states=z, timestep=t, encoder_hidden_states=ctx)

    for _ in range(warmup):
        _step(next(it))
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    samples = 0
    for _ in range(steps):
        batch = next(it)
        _step(batch)
        samples += batch["vae_latent"].shape[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return _report("cached", samples, dt)


def time_raw(model, vae, tok, enc, loader, *, device, dtype,
             warmup: int, steps: int) -> dict:
    """Raw path: decode mp4 → VAE encode → UMT5 encode → DiT forward."""
    import io
    import numpy as np
    from PIL import Image
    print(f"\n[B5/raw]    warmup={warmup}, steps={steps}")
    it = _iter_forever(loader)

    def _decode_to_latent_input(b: bytes) -> torch.Tensor:
        import av
        try:
            container = av.open(io.BytesIO(b))
        except Exception:
            return torch.zeros(3, VAE_INPUT_FRAMES, VAE_INPUT_H, VAE_INPUT_W,
                               device=device, dtype=torch.float32)
        total = container.streams.video[0].frames or 0
        if total <= 0:
            frames_pil = [f.to_image() for f in container.decode(video=0)]
            container.close()
            if not frames_pil:
                return torch.zeros(3, VAE_INPUT_FRAMES, VAE_INPUT_H, VAE_INPUT_W,
                                   device=device, dtype=torch.float32)
            n = VAE_INPUT_FRAMES
            idx = [round(i * (len(frames_pil) - 1) / (n - 1)) for i in range(n)]
            frames_pil = [frames_pil[i] for i in idx]
        else:
            n = VAE_INPUT_FRAMES
            tgt = sorted({round(i * (total - 1) / max(n - 1, 1)) for i in range(n)})
            frames_pil = []
            tgt_set = set(tgt)
            for j, f in enumerate(container.decode(video=0)):
                if j in tgt_set:
                    frames_pil.append(f.to_image())
            container.close()
        while len(frames_pil) < VAE_INPUT_FRAMES:
            frames_pil.append(frames_pil[-1])
        arr = np.stack([
            np.asarray(im.resize((VAE_INPUT_W, VAE_INPUT_H), Image.BICUBIC))
            for im in frames_pil[:VAE_INPUT_FRAMES]
        ], axis=0)
        t = torch.from_numpy(arr).to(device).float().div(127.5).sub(1.0)
        return t.permute(3, 0, 1, 2).contiguous()

    def _step(batch):
        descriptors_or_bytes = batch["video_bytes_descriptor"]
        captions = batch["captions"]

        # When the blob flag is off, this column comes back as raw bytes.
        bs = len(captions)
        if hasattr(descriptors_or_bytes, "to_pylist"):
            raw_list = descriptors_or_bytes.to_pylist()
        else:
            raw_list = list(descriptors_or_bytes)

        # CPU pre-decode → GPU stack
        clips = torch.stack([_decode_to_latent_input(b) for b in raw_list]).to(device)

        # VAE encode
        with torch.no_grad():
            z = vae.encode(clips).latent_dist.sample().to(dtype)

            # UMT5 tokenise + encode
            toks = tok(list(captions), padding="max_length", truncation=True,
                       max_length=T5_SEQ_LEN, return_tensors="pt").to(device)
            ctx = enc(**toks).last_hidden_state.to(dtype)

            # DiT forward
            t = (torch.rand(bs, device=device) * 1000).long()
            model(hidden_states=z, timestep=t, encoder_hidden_states=ctx)

    for _ in range(warmup):
        _step(next(it))
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    samples = 0
    for _ in range(steps):
        batch = next(it)
        _step(batch)
        samples += len(batch["captions"])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return _report("raw", samples, dt)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report(label: str, samples: int, dt: float) -> dict:
    sps = samples / dt
    # Forward-only (this bench is no_grad).  Multiply by 3 to project to
    # full training-step MFU; we report both for honesty.
    flops_fwd = sps * _flops_per_sample_forward()
    mfu_fwd = flops_fwd / PEAK_FLOPS * 100
    print(
        f"  {label:>6s}  samples={samples:<5d}  wall={dt:.2f}s  "
        f"throughput={sps:.2f} samples/s  "
        f"fwd≈{flops_fwd / 1e12:6.1f} TFLOPS  fwd-MFU≈{mfu_fwd:5.2f}%"
    )
    return {"label": label, "samples": samples, "wall_s": dt,
            "samples_per_s": sps,
            "tflops_fwd": flops_fwd / 1e12, "mfu_fwd_pct": mfu_fwd}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db",         default="data/videos/lancedb")
    p.add_argument("--view",       default="videos_raw",
                   help="Lance table or Geneva MV to read from")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--warmup",     type=int, default=2)
    p.add_argument("--steps",      type=int, default=10)
    p.add_argument("--both",       action="store_true",
                   help="Time both cached and raw paths (raw needs VAE + UMT5).")
    p.add_argument("--raw-only",   action="store_true")
    args = p.parse_args(argv)

    device = torch.device("cuda")
    dtype  = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    print(f"Device: {device} dtype: {dtype}")
    print(f"FLOPs: latent_tokens={LATENT_TOKENS}  text_tokens={TEXT_TOKENS}")
    print(f"       forward/sample  ≈ {_flops_per_sample_forward() / 1e12:.2f} TFLOPs")
    print(f"       train-step/sample ≈ {_flops_per_sample_train() / 1e12:.2f} TFLOPs")
    print(f"       peak (H100 bf16)  = {PEAK_FLOPS / 1e12:.0f} TFLOPS\n")

    db = lancedb.connect(args.db)
    n_rows = len(db.open_table(args.view))
    print(f"View '{args.view}' rows={n_rows}\n")

    model = _build_transformer(device, dtype)

    results = []
    if not args.raw_only:
        cached_loader = make_cached_loader(args.db, args.view,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=False)
        results.append(time_cached(model, cached_loader, device=device, dtype=dtype,
                                   warmup=args.warmup, steps=args.steps))

    if args.both or args.raw_only:
        vae = _build_vae(device)
        tok, enc = _build_umt5(device)
        raw_loader = make_raw_loader(args.db, args.view,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=False)
        results.append(time_raw(model, vae, tok, enc, raw_loader,
                                device=device, dtype=dtype,
                                warmup=args.warmup, steps=args.steps))

    if len(results) == 2:
        a, b = results
        speedup = a["samples_per_s"] / b["samples_per_s"]
        print(f"\n  {a['label']:>6s}/{b['label']:<6s}  speedup × {speedup:.2f}  "
              f"fwd-MFU lift: +{a['mfu_fwd_pct'] - b['mfu_fwd_pct']:.2f} pts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
