"""
ChronoMagic-Bench-style evaluation harness (scaffold).

The real ChronoMagic-Bench computes two metrics — **MTScore** (metamorphic
amplitude, how much the scene actually changes) and **CHScore** (temporal
coherence) — using a held-out set of 1,649 prompts.  Running the full
benchmark requires generating videos for every prompt, then scoring them
with the official CLIP-based scoring scripts.

This scaffold:

  1. Loads a Wan2.2 pipeline plus a LoRA adapter checkpoint produced by
     ``train_wan22_lora.py``.
  2. Generates ``--n-prompts`` short clips from a built-in list of
     phase-transition prompts.
  3. Computes our **MTScore proxy** (1 - cos(CLIP(first), CLIP(last)))
     on the generated frames — the same proxy used at curation time so
     the numbers are directly comparable to the ``metamorphic_score``
     column on the training data.
  4. Writes a ``results.json`` next to the checkpoint.

It is intentionally NOT a full ChronoMagic-Bench run — that requires
the official prompts list and the CLIPScore-based reference computation.
We're matching the published methodology in spirit, not by-the-letter.

Usage
-----
python -m videogen.eval_chronomagic \\
    --checkpoint checkpoints/wan22_lora/step-000200 \\
    --n-prompts 5 --num-frames 49 --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


_DEFAULT_PROMPTS = [
    "An ice cube slowly melting into water on a warm sunny day.",
    "Butter melting in a hot pan, golden bubbles forming as it browns.",
    "Hot wax dripping down the side of a candle, pooling at the base.",
    "Sugar dissolving in steaming hot tea, swirls of brown leaving the cube.",
    "Snow on a black rooftop melting away as the morning sun rises.",
    "A puddle of water slowly evaporating from a sun-warmed stone path.",
    "Honey dripping slowly from a wooden spoon, pooling on toast below.",
    "A chocolate truffle gently melting under warm studio lighting.",
]


def _load_pipeline(model_id: str, lora_dir: Path, dtype: str = "bfloat16"):
    import torch
    from diffusers import WanPipeline
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    print(f"Loading {model_id} (dtype={dtype}) …")
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=td).to("cuda")
    if lora_dir is not None and Path(lora_dir).exists():
        print(f"  attaching LoRA from {lora_dir}")
        pipe.load_lora_weights(str(lora_dir))
    return pipe


def _to_pil(img):
    """WanPipeline returns frames as numpy arrays (H, W, 3) uint8;
    open_clip's preprocess expects PIL.Image.  Coerce here."""
    import numpy as np
    from PIL import Image
    if isinstance(img, Image.Image):
        return img
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _mtscore_proxy(frames) -> float:
    """1 − cos(CLIP(first), CLIP(last)).  Accepts PIL or numpy frames."""
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device="cuda")
    model.eval()
    first, last = _to_pil(frames[0]), _to_pil(frames[-1])
    with torch.no_grad():
        feats = model.encode_image(
            torch.stack([preprocess(first), preprocess(last)]).cuda()
        )
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos = (feats[0] * feats[1]).sum().item()
    return float(max(0.0, 1.0 - cos))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",   default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Directory containing the peft adapter_model.safetensors.  "
                        "Omit to evaluate the base model.")
    p.add_argument("--prompts-file", type=Path, default=None,
                   help="One prompt per line.  Defaults to a small built-in set.")
    p.add_argument("--n-prompts", type=int, default=4)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--height",     type=int, default=480)
    p.add_argument("--width",      type=int, default=704)
    p.add_argument("--steps",      type=int, default=20)
    p.add_argument("--guidance",   type=float, default=6.0)
    p.add_argument("--output",     type=Path, default=Path("eval_results.json"))
    p.add_argument("--dtype",      default="bfloat16",
                   choices=["bfloat16", "float16"])
    args = p.parse_args(argv)

    if args.prompts_file is not None:
        prompts = [ln.strip() for ln in args.prompts_file.read_text().splitlines() if ln.strip()]
    else:
        prompts = _DEFAULT_PROMPTS
    prompts = prompts[: args.n_prompts]

    print(f"Eval set: {len(prompts)} prompts")
    pipe = _load_pipeline(args.model_id, args.checkpoint, dtype=args.dtype)

    out = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] {prompt[:60]} …")
        t0 = time.perf_counter()
        result = pipe(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height, width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
        )
        dt = time.perf_counter() - t0
        frames = result.frames[0] if hasattr(result, "frames") else result
        mt = _mtscore_proxy(frames)
        print(f"  {dt:.1f}s  MTScore-proxy={mt:.3f}")
        out.append({"prompt": prompt, "mtscore_proxy": mt,
                    "wall_clock_s": dt, "num_frames": args.num_frames,
                    "height": args.height, "width": args.width})

    summary = {
        "model_id":   args.model_id,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "n_prompts":  len(out),
        "mtscore_proxy_mean": sum(r["mtscore_proxy"] for r in out) / max(len(out), 1),
        "results": out,
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nResults → {args.output}")
    print(f"  mean MTScore-proxy: {summary['mtscore_proxy_mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
