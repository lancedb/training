"""
VBench-style evaluation harness (scaffold).

The full VBench suite covers ~16 evaluation dimensions (subject consistency,
motion smoothness, dynamic degree, aesthetic quality, …) via specialised
scorers downloaded from the VBench repo.  Running the full eval involves:

  1. Generating videos for VBench's canonical prompt list.
  2. Invoking the upstream scorers (RAFT, ViCLIP, Q-Align, DOVER, etc.).

This scaffold focuses on **three dimensions that are cheap to compute**
without pulling in the full VBench wheel, picked because they are the
dimensions most relevant to our phase-transition use case:

  * **dynamic_degree**       — mean RAFT-style flow magnitude (motion strength)
  * **subject_consistency**  — mean cosine similarity of per-frame CLIP
                              embeddings (high = same subject, low = drift)
  * **temporal_smoothness**  — variance of frame-to-frame CLIP cosine
                              (high = jumpy, low = smooth)

We do not chase parity with VBench's exact numbers; we chase comparability
across our own runs (before/after LoRA, different ranks, etc.).

Usage
-----
python -m videogen.eval_vbench \\
    --checkpoint checkpoints/wan22_lora/step-000200 \\
    --n-prompts 4 --output vbench_proxy.json
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
    "Snow on a black rooftop melting away as the morning sun rises.",
]


def _load_pipeline(model_id: str, lora_dir: Path, dtype: str):
    import torch
    from diffusers import WanPipeline
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    print(f"Loading {model_id} (dtype={dtype}) …")
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=td).to("cuda")
    if lora_dir is not None and Path(lora_dir).exists():
        print(f"  attaching LoRA from {lora_dir}")
        pipe.load_lora_weights(str(lora_dir))
    return pipe


def _clip_embed_frames(frames):
    import open_clip, torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device="cuda")
    model.eval()
    tensors = torch.stack([preprocess(f) for f in frames]).cuda()
    with torch.no_grad():
        emb = model.encode_image(tensors)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return emb  # (T, 512)


def _motion(frames) -> float:
    import numpy as np
    if len(frames) < 2:
        return 0.0
    arrs = [np.asarray(f, dtype=np.float32) for f in frames]
    return float(
        np.mean([np.mean(np.abs(arrs[i + 1] - arrs[i]))
                 for i in range(len(arrs) - 1)]) / 255.0 * 100.0
    )


def _vbench_proxies(frames) -> dict:
    import torch
    emb = _clip_embed_frames(frames)  # (T, 512)
    cos = (emb[:-1] * emb[1:]).sum(dim=-1).clamp(min=-1, max=1)
    return {
        "dynamic_degree":      _motion(frames),
        "subject_consistency": float(cos.mean().item()),
        "temporal_smoothness": float(1.0 - cos.var().item()),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",   default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--prompts-file", type=Path, default=None)
    p.add_argument("--n-prompts",  type=int,   default=4)
    p.add_argument("--num-frames", type=int,   default=49)
    p.add_argument("--height",     type=int,   default=480)
    p.add_argument("--width",      type=int,   default=704)
    p.add_argument("--steps",      type=int,   default=20)
    p.add_argument("--guidance",   type=float, default=6.0)
    p.add_argument("--output",     type=Path,  default=Path("vbench_proxy.json"))
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

    rows = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] {prompt[:60]} …")
        t0 = time.perf_counter()
        result = pipe(prompt=prompt, num_frames=args.num_frames,
                      height=args.height, width=args.width,
                      num_inference_steps=args.steps,
                      guidance_scale=args.guidance)
        frames = result.frames[0] if hasattr(result, "frames") else result
        metrics = _vbench_proxies(frames)
        metrics["wall_clock_s"] = time.perf_counter() - t0
        metrics["prompt"] = prompt
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items() if k != "prompt"))
        rows.append(metrics)

    means = {
        k: sum(r[k] for r in rows) / len(rows)
        for k in ("dynamic_degree", "subject_consistency", "temporal_smoothness")
    }
    args.output.write_text(json.dumps({
        "model_id":   args.model_id,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "means":      means,
        "results":    rows,
    }, indent=2))
    print(f"\nResults → {args.output}")
    for k, v in means.items():
        print(f"  mean {k}: {v:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
