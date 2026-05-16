"""
Side-by-side base vs LoRA evaluation.

Runs ``eval_chronomagic`` and ``eval_vbench`` twice — once on the base
Wan2.2 pipeline, once with a LoRA adapter attached — and prints a delta
table.

This is the standard "did the fine-tune actually help" sanity check:

  MTScore-proxy        — should go *up*  (more metamorphic change)
  dynamic_degree       — should go *up*  (more motion in the output)
  subject_consistency  — should stay similar (~0.7-0.95 healthy range)
  temporal_smoothness  — should stay similar (≥0.99 = no jumps)

If subject_consistency or temporal_smoothness collapses while MTScore
shoots up, the LoRA over-fit and is hallucinating frame-to-frame.

Usage
-----
python -m videogen.eval_compare \\
    --checkpoint checkpoints/wan22_lora/step-002000 \\
    --n-prompts 8 --steps 20 --output compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable


_DEFAULT_PROMPTS = [
    "An ice cube slowly melting into water on a warm sunny day.",
    "Butter melting in a hot pan, golden bubbles forming as it browns.",
    "Hot wax dripping down the side of a candle, pooling at the base.",
    "Snow on a black rooftop melting away as the morning sun rises.",
]


def _load_pipeline(model_id: str, dtype: str, lora_dir: Path | None):
    import torch
    from diffusers import WanPipeline
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=td).to("cuda")
    if lora_dir is not None and Path(lora_dir).exists():
        pipe.load_lora_weights(str(lora_dir))
    return pipe


def _score_clip(pipe, prompt: str, *, num_frames: int, height: int,
                width: int, steps: int, guidance: float) -> dict:
    """Generate one clip and score it with our cheap proxies."""
    from videogen.eval_chronomagic import _mtscore_proxy
    from videogen.eval_vbench import _vbench_proxies
    t0 = time.perf_counter()
    result = pipe(prompt=prompt, num_frames=num_frames,
                  height=height, width=width,
                  num_inference_steps=steps, guidance_scale=guidance)
    frames = result.frames[0] if hasattr(result, "frames") else result
    return {
        "wall_clock_s": time.perf_counter() - t0,
        "mtscore_proxy": _mtscore_proxy(frames),
        **_vbench_proxies(frames),
    }


def _mean(rows: Iterable[dict], key: str) -> float:
    rows = list(rows)
    return sum(r[key] for r in rows) / max(len(rows), 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",   default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to peft adapter dir (adapter_model.safetensors).")
    p.add_argument("--prompts-file", type=Path, default=None)
    p.add_argument("--n-prompts",  type=int,   default=4)
    p.add_argument("--num-frames", type=int,   default=49)
    p.add_argument("--height",     type=int,   default=480)
    p.add_argument("--width",      type=int,   default=704)
    p.add_argument("--steps",      type=int,   default=20)
    p.add_argument("--guidance",   type=float, default=6.0)
    p.add_argument("--dtype",      default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--output",     type=Path, default=Path("compare.json"))
    args = p.parse_args(argv)

    if args.prompts_file is not None:
        prompts = [ln.strip() for ln in args.prompts_file.read_text().splitlines() if ln.strip()]
    else:
        prompts = _DEFAULT_PROMPTS
    prompts = prompts[: args.n_prompts]

    print(f"Eval set: {len(prompts)} prompts")
    print(f"Checkpoint: {args.checkpoint}\n")

    def _eval_with(label: str, lora: Path | None) -> list[dict]:
        print(f"=== {label} ===")
        pipe = _load_pipeline(args.model_id, args.dtype, lora)
        rows = []
        for i, prompt in enumerate(prompts):
            row = _score_clip(pipe, prompt,
                              num_frames=args.num_frames,
                              height=args.height, width=args.width,
                              steps=args.steps, guidance=args.guidance)
            row["prompt"] = prompt
            rows.append(row)
            print(f"  [{i + 1}/{len(prompts)}] mtscore={row['mtscore_proxy']:.3f}  "
                  f"motion={row['dynamic_degree']:.2f}  "
                  f"consistency={row['subject_consistency']:.3f}  "
                  f"smoothness={row['temporal_smoothness']:.3f}  "
                  f"({row['wall_clock_s']:.1f}s)")
        del pipe
        import torch
        torch.cuda.empty_cache()
        return rows

    base_rows = _eval_with("baseline (no LoRA)",    None)
    ft_rows   = _eval_with("fine-tuned (w/ LoRA)",  args.checkpoint)

    metrics = ("mtscore_proxy", "dynamic_degree",
               "subject_consistency", "temporal_smoothness")
    print("\n" + "─" * 64)
    print(f"  {'metric':<24} {'baseline':>10} {'fine-tuned':>12} {'Δ':>10}")
    print("  " + "─" * 60)
    summary = {}
    for k in metrics:
        b = _mean(base_rows, k)
        f = _mean(ft_rows, k)
        d = f - b
        sign = "+" if d >= 0 else ""
        print(f"  {k:<24} {b:>10.4f} {f:>12.4f} {sign}{d:>9.4f}")
        summary[k] = {"baseline": b, "fine_tuned": f, "delta": d}

    out = {"model_id": args.model_id, "checkpoint": str(args.checkpoint),
           "summary": summary, "baseline": base_rows, "fine_tuned": ft_rows}
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWritten {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
