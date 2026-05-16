"""
Side-by-side base vs LoRA evaluation.

Runs ``eval_chronomagic`` and ``eval_vbench`` twice — once on the base
Wan2.2 pipeline, once with a LoRA adapter attached — and prints a delta
table.  Optionally saves the generated clips and a side-by-side
``compare.mp4`` per prompt plus an ``index.html`` for easy viewing.

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

# With side-by-side mp4 + html viewer:
python -m videogen.eval_compare \\
    --checkpoint checkpoints/wan22_lora/step-002000 \\
    --n-prompts 8 --steps 20 \\
    --save-videos eval_outputs/run01
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


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


# ---------------------------------------------------------------------------
# Pipeline + scoring
# ---------------------------------------------------------------------------

def _load_pipeline(model_id: str, dtype: str, lora_dir: Path | None):
    import torch
    from diffusers import WanPipeline
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=td).to("cuda")
    if lora_dir is not None and Path(lora_dir).exists():
        pipe.load_lora_weights(str(lora_dir))
    return pipe


def _score_clip(pipe, prompt: str, *, num_frames: int, height: int,
                width: int, steps: int, guidance: float,
                save_path: Path | None = None, fps: int = 12) -> dict:
    """Generate one clip, optionally save it, and score with our proxies."""
    from videogen.eval_chronomagic import _mtscore_proxy
    from videogen.eval_vbench import _vbench_proxies
    t0 = time.perf_counter()
    result = pipe(prompt=prompt, num_frames=num_frames,
                  height=height, width=width,
                  num_inference_steps=steps, guidance_scale=guidance)
    frames = result.frames[0] if hasattr(result, "frames") else result
    wall = time.perf_counter() - t0
    if save_path is not None:
        _save_clip(frames, save_path, fps=fps)
    return {
        "wall_clock_s": wall,
        "mtscore_proxy": _mtscore_proxy(frames),
        **_vbench_proxies(frames),
    }


def _mean(rows: Iterable[dict], key: str) -> float:
    rows = list(rows)
    return sum(r[key] for r in rows) / max(len(rows), 1)


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------

def _save_clip(frames, path: Path, *, fps: int = 12) -> None:
    """Write a list of frames (PIL or np uint8) to mp4 via imageio[ffmpeg]."""
    import imageio.v3 as iio
    import numpy as np
    arrs = []
    for f in frames:
        if hasattr(f, "save"):  # PIL.Image
            a = np.asarray(f)
        else:
            a = np.asarray(f)
            if a.dtype != np.uint8:
                a = (a * 255).clip(0, 255).astype(np.uint8)
        arrs.append(a)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), np.stack(arrs), fps=fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p", "-crf", "23"])


def _ffmpeg_path() -> str:
    """Return a usable ffmpeg binary path."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit("ffmpeg not found — `uv pip install imageio-ffmpeg`")


def _hstack_videos(a: Path, b: Path, out: Path,
                   *, label_left: str = "baseline",
                   label_right: str = "fine-tuned") -> None:
    """Side-by-side via ffmpeg hstack + drawtext labels."""
    ff = _ffmpeg_path()
    # Escape colon/semicolon for ffmpeg filter strings
    def _esc(s: str) -> str:
        return s.replace(":", r"\:").replace("'", r"\'")
    filter_complex = (
        f"[0:v]drawtext=text='{_esc(label_left)}':fontcolor=white:fontsize=18:"
        f"box=1:boxcolor=black@0.5:x=10:y=10[a];"
        f"[1:v]drawtext=text='{_esc(label_right)}':fontcolor=white:fontsize=18:"
        f"box=1:boxcolor=black@0.5:x=10:y=10[b];"
        f"[a][b]hstack=inputs=2"
    )
    cmd = [
        ff, "-y",
        "-i", str(a), "-i", str(b),
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-loglevel", "error",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def _slugify(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.lower()).strip("_")
    return s[:maxlen] or "clip"


def _write_index_html(out_dir: Path, rows: list[dict]) -> Path:
    """Write a tiny index.html that embeds each compare.mp4."""
    cards = []
    for r in rows:
        prompt = r["prompt"]
        slug = r["slug"]
        m = r["metrics"]
        cards.append(f"""
<section>
  <h3>{prompt}</h3>
  <video controls width="800" preload="metadata">
    <source src="{slug}/compare.mp4" type="video/mp4">
  </video>
  <p>Δ mtscore <b>{m['mtscore_proxy']['delta']:+.4f}</b>
   · Δ motion <b>{m['dynamic_degree']['delta']:+.4f}</b>
   · Δ consistency <b>{m['subject_consistency']['delta']:+.4f}</b>
   · Δ smoothness <b>{m['temporal_smoothness']['delta']:+.6f}</b></p>
  <p>
    <a href="{slug}/baseline.mp4">baseline.mp4</a> ·
    <a href="{slug}/fine_tuned.mp4">fine_tuned.mp4</a> ·
    <a href="{slug}/compare.mp4">compare.mp4</a>
  </p>
</section>""")
    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>videogen eval — baseline vs fine-tuned</title>
<style>
 body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2em auto; }
 section { border-top: 1px solid #ddd; padding: 1em 0; }
 video { width: 100%; max-width: 800px; }
 h1 { font-size: 1.4em; } h3 { font-size: 1.05em; margin: 0.5em 0; }
</style></head>
<body>
<h1>videogen eval — baseline (left) vs fine-tuned (right)</h1>
""" + "\n".join(cards) + "</body></html>\n"
    p = out_dir / "index.html"
    p.write_text(html)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    p.add_argument("--fps",        type=int,   default=12,
                   help="Frame rate for saved mp4s.")
    p.add_argument("--dtype",      default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--output",     type=Path, default=Path("compare.json"))
    p.add_argument("--save-videos", type=Path, default=None,
                   help="Directory to write per-prompt baseline.mp4, "
                        "fine_tuned.mp4, compare.mp4 + index.html.")
    args = p.parse_args(argv)

    if args.prompts_file is not None:
        prompts = [ln.strip() for ln in args.prompts_file.read_text().splitlines() if ln.strip()]
    else:
        prompts = _DEFAULT_PROMPTS
    prompts = prompts[: args.n_prompts]

    print(f"Eval set: {len(prompts)} prompts")
    print(f"Checkpoint: {args.checkpoint}")
    if args.save_videos:
        args.save_videos.mkdir(parents=True, exist_ok=True)
        print(f"Save videos: {args.save_videos}")
    print()

    slugs = [_slugify(pr) for pr in prompts]

    def _eval_with(label: str, lora: Path | None, video_suffix: str) -> list[dict]:
        print(f"=== {label} ===")
        pipe = _load_pipeline(args.model_id, args.dtype, lora)
        rows = []
        for i, prompt in enumerate(prompts):
            sub = args.save_videos / slugs[i] if args.save_videos else None
            save_path = (sub / f"{video_suffix}.mp4") if sub else None
            row = _score_clip(pipe, prompt,
                              num_frames=args.num_frames,
                              height=args.height, width=args.width,
                              steps=args.steps, guidance=args.guidance,
                              save_path=save_path, fps=args.fps)
            row["prompt"] = prompt
            rows.append(row)
            print(f"  [{i + 1}/{len(prompts)}] mtscore={row['mtscore_proxy']:.3f}  "
                  f"motion={row['dynamic_degree']:.3f}  "
                  f"consistency={row['subject_consistency']:.3f}  "
                  f"smoothness={row['temporal_smoothness']:.4f}  "
                  f"({row['wall_clock_s']:.1f}s)")
        del pipe
        import torch
        torch.cuda.empty_cache()
        return rows

    base_rows = _eval_with("baseline (no LoRA)",    None,            "baseline")
    ft_rows   = _eval_with("fine-tuned (w/ LoRA)",  args.checkpoint, "fine_tuned")

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

    # ---- Side-by-side mp4 + index.html ------------------------------------
    if args.save_videos:
        print("\nStitching side-by-side mp4s …")
        html_rows = []
        for i, prompt in enumerate(prompts):
            sub = args.save_videos / slugs[i]
            base_p = sub / "baseline.mp4"
            ft_p   = sub / "fine_tuned.mp4"
            cmp_p  = sub / "compare.mp4"
            if base_p.exists() and ft_p.exists():
                try:
                    _hstack_videos(base_p, ft_p, cmp_p)
                except Exception as e:
                    print(f"  [{slugs[i]}] ffmpeg hstack failed: {e}")
                    continue
                # Per-prompt metric deltas
                per_metric = {}
                for k in metrics:
                    per_metric[k] = {
                        "baseline":   base_rows[i][k],
                        "fine_tuned": ft_rows[i][k],
                        "delta":      ft_rows[i][k] - base_rows[i][k],
                    }
                html_rows.append({"prompt": prompt, "slug": slugs[i],
                                  "metrics": per_metric})
                print(f"  [{slugs[i]}] ✓  {cmp_p}")
        idx = _write_index_html(args.save_videos, html_rows)
        print(f"\nIndex page: file://{idx.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
