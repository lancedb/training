"""
Wan2.2-TI2V-5B LoRA fine-tune on a Lance-cached training view.

This is the **headline training loop**: it reads the two pre-computed
columns (``t5_hidden_states`` + ``vae_latent``) from a Geneva
materialised view and feeds them straight to the Wan DiT.  The VAE and
the UMT5 text encoder are **never loaded** in this process — all the
H100's memory goes to the transformer, the LoRA adapter, and the
optimiser state.

Compared to the classic diffusion-pipe-style loop (decode mp4 → VAE
encode → UMT5 encode → DiT), this:

  1. Removes ~12 GB of VRAM (UMT5-XXL ~11 GB + Wan-VAE ~3 GB) that
     would otherwise sit idle on the GPU after one forward pass per row.
  2. Removes the per-step decode+encode CPU/GPU work that starves the
     DiT on consumer-class machines.
  3. Lets us run a noticeably bigger batch / longer context.

Usage (smoke-sized)
-------------------
python -m videogen.train_wan22_lora \\
    --db data/videos/lancedb \\
    --train-view videos_raw \\
    --steps 20 --batch-size 1 \\
    --rank 32 --alpha 32 --lr 1e-4 \\
    --output-dir checkpoints/smoke

The training script logs ``view.version`` with every checkpoint so the
weights link back to an exact data snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import lancedb
import torch
import torch.nn.functional as F

from videogen.dataloader import make_cached_loader


WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_transformer(*, dtype: torch.dtype, device: torch.device):
    """Load only the Wan DiT (no VAE, no text encoder)."""
    from diffusers import WanTransformer3DModel
    print(f"Loading WanTransformer3DModel ({WAN_MODEL_ID})…")
    t0 = time.time()
    model = WanTransformer3DModel.from_pretrained(
        WAN_MODEL_ID, subfolder="transformer", torch_dtype=dtype,
    ).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time() - t0:.1f}s  ({n / 1e9:.2f}B params, {dtype})")
    return model


def attach_lora(model, *, rank: int, alpha: int, targets: list[str]):
    """Wrap the transformer with peft LoRA on the named attention projections."""
    from peft import LoraConfig, get_peft_model

    # Freeze the base model first.
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        bias="none",
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={rank} α={alpha}  trainable={trainable / 1e6:.2f}M "
          f"({trainable / total * 100:.3f}% of {total / 1e9:.2f}B)")
    return model


# ---------------------------------------------------------------------------
# Flow-matching training step (rectified flow / Wan-style velocity target)
# ---------------------------------------------------------------------------

def flow_matching_step(
    model,
    batch: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One forward+backward step using a velocity (z1 - z0) target.

    Wan2.2 trains with rectified flow: at time ``t`` the noisy sample is
    ``z_t = (1 - t) * z_0 + t * z_1``, and the model predicts the
    velocity ``v = z_1 - z_0``.  We pass the unscaled timestep as a long
    tensor in the [0, 1000] range to match what the model was trained
    against (the scheduler is sigma-free in flow matching).
    """
    z0 = batch["vae_latent"].to(device=device, dtype=dtype, non_blocking=True)
    ctx = batch["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)

    bsz = z0.shape[0]
    # Uniform t in (0, 1); we keep both the continuous t (for mixing) and
    # the long-scaled timestep (for the model's time embedding).
    t = torch.rand(bsz, device=device, dtype=dtype).clamp(1e-3, 1.0 - 1e-3)
    t_long = (t * 1000.0).long()

    z1 = torch.randn_like(z0)
    t_b = t.view(bsz, 1, 1, 1, 1)
    z_t = (1.0 - t_b) * z0 + t_b * z1
    target = z1 - z0  # velocity

    pred = model(
        hidden_states=z_t,
        timestep=t_long,
        encoder_hidden_states=ctx,
    ).sample

    return F.mse_loss(pred.float(), target.float())


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    out_dir: Path,
    *,
    step: int,
    train_view_version: int,
    db_path: str,
    train_view: str,
    extra: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    step_dir = out_dir / f"step-{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # PEFT exposes save_pretrained for the adapter only.
    model.save_pretrained(step_dir)

    meta = {
        "step": step,
        "model_id": WAN_MODEL_ID,
        "db": db_path,
        "train_view": train_view,
        "train_view_version": train_view_version,
        "lora_targets": DEFAULT_LORA_TARGETS,
        **(extra or {}),
    }
    (step_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"  [ckpt] step={step:06d}  → {step_dir}  (view version {train_view_version})")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _infinite(loader: Iterable):
    """Re-iterate the loader forever."""
    while True:
        for batch in loader:
            yield batch


def _setup_distributed() -> tuple[torch.device, int, int]:
    """If launched via ``accelerate launch`` / ``torchrun``, init DDP and
    return (device, rank, world_size).  Otherwise return defaults."""
    import os
    if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world      = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), rank, world


def run(args) -> None:
    device, rank, world = _setup_distributed()
    is_main = rank == 0
    dtype  = torch.bfloat16 if (device.type == "cuda"
                                and torch.cuda.get_device_capability(device)[0] >= 8) else torch.float16
    if is_main:
        print(f"Device: {device}  dtype: {dtype}  world_size={world}")

    # ---- Data --------------------------------------------------------------
    db = lancedb.connect(args.db)
    train_tbl = db.open_table(args.train_view)
    train_view_version = train_tbl.version
    if is_main:
        print(f"Train view '{args.train_view}'  rows={len(train_tbl)}  "
              f"version={train_view_version}")

    loader = make_cached_loader(
        uri=args.db, table_name=args.train_view,
        batch_size=args.batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, shuffle=True,
        seed=args.seed + rank,  # different shuffle per rank
    )
    if is_main:
        print(f"  loader: bs={args.batch_size}  workers={args.num_workers}  "
              f"prefetch={args.prefetch_factor}  steps/epoch≈{len(loader)}")

    # ---- Model + LoRA ------------------------------------------------------
    model = build_transformer(dtype=dtype, device=device)
    model = attach_lora(model, rank=args.rank, alpha=args.alpha,
                        targets=args.lora_targets or DEFAULT_LORA_TARGETS)
    model.train()

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if is_main:
                print("  gradient checkpointing enabled")
        except Exception as e:
            if is_main:
                print(f"  gradient checkpointing not available: {e}")

    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index],
                    find_unused_parameters=False)
        if is_main:
            print(f"  wrapped in DDP across {world} ranks")

    # ---- Optimiser ---------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, fused=True,
    )

    # ---- Train -------------------------------------------------------------
    if is_main:
        print(f"\n=== Training {args.steps} steps ===")
    iterator = _infinite(loader)

    losses: list[float] = []
    t_train_start = time.perf_counter()
    last_log = t_train_start

    for step in range(1, args.steps + 1):
        batch = next(iterator)
        loss = flow_matching_step(model, batch, device=device, dtype=dtype)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        losses.append(loss.item())
        if is_main and (step % args.log_every == 0 or step == 1):
            now = time.perf_counter()
            dt = now - last_log
            last_log = now
            steps_per_sec = args.log_every / dt if step > 1 else 1.0 / dt
            recent = losses[-args.log_every:]
            print(f"  step {step:5d}/{args.steps}  loss {sum(recent)/len(recent):.4f}  "
                  f"({steps_per_sec:.2f} steps/s)")

        should_save = is_main and (
            step == args.steps or (args.save_every and step % args.save_every == 0)
        )
        if should_save:
            inner = model.module if hasattr(model, "module") else model
            save_checkpoint(
                inner, Path(args.output_dir),
                step=step,
                train_view_version=train_view_version,
                db_path=args.db,
                train_view=args.train_view,
                extra={
                    "recent_loss_mean": float(
                        sum(losses[-20:]) / max(len(losses[-20:]), 1)
                    ),
                    "world_size": world,
                },
            )

    if world > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if is_main:
        total = time.perf_counter() - t_train_start
        per_step_global = args.steps * args.batch_size * world / total
        print(f"\nDone — {args.steps} steps in {total:.1f}s "
              f"({args.steps / total:.2f} steps/s/rank, "
              f"global samples/s = {per_step_global:.2f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B LoRA training on a Lance-cached view.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db",         default="data/videos/lancedb")
    p.add_argument("--train-view", default="phase_transitions_train")
    p.add_argument("--output-dir", default="checkpoints/wan22_lora")

    p.add_argument("--steps",       type=int, default=200)
    p.add_argument("--batch-size",  type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)

    p.add_argument("--rank",        type=int, default=32)
    p.add_argument("--alpha",       type=int, default=32)
    p.add_argument("--lora-targets", nargs="+", default=None,
                   help="Module names to attach LoRA to.  Defaults to "
                        "to_q/to_k/to_v/to_out.0.")

    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action="store_true")

    p.add_argument("--log-every",   type=int, default=10)
    p.add_argument("--save-every",  type=int, default=0,
                   help="Save a checkpoint every N steps.  0 = only at end.")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
