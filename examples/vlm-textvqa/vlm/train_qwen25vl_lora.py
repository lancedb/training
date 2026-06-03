"""LoRA SFT of Qwen2.5-VL-3B on TextVQA, reading the cached Lance columns.

The vision tower is **not loaded** in this process.  Instead we:

  1. Pull ``vision_tower_hiddens`` (fp16[B, 400, 2048]) +
     ``input_ids`` (int64[B, 512]) + ``attention_mask`` + ``labels``
     from the Lance dataset (zero-copy where possible).

  2. Compute ``inputs_embeds = embed_tokens(input_ids)`` and use
     ``masked_scatter`` to overwrite every ``<|image_pad|>`` position
     with the corresponding cached vision hidden.

  3. Run the language model forward with ``inputs_embeds=``.  Pass
     ``labels=`` so HF computes the loss internally with the prompt
     span already masked to -100.

  4. LoRA on q/k/v/o of the LLM only.

Usage:

    python -m vlm.train_qwen25vl_lora \
        --db data/textvqa.lance --batch-size 2 --grad-accum 4 \
        --epochs 2 --out runs/textvqa_lora
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from .dataloader import make_cached_loader
from .schema import LLM_TOKENS_PER_IMAGE, MAX_TEXT_TOKENS, VISION_HIDDEN

LOG = logging.getLogger("vlm.train")

_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_IMAGE_PAD_TOKEN = "<|image_pad|>"


def _build_model(use_lora: bool, lora_r: int, load_4bit: bool = False):
    """Load Qwen2.5-VL with `model.model.visual = None`.

    The vision tower is ~1.3 GB; not loading it frees that much VRAM
    for activations / a bigger batch.

    ``load_4bit`` quantises the LLM weights to NF4 (bitsandbytes) so the
    3.75 B-param model + LoRA fits a free Colab T4 (16 GB).  On an H100
    leave it ``False`` and train in bf16.
    """
    kwargs = dict(attn_implementation="sdpa")
    if load_4bit:
        from transformers import BitsAndBytesConfig
        LOG.info("loading LLM in 4-bit (NF4) for low-VRAM training")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "cuda:0"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(_QWEN_MODEL_ID, **kwargs)

    # Free the vision tower weights.
    LOG.info("freeing vision tower (~%.0f MB)",
             sum(p.numel() * p.element_size()
                 for p in model.model.visual.parameters()) / 1e6)
    del model.model.visual
    model.model.visual = None
    torch.cuda.empty_cache()

    if load_4bit and use_lora:
        # QLoRA: cast norms to fp32, enable input grads, gradient checkpointing.
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
        )

    if use_lora:
        LOG.info("wrapping LLM with LoRA r=%d on q/k/v/o", lora_r)
        # peft applies LoRA to modules whose names contain these patterns.
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad = True

    return model


def _forward_cached(model, batch, image_pad_id: int):
    """Inject cached vision hiddens at <|image_pad|> positions and run LLM."""
    # Discover the actual nn.Module behind PEFT wrappers
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = base.model  # Qwen2_5_VLModel

    embed = inner.get_input_embeddings()
    inputs_embeds = embed(batch.input_ids)            # [B, T, D]
    B, T, D = inputs_embeds.shape

    # Mask = (input_ids == <|image_pad|>) broadcast over hidden dim
    mask = (batch.input_ids == image_pad_id).unsqueeze(-1).expand_as(inputs_embeds)

    # vision_hiddens: fp16[B, LLM_TOKENS_PER_IMAGE, D] -> bf16 to match LLM
    vision_flat = batch.vision_hiddens.to(inputs_embeds.dtype).reshape(-1, D)
    # masked_scatter consumes the matching number of elements row-major.
    inputs_embeds = inputs_embeds.masked_scatter(mask, vision_flat)

    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
    )
    return out.loss


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--db",         default="data/textvqa.lance")
    p.add_argument("--out",        default="runs/textvqa_lora")
    p.add_argument("--epochs",     type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--warmup",     type=float, default=0.03)
    p.add_argument("--lora-r",     type=int, default=64)
    p.add_argument("--no-lora",    action="store_true",
                   help="full fine-tune (debug only — does not fit comfortably)")
    p.add_argument("--load-4bit",  action="store_true",
                   help="QLoRA: quantise LLM to NF4 so it fits a Colab T4 (16 GB)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers (each reopens its own Permutation)")
    p.add_argument("--log-every",  type=int, default=10)
    p.add_argument("--max-steps",  type=int, default=0,
                   help="cap steps (0 = no cap)")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    LOG.info("config: %s", vars(args))

    # Tokeniser is needed only to look up the image_pad token id.
    tok = AutoTokenizer.from_pretrained(_QWEN_MODEL_ID)
    image_pad_id = tok.convert_tokens_to_ids(_IMAGE_PAD_TOKEN)
    LOG.info("image_pad id = %d", image_pad_id)

    # Model
    model = _build_model(use_lora=not args.no_lora, lora_r=args.lora_r,
                         load_4bit=args.load_4bit)
    model.train()

    # Optimiser only over trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    LOG.info("trainable params: %.1f M", n_trainable / 1e6)
    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # Data — LanceDB Permutation API, shuffled per epoch.
    loader = make_cached_loader(
        args.db, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, seed=args.seed,
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    LOG.info("steps_per_epoch=%d  total_steps=%d", steps_per_epoch, total_steps)
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=max(1, int(args.warmup * total_steps)),
        num_training_steps=total_steps,
    )

    device = torch.device("cuda:0")
    step = 0
    t0 = time.time()
    running_loss = 0.0
    samples_seen = 0
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w")

    for epoch in range(args.epochs):
        LOG.info("=== epoch %d/%d ===", epoch + 1, args.epochs)
        accum_count = 0
        optim.zero_grad(set_to_none=True)
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            loss = _forward_cached(model, batch, image_pad_id)
            (loss / args.grad_accum).backward()
            accum_count += 1
            running_loss += loss.item() * batch.input_ids.size(0)
            samples_seen += batch.input_ids.size(0)

            if accum_count >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                accum_count = 0
                step += 1

                if step % args.log_every == 0 or step == total_steps:
                    elapsed = time.time() - t0
                    sps = samples_seen / elapsed
                    rec = {
                        "step": step,
                        "epoch": epoch + (step / steps_per_epoch),
                        "loss":  running_loss / max(samples_seen, 1),
                        "lr":    sched.get_last_lr()[0],
                        "samples_per_s": sps,
                        "elapsed_s": elapsed,
                    }
                    LOG.info("step %4d  loss=%.4f  lr=%.2e  sps=%.2f",
                             step, rec["loss"], rec["lr"], sps)
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                    running_loss = 0.0
                    samples_seen = 0
                    t0 = time.time()

                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    log_f.close()

    # Save LoRA adapter (small)
    save_dir = out_dir / "lora"
    save_dir.mkdir(exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(save_dir))
        LOG.info("saved adapter to %s", save_dir)
    else:
        torch.save(model.state_dict(), save_dir / "model.pt")

    LOG.info("training done.  steps=%d", step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
