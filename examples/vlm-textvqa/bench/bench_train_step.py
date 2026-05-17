"""Cached vs raw train-step throughput.

Measures: forward + backward + step on a real LoRA-wrapped Qwen2.5-VL.

Two paths, same model, same batch size:

  * **cached** — read pre-computed ``vision_tower_hiddens`` from Lance,
    inject at <|image_pad|> positions, drop the vision tower entirely
    from the train process.

  * **raw**  — read jpeg bytes from Lance, run vision tower inline,
    then continue with the same LLM forward.

Reports samples/s and steady-state VRAM use for each path.

Usage:

    python -m bench.bench_train_step --db data/textvqa.lance \
        --bs 2 --steps 30 --mode both
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm.dataloader import LanceCachedLoader, LanceRawLoader
from vlm.schema import IMAGE_PX, LLM_TOKENS_PER_IMAGE, MAX_TEXT_TOKENS, VISION_HIDDEN

LOG = logging.getLogger("bench.train_step")

_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _build_model(drop_vision: bool, lora_r: int):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )
    if drop_vision:
        del model.model.visual
        model.model.visual = None
        torch.cuda.empty_cache()
    config = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.train()
    return model


def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Cached path
# ---------------------------------------------------------------------------

def _step_cached(model, batch, image_pad_id, optim):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    embed = base.model.get_input_embeddings()
    inputs_embeds = embed(batch.input_ids)
    mask = (batch.input_ids == image_pad_id).unsqueeze(-1).expand_as(inputs_embeds)
    vflat = batch.vision_hiddens.to(inputs_embeds.dtype).reshape(-1, inputs_embeds.shape[-1])
    inputs_embeds = inputs_embeds.masked_scatter(mask, vflat)
    out = model(inputs_embeds=inputs_embeds, attention_mask=batch.attention_mask,
                labels=batch.labels)
    out.loss.backward()
    optim.step(); optim.zero_grad(set_to_none=True)
    return out.loss.item()


# ---------------------------------------------------------------------------
# Raw path
# ---------------------------------------------------------------------------

def _raw_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(_QWEN_MODEL_ID)


def _step_raw(model, batch, processor, optim):
    images, questions, answers = batch.images, batch.questions, batch.answers
    # Build full SFT prompts then tokenise + run vision inline
    eos = processor.tokenizer.eos_token
    full_inputs = []
    full_labels = []
    pixel_values_all = []
    grid_thw_all = []
    for img, q, a in zip(images, questions, answers):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img,
             "min_pixels": IMAGE_PX * IMAGE_PX,
             "max_pixels": IMAGE_PX * IMAGE_PX},
            {"type": "text",  "text": q},
        ]}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
        prompt_ids = inputs["input_ids"][0].tolist()
        ans_ids = processor.tokenizer(a + eos, add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + ans_ids)[:MAX_TEXT_TOKENS]
        lab = ([-100] * len(prompt_ids) + ans_ids)[:MAX_TEXT_TOKENS]
        pad = MAX_TEXT_TOKENS - len(full_ids)
        full_ids += [processor.tokenizer.eos_token_id] * pad
        lab      += [-100] * pad
        full_inputs.append(full_ids)
        full_labels.append(lab)
        pixel_values_all.append(inputs["pixel_values"])
        grid_thw_all.append(inputs["image_grid_thw"])
    input_ids = torch.tensor(full_inputs, dtype=torch.long, device="cuda:0")
    labels    = torch.tensor(full_labels, dtype=torch.long, device="cuda:0")
    pixel_values = torch.cat(pixel_values_all, dim=0).to("cuda:0", torch.bfloat16)
    image_grid_thw = torch.cat(grid_thw_all, dim=0).to("cuda:0")
    attn_mask = (input_ids != processor.tokenizer.eos_token_id).long()
    # If the right-pad EOS collides with real EOS, that's fine for VRAM bench.

    out = model(
        input_ids=input_ids, attention_mask=attn_mask, labels=labels,
        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
    )
    out.loss.backward()
    optim.step(); optim.zero_grad(set_to_none=True)
    return out.loss.item()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _bench(mode: str, db: str, bs: int, n_steps: int, lora_r: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    drop_vision = (mode == "cached")
    model = _build_model(drop_vision=drop_vision, lora_r=lora_r)
    optim = torch.optim.AdamW(_trainable(model), lr=1e-5)
    from transformers import AutoTokenizer
    image_pad_id = AutoTokenizer.from_pretrained(_QWEN_MODEL_ID).convert_tokens_to_ids("<|image_pad|>")

    if mode == "cached":
        loader = iter(LanceCachedLoader(db, batch_size=bs, seed=0, infinite=True))
    else:
        loader = iter(LanceRawLoader(db, batch_size=bs, seed=0, infinite=True))
        processor = _raw_processor()

    # warmup
    LOG.info("[%s] warming up …", mode)
    for _ in range(2):
        batch = next(loader)
        if mode == "cached":
            batch = batch.to(torch.device("cuda:0"))
            _step_cached(model, batch, image_pad_id, optim)
        else:
            _step_raw(model, batch, processor, optim)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    LOG.info("[%s] starting timed steps …", mode)

    t0 = time.time()
    n_samples = 0
    for s in range(n_steps):
        batch = next(loader)
        if mode == "cached":
            batch = batch.to(torch.device("cuda:0"))
            loss = _step_cached(model, batch, image_pad_id, optim)
            n_samples += batch.input_ids.size(0)
        else:
            loss = _step_raw(model, batch, processor, optim)
            n_samples += len(batch.images)
    torch.cuda.synchronize()
    wall = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    return {
        "mode":          mode,
        "steps":         n_steps,
        "batch_size":    bs,
        "samples":       n_samples,
        "wall_s":        wall,
        "samples_per_s": n_samples / wall,
        "steps_per_s":   n_steps / wall,
        "peak_vram_mb":  peak_mb,
        "final_loss":    loss,
    }


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--db",     default="data/textvqa.lance")
    p.add_argument("--bs",     type=int, default=2)
    p.add_argument("--steps",  type=int, default=30)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--mode",   default="both",
                   choices=["cached", "raw", "both"])
    p.add_argument("--out",    default="bench_outputs/train_step.json")
    args = p.parse_args()

    runs = ["cached", "raw"] if args.mode == "both" else [args.mode]
    results = []
    for m in runs:
        results.append(_bench(m, args.db, args.bs, args.steps, args.lora_r))
        torch.cuda.empty_cache()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    LOG.info("wrote %s", out)

    print()
    print(f"{'mode':<8}  {'bs':>3}  {'steps':>5}  {'samples':>7}  "
          f"{'wall_s':>7}  {'samp/s':>7}  {'step/s':>7}  {'peakVRAM':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['mode']:<8}  {r['batch_size']:>3}  {r['steps']:>5}  "
              f"{r['samples']:>7}  {r['wall_s']:>7.1f}  "
              f"{r['samples_per_s']:>7.2f}  {r['steps_per_s']:>7.2f}  "
              f"{r['peak_vram_mb']:>7.0f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
