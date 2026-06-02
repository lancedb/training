"""Single-process Tier-3 backfill — the fallback for ``backfill_geneva``.

The default Tier-3 path is Geneva (``vlm/backfill_geneva.py --tier 3``),
which distributes the vision-tower forward across its actor pool.  This
module is the single-process equivalent: it uses Lance's
``add_columns(transform, read_columns=...)`` API to batch reads from the
Lance file, run the transform in-process, and write the output columns as
fresh fragments — no Ray, no actor pool.

Use this when you're on a single box (e.g. the Colab bake in
``vlm/colab_prepare.py``), or to sidestep Ray if an actor-pool issue
shows up.  It writes the same four columns as the Geneva path's
``vision_tower_hiddens`` + ``sft_tokens`` UDFs, just as flat columns.

One combined transform writes four columns from a single image decode +
processor call:

  * ``vision_tower_hiddens``  fp16[LLM_TOKENS_PER_IMAGE * VISION_HIDDEN]
  * ``input_ids``             int32[MAX_TEXT_TOKENS]   ── full chat template
  * ``attention_mask``        int8 [MAX_TEXT_TOKENS]
  * ``labels``                int32[MAX_TEXT_TOKENS]   ── prompt masked

The full Qwen chat prompt includes ``<|vision_start|> [image_pad]*N
<|vision_end|>`` between the system header and the question, so the
cached path can inject ``vision_tower_hiddens`` at those placeholder
positions in ``inputs_embeds`` and skip the vision tower entirely.

Usage:

    python -m vlm.backfill_direct --db data/textvqa.lance --batch-size 8
"""
from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from .schema import (
    IMAGE_PX,
    LLM_TOKENS_PER_IMAGE,
    MAX_TEXT_TOKENS,
    VISION_HIDDEN,
)

LOG = logging.getLogger("vlm.backfill_direct")

_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_IGNORE = -100


class _Tier3Pipeline:
    """Lazy-loaded Qwen processor + vision tower.  Batched."""

    def __init__(self) -> None:
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        from transformers import (
            Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer,
        )
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
        )
        self._vision = m.model.visual.eval()
        self._vision_dtype = next(self._vision.parameters()).dtype
        self._processor = AutoProcessor.from_pretrained(_QWEN_MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(_QWEN_MODEL_ID)
        self._eos = self._tokenizer.eos_token_id
        self._loaded = True
        LOG.info(
            "tier3 pipeline loaded (vision %.0f MB)",
            sum(p.numel() * p.element_size() for p in self._vision.parameters()) / 1e6,
        )

    @torch.no_grad()
    def run_batch(self, image_bytes_list: list[bytes],
                  questions: list[str], answers: list[str]) -> dict[str, np.ndarray]:
        self._load()
        n = len(image_bytes_list)

        # 1) Decode every image to a fixed-size RGB.
        images = [
            Image.open(io.BytesIO(b)).convert("RGB").resize(
                (IMAGE_PX, IMAGE_PX), Image.LANCZOS
            ) for b in image_bytes_list
        ]

        # 2) Build chat prompts and tokenise the whole batch in one shot.
        prompts = []
        for img, q in zip(images, questions):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img,
                 "min_pixels": IMAGE_PX * IMAGE_PX, "max_pixels": IMAGE_PX * IMAGE_PX},
                {"type": "text", "text": q},
            ]}]
            prompts.append(self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        proc = self._processor(text=prompts, images=images, return_tensors="pt", padding=True)
        # NOTE: processor returns padded input_ids of shape [n, max_len].
        prompt_ids_padded = proc["input_ids"]  # may be padded with eos / pad
        prompt_attn       = proc["attention_mask"]
        # Per-row valid prompt length (excluding right-padding)
        prompt_lens = prompt_attn.sum(dim=1).tolist()

        # 3) Vision tower forward — pixel_values is already a single
        # concatenated tensor; one call covers all images.
        v_out = self._vision(
            proc["pixel_values"].to(self._vision_dtype).cuda(),
            grid_thw=proc["image_grid_thw"].cuda(),
        ).pooler_output  # [n * LLM_TOKENS_PER_IMAGE, VISION_HIDDEN]
        v_fp16 = v_out.to(torch.float16).cpu().numpy().reshape(
            n, LLM_TOKENS_PER_IMAGE * VISION_HIDDEN
        )

        # 4) Append answer + EOS per row, build labels.
        flat_ids = np.full((n, MAX_TEXT_TOKENS), self._eos, dtype=np.int32)
        flat_atn = np.zeros((n, MAX_TEXT_TOKENS),            dtype=np.int8)
        flat_lab = np.full((n, MAX_TEXT_TOKENS), _IGNORE,    dtype=np.int32)

        for i in range(n):
            plen = int(prompt_lens[i])
            prompt_ids = prompt_ids_padded[i, :plen].tolist()
            ans_ids = self._tokenizer(answers[i] + self._tokenizer.eos_token,
                                       add_special_tokens=False)["input_ids"]
            full_ids = (prompt_ids + ans_ids)[:MAX_TEXT_TOKENS]
            lab      = ([_IGNORE] * len(prompt_ids) + ans_ids)[:MAX_TEXT_TOKENS]
            if len(full_ids) > MAX_TEXT_TOKENS:
                full_ids = full_ids[:MAX_TEXT_TOKENS]
                lab      = lab     [:MAX_TEXT_TOKENS]
            flat_ids[i, :len(full_ids)] = full_ids
            flat_atn[i, :len(full_ids)] = 1
            flat_lab[i, :len(lab)]      = lab

        return {
            "vision_tower_hiddens": v_fp16,
            "input_ids":            flat_ids,
            "attention_mask":       flat_atn,
            "labels":               flat_lab,
        }


def _make_tier3_transform():
    pipe = _Tier3Pipeline()
    v_dim = LLM_TOKENS_PER_IMAGE * VISION_HIDDEN

    def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
        images    = batch.column("image").to_pylist()
        questions = batch.column("question").to_pylist()
        answers   = batch.column("answer").to_pylist()

        out = pipe.run_batch(images, questions, answers)

        def _fsl(flat: np.ndarray, dtype, width: int) -> pa.FixedSizeListArray:
            return pa.FixedSizeListArray.from_arrays(
                pa.array(flat.reshape(-1), type=dtype), width
            )

        return pa.RecordBatch.from_arrays(
            [
                _fsl(out["vision_tower_hiddens"], pa.float16(), v_dim),
                _fsl(out["input_ids"],            pa.int32(),   MAX_TEXT_TOKENS),
                _fsl(out["attention_mask"],       pa.int8(),    MAX_TEXT_TOKENS),
                _fsl(out["labels"],               pa.int32(),   MAX_TEXT_TOKENS),
            ],
            names=["vision_tower_hiddens", "input_ids", "attention_mask", "labels"],
        )

    return transform


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--db",         default="data/textvqa.lance")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    db_path = str(Path(args.db).resolve())
    ds = lance.dataset(db_path)
    LOG.info("opened %s (rows=%d)", db_path, ds.count_rows())

    needed = {"vision_tower_hiddens", "input_ids", "attention_mask", "labels"}
    have = set(ds.schema.names)
    missing = needed - have
    if not missing:
        LOG.info("all tier-3 columns already present — nothing to do")
        return 0
    if missing != needed:
        # partial state: tear it down so add_columns is atomic
        LOG.warning("partial tier-3 columns present (%s); dropping and rerunning",
                    needed & have)
        ds.drop_columns(list(needed & have))
        ds = lance.dataset(db_path)

    transform = _make_tier3_transform()
    t0 = time.time()
    ds.add_columns(
        transform,
        read_columns=["image", "question", "answer"],
        batch_size=args.batch_size,
    )
    LOG.info("tier-3 backfill done in %.1fs", time.time() - t0)
    ds = lance.dataset(db_path)
    LOG.info("FINAL columns: %s", ds.schema.names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
