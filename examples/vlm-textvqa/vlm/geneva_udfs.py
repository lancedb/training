"""Geneva UDFs for the vlm-textvqa feature-engineering pipeline.

Three tiers — UDF code in this file, registry below.  Tier 1 is CPU-only
and runs in seconds.  Tier 2 decodes the JPEG image once for dhash.
Tier 3 is the headline column: ``vision_tower_hiddens`` from
Qwen2.5-VL's frozen ViT.

Tier-2 / Tier-3 GPU UDFs are stateful classes that load model weights
lazily in ``__call__`` so the driver process never tries to grab a GPU.
"""
from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import pyarrow as pa
from PIL import Image
from geneva.transformer import udf

from .schema import (
    IMAGE_PX,
    LLM_TOKENS_PER_IMAGE,
    MAX_TEXT_TOKENS,
    VISION_HIDDEN,
)


# ---------------------------------------------------------------------------
# Tier 1 — CPU, text-only
# ---------------------------------------------------------------------------

# Question-type classification.  TextVQA questions are scene-text Q&A so
# the long tail is small — these patterns cover ~95% of the train set.
_QUESTION_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("how_many",     re.compile(r"^\s*how\s+many\b",            re.IGNORECASE)),
    ("what_color",   re.compile(r"^\s*what\s+(is\s+the\s+)?color\b", re.IGNORECASE)),
    ("what_time",    re.compile(r"^\s*what\s+(is\s+the\s+)?time\b",  re.IGNORECASE)),
    ("what_number",  re.compile(r"^\s*what\s+number\b",         re.IGNORECASE)),
    ("what_brand",   re.compile(r"^\s*what\s+(is\s+the\s+)?(brand|company|make)\b", re.IGNORECASE)),
    ("what_letter",  re.compile(r"^\s*what\s+(letter|word|letters|words)\b", re.IGNORECASE)),
    ("what",         re.compile(r"^\s*what\b",                  re.IGNORECASE)),
    ("which",        re.compile(r"^\s*which\b",                 re.IGNORECASE)),
    ("who",          re.compile(r"^\s*who\b",                   re.IGNORECASE)),
    ("where",        re.compile(r"^\s*where\b",                 re.IGNORECASE)),
    ("when",         re.compile(r"^\s*when\b",                  re.IGNORECASE)),
    ("is_does",      re.compile(r"^\s*(is|are|does|do|can)\b",  re.IGNORECASE)),
]


@udf(data_type=pa.int32(), input_columns=["question"])
def question_length(question: str) -> int:
    return len(question) if question else 0


@udf(data_type=pa.int32(), input_columns=["answer"])
def answer_length(answer: str) -> int:
    return len(answer) if answer else 0


@udf(data_type=pa.string(), input_columns=["question"])
def question_type(question: str) -> str:
    if not question:
        return "other"
    for label, pat in _QUESTION_TYPE_PATTERNS:
        if pat.search(question):
            return label
    return "other"


@udf(data_type=pa.int32(), input_columns=["ocr_tokens"])
def ocr_token_count(ocr_tokens: list[str] | None) -> int:
    return len(ocr_tokens) if ocr_tokens else 0


TIER1_UDFS: dict[str, Any] = {
    "question_length": question_length,
    "answer_length":   answer_length,
    "question_type":   question_type,
    "ocr_token_count": ocr_token_count,
}


# ---------------------------------------------------------------------------
# Tier 2 — light: image decode + perceptual hash
# ---------------------------------------------------------------------------
#
# 64-bit difference hash: shrink to 9x8 grayscale, compute horizontal diffs,
# pack as bits into a uint64.  Two near-duplicate images share most bits.

@udf(data_type=pa.uint64(), input_columns=["image"])
def dhash(image: bytes) -> int:
    if not image:
        return 0
    img = Image.open(io.BytesIO(image)).convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()  # 8 * 8 = 64 bits
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out & ((1 << 64) - 1)


# ---------------------------------------------------------------------------
# Tier 3 — heavy GPU.  Headline columns.
# ---------------------------------------------------------------------------
#
# vision_tower_hiddens: run Qwen2.5-VL's frozen ViT once at backfill,
#   store the merger output as fp16[LLM_TOKENS_PER_IMAGE x VISION_HIDDEN].
#
# input_ids / attention_mask / labels: pre-tokenise the SFT prompt
#   ("question -> answer") with prompt tokens masked to -100 in `labels`.
#
# Both are stateful class UDFs so the model loads lazily in the worker
# process, not in the driver.

_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_IGNORE = -100
_IMAGE_PAD = "<|image_pad|>"


class VisionTowerEmbedder:
    """Decode JPEG -> fixed-size square image -> Qwen ViT merger output."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self._torch = torch
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
        ).model.visual.eval()
        self._processor = AutoProcessor.from_pretrained(_QWEN_MODEL_ID)
        self._dtype = next(self._model.parameters()).dtype

    def __call__(self, image: bytes) -> list[float]:
        self._lazy_load()
        if not image:
            return [0.0] * (LLM_TOKENS_PER_IMAGE * VISION_HIDDEN)
        img = Image.open(io.BytesIO(image)).convert("RGB")
        # Square-resize to IMAGE_PX so all rows share the cached shape.
        img = img.resize((IMAGE_PX, IMAGE_PX), Image.LANCZOS)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img,
             "min_pixels": IMAGE_PX * IMAGE_PX,
             "max_pixels": IMAGE_PX * IMAGE_PX},
            {"type": "text", "text": "x"},
        ]}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")
        with self._torch.no_grad():
            out = self._model(
                inputs["pixel_values"].to(self._dtype),
                grid_thw=inputs["image_grid_thw"],
            )
        pooled = out.pooler_output  # [N_tokens, VISION_HIDDEN]
        assert pooled.shape == (LLM_TOKENS_PER_IMAGE, VISION_HIDDEN), (
            f"unexpected pooler shape {tuple(pooled.shape)} "
            f"(expected ({LLM_TOKENS_PER_IMAGE}, {VISION_HIDDEN}))"
        )
        return pooled.to(self._torch.float16).flatten().cpu().numpy().tolist()


# Wrap as a Geneva UDF with explicit fixed-size output.
vision_tower_hiddens = udf(
    data_type=pa.list_(pa.float16(), LLM_TOKENS_PER_IMAGE * VISION_HIDDEN),
    input_columns=["image"],
)(VisionTowerEmbedder)


class SFTTokenizer:
    """Pre-tokenise the SFT pair into (input_ids, attention_mask, labels).

    The prompt is the **vision-aware** Qwen2.5-VL chat template: an image
    message followed by the question, so the template emits
    ``<|vision_start|>`` + ``LLM_TOKENS_PER_IMAGE`` × ``<|image_pad|>`` +
    ``<|vision_end|>`` before the question.  Those ``<|image_pad|>``
    positions are exactly where the train loop ``masked_scatter``-injects
    the cached ``vision_tower_hiddens`` — so the model actually sees the
    image.  (The image side is fixed to ``IMAGE_PX`` → ``LLM_TOKENS_PER_IMAGE``
    merged tokens, so we expand the single template placeholder to that many
    without needing the pixels here.)

    Returns a single struct so Geneva writes three columns from one call.
    Prompt tokens (incl. the image pads) are masked to -100 in ``labels`` so
    loss is only on the answer span.
    """

    def __init__(self) -> None:
        self._proc = None
        self._tok = None

    def _lazy_load(self) -> None:
        if self._tok is not None:
            return
        from transformers import AutoProcessor
        self._proc = AutoProcessor.from_pretrained(_QWEN_MODEL_ID)
        self._tok = self._proc.tokenizer

    def __call__(self, question: str, answer: str) -> dict[str, list[int]]:
        self._lazy_load()
        question = question or ""
        answer = answer or ""

        # Vision-aware chat template (image message + question). The template
        # writes one <|image_pad|>; expand it to LLM_TOKENS_PER_IMAGE so the
        # placeholders line up 1:1 with the cached vision hiddens.
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]}]
        prompt = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt = prompt.replace(_IMAGE_PAD, _IMAGE_PAD * LLM_TOKENS_PER_IMAGE)
        prompt_ids = self._tok(prompt, add_special_tokens=False)["input_ids"]
        ans_ids    = self._tok(answer + self._tok.eos_token,
                               add_special_tokens=False)["input_ids"]

        full_ids = (prompt_ids + ans_ids)[:MAX_TEXT_TOKENS]
        full_lab = ([_IGNORE] * len(prompt_ids) + ans_ids)[:MAX_TEXT_TOKENS]
        attn     = [1] * len(full_ids)

        # Right-pad with EOS to MAX_TEXT_TOKENS.
        pad = MAX_TEXT_TOKENS - len(full_ids)
        if pad > 0:
            full_ids += [self._tok.eos_token_id] * pad
            full_lab += [_IGNORE] * pad
            attn     += [0] * pad

        return {
            "input_ids":      full_ids,
            "attention_mask": attn,
            "labels":         full_lab,
        }


_SFT_STRUCT = pa.struct([
    pa.field("input_ids",      pa.list_(pa.int32(), MAX_TEXT_TOKENS)),
    pa.field("attention_mask", pa.list_(pa.int8(),  MAX_TEXT_TOKENS)),
    pa.field("labels",         pa.list_(pa.int32(), MAX_TEXT_TOKENS)),
])

sft_tokens = udf(
    data_type=_SFT_STRUCT,
    input_columns=["question", "answer"],
)(SFTTokenizer)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TIER2_UDFS: dict[str, Any] = {
    "dhash": dhash,
}

TIER3_UDFS: dict[str, Any] = {
    "vision_tower_hiddens": vision_tower_hiddens,
    "sft_tokens":           sft_tokens,
}

ALL_UDFS: dict[str, Any] = {**TIER1_UDFS, **TIER2_UDFS, **TIER3_UDFS}
