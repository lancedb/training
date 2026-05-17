"""Schemas for the vlm-textvqa Lance dataset.

The source corpus comes from ``hf://datasets/lance-format/textvqa-lance``
which already ships image bytes, CLIP embeddings and OCR tokens.  We
add Geneva-backfilled columns on top of that.

Vision-tower output shape is locked to ``fp16[400, 2048]``:

    560 x 560 px image
    -> Qwen2.5-VL patch=14, spatial_merge=2  ->  40 x 40 = 1600 raw patches
    -> merger downsamples 4x  ->  400 LLM tokens
    -> projector lifts to LLM hidden size 2048

so the cached column is a single fixed-size-list per row that the train
loop can read zero-copy.  Total disk: 1.64 MB/row x 34,602 rows ~ 57 GB.

If you want higher OCR fidelity at the cost of disk + a smaller batch,
bump ``IMAGE_PX`` to 672 (-> 576 tokens, 82 GB) or 784 (-> 784 tokens,
110 GB) and rerun the Tier-3 backfill.
"""
from __future__ import annotations

import pyarrow as pa


# ---------------------------------------------------------------------------
# locked vision params
# ---------------------------------------------------------------------------
IMAGE_PX = 560                       # square side fed to Qwen vision tower
PATCH = 14
SPATIAL_MERGE = 2
SUPER_PATCH = PATCH * SPATIAL_MERGE  # 28 px per LLM token

LLM_TOKENS_PER_IMAGE = (IMAGE_PX // SUPER_PATCH) ** 2   # 400
VISION_HIDDEN = 2048                                    # = LLM hidden size

# Full SFT sequence budget:
#   ~25 tokens of chat template + vision_start/end + question + answer
#   + LLM_TOKENS_PER_IMAGE image-pad placeholders.
# 512 leaves ~87 tokens for question/answer text.
MAX_TEXT_TOKENS = 512


# ---------------------------------------------------------------------------
# base schema — matches lance-format/textvqa-lance
# ---------------------------------------------------------------------------
BASE_SCHEMA = pa.schema([
    pa.field("id",            pa.int64()),
    pa.field("image",         pa.large_binary()),
    pa.field("image_id",      pa.string()),
    pa.field("question_id",   pa.string()),
    pa.field("question",      pa.string()),
    pa.field("answers",       pa.list_(pa.string())),
    pa.field("answer",        pa.string()),
    pa.field("image_emb",     pa.list_(pa.float32(), 512)),
    pa.field("question_emb",  pa.list_(pa.float32(), 512)),
    pa.field("ocr_tokens",    pa.list_(pa.string())),
    pa.field("image_classes", pa.list_(pa.string())),
    pa.field("set_name",      pa.string()),
])


# ---------------------------------------------------------------------------
# Geneva-derived columns, by tier
# ---------------------------------------------------------------------------
TIER1_COLUMNS = {
    "question_length":  pa.int32(),
    "answer_length":    pa.int32(),
    "question_type":    pa.string(),   # what / how-many / what-color / ...
    "ocr_token_count":  pa.int32(),
}

TIER2_COLUMNS = {
    # dhash → 64-bit perceptual hash, stored as uint64 for fast XOR.
    "dhash":            pa.uint64(),
}

TIER3_COLUMNS = {
    "vision_tower_hiddens": pa.list_(
        pa.float16(), LLM_TOKENS_PER_IMAGE * VISION_HIDDEN
    ),
    "input_ids":      pa.list_(pa.int32(), MAX_TEXT_TOKENS),
    "attention_mask": pa.list_(pa.int8(),  MAX_TEXT_TOKENS),
    "labels":         pa.list_(pa.int32(), MAX_TEXT_TOKENS),
}

ALL_DERIVED_COLUMNS = {**TIER1_COLUMNS, **TIER2_COLUMNS, **TIER3_COLUMNS}


def full_schema() -> pa.Schema:
    """BASE_SCHEMA + every derived column.  Useful for sanity checks."""
    fields = list(BASE_SCHEMA)
    for name, dtype in ALL_DERIVED_COLUMNS.items():
        fields.append(pa.field(name, dtype))
    return pa.schema(fields)


def describe() -> str:
    lines = ["BASE_SCHEMA:"]
    for f in BASE_SCHEMA:
        lines.append(f"  {f.name:<16} {f.type}")
    for tier_name, cols in (
        ("TIER1", TIER1_COLUMNS),
        ("TIER2", TIER2_COLUMNS),
        ("TIER3", TIER3_COLUMNS),
    ):
        lines.append(f"\n{tier_name}:")
        for name, dt in cols.items():
            lines.append(f"  {name:<24} {dt}")
    lines.append(
        f"\nLocked: IMAGE_PX={IMAGE_PX}, LLM_TOKENS_PER_IMAGE={LLM_TOKENS_PER_IMAGE}, "
        f"VISION_HIDDEN={VISION_HIDDEN}, MAX_TEXT_TOKENS={MAX_TEXT_TOKENS}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
