"""
Lance schema for the videogen pipeline.

Two design choices borrowed from `object-detection/`:

  1. **No nested structs.**  Every Geneva-added column is a flat scalar or a
     flat `list<T>[N]`.  Keeps SQL filters straightforward and dodges the
     known nested-struct read-path issue.

  2. **Blob v2 for video bytes.**  We mark `video_bytes` as a Lance blob
     column so it is stored in dedicated regions on disk and lazy-loaded as
     a ``BlobFile`` — random-access reads don't drag the bytes through the
     ordinary page cache.

The schema is split into three groups:

  * BASE_FIELDS         — written by ``ingest_chronomagic.py``.
  * GENEVA_TIER1_FIELDS — CPU-only, annotation-derived. Safe to run on a busy GPU box.
  * GENEVA_GPU_FIELDS   — small + large GPU UDFs (CLIP, motion, MTScore, T5, VAE, dHash).

The Geneva column declarations are also re-exported in dictionaries so the
backfill orchestrator can iterate over them by tier.
"""

from __future__ import annotations

import pyarrow as pa


# ---------------------------------------------------------------------------
# Blob v2 metadata
# ---------------------------------------------------------------------------
#
# Legacy-compatible way to mark a `large_binary` column as a Lance blob is
# the field-level ``lance-encoding:blob = true`` metadata.
#
# **NOTE — known regression in our pinned stack.**  On
# ``lancedb==0.30.2 + pylance==3.0.0 + geneva==0.12.0`` the blob flag on a
# *parent* table corrupts string-column reads from any Geneva materialised
# view built over that parent (the Lance Arrow decoder asserts
# ``StringArray data should contain 2 buffers``).  Until that's fixed
# upstream we ship the schema with ``BLOB_META = {}`` so the same table can
# be both ingested *and* queried through MVs.  Re-enable by setting
# ``BLOB_META = _BLOB_META_ENABLED`` once the regression is gone.
#
# Functionally this only changes the on-disk layout of ``video_bytes``: with
# the flag off the bytes live inline in the column pages rather than in
# dedicated blob regions, so very large videos read slightly slower.  All
# user-facing APIs in this pipeline are identical either way.

_BLOB_META_ENABLED = {b"lance-encoding:blob": b"true"}
BLOB_META: dict[bytes, bytes] = {}


# ---------------------------------------------------------------------------
# Base table — written by ingest_chronomagic.py
# ---------------------------------------------------------------------------

BASE_FIELDS = [
    # Identity ---------------------------------------------------------------
    pa.field("clip_id",    pa.string()),       # source-stable id (e.g. youtube id + start)
    pa.field("source",     pa.string()),       # "chronomagic-pro" | "chronomagic-proh" | "synthetic"
    pa.field("split",      pa.string()),       # "train" | "val"

    # Media -----------------------------------------------------------------
    pa.field("video_bytes", pa.large_binary(), metadata=BLOB_META),
    pa.field("width",      pa.int32()),
    pa.field("height",     pa.int32()),
    pa.field("fps",        pa.float32()),
    pa.field("n_frames",   pa.int32()),
    pa.field("duration_s", pa.float32()),

    # Text ------------------------------------------------------------------
    pa.field("caption",    pa.string()),       # raw long-form prompt from ChronoMagic-Pro
    pa.field("source_url", pa.string()),       # original URL if available (provenance only)
]

BASE_SCHEMA = pa.schema(BASE_FIELDS)


# ---------------------------------------------------------------------------
# Geneva Tier 1 — CPU, annotation/text-derived.  Safe to backfill while a GPU
# job is already running on the same machine.
# ---------------------------------------------------------------------------

# Five transition flags + a length scalar.  The keyword UDFs share a single
# implementation (see geneva_udfs.py) but each output is its own column so SQL
# stays trivial.
PHASE_TRANSITIONS = (
    "melting", "freezing", "dissolving", "boiling", "evaporating",
)

GENEVA_TIER1_FIELDS = [
    pa.field("caption_length", pa.int32()),
    *[pa.field(f"keyword_{w}", pa.bool_()) for w in PHASE_TRANSITIONS],
    pa.field("any_phase_keyword", pa.bool_()),
]


# ---------------------------------------------------------------------------
# Geneva Tier 2 — light GPU (CLIP / RAFT / MTScore).  Required for curation
# beyond keywords, but each row is cheap.
# ---------------------------------------------------------------------------

GENEVA_TIER2_FIELDS = [
    pa.field("clip_emb_text",     pa.list_(pa.float32(), 512)),
    pa.field("clip_emb_video",    pa.list_(pa.float32(), 512)),
    pa.field("motion_strength",   pa.float32()),
    pa.field("metamorphic_score", pa.float32()),
]


# ---------------------------------------------------------------------------
# Geneva Tier 3 — heavy GPU.  Pre-tokenised UMT5-XXL hidden states +
# pre-encoded Wan-VAE latents.  These two columns are the **headline
# trick**: the train loop reads only them, so the VAE and text encoder
# are never loaded at train time.
#
# Shapes confirmed against Wan-AI/Wan2.2-TI2V-5B-Diffusers (commit
# probed in this branch — see PROPOSAL.md §"Schema"):
#   UMT5-XXL  encoder hidden: (512, 4096) fp16   → 4.2 MB/row
#   Wan-VAE   latent (49f@480×720):  (48, 13, 30, 45) fp16 → 1.65 MB/row
# Total cached-feature payload ≈ 5.9 MB / clip.
# ---------------------------------------------------------------------------

T5_SEQ_LEN = 512                # WanPipeline.encode_prompt default
T5_HIDDEN  = 4096               # UMT5-XXL d_model

VAE_LATENT_C = 48               # Wan2.2-VAE z_dim
VAE_LATENT_T = 13               # 49 frames / temporal_compression
VAE_LATENT_H = 30               # 480 / 16
VAE_LATENT_W = 45               # 720 / 16

# Reference clip shape that produced the latent shape above.  Tier-3
# UDFs resize/sample to this before encoding so every row's vae_latent
# is the same length.
VAE_INPUT_FRAMES = 49
VAE_INPUT_H      = 480
VAE_INPUT_W      = 720

T5_TOTAL  = T5_SEQ_LEN * T5_HIDDEN
VAE_TOTAL = VAE_LATENT_C * VAE_LATENT_T * VAE_LATENT_H * VAE_LATENT_W

GENEVA_TIER3_FIELDS = [
    pa.field("t5_hidden_states", pa.list_(pa.float16(), T5_TOTAL)),
    pa.field("vae_latent",       pa.list_(pa.float16(), VAE_TOTAL)),
]


# ---------------------------------------------------------------------------
# Geneva Tier 4 — dedup (GPU dHash forward pass + CPU NN lookup).
# ---------------------------------------------------------------------------

DHASH_BITS = 64 * 2  # first + last frame

GENEVA_TIER4_FIELDS = [
    pa.field("dhash_first_last", pa.list_(pa.float32(), DHASH_BITS)),
    pa.field("is_duplicate",     pa.bool_()),
]


# ---------------------------------------------------------------------------
# Convenience lookups
# ---------------------------------------------------------------------------

ALL_GENEVA_FIELDS = (
    GENEVA_TIER1_FIELDS
    + GENEVA_TIER2_FIELDS
    + GENEVA_TIER3_FIELDS
    + GENEVA_TIER4_FIELDS
)

TIER_FIELDS: dict[int, list[pa.Field]] = {
    1: GENEVA_TIER1_FIELDS,
    2: GENEVA_TIER2_FIELDS,
    3: GENEVA_TIER3_FIELDS,
    4: GENEVA_TIER4_FIELDS,
}

TIER_FIELD_NAMES: dict[int, list[str]] = {
    tier: [f.name for f in fields] for tier, fields in TIER_FIELDS.items()
}

# What the dataloader projects for the **headline (cached) path**.
CACHED_TRAIN_COLUMNS = ["t5_hidden_states", "vae_latent"]
# What the **baseline (raw) path** projects.
RAW_TRAIN_COLUMNS    = ["video_bytes", "caption"]
