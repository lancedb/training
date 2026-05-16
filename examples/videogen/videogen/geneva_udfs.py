"""
Geneva UDFs for the videogen feature-engineering pipeline.

Four tiers, matching the proposal:

  Tier 1 — CPU, annotation/text-derived (this file)
    caption_length, keyword_<transition> flags, any_phase_keyword.

  Tier 2 — light GPU (stubs only at this point)
    clip_emb_text, clip_emb_video, motion_strength, metamorphic_score.

  Tier 3 — heavy GPU (stubs only)
    t5_input_ids, t5_hidden_states, vae_latent.

  Tier 4 — dedup (stubs only)
    dhash_first_last, is_duplicate.

The GPU-tier UDFs are intentionally placeholder classes whose ``__call__``
raises ``NotImplementedError``.  We will fill them in once the H100 frees
up; in the meantime the rest of the pipeline (ingest, Tier 1 backfill,
materialised views, dataloader scaffolding, EDA) can be exercised end-to-end.

The decorator pattern follows the object-detection example exactly: stateful
class UDFs allocate weights lazily in ``__call__`` so the driver process
never tries to load a model.
"""

from __future__ import annotations

import re
from typing import Optional

import pyarrow as pa
from geneva.transformer import udf

from videogen.schema import (
    DHASH_BITS,
    PHASE_TRANSITIONS,
    T5_SEQ_LEN,
    T5_TOTAL,
    VAE_TOTAL,
)


# ---------------------------------------------------------------------------
# Tier 1 — caption / annotation-derived (CPU, no model load)
# ---------------------------------------------------------------------------
#
# All Tier-1 UDFs are pure functions over the ``caption`` column.  They are
# cheap to compute (microseconds per row), incremental-friendly (Geneva's
# default ``where`` filter only re-runs on rows where the column is NULL),
# and safe to launch while the GPU is busy.

# Match common surface forms — verb / progressive / gerund / participle.
# Word-boundary regex prevents "freezer" from matching keyword_freezing, etc.
_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    "melting":     re.compile(r"\b(melt|melts|melting|melted)\b",           re.IGNORECASE),
    "freezing":    re.compile(r"\b(freeze|freezes|freezing|froze|frozen)\b", re.IGNORECASE),
    "dissolving":  re.compile(r"\b(dissolve|dissolves|dissolving|dissolved)\b", re.IGNORECASE),
    "boiling":     re.compile(r"\b(boil|boils|boiling|boiled)\b",           re.IGNORECASE),
    "evaporating": re.compile(r"\b(evaporate|evaporates|evaporating|evaporated)\b", re.IGNORECASE),
}

# Sanity: every transition declared in schema has a regex here.
assert set(_KEYWORD_PATTERNS) == set(PHASE_TRANSITIONS), (
    "PHASE_TRANSITIONS in schema.py and _KEYWORD_PATTERNS here must match"
)


@udf(data_type=pa.int32(), input_columns=["caption"])
def caption_length(caption: str) -> int:
    """Character length of the caption.  Useful for filtering out very short
    or absurdly long captions during curation."""
    return len(caption) if caption is not None else 0


def _make_keyword_udf(transition: str) -> object:
    """Build a tier-1 boolean UDF for one phase transition."""
    pattern = _KEYWORD_PATTERNS[transition]

    @udf(data_type=pa.bool_(), input_columns=["caption"])
    def _fn(caption: str) -> bool:
        if caption is None:
            return False
        return pattern.search(caption) is not None

    return _fn


# One column per transition — flat scalar bools so SQL stays trivial.
KEYWORD_UDFS: dict[str, object] = {
    f"keyword_{t}": _make_keyword_udf(t) for t in PHASE_TRANSITIONS
}


@udf(data_type=pa.bool_(), input_columns=["caption"])
def any_phase_keyword(caption: str) -> bool:
    """Convenience OR over all phase-transition keyword patterns."""
    if caption is None:
        return False
    return any(p.search(caption) is not None for p in _KEYWORD_PATTERNS.values())


TIER1_UDFS: dict[str, object] = {
    "caption_length":    caption_length,
    **KEYWORD_UDFS,
    "any_phase_keyword": any_phase_keyword,
}


# ---------------------------------------------------------------------------
# Tier 2 — light GPU (CLIP text/video, RAFT motion, MTScore).  Stubs.
# ---------------------------------------------------------------------------
#
# All Tier 2/3/4 UDFs are written as stateful class UDFs:
#   __init__ runs on the driver — keep cheap, no model load
#   _load()  runs lazily on first __call__ on a worker — loads the model
#   __call__ runs on the worker for every batch
#
# At the moment the worker bodies raise NotImplementedError so the backfill
# orchestrator can register the columns + check the wiring, but won't
# actually consume GPU until we fill them in.

class _LazyGpuUdf:
    """Skeleton: stash device + model in lazy ``_load()``, do nothing else."""

    def __init__(self) -> None:
        self.device = None
        self.model: Optional[object] = None

    def _load(self) -> None:  # pragma: no cover — overridden by subclasses
        raise NotImplementedError("GPU body not yet implemented — see PROPOSAL.md")


@udf(
    data_type=pa.list_(pa.float32(), 512),
    input_columns=["caption"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _ClipEmbText(_LazyGpuUdf):
    """CLIP ViT-B/32 text encoder → 512-d L2-normalised float32."""

    def __call__(self, caption: pa.Array) -> pa.Array:
        self._load()
        raise NotImplementedError("CLIP text encoder not yet wired up")


@udf(
    data_type=pa.list_(pa.float32(), 512),
    input_columns=["video_bytes", "n_frames"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _ClipEmbVideo(_LazyGpuUdf):
    """CLIP ViT-B/32 image encoder over evenly-spaced frames → mean-pooled 512-d."""

    def __call__(self, video_bytes: pa.Array, n_frames: pa.Array) -> pa.Array:
        self._load()
        raise NotImplementedError("CLIP video encoder not yet wired up")


@udf(
    data_type=pa.float32(),
    input_columns=["video_bytes"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _MotionStrength(_LazyGpuUdf):
    """Mean RAFT optical-flow magnitude across central triplet of frames."""

    def __call__(self, video_bytes: pa.Array) -> pa.Array:
        self._load()
        raise NotImplementedError("RAFT motion UDF not yet wired up")


@udf(
    data_type=pa.float32(),
    input_columns=["video_bytes"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _MetamorphicScore(_LazyGpuUdf):
    """MTScore proxy: 1 - cos(CLIP(first), CLIP(last))."""

    def __call__(self, video_bytes: pa.Array) -> pa.Array:
        self._load()
        raise NotImplementedError("MTScore proxy not yet wired up")


TIER2_UDFS: dict[str, object] = {
    "clip_emb_text":     _ClipEmbText(),
    "clip_emb_video":    _ClipEmbVideo(),
    "motion_strength":   _MotionStrength(),
    "metamorphic_score": _MetamorphicScore(),
}


# ---------------------------------------------------------------------------
# Tier 3 — heavy GPU (T5 tokeniser + encoder, Wan-VAE encoder).  Stubs.
# ---------------------------------------------------------------------------

@udf(
    data_type=pa.list_(pa.int32(), T5_SEQ_LEN),
    input_columns=["caption"],
    num_gpus=0, num_cpus=1, cuda=False,
)
class _T5InputIds(_LazyGpuUdf):
    """T5-XXL tokeniser (CPU is fine — tokenisation is fast)."""

    def __call__(self, caption: pa.Array) -> pa.Array:
        self._load()
        raise NotImplementedError("T5 tokeniser not yet wired up")


@udf(
    data_type=pa.list_(pa.float16(), T5_TOTAL),
    input_columns=["t5_input_ids"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _T5HiddenStates(_LazyGpuUdf):
    """T5-XXL last hidden states.  Reads from t5_input_ids (Tier 3a) so the
    tokeniser doesn't run twice."""

    def __call__(self, t5_input_ids: pa.Array) -> pa.Array:
        self._load()
        raise NotImplementedError("T5 encoder not yet wired up")


@udf(
    data_type=pa.list_(pa.float16(), VAE_TOTAL),
    input_columns=["video_bytes", "n_frames", "fps"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _VaeLatent(_LazyGpuUdf):
    """Wan-VAE encoder over 49 sampled frames @ 480×720 → fp16 latent."""

    def __call__(self, video_bytes: pa.Array,
                 n_frames: pa.Array, fps: pa.Array) -> pa.Array:
        self._load()
        raise NotImplementedError("Wan-VAE encoder not yet wired up")


TIER3_UDFS: dict[str, object] = {
    "t5_input_ids":     _T5InputIds(),
    "t5_hidden_states": _T5HiddenStates(),
    "vae_latent":       _VaeLatent(),
}


# ---------------------------------------------------------------------------
# Tier 4 — dedup (GPU dHash on first+last frame, CPU NN lookup).  Stubs.
# ---------------------------------------------------------------------------

@udf(
    data_type=pa.list_(pa.float32(), DHASH_BITS),
    input_columns=["video_bytes"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _DHashFirstLast(_LazyGpuUdf):
    """Stack first-frame + last-frame 64-bit dHashes for video dedup."""

    def __call__(self, video_bytes: pa.Array) -> pa.Array:
        self._load()
        raise NotImplementedError("dHash UDF not yet wired up")


@udf(
    data_type=pa.bool_(),
    input_columns=["clip_id", "dhash_first_last"],
)
class _IsDuplicate:
    """For each row, the nearest non-self neighbour in dhash_first_last;
    duplicate if Hamming distance ≤ threshold."""

    def __init__(self, db_path: str = "data/videos/lancedb",
                 table_name: str = "videos_raw",
                 hamming_threshold: int = 12) -> None:
        self.db_path = db_path
        self.table_name = table_name
        self.hamming_threshold = hamming_threshold
        self._tbl = None

    def __call__(self, clip_id: pa.Array, dhash_first_last: pa.Array) -> pa.Array:
        raise NotImplementedError("is_duplicate UDF not yet wired up")


TIER4_UDFS: dict[str, object] = {
    "dhash_first_last": _DHashFirstLast(),
    # is_duplicate is instantiated by the backfill script so it can pick up
    # the right db path / table name / threshold from the CLI.
}


# ---------------------------------------------------------------------------
# Registry — used by ``backfill_geneva.py``
# ---------------------------------------------------------------------------

TIER_UDFS: dict[int, dict[str, object]] = {
    1: TIER1_UDFS,
    2: TIER2_UDFS,
    3: TIER3_UDFS,
    4: TIER4_UDFS,
}


# Explicit allowlist of *currently-implemented* UDF columns.  The backfill
# orchestrator refuses to run anything outside this set unless the user
# passes ``--force-stub``.  We extend this set as each tier comes online.
IMPLEMENTED_COLUMNS: set[str] = set(TIER1_UDFS)
