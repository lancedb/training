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
# Tier 2 — light GPU (CLIP text/video, motion, MTScore)
# ---------------------------------------------------------------------------
#
# Class-based UDFs:
#   __init__ runs on the driver — keep cheap, no model load.
#   _load()  runs lazily on first __call__ on a worker — loads the model.
#   __call__ runs on the worker for every batch — one CUDA forward pass.
#
# Three of the four UDFs use CLIP ViT-B/32 (text, video frames, first/last
# pair).  Geneva launches a separate Ray actor per backfill job so each
# job loads CLIP once and reuses it for every batch — but the three
# CLIP-using jobs do load CLIP independently of each other.  ~150 MB on
# the H100, trivially cheap; not worth optimising further.
#
# ``motion_strength`` is CPU-only — it's a frame-absdiff scalar that needs
# no model, so it can backfill in parallel with the GPU UDFs.


# Shared helpers ------------------------------------------------------------

def _decode_evenly_spaced_frames(video_bytes: bytes, n: int) -> list:
    """Decode ``n`` evenly-spaced RGB frames from an MP4 byte string.

    Returns a list of PIL Images (one per sampled frame).  ``n`` includes
    both endpoints — for example ``n=8`` returns frames 0, 1/7, 2/7, …, 1.

    If decoding fails we return an empty list; callers must handle that.
    """
    import io
    import av
    from PIL import Image

    try:
        container = av.open(io.BytesIO(video_bytes))
    except Exception:
        return []
    stream = container.streams.video[0]
    total = stream.frames or 0
    # Some containers (e.g. fragmented MP4) report 0 frames; fall back to
    # an unbounded decode and pick frames opportunistically.
    if total <= 0:
        frames = [f.to_image() for f in container.decode(video=0)]
        container.close()
        if not frames:
            return []
        if len(frames) <= n:
            return frames
        idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
        return [frames[i] for i in idx]

    targets = sorted({round(i * (total - 1) / max(n - 1, 1)) for i in range(n)})
    out = []
    target_set = set(targets)
    for j, frame in enumerate(container.decode(video=0)):
        if j in target_set:
            out.append(frame.to_image())
            if len(out) >= n:
                break
    container.close()
    return out


class _ClipBase:
    """Shared CLIP ViT-B/32 lifecycle (driver-cheap, worker-lazy)."""

    def __init__(self) -> None:
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device: Optional["torch.device"] = None  # noqa: F821

    def _load(self) -> None:
        if self.model is not None:
            return
        import open_clip
        import torch
        self.device = torch.device("cuda")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")


# 1. clip_emb_text ----------------------------------------------------------

@udf(
    data_type=pa.list_(pa.float32(), 512),
    input_columns=["caption"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _ClipEmbText(_ClipBase):
    """CLIP ViT-B/32 text encoder → 512-d L2-normalised float32."""

    def __call__(self, caption: pa.Array) -> pa.Array:
        import torch
        self._load()
        # Empty / None captions get a zero vector — keeps the column NOT NULL.
        texts = [c if c else " " for c in caption.to_pylist()]
        toks = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        torch.cuda.empty_cache()
        return pa.array(emb.cpu().float().tolist(), type=pa.list_(pa.float32(), 512))


# 2. clip_emb_video ---------------------------------------------------------

_VIDEO_FRAMES_PER_CLIP = 8  # evenly-spaced sample for the video embedding


@udf(
    data_type=pa.list_(pa.float32(), 512),
    input_columns=["video_bytes"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _ClipEmbVideo(_ClipBase):
    """Mean-pooled CLIP ViT-B/32 image embedding over evenly-spaced frames.

    Decoded on CPU via pyav, batched onto GPU for one CLIP forward per
    Geneva batch.  Returns the L2-normalised mean embedding per video.
    """

    def __call__(self, video_bytes: pa.Array) -> pa.Array:
        import torch
        self._load()

        per_video_frames: list[list] = []
        for b in video_bytes.to_pylist():
            frames = _decode_evenly_spaced_frames(b or b"", _VIDEO_FRAMES_PER_CLIP)
            per_video_frames.append(frames)

        # Flatten and remember offsets so we can mean-pool back.
        flat = []
        slices: list[tuple[int, int]] = []
        for frames in per_video_frames:
            start = len(flat)
            flat.extend(frames)
            slices.append((start, len(flat)))

        if flat:
            tensors = torch.stack([self.preprocess(im) for im in flat]).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(tensors)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            emb = torch.zeros((0, 512), device=self.device)

        outputs = []
        for (a, b) in slices:
            if b > a:
                v = emb[a:b].mean(dim=0)
                v = v / v.norm().clamp_min(1e-12)
            else:
                # No frames decoded — fall back to a zero vector.
                v = torch.zeros(512, device=self.device)
            outputs.append(v.cpu().float().tolist())
        torch.cuda.empty_cache()
        return pa.array(outputs, type=pa.list_(pa.float32(), 512))


# 3. motion_strength --------------------------------------------------------
#
# A pre-encoded video's "is something actually moving" signal.  We do NOT
# need RAFT for this — the per-pixel L1 difference between evenly-spaced
# frames is a robust enough proxy for curation (filter out static
# still-life clips, slow zoom-ins on a single object, etc.).

_MOTION_FRAMES = 6  # evenly-spaced for the absdiff average


@udf(
    data_type=pa.float32(),
    input_columns=["video_bytes"],
    num_cpus=2, num_gpus=0, cuda=False,
)
def motion_strength(video_bytes: bytes) -> float:
    """Mean per-pixel L1 difference between consecutive sampled frames, on a
    0-100 scale (255 → 100).  Higher = more motion."""
    import numpy as np
    frames = _decode_evenly_spaced_frames(video_bytes or b"", _MOTION_FRAMES)
    if len(frames) < 2:
        return 0.0
    arrs = [np.asarray(f, dtype=np.float32) for f in frames]
    diffs = [
        np.mean(np.abs(arrs[i + 1] - arrs[i]))
        for i in range(len(arrs) - 1)
    ]
    return float(np.mean(diffs) / 255.0 * 100.0)


# 4. metamorphic_score ------------------------------------------------------
#
# Proxy for ChronoMagic-Bench's MTScore: how visually different is the
# first frame from the last frame?  A static scene has MTScore ≈ 0; a
# time-lapse with a real phase transition has MTScore > 0.5 ish.

@udf(
    data_type=pa.float32(),
    input_columns=["video_bytes"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _MetamorphicScore(_ClipBase):
    """MTScore proxy: ``1 - cos(CLIP(first), CLIP(last))`` per video."""

    def __call__(self, video_bytes: pa.Array) -> pa.Array:
        import torch
        self._load()

        per_video: list[tuple[object, object]] = []
        for b in video_bytes.to_pylist():
            frames = _decode_evenly_spaced_frames(b or b"", 2)
            if len(frames) == 2:
                per_video.append((frames[0], frames[1]))
            else:
                per_video.append((None, None))

        flat = []
        for f0, f1 in per_video:
            if f0 is not None:
                flat.extend([f0, f1])

        if flat:
            tensors = torch.stack([self.preprocess(im) for im in flat]).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(tensors)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            emb = torch.zeros((0, 512), device=self.device)

        outputs = []
        cur = 0
        for f0, _ in per_video:
            if f0 is None:
                outputs.append(0.0)
                continue
            cos = (emb[cur] * emb[cur + 1]).sum().item()
            outputs.append(float(max(0.0, 1.0 - cos)))
            cur += 2
        torch.cuda.empty_cache()
        return pa.array(outputs, type=pa.float32())


TIER2_UDFS: dict[str, object] = {
    "clip_emb_text":     _ClipEmbText(),
    "clip_emb_video":    _ClipEmbVideo(),
    "motion_strength":   motion_strength,
    "metamorphic_score": _MetamorphicScore(),
}


# A placeholder kept for Tier 3/4 stubs below — they still raise.
class _LazyGpuUdf:
    """Skeleton: stash device + model in lazy ``_load()``, do nothing else."""

    def __init__(self) -> None:
        self.device = None
        self.model: Optional[object] = None

    def _load(self) -> None:  # pragma: no cover — overridden by subclasses
        raise NotImplementedError("GPU body not yet implemented — see PROPOSAL.md")


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
IMPLEMENTED_COLUMNS: set[str] = set(TIER1_UDFS) | set(TIER2_UDFS)
