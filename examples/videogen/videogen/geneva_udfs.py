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
    T5_HIDDEN,
    T5_SEQ_LEN,
    T5_TOTAL,
    VAE_INPUT_FRAMES,
    VAE_INPUT_H,
    VAE_INPUT_W,
    VAE_LATENT_C,
    VAE_LATENT_H,
    VAE_LATENT_T,
    VAE_LATENT_W,
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

    Robust to malformed clips — any exception during open / iteration /
    frame-to-image returns ``[]``.  The ChronoMagic train zip ships ~10
    clips that pyav rejects with ``[Errno 95] Operation not supported``
    mid-stream; we treat them as "no frames" and let the caller decide
    (UDFs above pad with a zero-vector or skip).
    """
    import io
    import av

    container = None
    try:
        container = av.open(io.BytesIO(video_bytes))
    except Exception:
        return []
    try:
        stream = container.streams.video[0]
        total = stream.frames or 0
        # Some containers (e.g. fragmented MP4) report 0 frames; fall
        # back to an unbounded decode and pick frames opportunistically.
        if total <= 0:
            frames = []
            for f in container.decode(video=0):
                try:
                    frames.append(f.to_image())
                except Exception:
                    return []
            if not frames:
                return []
            if len(frames) <= n:
                return frames
            idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
            return [frames[i] for i in idx]

        targets = sorted({round(i * (total - 1) / max(n - 1, 1))
                          for i in range(n)})
        target_set = set(targets)
        out = []
        for j, frame in enumerate(container.decode(video=0)):
            if j in target_set:
                try:
                    out.append(frame.to_image())
                except Exception:
                    return []
                if len(out) >= n:
                    break
        return out
    except Exception:
        return []
    finally:
        try:
            if container is not None:
                container.close()
        except Exception:
            pass


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
# Tier 3 — heavy GPU.  The **headline trick**: pre-tokenised UMT5-XXL
# hidden states + pre-encoded Wan-VAE latents stored as Lance columns.
# The training loop reads only these two columns and never loads the VAE
# or the text encoder.
# ---------------------------------------------------------------------------

WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


@udf(
    data_type=pa.list_(pa.float16(), T5_TOTAL),
    input_columns=["caption"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _T5HiddenStates:
    """UMT5-XXL last hidden states for the caption.

    Tokeniser + encoder live in the same UDF so we don't pay two model
    loads (UMT5 alone is ~5.7B params at fp16, ~11 GB VRAM).  Output is a
    flat fp16 list of length ``T5_SEQ_LEN * T5_HIDDEN`` per row;
    ``dataloader.collate`` reshapes back to ``(B, 512, 4096)``.
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.encoder = None
        self.device = None

    def _load(self) -> None:
        if self.encoder is not None:
            return
        import torch
        from transformers import T5TokenizerFast, UMT5EncoderModel
        self.device = torch.device("cuda")
        self.tokenizer = T5TokenizerFast.from_pretrained(
            WAN_MODEL_ID, subfolder="tokenizer",
        )
        self.encoder = UMT5EncoderModel.from_pretrained(
            WAN_MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16,
        ).eval().to(self.device)

    def __call__(self, caption: pa.Array) -> pa.Array:
        import numpy as np
        import torch
        self._load()

        texts = [c if c else " " for c in caption.to_pylist()]
        toks = self.tokenizer(
            texts,
            padding="max_length", truncation=True,
            max_length=T5_SEQ_LEN, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            h = self.encoder(**toks).last_hidden_state  # (B, 512, 4096) fp16

        # Flatten per-row to a python list-of-lists; pa builds a
        # fixed_size_list<float16>[T5_TOTAL] cleanly from this.
        flat = h.reshape(h.shape[0], -1).cpu().numpy().astype(np.float16)
        torch.cuda.empty_cache()
        return pa.array(flat.tolist(), type=pa.list_(pa.float16(), T5_TOTAL))


@udf(
    data_type=pa.list_(pa.float16(), VAE_TOTAL),
    input_columns=["video_bytes"],
    num_gpus=1, num_cpus=2, cuda=True,
)
class _VaeLatent:
    """Pre-encoded Wan-VAE latents for the cached training path.

    Pipeline per row:
      1. Decode mp4 bytes on CPU (pyav).
      2. Sample ``VAE_INPUT_FRAMES`` evenly-spaced frames.
      3. Resize to ``VAE_INPUT_H × VAE_INPUT_W``.
      4. Normalise to ``[-1, 1]`` (the AutoencoderKLWan input range).
      5. Stack into ``(1, 3, T, H, W)`` fp32 tensor on GPU.
      6. VAE encode → ``(1, 48, 13, 30, 45)`` fp32 latent.
      7. Cast to fp16, flatten, store.

    The Wan VAE itself is ~705M params at fp32; we keep it in fp32 for
    encode stability and only cast the output down to fp16 for storage.
    """

    def __init__(self) -> None:
        self.vae = None
        self.device = None

    def _load(self) -> None:
        if self.vae is not None:
            return
        import torch
        from diffusers import AutoencoderKLWan
        self.device = torch.device("cuda")
        self.vae = AutoencoderKLWan.from_pretrained(
            WAN_MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
        ).eval().to(self.device)

    def __call__(self, video_bytes: pa.Array) -> pa.Array:
        import numpy as np
        import torch
        from PIL import Image
        self._load()

        results: list[list[float]] = []
        for b in video_bytes.to_pylist():
            frames = _decode_evenly_spaced_frames(b or b"", VAE_INPUT_FRAMES)
            if len(frames) < VAE_INPUT_FRAMES:
                # Pad by replicating the last frame so we always emit a
                # fixed-size latent.  In production we'd skip these rows
                # via curation, but the column has to be NOT NULL here.
                if not frames:
                    results.append(np.zeros(VAE_TOTAL, dtype=np.float16).tolist())
                    continue
                while len(frames) < VAE_INPUT_FRAMES:
                    frames.append(frames[-1])

            # Resize + stack → (T, H, W, 3) uint8 → (1, 3, T, H, W) fp32 in [-1, 1]
            resized = [
                np.asarray(im.resize((VAE_INPUT_W, VAE_INPUT_H), Image.BICUBIC))
                for im in frames
            ]
            arr = np.stack(resized, axis=0)                              # (T,H,W,3)
            t = torch.from_numpy(arr).to(self.device).float().div(127.5).sub(1.0)
            t = t.permute(3, 0, 1, 2).unsqueeze(0).contiguous()           # (1,3,T,H,W)

            with torch.no_grad():
                enc = self.vae.encode(t)
                lat = enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc.latents
            # Sanity guard: ensure shape matches the schema constants.
            assert lat.shape == (1, VAE_LATENT_C, VAE_LATENT_T,
                                 VAE_LATENT_H, VAE_LATENT_W), \
                f"Unexpected latent shape {tuple(lat.shape)}"
            flat = lat.squeeze(0).reshape(-1).cpu().numpy().astype(np.float16)
            results.append(flat.tolist())

        torch.cuda.empty_cache()
        return pa.array(results, type=pa.list_(pa.float16(), VAE_TOTAL))


TIER3_UDFS: dict[str, object] = {
    "t5_hidden_states": _T5HiddenStates(),
    "vae_latent":       _VaeLatent(),
}


# ---------------------------------------------------------------------------
# Tier 4 — video dedup.  First-frame and last-frame perceptual hash on GPU,
# then a cheap CPU pass that flags rows whose nearest-neighbour Hamming
# distance is below a threshold.  Storing first + last gives us a 128-bit
# fingerprint that catches both "exact same clip" and "different clip of
# the same scene" duplicates that a single-frame hash would miss.
# ---------------------------------------------------------------------------


def _dhash_8x8(image_tensor) -> "torch.Tensor":
    """Compute a 64-bit dHash on a single (3, H, W) GPU tensor.

    Returns a length-64 float tensor of 0.0/1.0 values (so L2² == Hamming).
    """
    import torch
    import torch.nn.functional as F
    # ITU-R BT.601 luma
    luma = torch.tensor([0.299, 0.587, 0.114],
                        device=image_tensor.device,
                        dtype=image_tensor.dtype).view(3, 1, 1)
    gray = (image_tensor * luma).sum(dim=0, keepdim=True)            # (1, H, W)
    gray = F.interpolate(gray.unsqueeze(0), size=(8, 9), mode="bilinear",
                         align_corners=False).squeeze(0).squeeze(0)   # (8, 9)
    bits = (gray[:, :-1] > gray[:, 1:]).float()                       # (8, 8)
    return bits.flatten()                                             # (64,)


@udf(
    data_type=pa.list_(pa.float32(), DHASH_BITS),
    input_columns=["video_bytes"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class _DHashFirstLast:
    """Concat 64-bit dHash of the first and last frame → 128-bit fingerprint."""

    def __init__(self) -> None:
        self.device = None

    def _load(self) -> None:
        import torch
        self.device = torch.device("cuda")

    def __call__(self, video_bytes: pa.Array) -> pa.Array:
        import numpy as np
        import torch
        self._load()

        out: list[list[float]] = []
        for b in video_bytes.to_pylist():
            frames = _decode_evenly_spaced_frames(b or b"", 2)
            if len(frames) < 2:
                out.append(np.zeros(DHASH_BITS, dtype=np.float32).tolist())
                continue
            first_t = torch.from_numpy(np.asarray(frames[0])).to(self.device).float().permute(2, 0, 1) / 255.0
            last_t  = torch.from_numpy(np.asarray(frames[1])).to(self.device).float().permute(2, 0, 1) / 255.0
            h = torch.cat([_dhash_8x8(first_t), _dhash_8x8(last_t)])  # (128,)
            out.append(h.cpu().float().tolist())
        torch.cuda.empty_cache()
        return pa.array(out, type=pa.list_(pa.float32(), DHASH_BITS))


@udf(
    data_type=pa.bool_(),
    input_columns=["clip_id", "dhash_first_last"],
)
class _IsDuplicate:
    """Flag duplicates by nearest-neighbour Hamming on ``dhash_first_last``.

    For binary 0/1 vectors stored under L2 metric, ``_distance`` equals
    the Hamming distance.  We do a per-row vector search excluding the row
    itself and threshold the result.
    """

    def __init__(self, db_path: str = "data/videos/lancedb",
                 table_name: str = "videos_raw",
                 hamming_threshold: int = 12) -> None:
        self.db_path = db_path
        self.table_name = table_name
        self.hamming_threshold = hamming_threshold
        self._tbl = None

    def _open(self):
        if self._tbl is None:
            import lancedb as _ldb
            self._tbl = _ldb.connect(self.db_path).open_table(self.table_name)
        return self._tbl

    def __call__(self, clip_id: pa.Array, dhash_first_last: pa.Array) -> pa.Array:
        tbl = self._open()
        results = []
        ids   = clip_id.to_pylist()
        hashes = dhash_first_last.to_pylist()
        for iid, h in zip(ids, hashes):
            if h is None:
                results.append(False)
                continue
            try:
                hits = (
                    tbl.search(h, vector_column_name="dhash_first_last")
                    .metric("l2")
                    .where(f"clip_id != '{iid}'")
                    .limit(1)
                    .to_arrow()
                )
            except Exception:
                results.append(False)
                continue
            is_dup = (
                len(hits) > 0
                and float(hits["_distance"][0].as_py()) <= float(self.hamming_threshold)
            )
            results.append(bool(is_dup))
        return pa.array(results, type=pa.bool_())


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
IMPLEMENTED_COLUMNS: set[str] = (
    set(TIER1_UDFS) | set(TIER2_UDFS) | set(TIER3_UDFS) | set(TIER4_UDFS)
    | {"is_duplicate"}  # instantiated dynamically by backfill_geneva.py
)
