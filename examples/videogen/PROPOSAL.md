# Video Generation Training with LanceDB — Proposal

End-to-end fine-tuning of an open video-diffusion model on a curated physics slice, using LanceDB + Geneva as the data backbone at **every** stage. The story: **iterate faster, use GPUs better, and do feature engineering at scale without leaving the data layer.**

> Status: proposal — implementation not started. Branch: `videogen`.

---

## TL;DR

| Decision | Choice |
|---|---|
| **Base model** | Wan2.2-TI2V-5B (T2V LoRA) |
| **Fine-tune task** | Time-lapse phase transitions (melting · freezing · dissolving · boiling · evaporating) |
| **Source corpus** | ChronoMagic-Pro (460K time-lapse clips + captions) → curate to ~15-25K target clips |
| **Default hardware** | 1×H100 80GB (LoRA r=64, bs=1, grad-accum=4, 49 frames × 480×720) |
| **"Big run" hardware** | 4×H100 (LoRA + full-FT comparison, larger batch) — appendix only |
| **Headline trick** | Geneva-precomputed **VAE latents + T5 embeddings as Lance columns** → training loop becomes a byte-stream of cached tensors; VAE/T5 never load in train job |
| **Evaluation** | ChronoMagic-Bench (MTScore, CHScore) + VBench dynamic-degree / subject-consistency / motion-smoothness |

The PDF baseline (CogVideoX-2B + LoRA on 20K fluid-physics clips, ~1.5-3h/epoch on H100, ~$50/run) gives us the wall-clock to beat. We expect to beat it on:

1. **time-to-curate** (SQL/vector vs writing a custom mp4 walker),
2. **steady-state step time** (cached latents vs decode + VAE in the loop),
3. **iteration cost** when the recipe changes (re-encode one column vs re-derive a whole filesystem cache).

---

## Why this example exists

Video models are the worst case for a training data layer:

- Every sample is **megabytes** (49-frame clips at 480×720 are ~5 MB encoded).
- The standard pipeline reads `.mp4`, decodes on CPU, feeds raw frames to a VAE, then to T5, then to the DiT — three stalls per step.
- Practitioners "fix" this by precomputing VAE/T5 into per-clip `.pt` files on disk, which means **two filesystem hierarchies to keep in sync, two versions to track, one custom loader per project**.
- Curation is usually a manual `pandas` notebook that produces a CSV of paths; reruns are O(corpus).

LanceDB collapses all of that into one columnar table where every artefact — raw video bytes, captions, T5 token IDs, T5 hidden states, VAE latents, CLIP vectors, motion scores, your custom physics-relevance score — is a column. **Schema evolution adds columns without rewriting rows. Geneva backfills are incremental and crash-safe. Materialized views = SQL-defined training splits. The PyTorch DataLoader is `Permutation.identity(table).select_columns([...])`.**

---

## Pipeline architecture

```
┌─ ChronoMagic-Pro (HF) — 460K time-lapse clips + captions
│
│                       │ ingest_videos.py
│                       ▼
│  ┌──────────────────────────────────────────────────────────────────┐
│  │                    videos_raw  [Lance table]                     │
│  │  clip_id · video_bytes (blob v2) · caption · width · height ·    │
│  │  fps · n_frames · duration · source_url · split                  │
│  │  ┌────────────────────────────────────────────────────────────┐  │
│  │  │ blob v2 — video bytes in dedicated regions, lazy BlobFile  │  │
│  │  │ stable row IDs — incremental view refresh + MV checkpoints │  │
│  │  │ data_storage_version="2.2"                                 │  │
│  │  └────────────────────────────────────────────────────────────┘  │
│  └──────────────────────────────────────────────────────────────────┘
│                       │
│                       │ backfill_geneva.py  (incremental, Ray, checkpointed)
│                       │
│                       │  Tier 1 — CPU
│                       │    caption_length · keyword flags (melting,
│                       │    freezing, dissolving, boiling, evaporating)
│                       │    aesthetic-rejector heuristics
│                       │
│                       │  Tier 2 — GPU, small
│                       │    clip_emb_video   [list<f32>[512]]  CLIP-ViT-B/32 mean-pooled
│                       │    clip_emb_text    [list<f32>[512]]
│                       │    motion_strength  [f32]            mean RAFT optical-flow magnitude
│                       │    metamorphic_score [f32]           MTScore proxy on first/last
│                       │
│                       │  Tier 3 — GPU, expensive  *** headline trick ***
│                       │    t5_input_ids     [list<i32>[226]]  T5 tokenisation only
│                       │    t5_hidden_states [list<f16>[226*4096]]  T5-XXL last hidden
│                       │    vae_latent       [list<bf16>[16*13*60*90]]  Wan VAE encoded
│                       │
│                       │  Tier 4 — GPU, dedup
│                       │    dhash_first_last [list<f32>[128]]   2× 64-bit perceptual hash
│                       │    is_duplicate     [bool]
│                       │
│                       │  ┌──────────────────────────────────────────────────────┐
│                       │  │ Lance: each new column added without table rewrite  │
│                       │  │ Geneva: stateful class UDF on Ray, cuda=True,         │
│                       │  │   model loaded lazily in __call__ per worker,         │
│                       │  │   checkpointed every N rows                          │
│                       │  └──────────────────────────────────────────────────────┘
│                       │
│                       │  Curation queries (SQL + FTS + vector, all on one table)
│                       │  ┌──────────────────────────────────────────────────────┐
│                       │  │ FTS:    "melting candle" / "ice cream melts"          │
│                       │  │ Vector: CLIP "honey dripping from spoon"             │
│                       │  │ SQL:    motion_strength BETWEEN 2.0 AND 12.0          │
│                       │  │   AND  metamorphic_score > 0.6 AND duration BETWEEN 4 AND 8 │
│                       │  │   AND  NOT is_duplicate                              │
│                       │  └──────────────────────────────────────────────────────┘
│                       │
│                       │ manage_views.py
│              ┌────────┼────────────┐
│              ▼        ▼            ▼
│         melting_v1  freezing_v1  dissolving_v1   (Geneva materialized views — each train/val)
│              └────────┼────────────┘
│                       │
│                       │ train_wan22_lora.py
│                       │  ┌────────────────────────────────────────────────────┐
│                       │  │ Permutation API — random-access, no copy           │
│                       │  │ Reads ONLY t5_hidden_states + vae_latent columns    │
│                       │  │ No VAE / no T5 loaded — only the DiT + LoRA + opt  │
│                       │  │ → all VRAM goes to the model that's actually being  │
│                       │  │   trained → bigger batch / longer ctx / higher MFU │
│                       │  └────────────────────────────────────────────────────┘
│                       ▼
│         checkpoints/<view-name>/{step}/  (logs table.version with every save)
└──────────────────────────────────────────────────────────────────────────────
```

---

## Schema

Two tables: `videos_raw` (source of truth, all derived columns live here) and `videos_train_*` (Geneva materialized views).

### `videos_raw`

```python
import pyarrow as pa
import lance

SCHEMA = pa.schema([
    pa.field("clip_id",     pa.string()),
    pa.field("source",      pa.string()),       # "chronomagic-pro" / "open-sora" / custom
    pa.field("split",       pa.string()),       # train / val
    pa.field("caption",     pa.string()),       # raw long-form prompt
    # blob v2 (Lance 2.2) for video bytes — stored in dedicated regions
    lance.blob_field("video_bytes", pa.large_binary()),
    pa.field("width",       pa.int32()),
    pa.field("height",      pa.int32()),
    pa.field("fps",         pa.float32()),
    pa.field("n_frames",    pa.int32()),
    pa.field("duration_s",  pa.float32()),
])
```

Created with `storage_options={"new_table_enable_stable_row_ids": "true"}` and `data_storage_version="2.2"` so MV refresh is incremental and blob v2 takes effect.

### Geneva-added columns (all flat, no nested structs)

| Column | Type | Tier | Notes |
|---|---|---|---|
| `caption_length` | `int32` | 1 CPU | trivial |
| `keyword_melting/_freezing/_dissolving/_boiling/_evaporating` | `bool` | 1 CPU | regex over caption |
| `clip_emb_video` | `list<float32>[512]` | 2 GPU | mean-pooled CLIP frames; cosine IVF-PQ index |
| `clip_emb_text` | `list<float32>[512]` | 2 GPU | CLIP text encoder over caption |
| `motion_strength` | `float32` | 2 GPU | mean RAFT flow magnitude, central-frame triplet |
| `metamorphic_score` | `float32` | 2 GPU | MTScore proxy: 1 - cos(CLIP(first), CLIP(last)) |
| `t5_hidden_states` | `list<float16>[512*4096]` | 3 GPU | UMT5-XXL last hidden, fp16; **4.19 MB / row** |
| `vae_latent` | `list<float16>[48*13*30*44]` | 3 GPU | Wan2.2 VAE; **1.65 MB / row** at 49f×480×704 |
| `dhash_first_last` | `list<float32>[128]` | 4 GPU | 2× dHash for dedup |
| `is_duplicate` | `bool` | 4 CPU | nearest-neighbour Hamming ≤ threshold |

**Per-row cost after Tier 3:** ~5 MB raw video + ~6 MB cached latents/embeddings = ~11 MB. 25K rows ≈ 275 GB. Comfortably fits a single 1-TB NVMe; trivial in object storage.

(Smoke-test measurement on 20 synthetic clips: 11.2 MB/row including
Geneva checkpoint overhead.  Both `t5_hidden_states` and `vae_latent`
round-trip cleanly to `(B, 512, 4096)` fp16 and `(B, 48, 13, 30, 44)`
fp16 via the Permutation API.)

> Note on column shape: `t5_hidden_states` and `vae_latent` are stored as flat `list<f16|bf16>[N]` (not nested fixed-size-lists). This matches the object-detection example's "no nested structs" rule and avoids the known nested-struct read path. Reshape happens in `collate_fn`.

---

## The headline trick: pre-tokenized features as Lance columns

The classic video-diffusion training step:

```
mp4 → ffmpeg decode → frames → VAE encode → latent
caption → T5 tokenise → T5 encode → hidden
(latent, hidden) → DiT forward + LoRA backward
```

Steps 1-4 are **per-epoch repeated work** that doesn't depend on weights. Diffusion-pipe & friends cache them as `.pt` files in a separate `cache/` folder. Two problems:

1. **Filesystem cache drifts.** Recipe change (different fps, different VAE) means re-deriving the whole cache by hand. No version link to the source.
2. **Cache and source are two stores.** Curation queries can't see what's cached.

With Lance + Geneva, each cache becomes a **column** on the same table:

```python
@udf(
    data_type=pa.list_(pa.float16(), 226 * 4096),
    input_columns=["caption"],
    num_gpus=1, num_cpus=1, cuda=True,
)
class T5HiddenStates:
    def __init__(self):
        self.tok = None
        self.model = None
        self.device = None

    def _load(self):
        if self.model is not None: return
        import torch
        from transformers import T5EncoderModel, T5Tokenizer
        self.device = torch.device("cuda")
        self.tok = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        self.model = T5EncoderModel.from_pretrained(
            "google/t5-v1_1-xxl", torch_dtype=torch.float16,
        ).eval().to(self.device)

    def __call__(self, caption: pa.Array) -> pa.Array:
        import torch
        self._load()
        toks = self.tok(
            caption.to_pylist(), padding="max_length", truncation=True,
            max_length=226, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**toks).last_hidden_state  # (B, 226, 4096) fp16
        flat = out.reshape(out.shape[0], -1).cpu().numpy().tolist()
        return pa.array(flat, type=pa.list_(pa.float16(), 226 * 4096))
```

(Equivalent pattern for `vae_latent`, swapping in the Wan2.2 VAE.)

At train time the DataLoader **only** projects the columns it needs:

```python
perm = (
    Permutation.identity(view)
    .select_columns(["t5_hidden_states", "vae_latent"])
    .with_format("arrow")
)
```

The VAE and T5 never enter the training process. VRAM saved:

| Component | VRAM @ bf16 | Status |
|---|---|---|
| Wan2.2 DiT (5.0B params) | ~10 GB | trained (frozen + LoRA) |
| Wan2.2 VAE (705M params) | ~3 GB | **not loaded** |
| UMT5-XXL encoder (5.7B params) | ~11 GB | **not loaded** |
| LoRA r=32 + AdamW state | ~0.3 GB | trained |
| Activations + grads | ~5 GB | trained |
| **Total (cached)** | **~15 GB** | fits 1×H100 with huge headroom |
| Total (uncached, classic) | ~30 GB | tight on 1×H100, OOM-prone at higher batch |

That ~14 GB of headroom buys: bigger batch (1 → 2-4), longer sequences (49f → 81f), no gradient checkpointing, or a `torch.compile` cache.

---

## Curation: SQL + FTS + vector — one table, no manifests

The PDF builds the training set from a flat directory of 20K clips. Our equivalent is a single SQL query over `videos_raw`:

```python
# 1) keyword + caption-shape filter (FTS, cheap)
candidates = tbl.search(
    "melting OR freezing OR dissolving OR boiling OR evaporating",
    query_type="fts",
).limit(50_000).to_arrow()

# 2) Vector recall for unlabeled keywords ("phase change",
#    "viscous flow", "crystal formation")
extra = tbl.search(
    clip.encode_text("ice melting into water"),
    vector_column_name="clip_emb_text",
).metric("cosine").where("motion_strength BETWEEN 1.0 AND 15.0").limit(10_000).to_arrow()

# 3) Final SQL gate over the union — runs over backfilled scalar columns
view_sql = """
   (keyword_melting OR keyword_freezing OR keyword_dissolving
                    OR keyword_boiling OR keyword_evaporating)
   AND duration_s BETWEEN 4 AND 8
   AND motion_strength BETWEEN 2.0 AND 12.0
   AND metamorphic_score > 0.6
   AND NOT is_duplicate
   AND split = 'train'
"""
gconn.create_materialized_view("phase_transitions_train", tbl.search().where(view_sql))
```

Each MV is a **first-class table** with its own version. Training scripts open it by name; no path manifests change hands.

When new clips arrive (`videos_raw.add(reader)`), `mv.refresh()` is **incremental** — only the new rows are tested against the filter, only new latents are backfilled (stable row IDs + Geneva checkpointing).

---

## Training loop

Same shape as the PDF; only the data path is different.

```python
from lancedb.permutation import Permutation
import lancedb
import torch
from torch.utils.data import DataLoader

class WanCachedDataset(torch.utils.data.Dataset):
    def __init__(self, uri, view_name):
        self.uri, self.view_name, self._perm = uri, view_name, None
        self.length = len(lancedb.connect(uri).open_table(view_name))

    def __len__(self): return self.length
    def __getstate__(self):
        s = self.__dict__.copy(); s["_perm"] = None; return s

    def _ensure(self):
        if self._perm is None:
            db = lancedb.connect(self.uri)
            self._perm = (
                Permutation.identity(db.open_table(self.view_name))
                .select_columns(["t5_hidden_states", "vae_latent"])
                .with_format("arrow")
            )

    def __getitems__(self, indices):
        self._ensure()
        return self._perm.__getitems__(indices)


def collate(batch):
    import numpy as np
    # zero-copy reshape from the flat list columns
    t5 = torch.frombuffer(
        batch.column("t5_hidden_states").to_numpy(zero_copy_only=False).tobytes(),
        dtype=torch.float16,
    ).reshape(-1, 226, 4096)
    lat = torch.frombuffer(
        batch.column("vae_latent").to_numpy(zero_copy_only=False).tobytes(),
        dtype=torch.bfloat16,
    ).reshape(-1, 16, 13, 60, 90)
    return {"prompt_embeds": t5, "vae_latent": lat}


loader = DataLoader(
    WanCachedDataset(uri, "phase_transitions_train"),
    batch_size=1, num_workers=8, pin_memory=True,
    persistent_workers=True, prefetch_factor=4,
    multiprocessing_context="spawn",
    collate_fn=collate,
)

# Standard Wan2.2 LoRA loop — flow matching loss
for batch in loader:
    z0    = batch["vae_latent"].to("cuda")
    ctx   = batch["prompt_embeds"].to("cuda")
    t     = torch.rand(z0.shape[0], device="cuda")
    noise = torch.randn_like(z0)
    z_t   = (1 - t) * z0 + t * noise
    target = noise - z0                            # flow matching velocity target
    pred = transformer(z_t, t, encoder_hidden_states=ctx)
    loss = F.mse_loss(pred.float(), target.float())
    accelerator.backward(loss); optimizer.step(); optimizer.zero_grad()
```

LoRA targets: `to_q, to_k, to_v, to_out.0`, rank 64, alpha 64 (matches PDF/CogVideoX recommendations and Wan2.2 best practice).

The training script logs `view.version` with every checkpoint — exact data snapshot ↔ weights link.

---

## Benchmarks we will publish

Each row a separate measurement. Baseline ≈ "what the PDF does" (per-row mp4 decode + VAE + T5 in the dataloader). Lance-cached ≈ "this proposal".

Numbers in *italics* are measured on this branch on 1×H100 against
synthetic data (smoke-test sized).  Full-corpus numbers replace these
once we run on a 25K-clip ChronoMagic-Pro subset.

| # | Stage | Metric | Baseline | Lance + Geneva | Smoke-test result |
|---|---|---|---|---|---|
| **B1** | Ingest | rows/sec | dir walk + ffprobe | RecordBatch stream into Lance | *144,654-row ChronoMagic-ProH manifest ingested in **&lt;5 s** (no clips, captions only) via `videogen.ingest_chronomagic --manifest`.* |
| **B2** | Curation query | wall-clock | grep + manual inspection | SQL + FTS over Geneva cols | *FTS index on 144K-row ChronoMagic-ProH built in **2.3 s**; per-query latency **12-25 ms** for 1000-hit limit; SQL count_rows on Tier-1 keyword: **1.85 ms** at 100 rows.  See `bench/bench_curation.py`.* |
| **B3** | Feature backfill | wall-clock per 1K clips | diffusion-pipe `.pt` cache | Geneva UDF backfill, Ray | *T5 ~1.5 s/row + VAE ~1.5 s/row on H100* |
| **B4** | Incremental refresh | wall-clock for +M rows | re-derive whole cache | Geneva skips filled rows via NULL filter | *validated; Tier 1 too cheap to show win — bench harness in `bench_backfill.py`* |
| **B5** | Dataloader throughput | samples/sec bs=1 | mp4 + on-the-fly VAE + UMT5 | cached columns via Permutation | ***7.58× speedup*** (7.76 vs 1.02 samples/s) |
| **B6** | GPU fwd-MFU (DiT only) | % on 1×H100 | baseline | cached | ***29.15% vs 3.85%, +25.3 pts*** |
| **B7** | Wall-clock per epoch on 20K clips | hours | PDF: 1.5-3 h | cached + bs=2 | TBD — extrapolating B5: ~0.7 h |
| **B8** | Storage footprint | GB for 25K clips | 2 dirs (mp4s + cache.pt) | 1 Lance table | *~11.2 MB/row → ~280 GB for 25K* |
| **B9** | Recipe-change cost | wall-clock for new VAE | re-derive whole cache by hand | Geneva backfill on the one changed column | TBD |
| **B10** | End-to-end wall-clock | curate → cache → train → eval | PDF baseline | this pipeline | ***240 s on 8 clips, 1×H100*** (see breakdown below) |

**B10 stage breakdown (8 synthetic clips, 1×H100, 4 train steps):**

| Stage | Wall-clock | % |
|---|---:|---:|
| ingest | 1.74 s | 0.7% |
| tier1 (CPU keywords × 7) | 44.03 s | 18.4% |
| tier2 (CLIP + motion + MTScore) | 45.03 s | 18.8% |
| tier3 t5 (UMT5-XXL encode) | 32.65 s | 13.6% |
| tier3 vae (Wan2.2 VAE encode) | 28.82 s | 12.0% |
| tier4 dhash | 16.49 s | 6.9% |
| tier4 idx (L2 IVF) | 1.35 s | 0.6% |
| tier4 dup (NN lookup) | 14.92 s | 6.2% |
| curate t1 (12 MVs) | 40.98 s | 17.1% |
| train (4 steps) | 13.64 s | 5.7% |
| **total** | **239.65 s** | 100% |

Most of the per-stage time at this small N is Ray actor spin-up (~5-10 s
per backfill).  At realistic N≈20-25 K the per-row UDF work dominates;
extrapolating from B3 (~1.5 s/row for the heavy GPU UDFs) the E2E for
20K clips lands at **roughly 4-6 hours on 1×H100**, well within the
"single-H100 demo" budget.

Benchmarks (B5)-(B7) follow the **MFU methodology** already in `examples/ViT/mfu_bench_fp16/bench.py` (warmup + timed window, FLOP estimate, H100 peak = 989 TFLOPS bf16). The harness lives under `examples/videogen/bench/`.

Comparison reference points: `examples/ViT/` already shows LanceDB ≥ 3× MFU vs raw S3 and ~1.8× vs S3-Parquet on ViT-H @ 350-image batch. We expect a larger gap on video because the per-row payload is bigger (mp4 → bytes → frames decode) and the latent-column trick eliminates the VAE/T5 entirely.

---

## Evaluation

Time-lapse phase transitions have an established benchmark, so we don't have to invent one:

- **ChronoMagic-Bench** — 1,649 prompts × MTScore (metamorphic amplitude) + CHScore (temporal coherence). Direct measure of "does the model actually represent the transition."
- **VBench** — subject consistency, motion smoothness, dynamic degree, aesthetic quality. Standard for T2V papers.
- **Held-out phase-transition prompts** — qualitative side-by-side: baseline Wan2.2 vs ours, for melting / freezing / dissolving / boiling / evaporating × 4 prompts each. 20 videos, public.

We log MTScore + CHScore on the val MV every N steps so the loss curve isn't the only signal (the PDF noted its loss was stuck at a local minimum but generations still improved — same trap we want to avoid).

---

## Directory layout (proposed)

```
examples/videogen/
├── PROPOSAL.md                       ← this doc
├── README.md                         ← runnable quickstart (later)
├── pyproject.toml                    ← lancedb, geneva, diffusers, accelerate, ...
├── videogen/
│   ├── schema.py                     ← BDD-style: VIDEO_SCHEMA + GENEVA_COLUMNS
│   ├── ingest_chronomagic.py         ← stream HF → RecordBatch → Lance (blob v2)
│   ├── geneva_udfs.py                ← CLIP, motion (RAFT), MTScore, T5, VAE, dHash
│   ├── backfill_geneva.py            ← orchestration (tier 1 → 2 → 3 → 4)
│   ├── manage_views.py               ← per-phase MVs + dedup clause
│   ├── dataloader.py                 ← Permutation + collate for cached cols
│   ├── train_wan22_lora.py           ← LoRA training loop
│   ├── eval_chronomagic.py           ← MTScore + CHScore
│   ├── eval_vbench.py                ← VBench dimensions
│   └── verify_pipeline.py            ← status-table sentinel (like object-detection)
├── bench/
│   ├── bench_dataloader.py           ← B5: clips/sec, GPU MFU
│   ├── bench_ingest.py               ← B1
│   ├── bench_backfill.py             ← B3, B4
│   ├── bench_curation.py             ← B2
│   ├── bench_storage.py              ← B8
│   └── bench_e2e.py                  ← B10
├── notebooks/
│   └── eda_phase_transitions.ipynb   ← FTS + CLIP discovery + curation
└── data/videos/lancedb/              ← gitignored
    └── videos_raw.lance
```

---

## Milestones

| # | Milestone | Output | ETA on 1×H100 / 1 dev |
|---|---|---|---|
| M1 | Ingest 50K-clip subset of ChronoMagic-Pro into `videos_raw` with blob v2 | working table, B1 numbers | ~1 day |
| M2 | Tier 1 + Tier 2 Geneva backfill (CPU + light GPU), CLIP index, MV `phase_transitions_*` | curation works in a notebook, B2 numbers | ~2 days |
| M3 | Tier 3 — T5 + VAE Geneva UDFs, full backfill on the curated MV | `t5_hidden_states`, `vae_latent` columns populated; B3, B4 numbers | ~2 days |
| M4 | Dataloader + training script + 1×H100 LoRA run on 20K clips | first checkpoint, B5/B6/B7 numbers | ~2 days |
| M5 | Eval harness (ChronoMagic-Bench + VBench), held-out comparison videos | results table, qualitative reel | ~2 days |
| M6 | 4×H100 "big run" appendix (LoRA + optional full-FT) | second checkpoint, ablation table | ~2 days |
| M7 | Blog-shaped README + benchmark plots, PR | merge-ready | ~1 day |

**Total:** ~2 weeks single dev. Compute budget: <$1K @ 1×H100 spot for the full milestone sequence; the 4×H100 callout adds another ~$300-500.

---

## Open questions / decisions still pending

1. **VAE-latent precision.** bf16 is the default for Wan2.2; we'll measure end-task quality at fp16 to halve column size and see if drift matters.
2. **Permutation API for very large columns.** `vae_latent` rows are ~2 MB; ensure the Permutation random-access pattern doesn't get hurt vs scan. If it does, fall back to `with_format("arrow")` over `__getitems__` with shuffled offsets (already what the object-detection example does). Result of B5 will tell us.
3. **Dedup hash for video.** First-frame + last-frame dHash should catch the obvious clip-of-clip duplicates ChronoMagic-Pro has from YouTube re-uploads, but we may need a video-CLIP variant for visually similar but temporally different clips. Bench it; if recall < ~95% on a planted set, add `clip_emb_video` as the dedup vector.
4. **Should we publish the curated MV as a HF dataset (Lance-format)?** The lerobot example already publishes `lance-format/lerobot_xvla-soft-fold`. Doing the same here (e.g. `lance-format/phase-transitions-25k`) lets readers reproduce without re-doing curation — a nice cherry on top.

These are intentionally implementation-time decisions; the proposal does not require resolving them now.

---

## What this proposal does **not** do

- No new format or API. Every API used here (`lancedb.connect`, `Permutation`, `geneva.connect`, `@udf`, `create_materialized_view`) is already in the existing examples.
- No claim that LanceDB makes the model better. The win is **time-per-iteration** and **GPU utilization**, not mAP/MTScore (though we expect a small lift because better curation → cleaner training data).
- No multi-node / KubeRay setup in the headline. Geneva supports it, but the story is "single H100, hours not days" — multi-node is an appendix.
