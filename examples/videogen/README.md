# Video Generation Training with LanceDB

End-to-end fine-tune of a video-diffusion model (Wan2.2-TI2V-5B) on a curated
time-lapse phase-transition slice — using LanceDB + Geneva at every stage
from raw clips to high-MFU training.

See [PROPOSAL.md](PROPOSAL.md) for the full design + benchmark plan.

> Status: **Tier-1 (CPU) + Tier-2 (light GPU) + Tier-3 (heavy GPU) pipeline runnable end-to-end.**
> Tier 2 = CLIP ViT-B/32 text + video embeddings, frame-absdiff motion score,
> CLIP-based MTScore proxy.
> Tier 3 = the **headline trick**: pre-tokenised UMT5-XXL hidden states and
> pre-encoded Wan2.2-VAE latents stored as Lance columns; the training loop
> reads only these and never loads the VAE or text encoder.
> Tier 4 (dedup) is still scaffolded as stubs.  See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Setup

```bash
cd examples/videogen
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
```

> The dependency set in `pyproject.toml` is the **full training stack**
> (torch, diffusers, transformers, etc).  For CPU-only work you can drop
> them: `uv pip install lancedb pylance geneva pyarrow pandas imageio[ffmpeg]`.

## Quickstart — CPU only, ~30 seconds

Runs the full CPU pipeline on 500 synthetic clips.  No GPU.

```bash
# 1) Ingest
python -m videogen.ingest_chronomagic --synthetic 500 --overwrite

# 2) Tier 1 backfill (CPU keyword UDFs)
python -m videogen.backfill_geneva --tier 1

# 3) Curate per-transition materialised views
python -m videogen.manage_views --action curate

# 4) Status sentinel
python -m videogen.verify_pipeline
```

## Quickstart — Tier 3 (heavy GPU, ~30-60s per 20 clips on H100)

Pre-encodes the cached training features.  First run downloads Wan2.2-TI2V-5B
weights (~15 GB) into `~/.cache/huggingface`.

```bash
# 8) Tier 3 backfill — UMT5-XXL hidden states + Wan2.2-VAE latents
python -m videogen.backfill_geneva --columns t5_hidden_states
python -m videogen.backfill_geneva --columns vae_latent

# 9) Use the cached dataloader path — VAE and UMT5 never load in this process
python - <<'PY'
from videogen.dataloader import make_cached_loader
loader = make_cached_loader("data/videos/lancedb", "phase_transitions_train",
                            batch_size=2, num_workers=0)
batch = next(iter(loader))
print("prompt_embeds:", batch["prompt_embeds"].shape, batch["prompt_embeds"].dtype)
print("vae_latent   :", batch["vae_latent"].shape,    batch["vae_latent"].dtype)
# → torch.Size([2, 512, 4096]) float16
# → torch.Size([2, 48, 13, 30, 45]) float16
PY
```

## Quickstart — Tier 2 (GPU, ~10s for 100 clips on H100)

Adds CLIP text + video embeddings, motion strength, and the MTScore proxy.

```bash
# 5) Tier 2 backfill (CLIP, motion, MTScore)
python -m videogen.backfill_geneva --tier 2

# 6) Tier-2 quality-gated views (curated_train / curated_val)
python -m videogen.manage_views --action curate-2

# 7) Vector + SQL discovery
python - <<'PY'
import lancedb, open_clip, torch
tbl = lancedb.connect("data/videos/lancedb").open_table("videos_raw")

# Text → video retrieval via CLIP
m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device="cuda")
tok = open_clip.get_tokenizer("ViT-B-32")
with torch.no_grad():
    q = m.encode_text(tok(["ice melting into water"]).cuda())
    q = (q / q.norm(dim=-1, keepdim=True)).cpu().float()[0].tolist()
print(tbl.search(q, vector_column_name="clip_emb_video")
         .metric("cosine").limit(5).to_pandas()[["clip_id", "caption", "_distance"]])
PY
```

You should see something like:

```
  PASS    source table                                500 rows  version=22
  PASS    T1 column: caption_length                   500 / 500 filled (100%)
  PASS    T1 column: keyword_melting                  500 / 500 filled (100%)
  …
  PASS    view: phase_transitions_train               450 rows  version=4
  PASS    view: phase_transitions_val                 50 rows  version=4
```

## Real data — ChronoMagic-Pro (no clip download)

You don't need the 2 TB clip archive to start curating; the captions are
the small bit.

```bash
# Pull just the caption manifest (~few MB)
python -m videogen.download_manifest --variant proh --out data/chronomagic_proh.parquet

# Ingest captions only (video_bytes left empty until you download clips)
python -m videogen.ingest_chronomagic \
    --manifest data/chronomagic_proh.parquet \
    --limit 25000 --overwrite

# Tier 1 backfill + curate as before
python -m videogen.backfill_geneva --tier 1
python -m videogen.manage_views --action curate
```

Now jump into `notebooks/eda_phase_transitions.ipynb` to explore.

## Pipeline shape

```
chronomagic-pro.parquet ─┐
synthetic mp4s ──────────┤
on-disk mp4 dir ─────────┘
                  │
                  │  ingest_chronomagic.py        (CPU)
                  ▼
         videos_raw [Lance]
                  │  backfill_geneva --tier 1     (CPU)  ← runnable today
                  │   keyword_* · caption_length
                  │
                  │  backfill_geneva --tier 2     (GPU)  ← runnable today
                  │   clip_emb_* · motion · MTScore
                  │
                  │  backfill_geneva --tier 3     (GPU)  ← runnable today
                  │   t5_hidden_states · vae_latent     ← headline trick
                  │
                  │  backfill_geneva --tier 4     (GPU + CPU)
                  │   dhash · is_duplicate
                  │
                  │  manage_views --action curate(-2)
                  ▼
        phase_<transition>_{train,val}   [Geneva MVs]
                  │
                  │  dataloader.make_cached_loader(...)
                  ▼
            train_wan22_lora.py   (deferred — GPU)
```

## Layout

```
videogen/                  Pipeline package
├── schema.py              Lance schema (per-tier field declarations)
├── ingest_chronomagic.py  Streaming ingest: synthetic / manifest / manifest+mp4
├── download_manifest.py   Pull just the HF caption parquet (no clips)
├── geneva_udfs.py         Tier 1 (CPU keyword) + Tier 2 (CLIP/motion/MTScore) + Tier 3 (UMT5 + Wan-VAE) + Tier 4 stubs
├── backfill_geneva.py     Geneva backfill orchestrator (--tier, --columns)
├── manage_views.py        Materialised views per phase transition
├── dataloader.py          Permutation-based loaders (cached + raw paths)
├── spec_queries.py        Curation helpers (count, preview, FTS)
└── verify_pipeline.py     End-to-end status sentinel

bench/                     Benchmark harness — see PROPOSAL.md §"Benchmarks"
├── bench_ingest.py        B1  ingest rows/sec
├── bench_curation.py      B2  SQL+FTS query timings
└── bench_storage.py       B8  on-disk footprint per table/view

notebooks/
└── eda_phase_transitions.ipynb   Curation EDA

PROPOSAL.md                Full design doc + benchmark plan
KNOWN_ISSUES.md            Upstream regressions we're working around
```

## What's not done yet

These are tracked in `PROPOSAL.md §"Milestones"`:

* Tier 4 Geneva UDF bodies — dHash (GPU) + is_duplicate (CPU NN lookup).
* `train_wan22_lora.py` — Wan2.2 LoRA training loop reading from cached MVs.
* `bench_dataloader.py` — B5/B6 clips/sec + GPU MFU.
* `bench_backfill.py` — B3/B4 incremental backfill timings.
* `bench_e2e.py` — B10 end-to-end wall-clock.
* `eval_chronomagic.py` / `eval_vbench.py` — quantitative eval.
