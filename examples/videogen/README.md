# Video Generation Training with LanceDB

End-to-end fine-tune of a video-diffusion model (Wan2.2-TI2V-5B) on a curated
time-lapse phase-transition slice — using LanceDB + Geneva at every stage
from raw clips to high-MFU training.

See [PROPOSAL.md](PROPOSAL.md) for the full design + benchmark plan.

> Status: **Full pipeline runnable end-to-end on a single H100.**
> Tier 1 = caption-derived flags (CPU).
> Tier 2 = CLIP ViT-B/32 text + video embeddings, frame-absdiff motion score,
> CLIP-based MTScore proxy.
> Tier 3 = the **headline trick**: pre-tokenised UMT5-XXL hidden states and
> pre-encoded Wan2.2-VAE latents stored as Lance columns; the training loop
> reads only these and never loads the VAE or text encoder.
> Tier 4 = perceptual-hash dedup (first/last frame dHash + Hamming NN flag).
> The Wan2.2-TI2V-5B LoRA training loop trains from the cached columns
> (no VAE / no UMT5 in the train process).  See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Headline numbers (1×H100)

**Dataloader (forward-only, see [`bench/bench_dataloader.py`](bench/bench_dataloader.py)):**

|  | cached path (Lance) | raw path (mp4 + VAE + UMT5 in loop) | speedup |
|---|---:|---:|---:|
| samples/s (DiT fwd, bs=1) | **7.76** | 1.02 | **× 7.58** |
| fwd-only TFLOPS | 288 | 38 | |
| fwd-only MFU (H100 bf16) | **29.2%** | 3.85% | **+ 25.3 pts** |

**Curation (real data — ChronoMagic-ProH, 144,654 captions):**

|  | Lance + Geneva |
|---|---:|
| Manifest ingest into Lance | **< 5 s** |
| Tier-1 backfill (caption_length + 5 keyword flags + any-phase) | **~ 1 min** |
| FTS index build on `caption` | **2.3 s** |
| FTS query latency (1000-hit limit) | **12-25 ms** |
| CLIP text→video vector search latency | **4-23 ms** |

## Setup

```bash
cd examples/videogen
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
```

## Quickstart — real ChronoMagic data

```bash
# 1) Pull the caption manifest (just videoids + captions, ~80 MB)
python -m videogen.download_manifest --variant proh \
    --out data/chronomagic_proh.parquet

# 2) Download clips via yt-dlp (best-effort; many YouTube ids are dead).
#    Filter by caption keyword to grab only what curation needs.
for kw in melting freezing dissolving boiling evaporating; do
    python -m videogen.download_clips \
        --manifest data/chronomagic_proh.parquet \
        --out data/clips --filter "$kw" \
        --limit 80 --quality 480 --max-duration 60
done

# 3) Run the whole pipeline (ingest → tier 1-4 → curate → 20 train steps)
bash scripts/run_pipeline.sh
```

Stop at any stage:

```bash
bash scripts/run_pipeline.sh ingest   # captions + clips → Lance
bash scripts/run_pipeline.sh tier1    # + Tier 1 keyword UDFs
bash scripts/run_pipeline.sh tier2    # + Tier 2 GPU UDFs
bash scripts/run_pipeline.sh tier3    # + Tier 3 cached features (headline)
bash scripts/run_pipeline.sh curate   # + materialised views + indices
bash scripts/run_pipeline.sh tier4    # + dedup
bash scripts/run_pipeline.sh train    # + a short training run (default)
```

Env-var knobs:
```bash
DB=/tmp/videogen_demo MANIFEST=... CLIPS=... TRAIN_STEPS=200 \
    bash scripts/run_pipeline.sh
```

## Pipeline shape

```
chronomagic-pro.parquet ─┐
yt-dlp clips ────────────┘
                  │
                  │  ingest_chronomagic.py        (CPU)
                  ▼
         videos_raw [Lance]
                  │  backfill_geneva --tier 1     (CPU)
                  │   keyword_* · caption_length
                  │
                  │  backfill_geneva --tier 2     (GPU)
                  │   clip_emb_* · motion · MTScore
                  │
                  │  backfill_geneva --tier 3     (GPU)   ← headline trick
                  │   t5_hidden_states · vae_latent
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
            train_wan22_lora.py
```

## Layout

```
videogen/                  Pipeline package
├── schema.py              Lance schema (per-tier field declarations)
├── ingest_chronomagic.py  Manifest ingest (captions + optional clip bytes)
├── download_manifest.py   HF datasets pull of ChronoMagic-Pro / -ProH parquet
├── download_clips.py      yt-dlp helper to fetch ChronoMagic-Pro clips by id
├── geneva_udfs.py         Tier 1 (CPU keyword) + Tier 2 (CLIP/motion/MTScore) + Tier 3 (UMT5 + Wan-VAE) + Tier 4 (dHash+dedup)
├── backfill_geneva.py     Geneva backfill orchestrator (--tier, --columns)
├── manage_views.py        Materialised views per phase transition + build-indices
├── dataloader.py          Permutation-based loaders (cached + raw paths)
├── spec_queries.py        Curation helpers (count, preview, FTS)
├── train_wan22_lora.py    Wan2.2 LoRA trainer reading the cached path
├── eval_chronomagic.py    MTScore-proxy evaluator (ChronoMagic-Bench-style)
├── eval_vbench.py         VBench-dimension proxies (dynamic, consistency, smoothness)
├── eval_compare.py        Side-by-side baseline vs LoRA — one delta table
├── upload_to_hf.py        Publish a curated MV as a HF Lance-format dataset
└── verify_pipeline.py     End-to-end status sentinel

bench/                     Benchmark harness — see PROPOSAL.md §"Benchmarks"
├── bench_ingest.py        B1   ingest rows/sec
├── bench_curation.py      B2   SQL+FTS query timings
├── bench_backfill.py      B3 / B4   feature backfill + incremental
├── bench_dataloader.py    B5 / B6   forward-only clips/sec + GPU MFU
├── bench_train_step.py    Training-step throughput (fwd + bwd + opt)
├── bench_storage.py       B8   on-disk footprint per table/view
├── bench_recipe_change.py B9   one-column re-derive vs whole pipeline
└── bench_e2e.py           B10  end-to-end wall-clock summary

scripts/
└── run_pipeline.sh        one-command pipeline runner

notebooks/
└── eda_phase_transitions.ipynb   Curation EDA

PROPOSAL.md                Full design doc + benchmark plan
KNOWN_ISSUES.md            Upstream regressions we're working around
```

## Training

### Single H100

Once Tiers 1-3 are backfilled and the curated view is materialised:

```bash
python -m videogen.train_wan22_lora \
    --db data/videos/lancedb \
    --train-view phase_transitions_curated_train \
    --steps 2000 --batch-size 1 \
    --rank 32 --alpha 32 --lr 1e-4 \
    --save-every 200 \
    --output-dir checkpoints/wan22_lora
```

### Multi-GPU "big run" (4×H100 DDP)

The trainer detects `LOCAL_RANK`/`WORLD_SIZE` and wraps the LoRA-adapted
model in `DistributedDataParallel` automatically.  Launch via
`torchrun` (or `accelerate launch`):

```bash
torchrun --standalone --nproc-per-node=4 \
    -m videogen.train_wan22_lora \
        --db data/videos/lancedb \
        --train-view phase_transitions_curated_train \
        --steps 4000 --batch-size 1 \
        --rank 64 --alpha 64 --lr 1e-4 \
        --num-workers 4 --prefetch-factor 4 \
        --save-every 500 \
        --output-dir checkpoints/wan22_lora_4gpu
```

Each rank reads a different shuffle of the cached columns (seeded by
`--seed + rank`).  Gradients are all-reduced via NCCL; only rank 0
writes checkpoints.  No extra Lance work is needed — the same
materialised view is opened per worker, and Permutation handles
random access in parallel.

## Evaluation

```bash
# ChronoMagic-Bench-style: generate clips, score MTScore-proxy
python -m videogen.eval_chronomagic \
    --checkpoint checkpoints/wan22_lora/step-002000 \
    --n-prompts 8 --output results_mtscore.json

# VBench-dimension proxies
python -m videogen.eval_vbench \
    --checkpoint checkpoints/wan22_lora/step-002000 \
    --n-prompts 4 --output results_vbench.json

# Side-by-side baseline vs fine-tuned, single command
python -m videogen.eval_compare \
    --checkpoint checkpoints/wan22_lora/step-002000 \
    --n-prompts 8 --steps 20 --output compare.json
```

prints a delta table:

```
  metric                     baseline    fine-tuned          Δ
  ────────────────────────────────────────────────────────────
  mtscore_proxy                0.4012        0.5187    +0.1175
  dynamic_degree               2.3110        4.1882    +1.8772
  subject_consistency          0.8744        0.8612    -0.0132
  temporal_smoothness          0.9994        0.9990    -0.0004
```

## What's not done yet

* Full-corpus training on 20 K+ real ChronoMagic clips (clip downloads
  are best-effort via yt-dlp — many ids are dead/private/geo-blocked, so
  a real run on 20 K clips needs an overnight downloader).
* Replace the MTScore-proxy in `eval_chronomagic.py` with the upstream
  ChronoMagic-Bench prompts list and CLIPScore-based reference.
