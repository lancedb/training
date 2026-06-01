# VLM Fine-Tune on Scene-Text Q&A with LanceDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lancedb/training/blob/vlm-textvqa/examples/vlm-textvqa/notebooks/colab_textvqa_lance.ipynb)

End-to-end LoRA SFT of **Qwen2.5-VL-3B-Instruct** for TextVQA, with one
LanceDB table backing the whole loop — raw image bytes to a tuned
adapter. The same three jobs as every other example in this repo:

- **Curate & engineer features** — image bytes, CLIP embeddings, OCR
  tokens, and Geneva-backfilled text/quality columns live in one
  schema-enforced table; slice with SQL / FTS / vector search.
- **Manage at scale** — the expensive vision-tower output is added as a
  column with **zero-copy schema evolution** (no table rewrite): compute
  it **once**, reuse it every epoch forever.
- **Load & train** — the train loop reads `vision_tower_hiddens` straight
  off Lance and skips the vision tower entirely — **~2× train-step
  throughput at −1.3 GB VRAM**.

**Headline trick:** freeze the vision tower at SFT time and pre-compute
its hidden states as a Lance column. The train process loads only the
LLM — no vision-tower forward in the loop, no image decode, no
tokenisation.

> Status: e2e validated on the full corpus (1×H100). Branch: `vlm-textvqa`.

## Run it on a free Colab T4

[`notebooks/colab_textvqa_lance.ipynb`](./notebooks/colab_textvqa_lance.ipynb)
runs the **whole loop on a single free T4** at demo scale: download a
pre-baked Lance subset → benchmark **Lance-vs-Parquet** read throughput →
**QLoRA** fine-tune from the cached columns → **before/after** answer grid
on held-out images. Qwen2.5-VL-3B fits 16 GB via 4-bit NF4 (`--load-4bit`)
plus the vision-tower-free cached path.

The notebook reads a small subset with the cached columns already
computed. Bake and host it once on any GPU box:

```bash
python -m vlm.colab_prepare \
    --out data/colab --train-rows 512 --val-rows 64 \
    --hf-repo <your-org>/textvqa-lance-colab --push      # needs HF_TOKEN
```

Then point the notebook at it via the `TEXTVQA_COLAB_REPO` env var (it
defaults to `lance-format/textvqa-lance-colab`).

## Real-run numbers (1×H100, 34,602 train rows, 200 eval rows)

| Stage              | Wall-clock | Notes |
|---|---:|---|
| ingest (stream)    | 4 min 51 s | 119 rows/s from HF |
| Tier-1 backfill    | 31 s       | 4 CPU UDFs via Geneva |
| Tier-2 backfill    | 5 min 15 s | dhash (image decode) via Geneva |
| Tier-3 backfill    | 31 min 5 s | vision tower + tokeniser, direct add_columns |
| layouts export     | 38 s       | raw_fs + WDS + parquet for baselines |
| train (3 epochs)   | 1 h 13 min | LoRA r=64, bs=2, grad-accum=4 |
| eval (base+tuned)  | 3 min 23 s | 200 val rows × 2 models |
| **total**          | **1 h 58 min** | — |

| Result | Value |
|---|---|
| **TextVQA acc, base**  | 0.793 |
| **TextVQA acc, tuned** | 0.815  (+2.2 pp) |
| **Cached-vs-raw train step** | **16.1 vs 7.9 samples/s** (-1.3 GB VRAM) |
| **1:1 dataloader (PIL decode in loop)** | Lance 60, HF 63, raw_fs 63, WDS 61 sps — all converge |
| **1:1 dataloader (raw bytes only)** | Lance 1.6 k, HF 14.7 k, raw_fs 19.3 k, WDS 2.5 k sps |

---

## TL;DR

| Decision | Choice |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` (3.1 B params; vision tower 668 M, LLM 3.75 B) |
| Fine-tune task | TextVQA scene-text Q&A — read text in images, answer questions |
| Source corpus | [`lance-format/textvqa-lance`](https://huggingface.co/datasets/lance-format/textvqa-lance) — 34,602 train rows, ships image bytes + CLIP embeddings + OCR tokens |
| Default hardware | 1×H100 80 GB, LoRA r=64, bs=2, grad-accum=4 |
| Cached column | `vision_tower_hiddens` fp16[400, 2048] = **1.64 MB/row** at 560×560 input |
| Eval | TextVQA-val accuracy + 12-prompt side-by-side markdown grid |

---

## Two benchmark stories

### 1.  1:1 read throughput (apples-to-apples)

`bench/bench_dataloader.py` measures throughput on a fixed workload —
read (image_bytes, question, answer) at bs=8 with shuffled batches.
**Honest result: at the same worker count, all four loaders converge.**
With PIL decode in the loop (the real workload), JPEG decoding is the
bottleneck and every loader lands at ~60 samples/s.  With workers,
`raw_fs` parallelises decode and wins; the others stay single-process
in this implementation.

For raw bytes alone (`--no-decode`), Lance is **not** the fastest —
parquet via HF datasets reads ~9× more bytes/sec, and direct file
reads ~12× more.  Lance's columnar random-access `take()` pays a
seek-and-fragment overhead on shuffled access that flat I/O doesn't.

| Baseline | What it is |
|---|---|
| **Lance**             | `LanceRawLoader` — fragment-level random access |
| **HF `datasets`**     | parquet shards, `.shuffle().iter(batch_size=...)` |
| **Raw filesystem**    | dir of JPEGs + manifest, PyTorch `DataLoader` |
| **WebDataset**        | tar-shard streaming |

The takeaway from this bench is not "Lance is faster on reads" — it's
"Lance is **competitive** with the alternatives on reads, and lets you
write extra columns alongside the raw bytes without rewriting the
whole table."  The real win is below.

### 2.  Lance lets you do something the others **can't**

`bench/bench_train_step.py` measures fwd+bwd+step throughput in two
modes against the same model:

| Component | Raw path | Cached path |
|---|---:|---:|
| Qwen2.5-VL ViT (~1.3 GB @ bf16) | runs every step | **not loaded** |
| Qwen2.5-LLM (3 B @ bf16)        | runs every step | runs every step |
| Image decode + processor        | runs every step | runs at backfill, cached |

The cached path reads `vision_tower_hiddens` from Lance and injects them
at `<|image_pad|>` positions in `inputs_embeds` via `masked_scatter`.
The full prompt was pre-tokenised at backfill time so the train loop
does not touch the processor either.

---

## Pipeline

```
lance-format/textvqa-lance (HF, 34,602 rows)
  │   image (bytes), question, answer, answers,
  │   image_emb (512-d), question_emb (512-d),
  │   ocr_tokens, image_classes, set_name, …
  ▼
vlm/ingest.py  →  textvqa.lance (local copy)
  │
  ├── Tier 1 (CPU, text-only — via Geneva)
  │     question_length · answer_length · question_type · ocr_token_count
  │
  ├── Tier 2 (image decode — via Geneva)
  │     dhash  (uint64, 64-bit perceptual hash)
  │
  └── Tier 3 (heavy GPU — direct lance.add_columns)
        vision_tower_hiddens  fp16[400 × 2048]
        input_ids             int32[512]   ── full chat template incl 400 image_pads
        attention_mask        int8 [512]
        labels                int32[512]   ── prompt masked to -100
  ▼
train_qwen25vl_lora.py   LoRA r=64 on q/k/v/o; vision tower = None
  ▼
eval.py                  base vs tuned, TextVQA accuracy, side-by-side md
```

---

## Schema (locked dimensions)

```
IMAGE_PX            = 560          # square input to Qwen vision tower
LLM_TOKENS_PER_IMAGE = 400         # (560/28)^2
VISION_HIDDEN       = 2048         # = LLM hidden size
MAX_TEXT_TOKENS     = 512
```

See `vlm/schema.py` for the full Arrow schema (BASE_SCHEMA + Tier-1/2/3
columns).

---

## Reproducing

```bash
cd examples/vlm-textvqa
.venv/bin/pip install -e .                              # install
export HF_TOKEN=hf_…                                    # so HF doesn't rate-limit you
RUN_DIR=runs/e2e_full bash scripts/run_pipeline.sh      # 4–5 h on 1×H100
```

For a smoke run, set `TRAIN_ROWS=256 EVAL_LIMIT=8 EPOCHS=1`.

Stage-by-stage:

```bash
.venv/bin/python -m vlm.ingest           --dst data/textvqa.lance
.venv/bin/python -m vlm.backfill_geneva  --db  data/textvqa.lance --tier 1
.venv/bin/python -m vlm.backfill_geneva  --db  data/textvqa.lance --tier 2
.venv/bin/python -m vlm.backfill_direct  --db  data/textvqa.lance --batch-size 16
.venv/bin/python -m vlm.train_qwen25vl_lora --db data/textvqa.lance --out runs/lora
.venv/bin/python -m vlm.eval             --db  data/textvqa_val.lance \
                                         --adapter runs/lora/lora --mode both
```

---

## Directory layout

```
examples/vlm-textvqa/
├── README.md                          ← this doc
├── PROPOSAL.md                        ← design doc
├── pyproject.toml
├── vlm/
│   ├── schema.py
│   ├── ingest.py                      ← stream HF -> local lance
│   ├── geneva_udfs.py                 ← Tier 1/2 UDFs
│   ├── backfill_geneva.py             ← Tier 1/2 runner
│   ├── backfill_direct.py             ← Tier 3 (vision + tokeniser, batched)
│   ├── colab_prepare.py              ← bake + push a small cached subset for Colab
│   ├── dataloader.py                  ← Lance cached + raw paths
│   ├── dataloader_baselines.py        ← HF datasets / raw FS / WebDataset
│   ├── train_qwen25vl_lora.py         ← LoRA SFT, cached path (--load-4bit for QLoRA)
│   └── eval.py                        ← accuracy + side-by-side (--load-4bit)
├── notebooks/
│   └── colab_textvqa_lance.ipynb     ← free-T4 end-to-end demo
├── bench/
│   ├── bench_dataloader.py            ← 1:1 throughput (4 loaders)
│   ├── bench_train_step.py            ← cached vs raw train step
│   └── bench_pipeline.py              ← e2e stage wall-clocks
└── scripts/
    └── run_pipeline.sh                ← single-command e2e runner
```

---

## Known constraints

- **HF rate-limits** anonymous requests within seconds.  Set `HF_TOKEN`.
- **Geneva 0.12.0 + pylance 3.0.0** are pinned together; the source HF
  Lance dataset uses a newer encoding so we stream-and-rewrite via
  `datasets.load_dataset` rather than open it directly.
- **Geneva actor pool stalls** on Tier-3-style multi-GB GPU UDFs.  We
  bypass it with `lance.add_columns(transform, read_columns=...)`,
  same pattern used in the videogen example.
