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
runs the **whole loop on a single free T4**, **Run-All with zero manual
steps** — the data is a public HF dataset, so there's nothing to configure:
download a pre-baked **curated** Lance subset → **explore it with LanceDB**
(distributions + a cross-modal vector-search demo over the shipped CLIP
features) → benchmark **Lance-vs-Parquet** read throughput → **QLoRA**
fine-tune from the cached columns → **before/after** accuracy on a held-out
curated val split. Qwen2.5-VL-3B fits 16 GB via 4-bit NF4 (`--load-4bit`)
plus the vision-tower-free cached path; peak VRAM stays **~5 GB**.

The subset is **[`lance-format/textvqa-lance-colab`](https://huggingface.co/datasets/lance-format/textvqa-lance-colab)**
— the curated **text-dense** slice (see [below](#curation-picking-the-slice-empirically)),
600 train rows with `vision_tower_hiddens` + SFT tokens pre-computed and 400
held-out val rows. The notebook defaults to it via the `TEXTVQA_COLAB_REPO`
env var.

Bake and host your own slice on any GPU box (the only manual prerequisite is
`HF_TOKEN` with write access):

```bash
python -m vlm.colab_prepare \
    --out data/colab --slice text_dense --train-rows 600 --val-rows 400 \
    --hf-repo <your-org>/textvqa-lance-colab --push      # needs HF_TOKEN
```

### Curation: picking the slice empirically

The base model is already strong on random TextVQA (~0.79), so a random
subset barely moves. We picked the slice that maximises the **before/after
gap** empirically (`vlm/slice_experiment.py`): of the candidates —
**scene-text** (questions that read specific text), **text-dense**
(top-quartile OCR-token count), and **random** — the **text-dense** slice
gives the clearest, most robust lift. Measured on the baked subset (4-bit,
held-out curated val):

| Slice | base | tuned | Δ |
|---|---:|---:|---:|
| **text-dense** (chosen) | **0.799** | **0.820** | **+2.1 pp** (256 rows); +2.3 pp on 400 |
| scene-text | 0.758 | 0.770 | +1.2 pp |
| random | 0.760 | 0.757 | ~0 |

Two findings worth their own line: the lift only appears with a **gentle
learning rate** (3e-5 + ~300 steps; the QLoRA-default 2e-4 over a few
hundred rows mildly forgets and *hurts*), and the cached payload must wire
the image in — the SFT prompt carries `LLM_TOKENS_PER_IMAGE` (400)
`<|image_pad|>` placeholders so the train loop `masked_scatter`s the cached
`vision_tower_hiddens` into them. EDA and curation use the columns the
source dataset **already ships** (CLIP `image_emb`/`question_emb`,
`ocr_tokens`, `image_classes`) — **no feature backfill**; the only thing
computed at bake time is `vision_tower_hiddens`.

## Real-run numbers (1×H100, 34,602 train rows, 200 eval rows)

| Stage              | Wall-clock | Notes |
|---|---:|---|
| ingest (stream)    | 4 min 51 s | 119 rows/s from HF |
| Tier-1 backfill    | 31 s       | 4 CPU UDFs via Geneva |
| Tier-2 backfill    | 5 min 15 s | dhash (image decode) via Geneva |
| Tier-3 backfill    | 31 min 5 s | vision tower + SFT tokens (Geneva GPU UDFs; this number was the single-process `direct` path) |
| train (3 epochs)   | 1 h 13 min | LoRA r=64, bs=2, grad-accum=4 |
| eval (base+tuned)  | 3 min 23 s | 200 val rows × 2 models |
| **total**          | **1 h 58 min** | — |

| Result | Value |
|---|---|
| **TextVQA acc, base**  | 0.793 |
| **TextVQA acc, tuned** | 0.815  (+2.2 pp) |
| **Cached-vs-raw train step** | **16.1 vs 7.9 samples/s** (−1.3 GB VRAM) |

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

## The win: cache the vision tower, train on what's left

Both dataloaders serve the table through the **LanceDB Permutation API**
(`vlm/dataloader.py`, same pattern as the
[object-detection example](https://github.com/lancedb/training/blob/main/object-detection/object_detection/dataloader.py)):
a `torch.utils.data.Dataset` that stores connection params, each worker
reopens its own `Permutation`, and `__getitems__(indices)` returns a
column-projected Arrow `RecordBatch` the collate turns into a model batch.
It never materialises the whole table into RAM, so the same loop scales
from the 600-row Colab subset to the full ~57 GB cached corpus, local or on
object storage. (The Colab bake writes the table as a single compacted
fragment — fastest for the shuffled `Permutation` reads.)

The point isn't raw read speed — it's *what you serve*:

| Component | `make_raw_loader` | `make_cached_loader` |
|---|---:|---:|
| Qwen2.5-VL ViT (~1.3 GB @ bf16) | runs every step | **not loaded** |
| Qwen2.5-LLM (3 B @ bf16)        | runs every step | runs every step |
| Image decode + processor        | runs every step | runs at backfill, cached |

`make_cached_loader` reads `vision_tower_hiddens` from Lance and the
train loop injects them at `<|image_pad|>` positions in `inputs_embeds`
via `masked_scatter`. The full prompt was pre-tokenised at backfill time,
so the loop never touches the vision tower or the processor — **16.1 vs
7.9 samples/s, −1.3 GB VRAM** vs the raw path. That's the feature
engineering (Tier 3) and the cheap column read paying off at train time.

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
  └── Tier 3 (heavy GPU — via Geneva, distributed across the actor pool)
        vision_tower_hiddens  fp16[400 × 2048]               (vision_tower_hiddens UDF)
        sft_tokens { input_ids int32[512], attention_mask    (sft_tokens UDF, one call)
                     int8[512], labels int32[512] }
        (TIER3_BACKEND=direct → flat columns via backfill_direct.py instead)
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
.venv/bin/python -m vlm.backfill_geneva  --db  data/textvqa.lance --tier 3   # Geneva GPU UDFs
# fallback (single box / no Ray): python -m vlm.backfill_direct --db data/textvqa.lance --batch-size 16
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
│   ├── ingest.py                      ← stream HF -> local lance (+ slice filter)
│   ├── slices.py                      ← canonical curation-slice definitions
│   ├── slice_experiment.py            ← empirical slice picker (base acc + lift)
│   ├── geneva_udfs.py                 ← Tier 1/2/3 UDFs (incl. vision_tower_hiddens, sft_tokens)
│   ├── backfill_geneva.py             ← Tier 1/2/3 runner (default Tier-3 path)
│   ├── backfill_direct.py             ← Tier 3 single-process path (no Ray; used by the bake)
│   ├── colab_prepare.py              ← bake + push a curated cached subset for Colab
│   ├── dataloader.py                  ← LanceDB Permutation loaders (cached + raw)
│   ├── train_qwen25vl_lora.py         ← LoRA SFT, cached path (--load-4bit for QLoRA)
│   ├── eval.py                        ← accuracy + side-by-side (--load-4bit)
│   └── verify_pipeline.py             ← sanity-check columns + artifacts
├── notebooks/
│   ├── build_notebook.py             ← regenerates the notebook from one config
│   └── colab_textvqa_lance.ipynb     ← free-T4 end-to-end demo
└── scripts/
    └── run_pipeline.sh                ← single-command e2e runner
```

---

## Known constraints

- **HF rate-limits** anonymous requests within seconds.  Set `HF_TOKEN`.
- **Geneva 0.12.0 + pylance 3.0.0** are pinned together; the source HF
  Lance dataset uses a newer encoding so we stream-and-rewrite via
  `datasets.load_dataset` rather than open it directly (this also lets the
  bake filter a slice cheaply).
- **Tier-3 GPU UDFs run two per actor by default** (`--concurrency 2`).
  Each actor lazy-loads the Qwen vision tower (~668 MB) in its worker, so
  tune concurrency to your GPU's memory. If you hit an actor-pool stall,
  fall back to the single-process path (`vlm/backfill_direct.py`, used by
  the Colab bake — same UDFs, no Ray).
- **The cached SFT tokens carry the image.** `vision_tower_hiddens` is only
  useful if the tokenised prompt has matching `<|image_pad|>` placeholders;
  `SFTTokenizer` emits `LLM_TOKENS_PER_IMAGE` (400) of them so the train
  loop `masked_scatter`s the cached hiddens into the right positions.
- **The bake writes a single Lance fragment** (`single_fragment=True` in
  `vlm/ingest.py`) so the cached table is one compact file — fastest for the
  shuffled `Permutation` reads the dataloader does.
