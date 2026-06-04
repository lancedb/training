# GPU runbook — finish & verify the VLM-TextVQA Colab end-to-end

Hand this whole file to a Claude Code agent running **on a GPU box** (≥16 GB; a T4
works, a bigger card bakes faster). Goal: produce a Colab notebook a user can open
and Run-All on a free T4 with **zero manual steps**, backed by a curated dataset
you create and upload to the `lance-format` HF org.

You are working in the `lancedb/training` repo, branch **`vlm-textvqa`**, example
dir **`examples/vlm-textvqa/`**. Commit and push your work to that branch when done.

## The one manual prerequisite (the user provides this)
- `export HF_TOKEN=...` — a token with **write** access to the `lance-format` HF org.
- A CUDA GPU. Everything else you install/run yourself.

## Context you need (don't rediscover it)
- Example fine-tunes **Qwen/Qwen2.5-VL-3B-Instruct** with LoRA on **TextVQA**.
- Source dataset **`lance-format/textvqa-lance`** ALREADY ships: `image` bytes,
  `question`, `answer`/`answers`, **`image_emb` + `question_emb` (512-d CLIP)**,
  `ocr_tokens`, `image_classes`. So EDA + curation need **no** feature backfill.
- The ONLY thing not in the dataset is the cached training payload
  **`vision_tower_hiddens`** (Qwen ViT output) + SFT tokens. That's the example's
  "headline trick": compute it once, train without the vision tower.
- Data layer is LanceDB: dataloaders use the **Permutation API**
  (`vlm/dataloader.py`: `make_cached_loader` / `make_raw_loader`). Reads go through
  `lancedb.connect().open_table()`. The cached Tier-3 backfill on a single box uses
  `vlm/backfill_direct.py` (no Ray; reuses the `VisionTowerEmbedder`/`SFTTokenizer`
  UDFs). QLoRA fits a T4 via `--load-4bit` on train + eval.
- Existing scaffolding to extend, not rewrite: `vlm/colab_prepare.py` (bake+upload),
  `notebooks/colab_textvqa_lance.ipynb` (the notebook), `vlm/ingest.py`,
  `vlm/eval.py`, `vlm/train_qwen25vl_lora.py`.

## Tasks

### 1. Pick the curation slice EMPIRICALLY (don't guess)
The base model is already strong on random data (~0.79), so a random subset shows a
small lift. Choose the slice that maximizes the **before/after gap**. Evaluate these
candidates and pick the winner with evidence:
- **scene-text**: questions reading specific text — regex on `question` like
  `^\s*(what\s+(number|time|brand|name|letter|word)|how much|how many)\b`.
- **text-dense**: top-quartile `len(ocr_tokens)` (lots of text in the image).
- **random**: control.

Procedure: stream ~1–2k rows of `train` + ~300 of `validation` from the HF dataset,
build each candidate slice, run **base-model** TextVQA accuracy on each slice's val
(no training — cheap). Then for the 1–2 lowest-base-accuracy slices, do a quick
1-epoch LoRA train (a few hundred rows) and measure tuned−base on held-out rows.
**Pick the slice with the largest positive delta** and enough rows (≥~400 train,
≥~64 val). Record the chosen filter + the measured base/tuned numbers — you'll cite
them in the README and notebook.

### 2. Bake + upload the curated subset to HF
Extend `vlm/colab_prepare.py` with a slice filter (e.g. `--slice scene_text|text_dense|random`
or a `--question-regex`) applied during/after ingest, before the Tier-3 backfill.
Then bake and upload:
- `textvqa_colab_train.lance` — the curated **train** slice WITH `vision_tower_hiddens`
  + SFT tokens (use the existing direct backfill).
- `textvqa_colab_val.lance` — a held-out **val** slice from the same filter (raw
  columns only; eval runs the full model on images).
- `cached_train.parquet` — parquet mirror of the cached columns (for the throughput cell).
- Upload all to **`lance-format/textvqa-lance-colab`** (create the dataset repo if
  needed; `exist_ok=True`). Keep it small (≈400–600 train rows ⇒ <1 GB).
Verify the upload by re-downloading via `snapshot_download` and opening with `lancedb`.

### 3. Add an EDA section to the notebook (before training)
Insert an EDA section that uses the columns ALREADY in the table and showcases
LanceDB (mirror the spirit of `object-detection/notebooks/eda_bdd100k.ipynb`):
- `question_type` (derive inline by regex) and answer-length distributions,
- `ocr_token_count` histogram, top `image_classes`,
- a **vector-search** demo: text→image using `question_emb`/`image_emb` via the
  LanceDB API (`tbl.search(...)`), showing a query + nearest images,
- a few sample image+Q/A thumbnails (reuse `vlm.eval._b64_thumb`).
All reads via `lancedb` (no raw `lance.dataset`).

### 4. Wire the notebook to the uploaded dataset
- Default `TEXTVQA_COLAB_REPO` to `lance-format/textvqa-lance-colab`.
- Keep the flow: setup → download → **EDA** → Lance-vs-Parquet throughput (Permutation
  API on the Lance side) → QLoRA train via `make_cached_loader` → before/after grid.
- Make the before/after grid evaluate on the **held-out curated val** so the lift is
  the curated one you measured in step 1.

### 5. Execute the notebook end-to-end on the GPU and fix until green
- Run headless: `jupyter nbconvert --to notebook --execute --inplace
  notebooks/colab_textvqa_lance.ipynb` (or papermill), with a fresh kernel.
- It MUST run top-to-bottom with no errors. Fix any breakage (4-bit/bnb quirks,
  `masked_scatter` dtype under 4-bit, struct-vs-flat `sft_tokens`, Permutation
  worker/spawn issues, memory). Re-run until clean.
- Confirm the before/after cell shows **tuned ≥ base** on the curated slice. If it
  doesn't, revisit the slice choice in step 1.
- Keep peak VRAM under ~15 GB so it fits a free T4 (4-bit; vision tower only loaded
  for eval, freed before/after training).

### 6. Update docs + commit
- Update `README.md`: note the dataset is `lance-format/textvqa-lance-colab`, the
  chosen slice + its measured before/after numbers, and that EDA + curation use the
  dataset's existing CLIP/OCR features (no backfill) while only `vision_tower_hiddens`
  is computed.
- Commit notebook (cleared or with outputs — your call; if cleared, also save a
  rendered `notebooks/colab_textvqa_lance.html` as run evidence), `colab_prepare.py`
  changes, and README. Push to `vlm-textvqa`. End the commit message with the
  standard Co-Authored-By line.

## Acceptance criteria (all must hold)
1. `lance-format/textvqa-lance-colab` exists and re-downloads + opens with `lancedb`.
2. `notebooks/colab_textvqa_lance.ipynb` runs end-to-end headless on the GPU with
   exit 0 and no errors, peak VRAM < ~15 GB.
3. The notebook needs **no** manual steps beyond selecting a GPU runtime + (if the
   dataset were private) an HF token — the data is public, so ideally zero.
4. EDA section renders distributions + a working vector-search demo.
5. Before/after grid shows a positive curated-slice lift; the README cites the
   measured base/tuned numbers.
6. Everything committed and pushed to `vlm-textvqa`.

## Gotchas / facts to respect
- Use the **LanceDB Permutation API** for the dataloader and `lancedb.connect()` for
  reads — this example was specifically aligned to that (don't reintroduce raw
  `lance.dataset().take()`).
- `lance-format/textvqa-lance` may be written in a newer Lance encoding than some
  pinned stacks; the robust path is to stream via `datasets.load_dataset(...,
  streaming=True)` and re-encode locally (see `vlm/ingest.py`), which also lets you
  filter the slice cheaply.
- The Colab bake is a single box → use the **direct** Tier-3 backfill
  (`vlm/backfill_direct.py`), not Geneva/Ray.
- Delete this file after the build if you don't want it in the example.
