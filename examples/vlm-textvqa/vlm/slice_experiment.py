"""Empirically pick the TextVQA curation slice that maximises the LoRA lift.

Self-contained harness that mirrors the *notebook* code path (4-bit QLoRA,
cached vision-tower hiddens, 4-bit eval) so the before/after numbers it
measures are the ones the notebook reproduces.

Candidates (see ``vlm/slices.py``):
  * scene_text  — questions that read specific text
                  (^what (number|time|brand|name|letter|word) | how much | how many)
  * text_dense  — top-quartile ocr_token_count (lots of text in the image)
  * random      — control

Procedure: stream a pool, one base-model pass over the val pool (grouped by
slice → base acc per slice), then for each candidate a cached QLoRA train +
tuned eval on its held-out val subset; pick the largest positive delta.

What it found (4-bit, held-out curated val): **text_dense** gives the
clearest, most robust lift — base 0.799 → tuned 0.820 (+2.1 pp on 256 rows,
+2.3 pp on 400); scene_text +1.2 pp; random ~0.  Two levers mattered: a
**gentle lr** (3e-5 + ~300 steps; the QLoRA-default 2e-4 over a few hundred
rows mildly forgets and hurts), and the cached SFT tokens carrying the 400
``<|image_pad|>`` placeholders so the vision hiddens are actually used.

    python -m vlm.slice_experiment --train-pool data/exp/train_pool.lance \
        --val-pool data/exp/val_pool.lance --n-train 600 --n-val 200
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import re
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from .eval import _generate, _load_model, _score_one
from .schema import BASE_SCHEMA

LOG = logging.getLogger("vlm.slice_experiment")

SCENE_TEXT_RE = re.compile(
    r"^\s*(what\s+(number|time|brand|name|letter|word)|how much|how many)\b",
    re.IGNORECASE,
)


def _split_db(db: str) -> tuple[str, str]:
    p = Path(db)
    name = p.name[:-len(".lance")] if p.name.endswith(".lance") else p.name
    return str(p.parent), name


def _open(db: str):
    uri, name = _split_db(db)
    return lancedb.connect(uri).open_table(name)


def _pool_to_rows(tbl) -> list[dict]:
    cols = ["id", "image", "question", "answer", "answers", "ocr_tokens"]
    t = tbl.search().select(cols).limit(tbl.count_rows()).to_arrow().to_pylist()
    return t


def _ocr_quartile_threshold(rows: list[dict]) -> int:
    counts = np.array([len(r["ocr_tokens"] or []) for r in rows])
    return int(np.quantile(counts, 0.75))


def _slice_membership(rows: list[dict], ocr_thresh: int) -> dict[str, list[int]]:
    """Return slice_name -> list of row indices into `rows`."""
    scene, dense = [], []
    for i, r in enumerate(rows):
        if SCENE_TEXT_RE.search(r["question"] or ""):
            scene.append(i)
        if len(r["ocr_tokens"] or []) >= ocr_thresh:
            dense.append(i)
    rnd = list(range(len(rows)))  # control = whole pool
    return {"scene_text": scene, "text_dense": dense, "random": rnd}


def _write_subset(rows: list[dict], idxs: list[int], db: str) -> str:
    """Write the selected pool rows to a fresh Lance table (base schema)."""
    uri, name = _split_db(db)
    sub = [rows[i] for i in idxs]
    db_conn = lancedb.connect(uri)
    if name in db_conn.list_tables():
        db_conn.drop_table(name)
    # rows came from a select() that dropped emb columns; refill defaults so
    # the base schema validates (train/backfill only read image/q/a).
    arrays = []
    for f in BASE_SCHEMA:
        if f.name in sub[0]:
            arrays.append(pa.array([r[f.name] for r in sub], type=f.type))
        elif f.name in ("image_emb", "question_emb"):
            arrays.append(pa.array([[0.0] * 512 for _ in sub], type=f.type))
        elif f.name == "image_classes":
            arrays.append(pa.array([[] for _ in sub], type=f.type))
        elif f.name in ("image_id", "question_id", "set_name"):
            arrays.append(pa.array(["" for _ in sub], type=f.type))
        else:
            arrays.append(pa.array([None for _ in sub], type=f.type))
    batch = pa.RecordBatch.from_arrays(arrays, schema=BASE_SCHEMA)
    db_conn.create_table(
        name, data=pa.Table.from_batches([batch]),
        storage_options={"new_table_enable_stable_row_ids": "true"},
    )
    return db


def _backfill(db: str, batch_size: int = 16) -> None:
    import lance
    from .backfill_direct import _make_tier3_transform, _TOKEN_FIELDS
    uri, name = _split_db(db)
    ds = lance.dataset(f"{uri}/{name}.lance")
    have = set(ds.schema.names)
    need = {"vision_tower_hiddens", *_TOKEN_FIELDS}
    if need <= have:
        return
    if need & have:
        ds.drop_columns(list(need & have))
        ds = lance.dataset(f"{uri}/{name}.lance")
    ds.add_columns(_make_tier3_transform(),
                   read_columns=["image", "question", "answer"],
                   batch_size=batch_size)


def _train(db: str, out: str, max_steps: int, lr: float = 3e-5) -> str:
    from .dataloader import make_cached_loader
    from .train_qwen25vl_lora import (
        _IMAGE_PAD_TOKEN, _QWEN_MODEL_ID, _build_model, _forward_cached,
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_QWEN_MODEL_ID)
    image_pad_id = tok.convert_tokens_to_ids(_IMAGE_PAD_TOKEN)
    model = _build_model(use_lora=True, lora_r=16, load_4bit=True)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95))
    device = torch.device("cuda:0")
    grad_accum = 4
    loader = make_cached_loader(db, batch_size=2, num_workers=0, shuffle=True, seed=0)
    step = accum = 0
    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    done = False
    for epoch in range(3):
        if done:
            break
        for batch in loader:
            batch = batch.to(device)
            loss = _forward_cached(model, batch, image_pad_id)
            (loss / grad_accum).backward()
            accum += 1
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step(); optim.zero_grad(set_to_none=True)
                accum = 0; step += 1
                if step % 10 == 0:
                    LOG.info("  train step %d loss=%.4f (%.1fs)", step, loss.item(), time.time() - t0)
                if max_steps and step >= max_steps:
                    done = True
                    break
    adapter = str(Path(out) / "lora")
    Path(adapter).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter)
    peak = torch.cuda.max_memory_allocated() / 1e9
    LOG.info("trained %d steps; peak VRAM %.1f GB", step, peak)
    del model, optim, loader
    import gc; gc.collect(); torch.cuda.empty_cache()
    return adapter


def _eval_rows(adapter, rows: list[dict]) -> list[float]:
    model, proc = _load_model(adapter_dir=adapter, load_4bit=True)
    scores = []
    for r in rows:
        img = Image.open(io.BytesIO(r["image"])).convert("RGB")
        pred = _generate(model, proc, img, r["question"])
        scores.append(_score_one(pred, r["answers"]))
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return scores


def main() -> int:
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--train-pool", default="data/exp/train_pool.lance")
    p.add_argument("--val-pool", default="data/exp/val_pool.lance")
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out", default="runs/slice_experiment")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    train_rows = _pool_to_rows(_open(args.train_pool))
    val_rows = _pool_to_rows(_open(args.val_pool))
    LOG.info("pool sizes: train=%d val=%d", len(train_rows), len(val_rows))

    ocr_thresh = _ocr_quartile_threshold(train_rows)
    LOG.info("ocr top-quartile threshold (train pool) = %d tokens", ocr_thresh)
    train_mem = _slice_membership(train_rows, ocr_thresh)
    val_mem = _slice_membership(val_rows, ocr_thresh)
    for s in ("scene_text", "text_dense", "random"):
        LOG.info("slice %-11s train=%d val=%d", s, len(train_mem[s]), len(val_mem[s]))

    # ---- base pass over the whole val pool (4-bit), grouped by slice ----
    base_cache = out / "base_val_scores.json"
    if base_cache.exists():
        base_scores = json.loads(base_cache.read_text())
        LOG.info("loaded cached base val scores (%d)", len(base_scores))
    else:
        LOG.info("=== base pass over val pool (%d rows, 4-bit) ===", len(val_rows))
        base_scores = _eval_rows(None, val_rows)
        base_cache.write_text(json.dumps(base_scores))
    base_scores = np.array(base_scores)

    results = {}
    for s in ("scene_text", "text_dense", "random"):
        vidx = val_mem[s][:args.n_val]
        base_acc = float(base_scores[vidx].mean()) if vidx else float("nan")
        results[s] = {"base_acc": base_acc, "n_val": len(vidx),
                      "n_train_avail": len(train_mem[s])}
    LOG.info("base acc by slice: %s",
             {k: round(v["base_acc"], 4) for k, v in results.items()})

    # rank candidates by lowest base acc (most headroom); always include all 3
    order = sorted(("scene_text", "text_dense", "random"),
                   key=lambda s: results[s]["base_acc"])
    LOG.info("candidate order (low base first): %s", order)

    for s in order:
        tidx = train_mem[s][:args.n_train]
        vidx = val_mem[s][:args.n_val]
        if len(tidx) < 100 or len(vidx) < 16:
            LOG.warning("slice %s too small (train=%d val=%d) — skip train", s, len(tidx), len(vidx))
            continue
        LOG.info("=== slice %s: train=%d val=%d ===", s, len(tidx), len(vidx))
        sub_db = f"data/exp/slice_{s}.lance"
        _write_subset(train_rows, tidx, sub_db)
        t0 = time.time()
        _backfill(sub_db)
        LOG.info("  backfill done %.1fs", time.time() - t0)
        adapter = _train(sub_db, str(out / s), args.max_steps)
        val_subset = [val_rows[i] for i in vidx]
        tuned = np.array(_eval_rows(adapter, val_subset))
        tuned_acc = float(tuned.mean())
        base_acc = results[s]["base_acc"]
        results[s].update({"tuned_acc": tuned_acc, "delta": tuned_acc - base_acc,
                           "n_train": len(tidx)})
        LOG.info("  slice %s: base=%.4f tuned=%.4f delta=%+.4f",
                 s, base_acc, tuned_acc, tuned_acc - base_acc)

    # pick winner: largest positive delta with enough rows
    eligible = {s: r for s, r in results.items()
                if r.get("delta") is not None and "delta" in r
                and r["n_train_avail"] >= 400 and r["n_val"] >= 64}
    winner = max(eligible, key=lambda s: eligible[s]["delta"]) if eligible else None
    peak = torch.cuda.max_memory_allocated() / 1e9
    summary = {"ocr_thresh": ocr_thresh, "results": results, "winner": winner,
               "peak_vram_gb": peak}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("=== SUMMARY ===\n%s", json.dumps(summary, indent=2))
    LOG.info("WINNER: %s", winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
