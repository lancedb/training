"""Bake a small, Colab-ready TextVQA subset and (optionally) push it to HF.

The Colab notebook (`notebooks/colab_textvqa_lance.ipynb`) needs a tiny
dataset that already has the **expensive Tier-3 columns pre-computed**, so
a free T4 never has to run the vision tower at training time.  Producing
those columns needs a GPU, so we do it **once** here and host the result;
the notebook then just downloads it.

This script:

  1. Ingests a subset of the TextVQA train split  → ``textvqa_colab_train.lance``
     and runs the Tier-3 backfill on it (vision_tower_hiddens + input_ids
     + attention_mask + labels).
  2. Ingests a subset of the val split           → ``textvqa_colab_val.lance``
     (raw image/question/answer only — eval runs the full model on unseen
     images, so it does not need cached hiddens).
  3. Optionally uploads the whole output dir to a HF dataset repo.

The notebook's throughput cell mirrors the raw columns to a Parquet file
itself (from the downloaded Lance table), so we don't host one here.

Run this on a box with a GPU (any 16 GB+ card is plenty for a few hundred
rows), then point the notebook at ``--hf-repo``.

Usage:

    python -m vlm.colab_prepare \
        --out data/colab \
        --train-rows 512 --val-rows 64 \
        --tier3-batch-size 8 \
        --hf-repo lance-format/textvqa-lance-colab --push
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import lance

from .backfill_direct import _make_tier3_transform
from .ingest import _ingest_split
from . import slices

LOG = logging.getLogger("vlm.colab_prepare")

_CACHED_COLUMNS = ["vision_tower_hiddens", "input_ids", "attention_mask", "labels"]


def _backfill_tier3(db_path: str, batch_size: int) -> None:
    ds = lance.dataset(db_path)
    have = set(ds.schema.names)
    if set(_CACHED_COLUMNS).issubset(have):
        LOG.info("tier-3 columns already present on %s — skipping", db_path)
        return
    # Tear down any partial state so add_columns stays atomic.
    partial = [c for c in _CACHED_COLUMNS if c in have]
    if partial:
        LOG.warning("dropping partial tier-3 columns: %s", partial)
        ds.drop_columns(partial)
        ds = lance.dataset(db_path)
    t0 = time.time()
    ds.add_columns(
        _make_tier3_transform(),
        read_columns=["image", "question", "answer"],
        batch_size=batch_size,
    )
    LOG.info("tier-3 backfill done in %.1fs", time.time() - t0)


def _push_to_hub(folder: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    LOG.info("creating / updating dataset repo %s", repo_id)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    LOG.info("uploading %s -> %s (this can take a few minutes)", folder, repo_id)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="dataset")
    LOG.info("done: https://huggingface.co/datasets/%s", repo_id)


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/colab",
                   help="output dir holding the baked subset")
    p.add_argument("--train-rows", type=int, default=512,
                   help="rows in the cached train subset (each ~1.64 MB cached)")
    p.add_argument("--val-rows", type=int, default=64,
                   help="rows in the eval subset (raw, no cached columns)")
    p.add_argument("--tier3-batch-size", type=int, default=8)
    p.add_argument("--slice", default="scene_text", choices=[*slices.SLICES],
                   help="curation slice to keep (see vlm/slices.py). "
                        "scene_text = questions that read specific text.")
    p.add_argument("--question-regex", default=None,
                   help="override --slice with a custom regex on `question` "
                        "(case-insensitive); rows whose question matches are kept")
    p.add_argument("--scan-factor", type=int, default=25,
                   help="how many source rows to scan per kept row (slices are "
                        "sparse, so we stream more than we keep)")
    p.add_argument("--hf-repo", default=None,
                   help="HF dataset repo id to upload to, e.g. org/name")
    p.add_argument("--push", action="store_true",
                   help="actually upload to --hf-repo (needs HF_TOKEN)")
    args = p.parse_args()

    # Build the slice predicate over raw HF rows (question + ocr_tokens).
    if args.question_regex is not None:
        import re
        _rx = re.compile(args.question_regex, re.IGNORECASE)
        predicate = lambda row: bool(_rx.search(row.get("question") or ""))
        slice_label = f"regex:{args.question_regex}"
    elif args.slice == "random":
        predicate = None
        slice_label = "random"
    else:
        predicate = (lambda row: slices.matches(
            args.slice, row.get("question") or "", row.get("ocr_tokens")))
        slice_label = args.slice
    LOG.info("curation slice: %s", slice_label)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_lance = out / "textvqa_colab_train.lance"
    val_lance   = out / "textvqa_colab_val.lance"

    train_scan = None if predicate is None else args.train_rows * args.scan_factor
    val_scan   = None if predicate is None else args.val_rows * args.scan_factor

    LOG.info("=== 1/3 ingest train subset (%d rows, slice=%s) ===",
             args.train_rows, slice_label)
    _ingest_split("train", train_lance, args.train_rows,
                  predicate=predicate, scan_cap=train_scan,
                  single_fragment=True)  # one compact file for fast Permutation reads

    LOG.info("=== 2/3 tier-3 backfill on train subset ===")
    _backfill_tier3(str(train_lance), args.tier3_batch_size)

    LOG.info("=== 3/3 ingest val subset (%d rows, slice=%s) ===",
             args.val_rows, slice_label)
    _ingest_split("validation", val_lance, args.val_rows,
                  predicate=predicate, scan_cap=val_scan)

    # Record what slice this bake is, so the notebook/README can cite it.
    import json
    (out / "slice_info.json").write_text(json.dumps({
        "slice": slice_label,
        "train_rows": lance.dataset(str(train_lance)).count_rows(),
        "val_rows": lance.dataset(str(val_lance)).count_rows(),
    }, indent=2))

    LOG.info("baked subset ready under %s", out)
    LOG.info("  %-28s %d rows (cached)", train_lance.name,
             lance.dataset(str(train_lance)).count_rows())
    LOG.info("  %-28s %d rows (raw)", val_lance.name,
             lance.dataset(str(val_lance)).count_rows())

    if args.hf_repo and args.push:
        _push_to_hub(out, args.hf_repo)
    elif args.hf_repo:
        LOG.info("dry-run: pass --push to upload to %s", args.hf_repo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
