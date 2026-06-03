"""Single-process Tier-3 backfill — the no-Ray fallback for ``backfill_geneva``.

The default Tier-3 path is Geneva (``vlm/backfill_geneva.py --tier 3``),
which distributes the work across its actor pool.  This module runs the
**exact same UDFs** — ``VisionTowerEmbedder`` and ``SFTTokenizer`` from
``vlm/geneva_udfs.py`` — in a single process via Lance's
``add_columns(transform, read_columns=...)`` API.  No Ray, no actor pool.

There is no separate implementation of the Tier-3 compute here: the
transform just instantiates the Geneva UDF classes and calls them per
row.  That keeps one source of truth for what the columns contain, while
giving you a path that runs on a single box (e.g. the Colab bake in
``vlm/colab_prepare.py``) or sidesteps Ray if an actor-pool issue shows
up.  It writes the four columns flat:

  * ``vision_tower_hiddens``  fp16[LLM_TOKENS_PER_IMAGE * VISION_HIDDEN]
  * ``input_ids``             int32[MAX_TEXT_TOKENS]
  * ``attention_mask``        int8 [MAX_TEXT_TOKENS]
  * ``labels``                int32[MAX_TEXT_TOKENS]

(The Geneva path writes ``vision_tower_hiddens`` + an ``sft_tokens``
struct; ``LanceCachedLoader`` reads either layout.)

Usage:

    python -m vlm.backfill_direct --db data/textvqa.lance --batch-size 8
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

from .geneva_udfs import SFTTokenizer, VisionTowerEmbedder
from .schema import LLM_TOKENS_PER_IMAGE, MAX_TEXT_TOKENS, VISION_HIDDEN

LOG = logging.getLogger("vlm.backfill_direct")

_TOKEN_FIELDS = ("input_ids", "attention_mask", "labels")


def _make_tier3_transform():
    """Build a Lance ``add_columns`` transform that runs the Geneva UDFs.

    The UDF classes are stateful — they lazy-load their model on the
    first call — so we instantiate them once per process here.
    """
    embed = VisionTowerEmbedder()
    tokenize = SFTTokenizer()
    v_dim = LLM_TOKENS_PER_IMAGE * VISION_HIDDEN

    def _fsl(flat: np.ndarray, dtype, width: int) -> pa.FixedSizeListArray:
        return pa.FixedSizeListArray.from_arrays(
            pa.array(flat.reshape(-1), type=dtype), width
        )

    def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
        images    = batch.column("image").to_pylist()
        questions = batch.column("question").to_pylist()
        answers   = batch.column("answer").to_pylist()

        # Same UDFs Geneva runs — just called in-process, per row.
        vis = np.asarray([embed(img) for img in images], dtype=np.float16)
        tok = [tokenize(q, a) for q, a in zip(questions, answers)]

        def _stack(field: str, dtype) -> np.ndarray:
            return np.asarray([t[field] for t in tok], dtype=dtype)

        return pa.RecordBatch.from_arrays(
            [
                _fsl(vis,                              pa.float16(), v_dim),
                _fsl(_stack("input_ids",      np.int32), pa.int32(), MAX_TEXT_TOKENS),
                _fsl(_stack("attention_mask", np.int8),  pa.int8(),  MAX_TEXT_TOKENS),
                _fsl(_stack("labels",         np.int32), pa.int32(), MAX_TEXT_TOKENS),
            ],
            names=["vision_tower_hiddens", *_TOKEN_FIELDS],
        )

    return transform


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--db",         default="data/textvqa.lance")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    db_path = str(Path(args.db).resolve())
    ds = lance.dataset(db_path)
    LOG.info("opened %s (rows=%d)", db_path, ds.count_rows())

    needed = {"vision_tower_hiddens", *_TOKEN_FIELDS}
    have = set(ds.schema.names)
    missing = needed - have
    if not missing:
        LOG.info("all tier-3 columns already present — nothing to do")
        return 0
    if missing != needed:
        # partial state: tear it down so add_columns is atomic
        LOG.warning("partial tier-3 columns present (%s); dropping and rerunning",
                    needed & have)
        ds.drop_columns(list(needed & have))
        ds = lance.dataset(db_path)

    transform = _make_tier3_transform()
    t0 = time.time()
    ds.add_columns(
        transform,
        read_columns=["image", "question", "answer"],
        batch_size=args.batch_size,
    )
    LOG.info("tier-3 backfill done in %.1fs", time.time() - t0)
    ds = lance.dataset(db_path)
    LOG.info("FINAL columns: %s", ds.schema.names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
