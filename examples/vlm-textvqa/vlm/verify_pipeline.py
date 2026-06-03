"""Quick sanity check on an end-to-end pipeline output.

Reads the Lance table, the trained adapter (if present), and the eval
JSON; reports whether every expected piece is there and consistent.

Usage:

    python -m vlm.verify_pipeline --db data/textvqa.lance --run-dir runs/e2e_full
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lancedb
import numpy as np

from .schema import (
    BASE_SCHEMA, LLM_TOKENS_PER_IMAGE, MAX_TEXT_TOKENS,
    TIER1_COLUMNS, TIER2_COLUMNS,
    VISION_HIDDEN,
)

LOG = logging.getLogger("vlm.verify")

_TOKEN_FIELDS = ("input_ids", "attention_mask", "labels")


def _split_db(db: str) -> tuple[str, str]:
    p = Path(db)
    name = p.name[:-len(".lance")] if p.name.endswith(".lance") else p.name
    return str(p.parent), name


def _check_columns(schema) -> list[str]:
    """Return list of issues (empty list = pass)."""
    issues: list[str] = []
    have = {f.name: f.type for f in schema}

    for f in BASE_SCHEMA:
        if f.name not in have:
            issues.append(f"missing base column {f.name}")

    for tier, cols in {"Tier 1": TIER1_COLUMNS, "Tier 2": TIER2_COLUMNS}.items():
        for name, dtype in cols.items():
            if name not in have:
                issues.append(f"missing {tier} column {name} ({dtype})")
            elif str(have[name]) != str(dtype):
                issues.append(f"{tier} column {name}: type {have[name]} != expected {dtype}")

    # Tier 3: vision_tower_hiddens + tokens as flat columns OR an sft_tokens struct.
    if "vision_tower_hiddens" not in have:
        issues.append("missing Tier 3 column vision_tower_hiddens")
    flat = all(t in have for t in _TOKEN_FIELDS)
    struct = "sft_tokens" in have
    if not (flat or struct):
        issues.append("missing Tier 3 tokens (neither flat input_ids/... nor sft_tokens struct)")
    return issues


def _check_tier3_row(tbl) -> list[str]:
    issues = []
    have = set(tbl.schema.names)
    struct = "sft_tokens" in have and not all(t in have for t in _TOKEN_FIELDS)
    cols = ["vision_tower_hiddens"] + (["sft_tokens"] if struct else list(_TOKEN_FIELDS))
    row = tbl.search().select(cols).limit(1).to_arrow().to_pydict()

    v = np.asarray(row["vision_tower_hiddens"][0], dtype=np.float16)
    if v.shape != (LLM_TOKENS_PER_IMAGE * VISION_HIDDEN,):
        issues.append(f"vision shape {v.shape} != ({LLM_TOKENS_PER_IMAGE*VISION_HIDDEN},)")
    if not np.isfinite(v).all():
        issues.append("vision_tower_hiddens contains non-finite values")
    if abs(v.mean()) > 5.0:
        issues.append(f"vision mean {v.mean():.3f} is suspiciously large")

    tokens = row["sft_tokens"][0] if struct else row
    for col in _TOKEN_FIELDS:
        arr = np.asarray(tokens[col][0] if not struct else tokens[col])
        if arr.shape != (MAX_TEXT_TOKENS,):
            issues.append(f"{col} shape {arr.shape} != ({MAX_TEXT_TOKENS},)")
    return issues


def _check_run_dir(run_dir: Path) -> list[str]:
    issues = []
    expected = ["lora/lora", "eval/accuracy.json", "eval/side_by_side.md"]
    for rel in expected:
        if not (run_dir / rel).exists():
            issues.append(f"missing artifact {rel}")
    return issues


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--db",      default="data/textvqa.lance")
    p.add_argument("--run-dir", default=None,
                   help="if set, also verify train/eval artifacts")
    args = p.parse_args()

    issues: list[str] = []
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"FAIL: db not found: {db_path}")
        return 1

    uri, table_name = _split_db(str(db_path))
    tbl = lancedb.connect(uri).open_table(table_name)
    LOG.info("opened %s/%s (rows=%d, cols=%d)", uri, table_name,
             tbl.count_rows(), len(tbl.schema.names))

    issues.extend(_check_columns(tbl.schema))
    if not any("Tier 3" in i for i in issues):
        issues.extend(_check_tier3_row(tbl))

    if args.run_dir:
        rd = Path(args.run_dir).resolve()
        if not rd.exists():
            issues.append(f"run dir not found: {rd}")
        else:
            issues.extend(_check_run_dir(rd))
            acc_file = rd / "eval/accuracy.json"
            if acc_file.exists():
                acc = json.loads(acc_file.read_text())
                LOG.info("eval accuracies: %s",
                         {k: round(v.get("accuracy", 0), 4) for k, v in acc.items()})

    if issues:
        print(f"\nFAIL — {len(issues)} issues:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("\nOK — pipeline verifies clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
