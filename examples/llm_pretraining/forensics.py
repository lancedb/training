"""Interrogate the training table: hybrid search, generation attribution,
semantic near-duplicates.

Requires the `embedding` column (geneva_backfill.py --columns embedding) and
a vector index (created here on first run).

Usage
-----
python forensics.py index --db ~/blogrun/db
python forensics.py hybrid --db ~/blogrun/db --query "carbon cycle photosynthesis"
python forensics.py attribute --db ~/blogrun/db --text "Photosynthesis is the process..."
python forensics.py neardups --db ~/blogrun/db --sample 2000
"""

from __future__ import annotations

import argparse

from common import DEFAULT_DB, DEFAULT_TABLE, banner, connect_table


def _embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts, normalize_embeddings=True)


def cmd_index(tbl, args) -> None:
    banner("VECTOR INDEX (IVF-PQ, cosine) on `embedding`")
    kwargs = {"num_partitions": args.partitions} if args.partitions else {}
    tbl.create_index(metric="cosine", vector_column_name="embedding", **kwargs)
    print("index created")


def cmd_hybrid(tbl, args) -> None:
    banner(f"HYBRID SEARCH (BM25 + vector): {args.query!r}")
    qvec = _embed([args.query])[0]
    hits = (
        tbl.search(query_type="hybrid", vector_column_name="embedding")
        .vector(qvec)
        .text(args.query)
        .select(["id", "score", "text"])
        .limit(args.k)
        .to_list()
    )
    for h in hits:
        print(f"  id={h['id']:<8} edu={h['score']:.2f}  {h['text'][:90]!r}")


def cmd_attribute(tbl, args) -> None:
    banner("GENERATION ATTRIBUTION: nearest training docs to generated text")
    print(f"generated: {args.text[:120]!r}\n")
    qvec = _embed([args.text])[0]
    hits = (
        tbl.search(qvec, vector_column_name="embedding")
        .metric("cosine")
        .select(["id", "text"])
        .limit(args.k)
        .to_list()
    )
    for h in hits:
        sim = 1 - h["_distance"]
        print(f"  cos={sim:.3f} id={h['id']:<8} {h['text'][:100]!r}")
    top = 1 - hits[0]["_distance"] if hits else 0.0
    verdict = "near-verbatim source likely" if top > 0.9 else "no close match — remixed"
    print(f"\n  top similarity {top:.3f} -> {verdict}")


def cmd_neardups(tbl, args) -> None:
    banner(f"SEMANTIC NEAR-DUPLICATES (sample of {args.sample})")
    rows = (
        tbl.search()
        .where("NOT is_dup")
        .select(["id", "embedding", "text"])
        .limit(args.sample)
        .to_list()
    )
    flagged = 0
    worst = None
    for r in rows:
        hits = (
            tbl.search(r["embedding"], vector_column_name="embedding")
            .metric("cosine")
            .where(f"id != {r['id']}", prefilter=True)
            .select(["id"])
            .limit(1)
            .to_list()
        )
        if hits and 1 - hits[0]["_distance"] >= args.threshold:
            flagged += 1
            if worst is None or hits[0]["_distance"] < worst[2]:
                worst = (r["id"], hits[0]["id"], hits[0]["_distance"], r["text"][:80])
    print(f"  {flagged}/{len(rows)} docs have a neighbor with cosine >= {args.threshold}")
    print(f"  ({flagged / max(len(rows), 1):.1%} semantic near-dup rate beyond exact dedup)")
    if worst:
        print(f"  closest pair: id={worst[0]} ~ id={worst[1]} cos={1 - worst[2]:.3f}")
        print(f"    {worst[3]!r}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["index", "hybrid", "attribute", "neardups"])
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--query", default="carbon cycle photosynthesis")
    p.add_argument("--text", default="Photosynthesis is the process by which plants produce energy.")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--threshold", type=float, default=0.92)
    p.add_argument("--partitions", type=int, default=0)
    args = p.parse_args(argv)

    tbl = connect_table(args.db, args.table)
    {
        "index": cmd_index,
        "hybrid": cmd_hybrid,
        "attribute": cmd_attribute,
        "neardups": cmd_neardups,
    }[args.cmd](tbl, args)


if __name__ == "__main__":
    main()
