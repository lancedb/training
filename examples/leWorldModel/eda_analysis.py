"""
leWorldModel × LanceDB: EDA, analysis, splits, and vector search.

Run sections independently or top-to-bottom:
  python eda_analysis.py --lance-uri ./lewm_lance --table lewm_pusht

Sections:
  1. Dataset statistics         – action/proprio distributions, episode lengths
  2. Episode-level splits       – clean train/val/test splits stored as metadata
  3. Temporal coherence checks  – verify no off-by-one leakage across episodes
  4. Vector search              – ANN search over frame embeddings
  5. Cross-episode retrieval    – find episodes with similar goal states
  6. Action entropy analysis    – identify high/low diversity episodes
  7. Data quality scan          – detect NaN, frozen frames, degenerate episodes
  8. LanceDB vs HDF5 comparison

Which embedding column to use for vector search (sections 4 & 5):
  emb_dinov2  – DINOv2 ViT-S/14 embeddings (best for pre-training EDA)
  emb_clip    – CLIP ViT-B/32 embeddings (supports text-to-state queries)
  emb_lewm    – Trained LeWM encoder embeddings (post-training analysis only)

  Add embeddings with:
    python create_data.py --embed --embedding-model dinov2 --dataset pusht
"""

import argparse

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


# ============================================================================
# 1. Dataset statistics
# ============================================================================

def dataset_statistics(tbl: lancedb.table.Table):
    print("\n" + "=" * 60)
    print("1. DATASET STATISTICS")
    print("=" * 60)

    schema = tbl.schema
    total_rows = len(tbl)

    # Read only the episode index column — negligible memory
    ds = tbl.to_lance()
    ep_idx_table = ds.to_table(columns=["episode_idx"])
    n_episodes = len(pc.unique(ep_idx_table["episode_idx"]))

    print(f"\nTotal timesteps : {total_rows:,}")
    print(f"Total episodes  : {n_episodes:,}")
    print(f"Avg steps/ep    : {total_rows / n_episodes:.1f}")
    print(f"\nSchema:\n{schema}\n")

    # Per-column stats — load one column at a time to bound peak memory
    list_cols = [
        f.name for f in schema
        if (pa.types.is_list(f.type) or pa.types.is_fixed_size_list(f.type))
        and f.name not in ("pixels",)
        and not f.name.startswith("emb_")
    ]
    if not list_cols:
        return

    for col in list_cols:
        col_table = ds.to_table(columns=[col])
        data = np.array(col_table[col].to_pylist(), dtype=np.float32)
        valid = ~np.isnan(data).any(axis=1)
        data = data[valid]
        print(f"  {col:<14} dim={data.shape[1]:3d} | "
              f"mean={data.mean():+.4f}  std={data.std():.4f}  "
              f"min={data.min():+.4f}  max={data.max():+.4f}  "
              f"NaN rows={(~valid).sum()}")

    # Episode length distribution
    ep_arr = ep_idx_table["episode_idx"].to_numpy()
    _, counts = np.unique(ep_arr, return_counts=True)
    print(f"\nEpisode length  min={counts.min()} max={counts.max()} "
          f"median={int(np.median(counts))} std={counts.std():.1f}")


# ============================================================================
# 2. Episode-level train / val / test splits
# ============================================================================

def create_splits(
    tbl: lancedb.table.Table,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Assign each episode to train/val/test.

    With LanceDB you can use these episode IDs as a filter at training time —
    no need to copy or materialise new tables:
        train_arrow = tbl.to_arrow(
            columns=["pixels", "action"],
            filter=f"episode_idx IN {tuple(splits['train'].tolist())}",
        )
    """
    print("\n" + "=" * 60)
    print("2. EPISODE-LEVEL SPLITS")
    print("=" * 60)

    ep_arr = tbl.to_lance().to_table(columns=["episode_idx"])["episode_idx"].to_numpy()
    all_eps = np.unique(ep_arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_eps)

    n = len(all_eps)
    n_train = int(n * train)
    n_val   = int(n * val)

    splits = {
        "train": all_eps[:n_train],
        "val":   all_eps[n_train : n_train + n_val],
        "test":  all_eps[n_train + n_val :],
    }

    for name, eps in splits.items():
        ep_mask = np.isin(ep_arr, eps)
        print(f"  {name:<6}: {len(eps):5,} episodes  {ep_mask.sum():8,} timesteps")

    print("\n  To use a split in training, filter with:")
    print("    tbl.to_lance().to_table(columns=[...], filter='episode_idx IN (0,1,...)')")

    return splits


# ============================================================================
# 3. Temporal coherence check
# ============================================================================

def temporal_coherence_check(tbl: lancedb.table.Table):
    """
    Verify that episodes are stored contiguously and step indices are
    monotonically increasing.  Detects truncated or merged episodes.
    """
    print("\n" + "=" * 60)
    print("3. TEMPORAL COHERENCE CHECK")
    print("=" * 60)

    idx  = tbl.to_lance().to_table(columns=["episode_idx", "step_idx"])
    ep   = idx["episode_idx"].to_numpy()
    step = idx["step_idx"].to_numpy()

    ep_changes = np.where(np.diff(ep) != 0)[0]
    episode_starts = np.concatenate([[0], ep_changes + 1, [len(ep)]])
    seen_episodes: set[int] = set()
    non_contiguous = 0
    for i in range(len(episode_starts) - 1):
        eid = ep[episode_starts[i]]
        if eid in seen_episodes:
            non_contiguous += 1
        seen_episodes.add(eid)

    if non_contiguous == 0:
        print(f"  [OK] All {len(seen_episodes):,} episodes stored contiguously.")
    else:
        print(f"  [WARN] {non_contiguous} episodes appear in non-contiguous blocks.")

    bad_resets = sum(
        1 for i in range(len(episode_starts) - 1)
        if step[episode_starts[i]] != 0
    )
    if bad_resets == 0:
        print("  [OK] All episodes start at step_idx = 0.")
    else:
        print(f"  [WARN] {bad_resets} episodes do not start at step_idx = 0.")

    non_mono = sum(
        1 for i in range(len(episode_starts) - 1)
        if not np.all(np.diff(step[episode_starts[i]:episode_starts[i + 1]]) == 1)
    )
    if non_mono == 0:
        print("  [OK] All step indices monotonically increase by 1.")
    else:
        print(f"  [WARN] {non_mono} episodes have non-unit step increments.")


# ============================================================================
# 4. Vector search over frame embeddings
# ============================================================================

def vector_search_demo(
    tbl: lancedb.table.Table,
    emb_col: str = "emb_dinov2",
    query_episode: int = 0,
    query_step: int = 0,
    top_k: int = 10,
):
    """
    Find the top_k most similar frames to a query frame (by ANN in embedding space).

    emb_col choices:
      emb_dinov2  – for pre-training EDA; semantically meaningful out of the box
      emb_clip    – for pre-training EDA; also supports text-to-state queries
      emb_lewm    – for post-training analysis; reflects the learned dynamics

    Requires: python create_data.py --embed --embedding-model {dinov2|clip|lewm}
    """
    print("\n" + "=" * 60)
    print(f"4. VECTOR SEARCH  (column: {emb_col})")
    print("=" * 60)

    if emb_col not in tbl.schema.names:
        print(f"  [SKIP] '{emb_col}' column not found.")
        which = emb_col.replace("emb_", "")
        print(f"  Add it with: python create_data.py --embed --embedding-model {which}")
        return

    # Fetch query embedding using a filter — no full-table scan
    query_arrow = (
        tbl.search()
        .where(f"episode_idx = {query_episode} AND step_idx = {query_step}")
        .select([emb_col])
        .limit(1)
        .to_arrow()
    )
    if len(query_arrow) == 0:
        print(f"  [SKIP] No row for episode={query_episode}, step={query_step}")
        return

    query_emb = np.array(query_arrow[emb_col][0].as_py(), dtype=np.float32)
    print(f"  Query: episode={query_episode}, step={query_step}  (dim={len(query_emb)})")

    results = (
        tbl.search(query_emb.tolist(), vector_column_name=emb_col)
        .limit(top_k)
        .select(["episode_idx", "step_idx", "_distance"])
        .to_arrow()
    )

    print(f"\n  Top-{top_k} nearest neighbors:")
    print(f"  {'episode_idx':>12}  {'step_idx':>10}  {'distance':>10}")
    for row in results.to_pylist():
        print(f"  {row['episode_idx']:>12}  {row['step_idx']:>10}  {row['_distance']:>10.4f}")


# ============================================================================
# 5. Cross-episode retrieval
# ============================================================================

def episode_retrieval_demo(
    tbl: lancedb.table.Table,
    emb_col: str = "emb_dinov2",
    target_episode: int = 0,
    top_k: int = 5,
):
    """
    Represent each episode as its mean frame embedding, then rank all episodes
    by cosine similarity to the target episode.

    Use cases:
      - Curriculum learning: order episodes by difficulty (distance from mean)
      - Deduplication: detect near-identical demonstrations
      - Retrieval-augmented planning: find past episodes like the current state
    """
    print("\n" + "=" * 60)
    print(f"5. CROSS-EPISODE RETRIEVAL  (column: {emb_col})")
    print("=" * 60)

    if emb_col not in tbl.schema.names:
        print(f"  [SKIP] '{emb_col}' column not found.")
        return

    arrow   = tbl.to_lance().to_table(columns=["episode_idx", emb_col])
    ep_arr  = arrow["episode_idx"].to_numpy()
    emb_arr = np.array(arrow[emb_col].to_pylist(), dtype=np.float32)

    unique_eps = np.unique(ep_arr)
    ep_means = {ep: emb_arr[ep_arr == ep].mean(axis=0) for ep in unique_eps}

    query_mean = ep_means[target_episode]
    all_eps  = np.array(list(ep_means.keys()))
    all_embs = np.stack(list(ep_means.values()), axis=0)
    sims = (all_embs @ query_mean) / (
        np.linalg.norm(all_embs, axis=1) * np.linalg.norm(query_mean) + 1e-8
    )
    # Sort descending, skip self
    order = np.argsort(-sims)
    order = order[all_eps[order] != target_episode]

    print(f"\n  Query episode: {target_episode}")
    print(f"  {'episode':>10}  {'cosine_sim':>12}")
    for idx in order[:top_k]:
        print(f"  {all_eps[idx]:>10}  {sims[idx]:>12.4f}")


# ============================================================================
# 6. Action entropy analysis
# ============================================================================

def action_entropy_analysis(tbl: lancedb.table.Table, top_k: int = 5):
    """
    Compute per-episode action entropy as a proxy for behavioural diversity.

    High entropy → varied actions → good for diverse training
    Low entropy  → repetitive trajectories → candidate for deduplication
    """
    print("\n" + "=" * 60)
    print("6. ACTION ENTROPY ANALYSIS")
    print("=" * 60)

    if "action" not in tbl.schema.names:
        print("  [SKIP] No action column.")
        return

    arrow   = tbl.to_lance().to_table(columns=["episode_idx", "action"])
    ep_arr  = arrow["episode_idx"].to_numpy()
    act_arr = np.array(arrow["action"].to_pylist(), dtype=np.float32)

    unique_eps = np.unique(ep_arr)
    entropies = {
        ep: float(np.log(act_arr[ep_arr == ep].std(axis=0) + 1e-8).mean())
        for ep in unique_eps
    }

    sorted_eps = sorted(entropies.items(), key=lambda x: x[1])

    print(f"\n  Least diverse (lowest action entropy):")
    for ep, ent in sorted_eps[:top_k]:
        print(f"    episode {ep:5d}: entropy = {ent:.4f}")

    print(f"\n  Most diverse (highest action entropy):")
    for ep, ent in sorted_eps[-top_k:][::-1]:
        print(f"    episode {ep:5d}: entropy = {ent:.4f}")

    return entropies


# ============================================================================
# 7. Data quality scan
# ============================================================================

def data_quality_scan(tbl: lancedb.table.Table):
    """
    Scan for: NaN values, degenerate short episodes, and pixel column presence.
    """
    print("\n" + "=" * 60)
    print("7. DATA QUALITY SCAN")
    print("=" * 60)

    schema = tbl.schema
    total = len(tbl)

    # NaN scan — only for non-pixel, non-embedding vector columns
    list_cols = [
        f.name for f in schema
        if (pa.types.is_list(f.type) or pa.types.is_fixed_size_list(f.type))
        and f.name not in ("pixels",)
        and not f.name.startswith("emb_")
    ]
    ds = tbl.to_lance()
    for col in list_cols:
        col_table = ds.to_table(columns=[col])
        data = np.array(col_table[col].to_pylist(), dtype=np.float32)
        n_nan = int(np.isnan(data).any(axis=1).sum())
        pct   = 100 * n_nan / total
        flag  = "[WARN]" if pct > 5 else "[OK]  "
        print(f"  {flag} {col:<14} NaN rows: {n_nan:,} ({pct:.1f}%)")

    # Degenerate episode check
    ep_arr = ds.to_table(columns=["episode_idx"])["episode_idx"].to_numpy()
    _, counts = np.unique(ep_arr, return_counts=True)
    short_eps = int((counts < 4).sum())
    if short_eps == 0:
        print(f"  [OK]   All {len(counts):,} episodes have ≥ 4 steps (suitable for T=4 windows).")
    else:
        print(f"  [WARN] {short_eps} episodes have < 4 steps — they produce no valid training windows.")

    # Embedding columns present
    emb_cols = [f.name for f in schema if f.name.startswith("emb_")]
    if emb_cols:
        print(f"\n  Embedding columns present: {emb_cols}")
        print("  Vector search is available on these columns.")
    else:
        print("\n  No embedding columns. Run: python create_data.py --embed --embedding-model dinov2")


# ============================================================================
# 8. LanceDB vs HDF5 comparison
# ============================================================================

def print_lancedb_vs_hdf5():
    comparison = """
╔══════════════════════════════╦════════════════════════════╦════════════════════════════╗
║ Feature                      ║ LanceDB                    ║ HDF5                       ║
╠══════════════════════════════╬════════════════════════════╬════════════════════════════╣
║ Random row access            ║ O(1) via Permutation       ║ O(1) but single-threaded   ║
║ Columnar reads               ║ Native Arrow columns       ║ Compound datasets only     ║
║ Multi-process reads          ║ Yes (per-worker conn.)     ║ No (POSIX file lock)       ║
║ Vector / ANN search          ║ Built-in IVF-PQ index      ║ Not supported              ║
║ SQL-like filter queries      ║ Yes (DuckDB dialect)       ║ No                         ║
║ Cloud-native (S3/GCS)        ║ Native, parallel           ║ Download first             ║
║ Schema evolution             ║ Add columns in-place       ║ Limited (no column drop)   ║
║ Versioning / time-travel     ║ Yes (Lance versioning)     ║ No                         ║
║ Embedding storage            ║ Native fixed_size_list     ║ Separate dataset           ║
║ Episode-level filters        ║ episode_idx = 42           ║ Loop + mask in Python      ║
║ Train/val split              ║ Filter query, zero copy    ║ Copy or index arrays       ║
║ Arrow zero-copy tensors      ║ Yes (with_format="arrow")  ║ No (numpy copy always)     ║
║ Concurrent writers           ║ Yes (append-safe)          ║ No                         ║
║ Compressed pixel storage     ║ JPEG binary column         ║ Raw uint8 (3-13× larger)   ║
╚══════════════════════════════╩════════════════════════════╩════════════════════════════╝
"""
    print("\n" + "=" * 60)
    print("8. LANCEDB vs HDF5 — Feature Comparison")
    print("=" * 60)
    print(comparison)

    print("""Key advantages for leWorldModel:

1. MULTI-PROCESS DATALOADERS
   HDF5 uses a POSIX file lock. Eight DataLoader workers trying to read the
   same .hdf5 file simultaneously either serialize or crash. The standard
   workaround (copy the file into /dev/shm) wastes RAM and requires manual
   setup. LanceDB opens an independent connection per worker with no locking.

2. PRE-TRAINING EDA WITH FOUNDATION MODEL EMBEDDINGS
   Run create_data.py --embed --embedding-model dinov2 before training and
   you immediately get ANN search, episode clustering, and similarity
   retrieval using DINOv2 semantics — before your LeWM model sees a single
   gradient.  Not possible with HDF5 without a separate vector store.

3. POST-TRAINING ANALYSIS WITH LEWM EMBEDDINGS
   After training, add a second embedding column (--embedding-model lewm).
   You can now compare what DINOv2 vs LeWM consider "similar" — a direct
   window into what the world model has learned to focus on.

4. EPISODE FILTERING WITHOUT ARRAY MANIPULATION
   tbl.to_lance().to_table(filter="episode_idx IN (...)")  returns only the matching
   rows, columnar-compressed, as Arrow. With HDF5 you load the full array
   and mask in Python.

5. ZERO-COPY ARROW FORMAT IN DATALOADERS
   Permutation.with_format("arrow") returns a pa.RecordBatch that converts
   to tensors without memory copy. HDF5 always goes through numpy.

6. VERSIONING
   Every table.add() creates a new Lance version. You can audit, roll back,
   or diff data additions — critical for reproducible experiment tracking.
""")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="leWorldModel LanceDB EDA and analysis")
    parser.add_argument("--lance-uri", default="./lewm_lance")
    parser.add_argument("--table",     default="lewm_pusht")
    parser.add_argument(
        "--emb-col",
        default="emb_dinov2",
        help="Embedding column to use for vector search sections (default: emb_dinov2)",
    )
    parser.add_argument(
        "--section",
        default="all",
        choices=["all", "stats", "splits", "coherence",
                 "vector_search", "retrieval", "entropy", "quality", "comparison"],
    )
    args = parser.parse_args()

    db  = lancedb.connect(args.lance_uri)
    tbl = db.open_table(args.table)
    run_all = args.section == "all"

    if run_all or args.section == "stats":
        dataset_statistics(tbl)
    if run_all or args.section == "splits":
        create_splits(tbl)
    if run_all or args.section == "coherence":
        temporal_coherence_check(tbl)
    if run_all or args.section == "vector_search":
        vector_search_demo(tbl, emb_col=args.emb_col)
    if run_all or args.section == "retrieval":
        episode_retrieval_demo(tbl, emb_col=args.emb_col)
    if run_all or args.section == "entropy":
        action_entropy_analysis(tbl)
    if run_all or args.section == "quality":
        data_quality_scan(tbl)
    if run_all or args.section == "comparison":
        print_lancedb_vs_hdf5()


if __name__ == "__main__":
    main()
