# Known issues — videogen example

## Lance MV string-read panic when parent has `lance-encoding:blob = true`

**Status:** workaround in place — `schema.BLOB_META` defaults to `{}`.
**Severity:** blocks Geneva-backed training views when the parent table marks
`video_bytes` as a Lance blob column. Functional impact: `video_bytes` is
stored inline in column pages instead of dedicated blob regions. We expect
worse storage/IO for very large videos but no API change.

### What we saw

After running:

```
python -m videogen.ingest_chronomagic --synthetic 500 --overwrite
python -m videogen.backfill_geneva --tier 1
python -m videogen.manage_views --action curate
```

Any string-column read off a Geneva materialised view panics with:

```
RuntimeError: task N panicked with message "assertion `left == right`
failed: StringArray data should contain 2 buffers only (offsets and values)
  left: 1
 right: 2",
/home/runner/work/lance/lance/rust/lance-encoding/src/decoder.rs:1477:40
```

Reproduces in all three read paths:
- `mv.search().select(["clip_id", "caption"]).limit(3).to_pandas()`
- `geneva.connect(...).open_table(name).to_arrow()`
- `lance.dataset(name + ".lance").scanner(columns=["clip_id"]).to_table()`

Removing the blob metadata from the parent's `video_bytes` field makes
*every* read above succeed.

### Versions

```
lancedb            == 0.30.2
pylance            == 3.0.0      (lance crate via this wheel)
lance-encoding     == 4.0.0      (per the panic message)
geneva             == 0.12.0
arrow-array        == 57.3.0
pyarrow            == 24.0.0
python             == 3.11.15  on  Linux x86_64
```

### Bisecting what we tried

Couldn't yet reduce it to a self-contained script. All of these passed in a
fresh process with the blob flag on:

- 3 cols `(clip_id:str, caption:str, video_bytes:blob)` × 4 rows
- 4 cols incl. extra `bool` or `int32` × 4 rows
- 11 cols matching the videogen schema × 4 rows
- 3 cols × {4, 16, 64, 256, 500} rows, captions up to 200 chars

The full videogen pipeline (`ingest_chronomagic --synthetic 500` →
`backfill_geneva --tier 1` → `manage_views --action curate`) reliably
triggers the panic, but stripping just the schema and creating an MV in
one process does not. Suspect interaction with one of:

- Geneva's `add_columns({…})` schema-evolution step that runs during
  `backfill_geneva` (the parent table gains 7 columns after creation).
- Multi-fragment MVs (Geneva writes them at concurrency=4).
- Whether the MV's `where` matches >50% of rows.

If you have spare time, the next bisects to try are:
1. Reproduce by **adding columns after table creation** (mimic Geneva).
2. Use multi-fragment input (`batch_size=64` in the ingest).
3. MV filter that matches ~90% of rows.

### Workaround applied

`videogen/schema.py` toggles `BLOB_META` between `{}` (default, current)
and `_BLOB_META_ENABLED`. With the empty dict, the blob flag is not
written and MVs read cleanly. Once the upstream regression lands a fix,
flip the constant back.

### Suggested issue title

> MV scan panics with "StringArray data should contain 2 buffers" when
> parent has `lance-encoding:blob = true` (lance-encoding 4.0.0)

### Suggested issue body skeleton

```
**Versions**
lancedb 0.30.2 / pylance 3.0.0 / lance-encoding 4.0.0 /
geneva 0.12.0 / pyarrow 24.0.0 / arrow-array 57.3.0

**What happens**
After running our videogen example (ingest → tier-1 backfill → MV curate),
any string-column read off the Geneva MV panics in the Lance Arrow
decoder with `StringArray data should contain 2 buffers only`.

**Repro**
We have a runnable end-to-end repro inside `examples/videogen/` of the
training repo, but have not yet been able to reduce it to a few lines
in a fresh process. Two known reductions:
- Pipeline above always panics.
- Pure schema + small MV in one process does *not* panic.

The diff between the two is Geneva's `add_columns` (which extends the
parent schema between table create and MV create) and/or multi-fragment
writes.

**Workaround**
Drop `lance-encoding:blob = true` field metadata from the parent's
`large_binary` column. MV reads then work.
```

The standalone scripts we used while bisecting are at:
- `/tmp/repro_mv_blob_string.py` (passes — does *not* reproduce)
- `/tmp/repro_min.py` (passes)
- `/tmp/repro_size.py` (passes up to 500 rows)
- The failing case is the full pipeline run from this directory.
