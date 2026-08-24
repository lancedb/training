"""LanceArtifactSink — a verifiers `ArtifactSink` backed by one Lance blob table.

Implements the sink protocol proposed for PrimeIntellect-ai/verifiers#2189 with a
columnar store instead of loose files: every collected archive is a row with a
queryable manifest (trace_id, source, status, size, sha256) and a Blob-v2 payload
that scans never touch. What that buys over a directory of tars:

  - one dataset for the whole run (or many runs), on local disk or object storage
  - the manifest is a query surface: `size > 8MB AND status = 'collected'` across
    every trace, in milliseconds, no directory walking
  - archives read back as digest-verified bytes via random access (~ms), or
    lazily as file handles
  - every append is a commit: concurrent env workers don't conflict, and the
    whole artifact record is versioned

Config (verifiers side):
    [env.taskset.task]
    artifact_sink = "lance_artifact_sink:make_sink?/data/artifacts.lance"

Requires: pip install pylance>=10.0  (and this module on PYTHONPATH).
"""

from __future__ import annotations

import hashlib
import time
from asyncio import to_thread

import lance
import pyarrow as pa

from verifiers.v1.utils.artifact_sink import ArtifactRef, ArtifactSink

SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("status", pa.string()),  # collected | missing
        pa.field("size", pa.int64()),
        pa.field("sha256", pa.string()),
        pa.field("created_at_ms", pa.int64()),
        lance.blob_field("tar", nullable=True),
    ]
)


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class LanceArtifactSink(ArtifactSink):
    def __init__(self, uri: str, spec: str | None = None) -> None:
        self.uri = uri
        self.spec = spec or f"{__name__}:make_sink?{uri}"
        try:
            lance.dataset(uri)
        except ValueError:
            lance.write_dataset(
                SCHEMA.empty_table(), uri, schema=SCHEMA, data_storage_version="2.2"
            )

    async def put(
        self, trace_id: str, source: str, data: bytes | None
    ) -> ArtifactRef | None:
        return await to_thread(self._put_sync, trace_id, source, data)

    async def get(self, ref: ArtifactRef) -> bytes:
        return await to_thread(self._get_sync, ref)

    def _put_sync(
        self, trace_id: str, source: str, data: bytes | None
    ) -> ArtifactRef | None:
        digest = hashlib.sha256(data).hexdigest() if data is not None else None
        row = pa.table(
            {
                "trace_id": [trace_id],
                "source": [source],
                "status": ["collected" if data is not None else "missing"],
                "size": pa.array([len(data) if data is not None else None], pa.int64()),
                "sha256": [digest],
                "created_at_ms": pa.array([int(time.time() * 1000)], pa.int64()),
                "tar": lance.blob_array([data]),
            },
            schema=SCHEMA,
        )
        # Appends never conflict with appends: concurrent env workers are safe.
        lance.write_dataset(row, self.uri, mode="append")
        if data is None:
            return None
        return ArtifactRef(
            sink=self.spec,
            key=f"{trace_id}|{source}",
            source=source,
            size=len(data),
            sha256=digest,
        )

    def _get_sync(self, ref: ArtifactRef) -> bytes:
        trace_id, _, source = ref.key.partition("|")
        ds = lance.dataset(self.uri)
        hits = ds.to_table(
            columns=[],
            filter=(
                f"trace_id = {_quote(trace_id)} AND source = {_quote(source)} "
                "AND status = 'collected'"
            ),
            with_row_id=True,
        )
        if hits.num_rows == 0:
            raise KeyError(f"artifact {ref.key!r} not found in {self.uri!r}")
        row_id = hits.column("_rowid")[-1].as_py()  # last write wins on retries
        blob = ds.take_blobs("tar", ids=[row_id])[0]
        with blob as f:
            data = f.read()
        if len(data) != ref.size or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise RuntimeError(
                f"artifact {ref.key!r} does not match its manifest digest"
            )
        return data

    def manifest(self, filter: str | None = None) -> pa.Table:
        """The queryable side: per-source rows across every trace in the run."""
        return lance.dataset(self.uri).to_table(
            columns=["trace_id", "source", "status", "size", "sha256", "created_at_ms"],
            filter=filter,
        )


def make_sink(args: str) -> LanceArtifactSink:
    """`load_sink` factory: `lance_artifact_sink:make_sink?<dataset uri>`."""
    if not args:
        raise ValueError("lance sink spec needs a dataset URI after '?'")
    return LanceArtifactSink(args)
