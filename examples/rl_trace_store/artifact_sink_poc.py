"""POC: the artifact sink that verifiers issue #2189 asks for, on Lance.

Context (PrimeIntellect-ai/verifiers, v1 Harbor flow):
  - the agent sandbox tars its outputs, capped by MAX_ARTIFACT_BYTES = 32 MiB
    (verifiers/v1/utils/artifacts.py)
  - the host keeps the tar bytes in trace.state.artifacts -- host RAM only,
    excluded from serialization, gone after the run
  - a second "grading" sandbox restores the tar and runs the verifier

Issue #2189 wants: a host-side artifact sink with a manifest, streaming to
disk instead of RAM retention, and artifacts inspectable after eval.

This file is that sink in ~60 lines: a Lance table keyed (trace_id, source)
with a blob column. Producers append tars as rollouts finish; graders lazily
open exactly one artifact as a file handle (no full-table reads, no RAM
retention); the manifest is queryable forever after.

Run:  python artifact_sink_poc.py
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tarfile
import time

import pyarrow as pa

import lance

SINK_SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string()),
        pa.field("source", pa.string()),  # e.g. "logs", "repo-delta", "screenshots"
        pa.field("status", pa.string()),  # collected | refused | failed
        pa.field("size_bytes", pa.int64()),
        pa.field("sha256", pa.string()),
        pa.field("created_at_ms", pa.int64()),
        lance.blob_field("tar", nullable=True),  # Blob v2: out-of-line, lazily readable
    ]
)

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024  # same ceiling as verifiers


class LanceArtifactSink:
    """Durable stand-in for `trace.state.artifacts: dict[str, bytes]`."""

    def __init__(self, uri: str):
        self.uri = uri
        if not os.path.exists(uri):
            lance.write_dataset(
                SINK_SCHEMA.empty_table(), uri, schema=SINK_SCHEMA, data_storage_version="2.2"
            )

    def collect(self, trace_id: str, source: str, tar_bytes: bytes) -> str:
        status = "collected" if len(tar_bytes) <= MAX_ARTIFACT_BYTES else "refused"
        row = pa.table(
            {
                "trace_id": [trace_id],
                "source": [source],
                "status": [status],
                "size_bytes": pa.array([len(tar_bytes)], pa.int64()),
                "sha256": [hashlib.sha256(tar_bytes).hexdigest()],
                "created_at_ms": pa.array([int(time.time() * 1000)], pa.int64()),
                "tar": lance.blob_array([tar_bytes if status == "collected" else None]),
            },
            schema=SINK_SCHEMA,
        )
        lance.write_dataset(row, self.uri, mode="append")
        return status

    def manifest(self, filter: str | None = None) -> pa.Table:
        ds = lance.dataset(self.uri)
        cols = ["trace_id", "source", "status", "size_bytes", "sha256", "created_at_ms"]
        return ds.to_table(columns=cols, filter=filter)

    def restore(self, trace_id: str, source: str):
        """Lazy file-like handle over ONE artifact -- for the grading sandbox."""
        ds = lance.dataset(self.uri)
        hit = ds.to_table(
            columns=[], filter=f"trace_id = '{trace_id}' AND source = '{source}'", with_row_id=True
        )
        if hit.num_rows == 0:
            raise KeyError(f"{trace_id}/{source}")
        return ds.take_blobs("tar", ids=[hit.column("_rowid")[0].as_py()])[0]


def fake_agent_tar(trace_id: str, mb: float) -> bytes:
    """What `collect()` produces inside the sandbox: a tar of /logs/artifacts."""
    buf = io.BytesIO()
    payload = (f"[{trace_id}] pytest output line\n" * 4000).encode() * max(1, int(mb * 10))
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("logs/artifacts/pytest.log")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


if __name__ == "__main__":
    root = "./sink_data"
    shutil.rmtree(root, ignore_errors=True)
    sink = LanceArtifactSink(os.path.join(root, "artifacts.lance"))

    # rollouts finish -> tars land in the sink instead of trace.state (RAM)
    t0 = time.perf_counter()
    for i in range(24):
        sink.collect(f"trace-{i:03d}", "logs", fake_agent_tar(f"trace-{i:03d}", mb=0.5 + i % 8))
    print(f"collected 24 artifacts in {time.perf_counter() - t0:.2f}s")

    # the grading sandbox pulls exactly one artifact, lazily
    t0 = time.perf_counter()
    with sink.restore("trace-017", "logs") as f:
        names = tarfile.open(fileobj=f, mode="r").getnames()
    print(f"restored trace-017 in {(time.perf_counter() - t0) * 1e3:.1f}ms -> {names}")

    # after the run: the manifest is still there, queryable
    big = sink.manifest(filter="size_bytes > 4000000")
    print(f"manifest query (artifacts > 4MB): {big.num_rows} rows, e.g. {big.to_pylist()[0]}")
