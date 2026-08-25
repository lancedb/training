## Summary

Collected artifacts are currently grading transport only: `collect()` archives
`/logs/artifacts/` plus declared paths and stores the tar bytes on
`trace.state.artifacts`, which is `exclude=True` from serialization. So for every
Harbor rollout the archives cost host RAM for the life of the trace and then
vanish — shared and single-agent runs pay the archive/memory cost for bytes
nothing consumes, and no run can be inspected afterwards. This is the follow-up
scope of #2189.

This adds an optional **artifact sink**: a place collected archives go instead of
`trace.state`. With a sink configured, each archive is streamed to durable storage
as it is collected and the in-memory value becomes a small, self-describing
`ArtifactRef`; `restore()` fetches referenced archives back on demand,
digest-verified, just before extraction. **With no sink configured the behavior is
byte-for-byte unchanged.**

## What changed

- **`verifiers/v1/utils/artifact_sink.py`** (new)
  - `ArtifactRef` — pydantic locator (`sink` spec, `key`, `source`, `size`, `sha256`).
  - `ArtifactSink` — ABC with `async put(trace_id, source, data) -> ArtifactRef | None`
    and `async get(ref) -> bytes`.
  - `DirectorySink` — archives + a per-trace `manifest.json` under a host directory
    (`<root>/<trace_id>/NNNN-<source>.tar`), mirroring Harbor's durable trial
    outputs; atomic tmp+rename writes; `get` verifies size and sha256 and refuses a
    tampered/truncated archive.
  - `load_sink(spec)` — a directory path builds a `DirectorySink`; a
    `pkg.module:factory` / `path.py:factory` (optional `?args`) plugs a custom sink,
    using the same grammar as task-fn plugins.
- **`collect()`** gains keyword-only `sink` / `trace_id`. With a sink, each archive
  is `put()` to it (shielded via `run_shielded`, like episode output, so a cancelled
  rollout can't leave a torn archive behind a manifest entry) and the dict value is
  its `ArtifactRef`; absent optional sources are recorded in the manifest and stay
  `None`. The 32 MiB collection budget and every existing rule are untouched.
- **`restore()`** dereferences `ArtifactRef`s (`load_sink(ref.sink).get(ref)`) before
  extraction; raw-bytes values still work exactly as before. Still refuses the
  subprocess runtime; still clears roots first.
- **`State.artifacts`** widens to `dict[str, bytes | ArtifactRef | None]`.
- **`TaskConfig.artifact_sink: str | None`** — the only new knob
  (`--env.taskset.task.artifact-sink <dir-or-spec>`).
- **`docs/v1/harbor.md`** documents the option.

Refs are self-describing (they carry the spec that reopens their sink), so the
existing grading topologies — isolated agentic judge, separate Harbor verifier —
keep working with no sink plumbing: they read `solution.state.artifacts` and pass
it to `restore()` exactly as today.

## Addresses #2189

- Host-side artifact sink with a manifest and per-source status ✔ (`DirectorySink` +
  `manifest.json`, `status: collected|missing`)
- Streaming/downloading to storage rather than retaining every payload in
  `Trace.state` ✔
- Reusing collected outputs as the source for isolated-verifier restoration ✔
  (refs restore transparently)
- Artifacts inspectable after an eval without live trace state ✔ (manifest + tars on
  disk)

Not claimed here (left for follow-ups the issue also lists): skipping redundant
grader upload for shared/single-agent runs, Harbor `destination`, separate
archival-vs-transport size limits.

## Measurements

Real `collect()` (real tar, host runtime) over 48 synthetic rollout workspaces,
1–28 MB each (~420 MB total), local disk, cold restore. `state` is today's
behavior; `dir` is `DirectorySink`; `lance` is an out-of-tree `ArtifactSink` on a
Lance blob table (not in this PR — shown to demonstrate the plugin surface).

| mode | RAM retained | peak RSS | collect 48 (s) | restore 1 archive p50 | manifest query | survives run |
|---|---|---|---|---|---|---|
| state (today) | **419.9 MB** | 410 MB | 2.88 | 0 ms | 0.05 ms¹ | ❌ |
| dir (this PR) | **0 MB** | 39 MB | 3.71 | 12.0 ms | 36.9 ms² | ✔ |
| lance (plugin) | **0 MB** | 305 MB³ | 4.27 | 36.8 ms | 22.8 ms | ✔ |

¹ `state`'s "query" only reads the dict that is the 420 MB it holds in RAM — it has
nothing to query once the run ends. ² directory manifest walk across all traces.
³ Lance's default metadata/index caches; tunable, and the point of that column is
the RAM-retention and survives-run columns, not peak RSS.

The headline: the sink removes the whole in-memory artifact payload (420 MB → 0)
and makes it inspectable after the run, at a per-archive restore cost in the tens of
milliseconds.

## Testing

- `tests/v1/test_artifact_sink.py` (new, 6 tests): default-path bytes unchanged;
  collect-into-`DirectorySink` with manifest/status assertions; `restore`
  dereferences refs (checked against a recording box, since `restore` refuses the
  host runtime by design); each distinct sink built once across a multi-artifact
  restore; tamper detection; `load_sink` for both directory and plugged-factory
  specs.
- `uv run pytest tests/v1` → 76 passed, 75 skipped (was 70+75; +6, no regressions;
  skips all need `PRIME_API_KEY`).
- `uv run pytest tests/ --ignore=tests/v1 -m "not prime"` → no new failures
  (the pre-existing `test_renderer_*` failures reproduce on the base commit,
  unrelated to this change).
- `uv run ruff check` and `ruff format --check` clean on all touched files.
- No live eval conducted; no new dependency added to `pyproject.toml`.

On `AGENTS.md`'s "prefer e2e over new unit modules": the sink's digest-verification
and spec-parsing logic isn't reachable through the `PRIME_API_KEY`-gated e2e path
(which fork CI skips), so I added a focused module rather than ship the security-
relevant tamper check untested. Happy to fold the collect→restore round-trip into
`test_e2e.py` and drop the rest if you'd prefer.
