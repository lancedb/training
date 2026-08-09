# Lance as the rollout & trace store for RL post-training

**Research report — August 9, 2026.** Where the rollout/trace data plane hurts in
today's RL libraries, verified in code with receipts; why Lance maps onto it;
measurements from a reproducible benchmark; and ranked integration proposals.

> Companion code: this repo. `./run_all.sh` reproduces every number quoted here.
> Receipts are `repo @ short-sha path:lines`; all repos were read at HEAD on 2026-08-09.

---

## 1. TL;DR

**The tip we chased is real, current, and documented in their own issue tracker.**
In Prime Intellect's `verifiers`, sandbox-graded rollouts tar the agent's outputs
(capped by `MAX_ARTIFACT_BYTES = 32 MiB`), hold the bytes in **host RAM only**
(`trace.state.artifacts`, explicitly excluded from serialization), and replay
them into a second grading sandbox. Open issue
[verifiers#2189](https://github.com/PrimeIntellect-ai/verifiers/issues/2189) asks
for exactly the missing component: a *"host-side artifact sink with manifest"*,
*"streaming … to disk instead of retaining payloads in `Trace.state`"*,
*"inspectable after eval."* That component is a Lance blob table —
`artifact_sink_poc.py` here implements it in ~60 lines (18 ms lazy restore of
one artifact, durable manifest).

**It is not one library's quirk — it is the industry pattern.** Across 12
RL/post-training codebases we surveyed (verl, OpenRLHF, TRL, SkyRL, ART, slime,
agent-lightning, AReaL, ROLL, NeMo-RL, open-instruct, tinker-cookbook),
rollouts live in transient RAM buffers and die at step end; trace logging is
ad-hoc JSONL / `.pt` / pickle; and the asks for persistence are already filed
(verl RFC #2539 "persistable replay buffer" — closed *not planned*;
agent-lightning's sqlite backend is literally `# TODO: Implement this`). One
participant already validated the columnar answer: ART migrated trajectory
storage JSONL→Parquet for *"~25x compression and ~20x faster queries."* Lance
is the next step on that same road: Parquet-class scans **plus** memory-class
random access, appends, versions, and blobs.

**Measured** (256 rollouts / 1.16 GB synthetic agentic traces, this repo):
Lance is the only store fast on all three RL access patterns at once —
**9.7 ms** cold fetch of one full trace (16× parquet), **6 ms** warm scalar
sweep (60× pickle), **34 ms** trainer batch of token columns (24× pickle) —
with zero rollouts held in host RAM (status quo holds all of them; at 2,048
concurrent 32 MB rollouts that's 64 GB of buffers). A live 3-writer
producer→verifier→trainer demo moves 581 MB through one dataset in 8 s with
5.4 ms/rollout verifier fetches.

**LanceDB already owns a seed asset**: `lance-context` ships a first-party
`RolloutStore` ("RolloutDB") with an RL-native schema (tokens, logprobs,
ref_logprobs, loss_mask, advantage, policy_version, blob-v2 payloads) and an
HTTP server. The play is not to invent a store — it's to **wire that store into
the seams the frameworks already expose** and lead with the verified pain.

**Recommended lead** (detail in §7–8): ship the verifiers artifact sink
(direct, in-demand, small), the ART/TRL writer swaps (mechanical, visible),
and the benchmark blog built from this repo; hold verl's replay-buffer tier as
the flagship follow-up.

## 2. The pattern: agentic RL has a data plane, and it's ad-hoc

Every post-training stack has the same three parties moving the same object
around — the **rollout** (multi-turn trace: messages, tool/execution outputs,
token ids, logprobs, rewards):

```
                 ┌────────────────┐
   sandboxed     │   generators   │  inference engine + env sandboxes
   execution ───►│ (rollout prod.)│
                 └───────┬────────┘
                         │  full trace (1–32 MB each)
                         ▼
                 ┌────────────────┐      ┌──────────────────┐
                 │  host / relay  │─────►│    verifiers     │  rubrics, judges,
                 │  (RAM, files)  │◄─────│ (eval sandboxes) │  test runners
                 └───────┬────────┘      └──────────────────┘
                         │  tokens + logprobs + advantages
                         ▼
                 ┌────────────────┐
                 │    trainer     │  FSDP/Megatron ranks
                 └────────────────┘
```

Three access patterns fall out of that shape, and they are the benchmark's
P1/P2/P3:

- **P1 — verifier fetch:** random access to ONE rollout's full trace.
- **P2 — sweep:** scalar columns (reward/verdict/step) over ALL rollouts.
- **P3 — trainer batch:** token/logprob columns for a sampled batch.

Today the plane is built from host RAM dicts, pickle/`.pt` files on shared
disks, JSON dumps, and object-store round trips — each fast for exactly one
pattern and pathological for the others (§6).

## 3. Ground truth: the Prime Intellect stack

We read the four repos end to end (`verifiers @ a298bcf`, `prime-rl @ 6e33f3f`,
`prime-environments @ cf220ee`, `prime` CLI/sandboxes monorepo `@ b51e353`).
The reported "sandbox → host → sandbox, 32 MB in memory" flow is **real, current,
and has an open issue asking for exactly the component Lance provides.**

### 3.1 The verified 32 MB artifact flow

For sandbox-graded tasks (Harbor tasksets with `[verifier].environment_mode =
"separate"`), a rollout ends like this:

1. **Agent sandbox → host.** The agent's outputs are tarred *inside its
   sandbox* and pulled to the host: `verifiers/v1/utils/artifacts.py:35-103`.
   The cap is `MAX_ARTIFACT_BYTES = 32 * 1024 * 1024`
   (`artifacts.py:21-23`), documented as *"Ceiling per collection. Sized for a
   delta, not a tree"*, and PR #2144 leaves it explicitly open: *"Open:
   `MAX_ARTIFACT_BYTES` is 32 MB on the same-image/delta-only assumption."*
2. **Host RAM, and only host RAM.** The tar bytes live at
   `trace.state.artifacts: dict[str, bytes | None]` (`verifiers/v1/state.py:12`),
   and `Trace.state` is `Field(..., exclude=True)` — *excluded from
   serialization* (`verifiers/v1/trace.py:362-363`). Artifacts are never
   persisted anywhere; they exist only while the rollout object is alive.
3. **Host → grading sandbox.** `finalize` boots a fresh verifier box,
   `await restore(box, solution.state.artifacts)` unpacks the tar, runs
   `bash /tests/test.sh`, reads back `reward.json` (≤1 MB)
   (`verifiers/v1/tasksets/harbor/env.py:70-124`, `taskset.py:49`). The agentic
   judge does the same (`verifiers/v1/envs/agentic_judge/env.py:173-177`).

So the anecdote decodes precisely: **32 MB is the per-rollout artifact cap, not
the typical size**, and the RAM residency isn't a fast-transfer optimization so
much as the absence of any store — there is nowhere else for the bytes to go.

### 3.2 They have already asked for the fix

[verifiers issue #2189](https://github.com/PrimeIntellect-ai/verifiers/issues/2189)
(*"v1: model Harbor artifacts as durable outputs, not only grading transport"*,
open) states the pain and the wanted component verbatim:

> "shared and single-agent Harbor rollouts pay archive/network/memory costs for
> bytes that are then unused" … "the 32 MB transport cap constrains the broader
> artifact surface"

and scopes the follow-up as:

> "Host-side artifact sink with manifest and per-source status" … "Streaming/
> downloading artifacts to disk instead of retaining payloads in `Trace.state`"
> … "Artifacts inspectable after eval without live trace state dependency."

That is a Lance blob table, feature for feature — see `artifact_sink_poc.py`
in this repo (~60 lines: `(trace_id, source)` manifest + Blob v2 column;
collect-on-completion; ~20 ms lazy `restore()` of one artifact; manifest still
queryable after the run). Related fragility: issue #2195 (isolated judges
"reach the judge sandbox empty-handed" when trace-state transport isn't wired).

### 3.3 Where rollouts already touch columnar storage

The rest of the stack strengthens the case — Prime already treats rollout
tables as an object-storage artifact, just without random access:

- **Platform uploads are per-step Parquet → R2.** prime-rl's `PrimeMonitor`
  converts rollouts to a fixed pyarrow `_SAMPLE_SCHEMA` (trajectory/completion
  as JSON strings) and `pq.write_table(..., compression="snappy")` per step,
  uploaded via presigned URL (`src/prime_rl/utils/monitor/prime.py:29-51,
  295-390`). The Environments Hub trace viewer then browses these samples —
  a whole-object-fetch pattern that Lance `take()` turns into row-level reads.
  Note `pyarrow>=21.0.0` is already a top-level prime-rl dependency.
- **Training traces are append-only JSONL per step** —
  `outputs/rollouts/step_N/{train,eval}/{all,effective}/traces.jsonl`
  (`orchestrator/orchestrator.py:556-563,654-659`, `utils/pathing.py:76-80`).
  Nothing can query them without rescanning files; their own release notes
  (v0.2.0) record streaming/memory pain with JSONL traces on "long or highly
  concurrent runs".
- **INTELLECT-2 exchanged rollouts as Parquet in object storage** between
  inference workers, TOPLOC validators, and the trainer: *"rollout data is
  exchanged between inference workers and the trainer using Parquet files"*
  (§2.1.1, [arXiv:2505.07291](https://arxiv.org/abs/2505.07291)) — validators
  literally "watch step folders" for uploaded rollout files. Decentralized RL
  made object-store rollout tables the *verification substrate*.
- **The hot path is ZMQ and should stay ZMQ.** Env server → orchestrator →
  trainer runs ZMQ+msgpack with in-memory buffers (`transport/zmq.py`,
  `train_sink.py:78-95`); the opt-in filesystem transport writes
  write-once/read-once msgpack `.bin` step files (`transport/filesystem.py:29-62`).
  We are *not* proposing Lance replace the in-flight hop (§7's anti-fit notes).

## 4. The wider landscape: 12 libraries, one missing tier

Full per-library receipts (file:line + issue links) are in the appendix of the
research notes; the pattern summary:

| Library (stars) | Rollout unit | Gen→train transport | Persistence after step | Trace logging |
|---|---|---|---|---|
| verl (22.9k) | `DataProto` / `AgentLoopOutput` | Ray object store (torch.save pickles) | none (async queue **drops when full**) | per-step JSONL (decoded text), wandb tables |
| OpenRLHF (9.9k) | `Experience` (~6 fp32 tensors/token) | Ray remote calls | `NaiveReplayBuffer` in CPU RAM, `.clear()` each step | 1 sample/step to wandb |
| TRL (19.0k) | batch dicts / `RolloutSample` | vLLM server: **JSON over HTTP** (logprobs as JSON floats) | none | **parquet per step** + HF Hub sync |
| SkyRL (2.1k) | `GeneratorOutput` | HTTP, orjson + base64 tensors | asyncio queue (staleness-bounded) | wandb-only table; eval JSONL |
| ART (10.6k) | `art.Trajectory` (full chat JSON) | JSON over HTTP | **parquet per step, versioned schema** | the parquet is the log |
| slime (7.8k) | `Sample` | `ray.put` / NIXL | debug: **one `.pt` per rollout** | wandb + custom hook |
| agent-lightning (17.5k) | OTel spans → `Triplet` | OTLP/HTTP to `LightningStore` | in-memory store **evicts spans at 70% RAM**; Mongo the only durable backend; `sqlite.py` = "TODO" | spans |
| AReaL (5.7k) | episode dicts | HTTP + queues | none | per-task JSONL in version dirs |
| ROLL (3.4k) | DataProto fork | Ray / TransferQueue | transient KV | wandb |
| NeMo-RL (1.9k) | `BatchedDataDict` | **TransferQueue data plane** (ZMQ/Mooncake) | *"Storage is transient … `kv_clear` drops it at step end"* | env-gated dumps |
| open-instruct (3.8k) | `GenerationResult` | Ray queues | none | wandb tables |
| tinker-cookbook | `Trajectory` (client-side) | HTTPS API | **JSONL per iteration** via `Storage` protocol | logtree HTML |

Six observations that repeat:

1. **Rollouts die with the step.** OpenRLHF clears its buffer every update;
   NeMo-RL's data plane drops keys at step end by design; verl's async queue
   silently discards samples when full. Nobody can answer *"what did my agent
   do at step 400?"* after a run.
2. **The ask is already filed.** verl RFC
   [#2539](https://github.com/verl-project/verl/issues/2539) — *"when rollout
   data becomes too large to fit in memory, out-of-memory (OOM) issues arise …
   When a trial fails, all the history rollout data is lost"* — proposed a
   persistable replay buffer and was **closed not planned**. OpenRLHF
   [#1065](https://github.com/OpenRLHF/OpenRLHF/issues/1065) tracks unresolved
   CPU-RAM growth; TRL [#3039](https://github.com/huggingface/trl/issues/3039)
   ends in a 978 GB anon-rss OOM kill, closed not planned.
3. **Transport hurts at agentic/multimodal scale.** verl RFC
   [#2847](https://github.com/volcengine/verl/issues/2847): Ray object-store
   relay = *"CPU memory bloat (2× copies), latency spikes, GPU starvation"*
   (measured 6.85 s to move 0.21 GB). TRL ships per-token logprobs as JSON;
   SkyRL base64-packs tensors into JSON.
4. **Trace logging is uniformly ad-hoc** — decoded-text JSONL (verl), `.pt`
   per rollout (slime), pickle (SkyRL), single-row wandb tables with the
   wandb#2981 full-re-upload workaround (verl, OpenRLHF, TRL, SkyRL). slime
   [#1519](https://github.com/THUDM/slime/issues/1519) shows users hand-rolling
   Streamlit viewers and "coverage gates" over misaligned dump shards.
5. **The seams for a storage backend already exist.** agent-lightning's
   `LightningStore` ABC, NeMo-RL's `DataPlaneClient` ABC ("future:
   nv-dataplane"), ROLL's transfer-backend registry, tinker-cookbook's
   `Storage` protocol, slime's `--custom-rollout-log-function-path`. The
   industry built the sockets; nobody shipped the durable plug.
6. **A participant already proved the columnar direction.** ART's own CLI:
   *"converts old .jsonl trajectory files to the new .parquet format, which
   provides ~25x compression and ~20x faster queries."* And their RULER
   judge re-reads stored trajectory groups — a re-query workload begging for
   random access.

## 5. Why Lance maps onto this (and where it doesn't)

Requirements extracted from §3–4, against verified Lance capabilities
(pylance 10.0.0, format 2.1/2.2; receipts in the lance docs/source):

| RL data-plane requirement | Lance capability (verified) |
|---|---|
| Fetch ONE trace out of millions, fast, from persisted storage | `take(indices, columns=...)` — no row groups; structural encodings; ~2,000× parquet in the canonical random-access benchmark; 9.7 ms cold on our 1.16 GB set |
| Sweep scalars (reward/verdict/step) across everything | Columnar scans + predicate pushdown + late materialization (filter on scalars, fetch heavy cols only for hits) |
| 1–32 MB artifacts/logs per rollout | **Blob v2** (`lance.blob_field`, format 2.2): out-of-line payloads, dedicated files >2 MiB, lazy file-like `take_blobs` (we un-tar straight from the handle), scans skip payload bytes |
| Many concurrent producers (env workers) | Optimistic concurrency; **append⊥append never conflicts** — verified 8 procs × 5 appends, 40/40 commits; MemWAL exists (experimental) for very high commit rates |
| Verifier/trainer see new rollouts promptly | MVCC versions; readers poll latest version (our demo: 3 writers + 2 readers, live) |
| Crash-safe, resumable, auditable runs | Every append is a version; tags (`ckpt-100-data`), branches, time travel, `commit_message` audit trail |
| Trainer reads tokens/logprobs only | Column projection (`take`/scans never touch blobs); `lance.torch` `LanceDataset` + sharded samplers; lance-ray for distributed |
| Curate RL→SFT datasets from traces | Filtered scans, zero-copy column adds (`add_columns`/`merge`), Geneva backfills, DuckDB/Polars SQL on the same table |
| Local NVMe now, object storage at scale | Same API over file/S3/GCS/Azure (+S3 Express); conditional-PUT commits; AIMD rate limiting |

**Honest limits** (all documented, details in lance perf docs):
- **Not the intra-step hot path.** ZMQ/TransferQueue/NIXL move padded tensors
  GPU↔GPU in milliseconds and delete them; a storage format adds nothing
  there. Lance's lane is every hop that wants the data to *outlive* the step.
- Many tiny commits → fragment/manifest sprawl; batch appends (per env-worker
  chunk, as our demo does) and run `compact_files`. Schema changes conflict
  with concurrent writes.
- Object-store point reads are tens-of-ms, not NVMe's 2–10 ms (S3 gives no
  benefit below ~100 KB reads); Enterprise NVMe caching is the mitigation.
- Row updates are copy-on-write (blob columns exist precisely to keep heavy
  bytes out of rewrites); versions accumulate until `cleanup_old_versions`.
- Blob v2 requires `data_storage_version="2.2"` (default is 2.1); measured
  trade-off in §6. Not fork-safe — spawn workers (all our code does).
- `lance-context`'s server layer is young (0.6.x), and blob v2 in the
  `lancedb` DB layer is still on the beta channel — the format itself (what we
  benchmarked) is GA.

## 6. Measurements

Setup: 256 synthetic agentic rollouts (16 steps × 16), **1.16 GB** total —
multi-turn tool-call chat JSON, 512–32k int32 token ids + float32 logprobs,
64 KB–32 MB log-like execution blobs (5% in the 24–32 MB class, matching the
verifiers artifact cap). Local NVMe, 4 vCPU, 15 GB RAM; cold = page cache
dropped; each backend isolated in its own process. `tracegen.py` is seeded —
byte-identical data across backends.

| backend | on disk | write MB/s | RAM held | P1 fetch 1 trace cold p50 | P2 sweep cold/warm | P3 batch×64 cold |
|---|---|---|---|---|---|---|
| host memory (status quo) | — | — | **1,161 MB** | 0 ms | ~0 ms | 0.1 ms |
| pickle / `.pt` per rollout | 1,157 MB | 140 | ~0 | 2.5 ms | 1,017 / 357 ms | 801 ms |
| JSON per rollout | 1,564 MB | 49 | ~0 | 18.7 ms | 7,414 / 4,373 ms | 3,742 ms |
| parquet per step | 631 MB | 98 | ~0 | 156.0 ms | **19 / 7 ms** | 74 ms |
| **Lance (Blob v2)** | 1,139 MB | 83 | ~0 | **9.7 ms** | 262 / **6 ms** | **34 ms** |

Readings:

- **Memory "wins" by holding everything** — that's the 32 MB × concurrency wall
  (2,048 concurrent rollouts ⇒ 64 GB of buffers).
- **Pickle/`.pt` has no columns**: reading rewards (4 bytes/row) or token
  batches deserializes every blob byte — 1.0 s and 0.8 s respectively.
- **Parquet has no rows**: one trace costs a row-group decode — 156 ms and
  ~hundreds of MB touched to return 4 MB.
- **Lance is fast on all three at once**, holding nothing in RAM. Encoding
  trade-off, measured: legacy inline blobs (2.1) write 183 MB/s with 54 ms cold
  sweeps; Blob v2 writes 83 MB/s with 262 ms cold sweeps but buys lazy
  file-handles, blob-skipping scans, and update-safety. Both dominate the
  baselines on P1+P3.

**Live pipeline** (`demo_pipeline.py`, 144 rollouts / 581 MB): 3 producer
processes append concurrently (0 conflicts, ~290 ms p50 per 8-rollout commit),
a verifier random-accesses every new trace (5.4 ms p50 / 60 ms p95 per
rollout, peak RSS 468 MB while touching 565 MB), a trainer consumes 410k
tokens from `(completion_ids, logprobs)` columns only (3.1 ms per batch, never
reads a blob) — 8.0 s wall, 19 recoverable versions, then a zero-copy filter
exports 66 high-reward rollouts for SFT.

**Artifact sink POC** (`artifact_sink_poc.py`): the #2189-shaped sink — 24
artifacts collected, one restored lazily into a tarfile reader in ~20 ms,
manifest (`size/sha256/status`) queryable after the run.

## 7. Integration proposals, ranked

Each has a named seam, verified at HEAD 2026-08-09. Effort ≈ PR size for a
working first cut. The vehicle for 2–6 should be **`lance-context`'s
`RolloutStore`** (LanceDB's existing first-party RL store — schema already has
tokens/logprobs/ref_logprobs/loss_mask/advantage/policy_version + blob-v2
payloads + HTTP server) plus a thin per-framework adapter, not new bespoke
stores.

| # | Target | Seam (receipt) | What ships | Effort | Why it wins |
|---|---|---|---|---|---|
| 1 | **verifiers artifact sink** | issue #2189; `v1/utils/artifacts.py:21-103`, `v1/state.py:12` | `LanceArtifactSink` behind their collect/restore, POC in this repo | S | They asked for it, in writing; fixes the literal 32 MB-in-RAM anecdote; opens the Prime relationship |
| 2 | **ART trajectory store** | `src/art/utils/trajectory_logging.py` (`write_trajectory_groups_parquet`, versioned schema) | Lance writer beside parquet; RULER re-scoring + benchmarking get random access | S | They already proved columnar (25×/20× claim); pyarrow tables in hand; agentic-RL flagship community |
| 3 | **agent-lightning `LanceLightningStore`** | `store/base.py:104` ABC; `store/sqlite.py` = "TODO: Implement this"; RAM eviction at 70% | The embedded persistent backend between InMemory and Mongo | M | Empty slot + 17.5k stars + Microsoft brand; spans (append-heavy, big text, query by rollout_id/time) are a Lance-shaped workload |
| 4 | **TRL completions log** | `grpo_trainer.py:3341-3357` (parquet per step + HF Hub `CommitScheduler`) | `log_completions="lance"` → one appendable dataset instead of file-per-step | S | Most-visible surface in the HF ecosystem; mechanical |
| 5 | **verl rollout persistence** | `ray_trainer.py:514-546` (`_log_rollout_data` JSONL, decoded text only); RFC #2539 (closed unplanned); v1 `ReplayBuffer`/TransferQueue seam | (i) JSONL→Lance keeping token ids/logprobs; (ii) flagship: Lance spill/persistence tier under the v1 replay buffer | M→L | 22.9k-star flagship; (ii) is the unmet RFC and also lands in ROLL/NeMo-RL via the shared TransferQueue seam |
| 6 | **prime-rl platform store** | `utils/monitor/prime.py:29-51,295-390` (per-step Parquet→R2); per-step `traces.jsonl` | Lance dataset per run: hub viewer gets row-level `take`, filters (`reward < 0`, `env = x`), later vector search over traces | M | Platform-scale story; INTELLECT-2 already ran rollouts-as-files-in-object-storage; pyarrow already a dep |

Also cheap and worth doing opportunistically: slime's
`--custom-rollout-log-function-path` hook (zero-fork Lance logger, directly
answers their #1519 analysis ask) and a `Storage`-protocol Lance writer for
tinker-cookbook.

**Anti-fit, stated up front in any pitch:** we do not touch ZMQ/TransferQueue/
NIXL in-flight hops, NCCL weight sync, or KV-cache transfer. Lance is the tier
those systems deliberately don't have: durable, queryable, memory-like random
access *across* steps and *after* runs.

## 8. Positioning plan

**The lead is the story this repo proves:** *"Agentic RL rollouts are 1–32 MB
multimodal records that today live in host RAM and die at step end — here's
the data-plane tier every framework is missing, receipts included."* Not
"Lance is fast" in the abstract; the pitch is anchored to `MAX_ARTIFACT_BYTES`,
RFC #2539, `# TODO: Implement this`, and ART's own 25× migration.

Sequencing:

1. **Benchmark blog + this repo public** — the P1/P2/P3 framing ("the only
   store fast on all three"), the 64 GB buffer-RAM math, the live demo GIF.
   Cite the existing house proof points (lerobot-lancedb 2–4×, stable-worldmodel
   3–4×, ViT MFU 37% vs 21% parquet) as the training-side track record.
2. **Two small PRs as existence proofs** — #1 verifiers artifact sink and #2
   ART Lance writer (or #4 TRL). Small, wanted, hard to say no to; each makes
   the blog's claims concrete in someone else's repo.
3. **Product story: RolloutDB** — promote `lance-context`'s `RolloutStore` as
   the named thing ("a versioned, columnar store purpose-built for RL rollout
   data, the way a vector DB is built for embeddings"), with per-framework
   adapters from the table above. The RL data flywheel closes the loop:
   rollouts → filter/judge in place → SFT/RM datasets → next run, all one
   table with versions.
4. **Platform conversation with Prime Intellect** — the Environments Hub
   viewer + platform sample store (#6) at hub scale, and the INTELLECT-line
   decentralized story (rollout files in object storage with validators
   watching — already their architecture, minus random access). Warm intro
   exists via the original anecdote.

What we did **not** verify (flag in any external claim): Prime's closed-source
platform internals beyond what prime-rl uploads; production scale numbers for
any lab; S3-backed latencies for our benchmark (local NVMe only — rerun
`bench.py` against an S3 URI before quoting object-store numbers).

---

*Method note: three research passes (Prime stack; 12-library survey; Lance
capability inventory) were run against HEAD clones on 2026-08-09 with
file:line receipts for every claim, then the benchmark/demo in this repo was
built to mirror the verified shapes (32 MB artifact class, JSON-string
trajectories, token/logprob columns). Synthetic traces are seeded and
regenerable; no proprietary data.*
