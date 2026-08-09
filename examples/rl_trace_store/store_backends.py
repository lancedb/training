"""Storage backends for the rollout-store benchmark.

Each backend implements the same contract so the benchmark exercises the
three access patterns every RL pipeline has:

  P1 verifier fetch   : full trace (messages + blob) for K random rollouts
  P2 reward sweep     : scalar columns (step, reward, verdict) for ALL rollouts
  P3 trainer batch    : token ids + logprobs for a random batch of rollouts

Backends:
  memory   -- python dict (what "keep rollouts in host RAM" looks like)
  pickle   -- one .pkl file per rollout (ad-hoc file relay; .pt is pickle too)
  json     -- one .json per rollout, blob base64-encoded (ad-hoc trace dumps)
  parquet  -- one parquet file per step (columnar baseline, default settings)
  lance    -- single Lance dataset, appended per step, blob-encoded trace
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from tracegen import Rollout

try:
    import lance
except ImportError:  # pragma: no cover
    lance = None


def du(path: str) -> int:
    out = subprocess.run(["du", "-sb", path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) if out.returncode == 0 else 0


ARROW_SCHEMA = pa.schema(
    [
        pa.field("rollout_id", pa.string()),
        pa.field("step", pa.int32()),
        pa.field("env_id", pa.string()),
        pa.field("example_id", pa.int32()),
        pa.field("seed", pa.int64()),
        pa.field("status", pa.string()),
        pa.field("reward", pa.float32()),
        pa.field("reward_components", pa.string()),
        pa.field("verdict", pa.bool_()),
        pa.field("messages", pa.large_string()),
        pa.field("completion_ids", pa.list_(pa.int32())),
        pa.field("logprobs", pa.list_(pa.float32())),
        pa.field("trace_blob", pa.large_binary()),
    ]
)

# Same schema, but the blob column is stored with Lance's blob encoding:
# values live out-of-line and reads are lazy file-like handles (take_blobs).
LANCE_SCHEMA = pa.schema(
    [
        f if f.name != "trace_blob"
        else pa.field("trace_blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"})
        for f in ARROW_SCHEMA
    ]
)


def rollouts_to_table(rollouts: list[Rollout], schema: pa.Schema) -> pa.Table:
    return pa.table(
        {
            "rollout_id": [r.rollout_id for r in rollouts],
            "step": pa.array([r.step for r in rollouts], pa.int32()),
            "env_id": [r.env_id for r in rollouts],
            "example_id": pa.array([r.example_id for r in rollouts], pa.int32()),
            "seed": pa.array([r.seed for r in rollouts], pa.int64()),
            "status": [r.status for r in rollouts],
            "reward": pa.array([r.reward for r in rollouts], pa.float32()),
            "reward_components": [r.reward_components for r in rollouts],
            "verdict": pa.array([r.verdict for r in rollouts], pa.bool_()),
            "messages": pa.array([r.messages for r in rollouts], pa.large_string()),
            "completion_ids": [r.completion_ids for r in rollouts],
            "logprobs": [r.logprobs for r in rollouts],
            "trace_blob": pa.array([r.trace_blob for r in rollouts], pa.large_binary()),
        },
        schema=schema,
    )


@dataclass
class Ref:
    """Stable address of a rollout: (step, index-within-step, global row)."""

    step: int
    idx: int
    row: int


class Backend:
    name: str = "base"
    persistent: bool = True

    def __init__(self, root: str, per_step: int):
        self.root = root
        self.per_step = per_step
        os.makedirs(root, exist_ok=True)

    def write_step(self, step: int, rollouts: list[Rollout]) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        pass

    def bytes_on_disk(self) -> int:
        return du(self.root)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    # -- read patterns ------------------------------------------------------
    def read_full(self, refs: list[Ref]) -> int:
        """P1: messages + full blob for each ref; returns bytes touched."""
        raise NotImplementedError

    def read_scalars_all(self, n_rows: int) -> float:
        """P2: (step, reward, verdict) over everything; returns mean reward."""
        raise NotImplementedError

    def read_training_batch(self, refs: list[Ref]) -> int:
        """P3: completion_ids + logprobs for refs; returns total tokens."""
        raise NotImplementedError


class MemoryBackend(Backend):
    """Rollouts held in host RAM -- the pattern we are replacing."""

    name = "memory"
    persistent = False

    def __init__(self, root: str, per_step: int):
        super().__init__(root, per_step)
        self.store: dict[int, Rollout] = {}

    def write_step(self, step: int, rollouts: list[Rollout]) -> None:
        for i, r in enumerate(rollouts):
            self.store[step * self.per_step + i] = r

    def bytes_on_disk(self) -> int:
        return 0

    def read_full(self, refs: list[Ref]) -> int:
        n = 0
        for ref in refs:
            r = self.store[ref.row]
            n += len(r.messages) + len(r.trace_blob)
        return n

    def read_scalars_all(self, n_rows: int) -> float:
        return float(np.mean([self.store[i].reward for i in range(n_rows)]))

    def read_training_batch(self, refs: list[Ref]) -> int:
        return int(sum(len(self.store[ref.row].completion_ids) for ref in refs))


class PickleBackend(Backend):
    """One pickle file per rollout (torch.save-style ad-hoc relay)."""

    name = "pickle"

    def _path(self, step: int, idx: int) -> str:
        return os.path.join(self.root, f"step_{step:04d}", f"rollout_{idx:04d}.pkl")

    def write_step(self, step: int, rollouts: list[Rollout]) -> None:
        os.makedirs(os.path.join(self.root, f"step_{step:04d}"), exist_ok=True)
        for i, r in enumerate(rollouts):
            with open(self._path(step, i), "wb") as f:
                pickle.dump(r, f, protocol=5)

    def _load(self, ref: Ref) -> Rollout:
        with open(self._path(ref.step, ref.idx), "rb") as f:
            return pickle.load(f)

    def read_full(self, refs: list[Ref]) -> int:
        n = 0
        for ref in refs:
            r = self._load(ref)
            n += len(r.messages) + len(r.trace_blob)
        return n

    def read_scalars_all(self, n_rows: int) -> float:
        rewards = []
        for step_dir in sorted(os.listdir(self.root)):
            for fn in sorted(os.listdir(os.path.join(self.root, step_dir))):
                with open(os.path.join(self.root, step_dir, fn), "rb") as f:
                    rewards.append(pickle.load(f).reward)  # full deserialize for 4 bytes
        return float(np.mean(rewards))

    def read_training_batch(self, refs: list[Ref]) -> int:
        return int(sum(len(self._load(ref).completion_ids) for ref in refs))


class JsonBackend(PickleBackend):
    """One JSON file per rollout, blob base64-encoded (ad-hoc trace dumps)."""

    name = "json"

    def _path(self, step: int, idx: int) -> str:
        return os.path.join(self.root, f"step_{step:04d}", f"rollout_{idx:04d}.json")

    def write_step(self, step: int, rollouts: list[Rollout]) -> None:
        os.makedirs(os.path.join(self.root, f"step_{step:04d}"), exist_ok=True)
        for i, r in enumerate(rollouts):
            doc = {
                "rollout_id": r.rollout_id, "step": r.step, "env_id": r.env_id,
                "example_id": r.example_id, "seed": r.seed, "status": r.status,
                "reward": r.reward, "reward_components": r.reward_components,
                "verdict": r.verdict, "messages": r.messages,
                "completion_ids": r.completion_ids.tolist(),
                "logprobs": r.logprobs.tolist(),
                "trace_blob": base64.b64encode(r.trace_blob).decode(),
            }
            with open(self._path(step, i), "w") as f:
                json.dump(doc, f)

    def _load(self, ref: Ref):
        with open(self._path(ref.step, ref.idx)) as f:
            d = json.load(f)
        d["trace_blob"] = base64.b64decode(d["trace_blob"])
        d["completion_ids"] = np.asarray(d["completion_ids"], dtype=np.int32)
        return _DictRollout(d)

    def read_scalars_all(self, n_rows: int) -> float:
        rewards = []
        for step_dir in sorted(os.listdir(self.root)):
            for fn in sorted(os.listdir(os.path.join(self.root, step_dir))):
                with open(os.path.join(self.root, step_dir, fn)) as f:
                    rewards.append(json.load(f)["reward"])
        return float(np.mean(rewards))


class _DictRollout:
    def __init__(self, d):
        self.messages = d["messages"]
        self.trace_blob = d["trace_blob"]
        self.completion_ids = d["completion_ids"]
        self.reward = d["reward"]


class ParquetBackend(Backend):
    """One parquet file per training step (default writer settings)."""

    name = "parquet"

    def _path(self, step: int) -> str:
        return os.path.join(self.root, f"step_{step:04d}.parquet")

    def write_step(self, step: int, rollouts: list[Rollout]) -> None:
        pq.write_table(rollouts_to_table(rollouts, ARROW_SCHEMA), self._path(step))

    def read_full(self, refs: list[Ref]) -> int:
        n = 0
        for ref in refs:
            t = pq.ParquetFile(self._path(ref.step)).read(columns=["messages", "trace_blob"])
            row = t.slice(ref.idx, 1)
            n += len(row.column("messages")[0].as_py()) + len(row.column("trace_blob")[0].as_py())
        return n

    def read_scalars_all(self, n_rows: int) -> float:
        ds = pads.dataset(self.root, format="parquet")
        t = ds.to_table(columns=["step", "reward", "verdict"])
        return float(np.mean(t.column("reward").to_numpy()))

    def read_training_batch(self, refs: list[Ref]) -> int:
        total = 0
        by_step: dict[int, list[Ref]] = {}
        for ref in refs:
            by_step.setdefault(ref.step, []).append(ref)
        for step, group in by_step.items():
            t = pq.ParquetFile(self._path(step)).read(columns=["completion_ids", "logprobs"])
            for ref in group:
                total += len(t.column("completion_ids")[ref.idx])
        return int(total)


class LanceBackend(Backend):
    """Single Lance dataset; every step is an append (and a version)."""

    name = "lance"

    def __init__(self, root: str, per_step: int):
        super().__init__(root, per_step)
        self.uri = os.path.join(self.root, "rollouts.lance")
        self.ds = None

    def write_step(self, step: int, rollouts: list[Rollout]) -> None:
        table = rollouts_to_table(rollouts, LANCE_SCHEMA)
        if self.ds is None and not os.path.exists(self.uri):
            self.ds = lance.write_dataset(table, self.uri, schema=LANCE_SCHEMA)
        else:
            self.ds = lance.write_dataset(table, self.uri, mode="append")

    def finalize(self) -> None:
        self.ds = lance.dataset(self.uri)

    def read_full(self, refs: list[Ref]) -> int:
        rows = [ref.row for ref in refs]
        t = self.ds.take(rows, columns=["messages"])
        n = sum(len(v.as_py()) for v in t.column("messages"))
        for blob in self.ds.take_blobs("trace_blob", indices=rows):
            with blob as f:
                n += f.size()
                f.read()  # actually pull the bytes so the comparison is fair
        return int(n)

    def read_scalars_all(self, n_rows: int) -> float:
        t = self.ds.to_table(columns=["step", "reward", "verdict"])
        return float(np.mean(t.column("reward").to_numpy()))

    def read_training_batch(self, refs: list[Ref]) -> int:
        t = self.ds.take([ref.row for ref in refs], columns=["completion_ids", "logprobs"])
        return int(sum(len(v) for v in t.column("completion_ids")))


BACKENDS = {
    b.name: b for b in [MemoryBackend, PickleBackend, JsonBackend, ParquetBackend, LanceBackend]
}
