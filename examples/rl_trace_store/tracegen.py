"""Synthetic agentic-RL rollout generator.

Models the trace an agentic RL environment produces per rollout:
  - a multi-turn chat (assistant tool calls + truncated tool results) as JSON
  - completion token ids + per-token logprobs (what the trainer consumes)
  - the full tool/execution output log as a large binary blob (what the
    verifier consumes) -- this is the field that makes real rollouts MBs big
  - scalar metadata: reward, verdict, env/step ids

Sizes follow a lognormal mix calibrated so the median rollout is ~1-2 MB and
the tail reaches the 32 MB class reported for sandboxed agent rollouts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

VOCAB = 151_936  # Qwen-class vocab
BLOB_MEDIAN = 1.2 * 1024 * 1024
BLOB_SIGMA = 1.1
BLOB_MIN = 64 * 1024
BLOB_MAX = 32 * 1024 * 1024  # the "32 MB rollout" class
HEAVY_FRACTION = 0.05  # ~5% of rollouts are 24-32 MB monsters


def _log_corpus(rng: np.random.Generator, size: int = 4 * 1024 * 1024) -> bytes:
    """A compressible, log-like corpus we slice per rollout (like real stdout)."""
    lines = []
    n = 0
    i = 0
    while n < size:
        line = (
            f"[2026-08-09T12:{i % 60:02d}:{(i * 7) % 60:02d}.{i % 1000:03d}] "
            f"worker-{i % 8} pid={1000 + i % 512} step={i} "
            f"loss={rng.random():.6f} grad_norm={rng.random() * 10:.4f} "
            f"path=/workspace/run/{rng.integers(0, 1 << 32):08x}/out.txt "
            f"status={'OK' if i % 7 else 'RETRY'} took={rng.integers(1, 900)}ms\n"
        )
        lines.append(line.encode())
        n += len(line)
        i += 1
    return b"".join(lines)[:size]


@dataclass
class Rollout:
    rollout_id: str
    step: int
    env_id: str
    example_id: int
    seed: int
    status: str
    reward: float
    reward_components: str  # json
    verdict: bool
    messages: str  # json chat
    completion_ids: np.ndarray  # int32
    logprobs: np.ndarray  # float32
    trace_blob: bytes

    def nbytes(self) -> int:
        return (
            len(self.trace_blob)
            + len(self.messages)
            + self.completion_ids.nbytes
            + self.logprobs.nbytes
            + len(self.reward_components)
            + 64
        )


@dataclass
class TraceGen:
    seed: int = 42
    rng: np.random.Generator = field(init=False)
    corpus: bytes = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.corpus = _log_corpus(np.random.default_rng(self.seed + 1))

    def _blob(self, size: int) -> bytes:
        """~70% sliced log corpus + ~30% random ascii (diffs/base64-ish)."""
        rng = self.rng
        parts = []
        remaining = size
        while remaining > 0:
            chunk = min(remaining, int(rng.integers(64 * 1024, 512 * 1024)))
            if rng.random() < 0.7:
                off = int(rng.integers(0, len(self.corpus) - chunk)) if chunk < len(self.corpus) else 0
                parts.append(self.corpus[off : off + chunk])
            else:
                parts.append(
                    rng.integers(32, 127, size=chunk, dtype=np.uint8).tobytes()
                )
            remaining -= chunk
        return b"".join(parts)[:size]

    def _messages(self, n_turns: int, blob_preview: bytes) -> str:
        msgs = [
            {"role": "system", "content": "You are an autonomous coding agent. Fix the failing tests."},
            {"role": "user", "content": "Repo mounted at /workspace. Make `pytest -q` pass."},
        ]
        preview = blob_preview[:2048].decode(errors="replace")
        for t in range(n_turns):
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"Turn {t}: inspecting failure and patching.",
                    "tool_calls": [
                        {
                            "id": f"call_{t}",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"cmd": f"pytest -q && git diff --stat # t{t}"}),
                            },
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{t}",
                    "content": f"(truncated; full log in trace blob)\n{preview}",
                }
            )
        return json.dumps(msgs)

    def make(self, step: int, idx: int) -> Rollout:
        rng = self.rng
        if rng.random() < HEAVY_FRACTION:
            blob_size = int(rng.integers(24 * 1024 * 1024, BLOB_MAX))
        else:
            blob_size = int(
                np.clip(rng.lognormal(np.log(BLOB_MEDIAN), BLOB_SIGMA), BLOB_MIN, BLOB_MAX)
            )
        n_tokens = int(np.clip(rng.lognormal(np.log(4096), 0.8), 512, 32_768))
        n_turns = int(rng.integers(4, 48))
        blob = self._blob(blob_size)
        reward_components = {
            "tests_passed": float(rng.random()),
            "format": float(rng.random()),
            "no_cheating": float(rng.integers(0, 2)),
        }
        reward = float(np.mean(list(reward_components.values())))
        return Rollout(
            rollout_id=f"ro-{step:04d}-{idx:04d}",
            step=step,
            env_id=f"swe-env-{idx % 8}",
            example_id=int(rng.integers(0, 10_000)),
            seed=int(rng.integers(0, 1 << 31)),
            status="completed",
            reward=reward,
            reward_components=json.dumps(reward_components),
            verdict=bool(reward > 0.5),
            messages=self._messages(n_turns, blob),
            completion_ids=rng.integers(0, VOCAB, size=n_tokens, dtype=np.int32),
            logprobs=(-rng.exponential(0.8, size=n_tokens)).astype(np.float32),
            trace_blob=blob,
        )

    def make_step(self, step: int, n: int) -> list[Rollout]:
        return [self.make(step, i) for i in range(n)]
