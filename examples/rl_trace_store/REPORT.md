# Lance as the rollout & trace store for RL post-training

**Research report — August 2026.** Where the rollout/trace data plane hurts in
today's RL libraries, verified in code; why Lance maps onto it; measurements
from a reproducible benchmark; and ranked integration proposals.

> Companion code: this folder. `./run_all.sh` reproduces every number quoted here.

---

## 1. TL;DR

<!-- TLDR: fill after agent receipts -->

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

<!-- AGENT A findings with receipts -->

## 4. The wider landscape

<!-- AGENT B table + highlights -->

## 5. Why Lance maps onto this

<!-- Capability → requirement matrix, with AGENT C receipts; honest limits -->

## 6. Measurements

<!-- benchmark + demo, from README, plus methodology notes -->

## 7. Integration proposals, ranked

<!-- concrete touchpoints with file:line receipts -->

## 8. Positioning plan

<!-- lead, content, sequencing -->
