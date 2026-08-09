"""End-to-end pipeline verification — runs offline on CPU in a few minutes.

Exercises every stage against a synthetic corpus and asserts the properties
the README claims:

1. ingest      — synthetic corpus with injected duplicates
2. curate      — EDA + FTS + zero-copy `is_dup` column
3. tokenize    — zero-copy `input_ids` column
4. elastic     — identical global batches at world_size 1 vs 2, filters honored
5. train       — tiny GPT, uninterrupted run
6. resume      — kill at step 12, resume, final state matches the
                 uninterrupted run

Usage
-----
python verify_e2e.py [--workdir ./verify_workdir]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import shutil

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    icon = {"PASS": "+", "FAIL": "x"}[status]
    print(f"  [{icon}] {name}  {detail}")
    results.append((status, name, detail))


def check(name: str, cond: bool, detail: str = "") -> None:
    record(PASS if cond else FAIL, name, detail)


def run_train(argv: list[str]) -> str:
    import train

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train.main(argv)
    out = buf.getvalue()
    print("    " + out.strip().splitlines()[-1])
    return out


def final_val_loss(output: str) -> float:
    m = re.search(r"val_loss=([0-9.]+)", output)
    assert m, f"no final val loss in output:\n{output}"
    return float(m.group(1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default="./verify_workdir")
    args = parser.parse_args(argv)

    shutil.rmtree(args.workdir, ignore_errors=True)
    db = f"{args.workdir}/db"
    common_args = ["--db", db]

    import curate
    import ingest
    import tokenize_data
    from common import TRAIN_FILTER, connect_table

    # ── 1. ingest ─────────────────────────────────────────────────────────
    ingest.main([*common_args, "--source", "synthetic", "--rows", "3000"])
    tbl = connect_table(db, "corpus")
    check("ingest: row count", tbl.count_rows() == 3000)

    # ── 2. curate ─────────────────────────────────────────────────────────
    curate.main(common_args)
    tbl = connect_table(db, "corpus")
    n_dup = tbl.count_rows("is_dup")
    check("curate: is_dup column exists", "is_dup" in tbl.schema.names)
    check("curate: duplicates flagged", n_dup > 0, f"{n_dup} dups")

    # ── 3. tokenize ───────────────────────────────────────────────────────
    tokenize_data.main([*common_args, "--tokenizer", "byte"])
    tbl = connect_table(db, "corpus")
    check("tokenize: input_ids column exists", "input_ids" in tbl.schema.names)
    row = tbl.search().select(["text", "input_ids"]).limit(1).to_list()[0]
    check(
        "tokenize: ids round-trip",
        bytes(row["input_ids"][1:-1]).decode("utf-8") == row["text"],
    )

    # ── 4. elastic determinism across world sizes ─────────────────────────
    from lancedb.streaming import StreamingDataset

    filt = f"NOT is_dup AND score >= 1.0 AND ({TRAIN_FILTER})"
    kwargs = dict(columns=["id"], filter=filt, num_splits=4, shuffle_seed=7)

    def global_batches(world_size: int) -> list[list[int]]:
        iters = [
            iter(
                StreamingDataset(
                    connect_table(db, "corpus"), rank=r, world_size=world_size, **kwargs
                )
            )
            for r in range(world_size)
        ]
        per_rank = 4 // world_size
        batches = []
        while True:
            step = []
            for it in iters:
                for _ in range(per_rank):
                    row = next(it, None)
                    if row is None:
                        return batches
                    step.append(row["id"])
            batches.append(sorted(step))

    b1, b2 = global_batches(1), global_batches(2)
    check("elastic: ws=1 == ws=2 global batches", b1 == b2, f"{len(b1)} steps")
    streamed = {i for step in b1 for i in step}
    expected = {
        r["id"]
        for r in connect_table(db, "corpus")
        .search()
        .select(["id"])
        .where(filt)
        .limit(10**9)
        .to_list()
    }
    dropped = len(expected) - len(streamed)
    check(
        "elastic: filter honored, full coverage",
        streamed <= expected and dropped < 4,
        f"{len(streamed)}/{len(expected)} rows ({dropped} dropped by split rounding)",
    )

    # ── 5. train (uninterrupted baseline) ─────────────────────────────────
    base_args = [
        *common_args,
        "--model",
        "tiny",
        "--seq-len",
        "256",
        "--batch-size",
        "4",
        "--warmup-steps",
        "4",
        "--lr-total-steps",
        "24",
        "--log-every",
        "8",
    ]
    out = run_train(
        [
            *base_args,
            "--steps",
            "24",
            "--ckpt-dir",
            f"{args.workdir}/ckpt_a",
            "--ckpt-every",
            "1000",
        ]
    )
    loss_a = final_val_loss(out)
    check("train: completes", True, f"val_loss={loss_a:.4f}")

    # ── 6. kill + resume == uninterrupted ─────────────────────────────────
    run_train(
        [
            *base_args,
            "--steps",
            "12",
            "--ckpt-dir",
            f"{args.workdir}/ckpt_b",
            "--ckpt-every",
            "12",
        ]
    )
    out = run_train(
        [
            *base_args,
            "--steps",
            "24",
            "--ckpt-dir",
            f"{args.workdir}/ckpt_b",
            "--ckpt-every",
            "1000",
            "--resume",
            "auto",
        ]
    )
    loss_b = final_val_loss(out)
    check(
        "resume: matches uninterrupted run",
        abs(loss_a - loss_b) < 1e-4,
        f"{loss_a:.6f} vs {loss_b:.6f}",
    )

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    n_pass = sum(1 for s, *_ in results if s == PASS)
    n_fail = len(results) - n_pass
    print(
        f"E2E {'COMPLETE' if n_fail == 0 else 'FAILED'}: {n_pass} passed  {n_fail} failed"
    )
    print("=" * 60)
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
