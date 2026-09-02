#!/usr/bin/env python3
r"""Parse lerobot-train logs correctly.

lerobot abbreviates large counters: `step:10K`, `smpl:2M`. A naive `step:(\d+)` matches the
"10" of "10K" and a `step >= 500` filter then silently drops almost every line -- which is how
a "steady state" data-wait figure ended up being computed from two log lines at step 500-750.
"""
import re, sys, collections

SUF = {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000}


def num(tok):
    m = re.fullmatch(r"([\d.]+)([KMG]?)", tok)
    return int(float(m.group(1)) * SUF[m.group(2)]) if m else None


def parse(path, skip_frac=0.2):
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    rows = []
    for m in re.finditer(r"step:([\d.]+[KMG]?)\s+.*?data_s:([\d.]+)\s+prep_s:([\d.]+)\s+"
                         r"updt_s:([\d.]+)\s+step_s:([\d.]+)\s+smp/s:(\d+)", txt):
        rows.append(dict(step=num(m.group(1)), data_s=float(m.group(2)),
                         prep_s=float(m.group(3)), updt_s=float(m.group(4)),
                         step_s=float(m.group(5)), smps=int(m.group(6))))
    if not rows:
        return None
    rows.sort(key=lambda r: r["step"])
    # Drop a warmup prefix by STEP, not by row count: early steps carry loader spin-up.
    cut = rows[-1]["step"] * skip_frac
    steady = [r for r in rows if r["step"] >= cut] or rows
    agg = lambda k: sum(r[k] for r in steady) / len(steady)
    return dict(n_lines=len(rows), n_steady=len(steady), last_step=rows[-1]["step"],
                warmup_cut=int(cut), smps=agg("smps"), step_s=agg("step_s"),
                data_s=agg("data_s"), updt_s=agg("updt_s"),
                data_wait_pct=100 * agg("data_s") / agg("step_s"))


def power(path):
    try:
        w = collections.defaultdict(list)
        for line in open(path):
            i, p = line.split(","); w[int(i)].append(float(p))
        v = [x for xs in w.values() for x in xs[len(xs) // 4:]]
        return sum(v) / len(v), len(v)
    except Exception:
        return None, 0


if __name__ == "__main__":
    for path in sys.argv[1:]:
        r = parse(path)
        print(f"\n=== {path}")
        if not r:
            print("  no step lines"); continue
        print(f"  {r['n_lines']} log lines, last step {r['last_step']:,}; "
              f"steady = {r['n_steady']} lines from step {r['warmup_cut']:,}")
        print(f"  steady {r['smps']:.0f} samples/s | step {r['step_s']:.3f}s "
              f"(data {r['data_s']:.3f}s, update {r['updt_s']:.3f}s) | "
              f"data wait {r['data_wait_pct']:.1f}%")
