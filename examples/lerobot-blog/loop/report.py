#!/usr/bin/env python
"""Aggregate eval shards into the result table: every arm against the base and the random control,
on all held-out episodes and on the two slices, with paired bootstrap confidence intervals.
"""
import argparse
import glob
import json

import numpy as np


def paired_ci(a: np.ndarray, b: np.ndarray, n=5000, seed=0):
    """95% CI of mean(b - a) over episodes, paired bootstrap."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="out/eval/shard_*.json")
    ap.add_argument("--sets", default="config/loop_sets.json")
    ap.add_argument("--metric", default="err_chunk_mae")
    ap.add_argument("--base", default="base")
    ap.add_argument("--control", default="random")
    ap.add_argument("--out", default="out/eval/report.json")
    a = ap.parse_args()

    merged = {}
    for f in sorted(glob.glob(a.shards)):
        for label, r in json.load(open(f)).items():
            merged.setdefault(label, {"path": r["path"], "per_episode": {}})["per_episode"].update(r["per_episode"])
    sets = json.load(open(a.sets))
    slices = sets["slices"]
    labels = list(merged)
    if a.base not in merged:
        raise SystemExit(f"no '{a.base}' checkpoint in shards: {labels}")

    def vec(label, eps):
        return np.array([merged[label]["per_episode"][str(e)][a.metric] for e in eps])

    report = {"metric": a.metric, "slices": {k: len(v) for k, v in slices.items()}, "rows": []}
    print(f"\nmetric: {a.metric}   (lower is better; deltas are vs {a.base}, 95% paired bootstrap CI)\n")
    hdr = f"{'arm':<16}" + "".join(f"{s:>34}" for s in slices)
    print(hdr); print("-" * len(hdr))
    for label in labels:
        row = {"arm": label}
        line = f"{label:<16}"
        for s, eps in slices.items():
            eps = [e for e in eps if str(e) in merged[label]["per_episode"] and str(e) in merged[a.base]["per_episode"]]
            base, cur = vec(a.base, eps), vec(label, eps)
            lo, hi = paired_ci(base, cur)
            improved = float((cur < base).mean()) if label != a.base else float("nan")
            row[s] = {"mean": round(float(cur.mean()), 5), "delta_vs_base": round(float((cur - base).mean()), 5),
                      "ci95": [round(lo, 5), round(hi, 5)], "frac_improved": round(improved, 3), "n": len(eps)}
            ctrl_label = a.control + label[label.rfind("_s"):] if "_s" in label else a.control
            if ctrl_label in merged and label != a.base and not label.startswith(a.control):
                ctrl = vec(ctrl_label, eps)
                clo, chi = paired_ci(ctrl, cur)
                row[s]["delta_vs_control"] = round(float((cur - ctrl).mean()), 5)
                row[s]["ci95_vs_control"] = [round(clo, 5), round(chi, 5)]
            cell = f"{cur.mean():.4f}"
            if label != a.base:
                cell += f" ({(cur - base).mean():+.4f} [{lo:+.4f},{hi:+.4f}]) {100 * improved:.0f}%"
            line += f"{cell:>34}"
        report["rows"].append(row)
        print(line)
    # average the seeds of each arm: mean over seeds of the per-episode metric, then the same stats
    groups = {}
    for label in labels:
        if "_s" in label and label != a.base:
            groups.setdefault(label[: label.rfind("_s")], []).append(label)
    if groups:
        print(f"{'seed-averaged':<16}" + "".join(f"{s:>34}" for s in slices)); print("-" * len(hdr))
        for arm, members in groups.items():
            row = {"arm": arm + "_avg", "members": members}
            line = f"{arm + '_avg':<16}"
            for s, eps in slices.items():
                eps = [e for e in eps if all(str(e) in merged[m]["per_episode"] for m in members + [a.base])]
                base = vec(a.base, eps)
                cur = np.mean([vec(m, eps) for m in members], axis=0)
                lo, hi = paired_ci(base, cur)
                row[s] = {"mean": round(float(cur.mean()), 5), "delta_vs_base": round(float((cur - base).mean()), 5),
                          "ci95": [round(lo, 5), round(hi, 5)], "frac_improved": round(float((cur < base).mean()), 3), "n": len(eps)}
                if a.control in groups and arm != a.control:
                    ctrl = np.mean([vec(m, eps) for m in groups[a.control]], axis=0)
                    clo, chi = paired_ci(ctrl, cur)
                    row[s]["delta_vs_control"] = round(float((cur - ctrl).mean()), 5)
                    row[s]["ci95_vs_control"] = [round(clo, 5), round(chi, 5)]
                    line += f"{cur.mean():.4f} vs {a.control} {(cur - ctrl).mean():+.4f} [{clo:+.4f},{chi:+.4f}]".rjust(34)
                else:
                    line += f"{cur.mean():.4f} ({(cur - base).mean():+.4f} vs base)".rjust(34)
            report["rows"].append(row)
            print(line)
    print()
    json.dump(report, open(a.out, "w"), indent=1)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
