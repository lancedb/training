import sys, statistics
from collections import defaultdict
per_ts=defaultdict(list)
for line in open(sys.argv[1], errors="ignore"):
    p=[x.strip() for x in line.split(",")]
    if len(p)==4 and p[2].isdigit(): per_ts[p[0].split(".")[0]].append(float(p[2]))
ts=sorted(per_ts)
active=[t for t in ts if statistics.mean(per_ts[t])>10]
skip=int(sys.argv[2]) if len(sys.argv)>2 else 45
win=active[skip:]
u=[x for t in win for x in per_ts[t]]
if u: print(f"gpu util: mean {statistics.mean(u):.1f}%  median {statistics.median(u):.0f}%  over {len(win)}s active window x 8 GPUs (first {skip}s active skipped)")
else: print(f"gpu util: too few active samples ({len(active)}s active)")
