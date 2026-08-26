import sys, time, warnings, lancedb
from lancedb.streaming import StreamingDataset
warnings.simplefilter("ignore")
tbl = lancedb.connect('/home/ubuntu/runs/small/db').open_table('corpus')
FILT = "NOT is_dup AND score >= 1.0 AND (id % 100 != 0)"
def run(label, seconds=12, packed=True, **kw):
    base = dict(columns=["input_ids"], filter=FILT, num_splits=32, shuffle_seed=0, read_batch_size=8)
    if packed:
        base.update(pack_sequences=1024, eos_id=50256, pad_id=50257, blocks_per_epoch=2_000_000)
    base.update(kw)
    ds = StreamingDataset(tbl, **base)
    it = iter(ds); next(it)
    t0 = time.perf_counter(); k = 0
    while time.perf_counter() - t0 < seconds:
        next(it); k += 1
    dt = time.perf_counter() - t0
    unit = "blk" if packed else "rows"
    print(f"{label:55s} {k/dt:8,.0f} {unit}/s  fetch {ds.fetch_time:7.1f}s tx {ds.transform_time:6.1f}s", flush=True)
    it.close() if hasattr(it, "close") else None
for spec in sys.argv[1:]:
    kw = eval(f"dict({spec})")
    run(spec, **kw)
