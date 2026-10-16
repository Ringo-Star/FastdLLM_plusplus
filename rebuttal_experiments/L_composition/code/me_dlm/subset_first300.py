#!/usr/bin/env python3
"""Derive n=300 results from a completed full-set (n=1319) run by taking the
FIRST 300 lines of its ordered gsm8k.jsonl and recomputing the exact
accuracy math from test_sp6.py (mask/edit accuracy, real_edit_steps,
inference_time). Valid because the harness runs single-rank (order preserved)
and --limit N == samples[:N] on the same data/math/gsm8k.jsonl."""
import json, os, sys, datetime

src, dst = sys.argv[1], sys.argv[2]
N = 300
rows = []
with open(os.path.join(src, "gsm8k.jsonl")) as f:
    for i, line in enumerate(f):
        if i >= N: break
        rows.append(json.loads(line))
assert len(rows) == N, f"only {len(rows)} rows in source"
os.makedirs(dst, exist_ok=True)
with open(os.path.join(dst, "gsm8k.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
acc = dict(
    mask_accuracy=sum(1 for r in rows if r["mask_correct"]) / N,
    edit_accuracy=sum(1 for r in rows if r["edit_correct"]) / N,
    real_edit_steps=sum(r["real_edit_step"] for r in rows) / N,
    inference_time=sum(r["inference_time"] for r in rows) / N,
)
with open(os.path.join(dst, "gsm8k_accuracy.json"), "w") as f:
    json.dump(acc, f, indent=4)
meta = json.load(open(os.path.join(src, "run_meta.json")))
meta["limit"] = N
meta["derived"] = dict(
    method="subset_rescore_first_300_of_completed_full_set",
    source_dir=os.path.basename(src.rstrip("/")),
    source_n=1319,
    derived_utc=datetime.datetime.utcnow().isoformat() + "Z",
    note="NFE and wall_seconds are inherited from the FULL n=1319 run and are NOT valid for the n=300 subset; use per-arm fresh-run NFE for efficiency comparisons.",
)
with open(os.path.join(dst, "run_meta.json"), "w") as f:
    json.dump(meta, f, indent=4)
with open(os.path.join(dst, "DONE"), "w") as f:
    f.write(datetime.datetime.utcnow().isoformat() + "Z derived-subset\n")
print(dst, json.dumps(acc))
