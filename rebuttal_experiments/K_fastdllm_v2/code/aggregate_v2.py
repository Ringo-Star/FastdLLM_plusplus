#!/usr/bin/env python3
"""Aggregate Fast-dLLM-v2 + Fréchet runs into the two deliverable CSVs.

Walks OUTROOT for */fpp_metrics.json, pairs each with its lm-eval results_*.json
(accuracy), and emits:
  tables/v2_subblock_transfer.csv    : one row per run
  tables/v2_matched_factor_pairs.csv : Fréchet(δ) vs matched Factor(1−δ) per (task,sub,δ)
Also prints a compact frontier summary.

Accuracy parsing (steering): use 'exact_match,flexible-extract' for gsm8k, and
'pass_at_1,none'/'pass@1' for mbpp; NEVER the *_stderr key.
"""
import os, sys, json, glob, csv, re

OUTROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/fastdllmpp_rebuttal/phaseA")
TABLES = os.path.join(OUTROOT, "tables")
os.makedirs(TABLES, exist_ok=True)


def find_accuracy(run_dir, task):
    cands = glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True)
    cands += glob.glob(os.path.join(run_dir, "**", "*results*.json"), recursive=True)
    for p in sorted(set(cands)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        res = d.get("results", {})
        for tkey, tv in res.items():
            if task.split("_")[0] not in tkey:
                continue
            # prefer flexible-extract for gsm8k/math, pass_at_1 for code
            for k in tv:
                if k.startswith("exact_match") and "flexible" in k and "stderr" not in k:
                    return float(tv[k]), k
            for k in tv:
                if ("pass_at_1" in k or "pass@1" in k) and "stderr" not in k:
                    return float(tv[k]), k
            for k in tv:
                if k.startswith("math_verify") and "stderr" not in k:
                    return float(tv[k]), k
            for k in tv:
                if k.startswith("exact_match") and "stderr" not in k:
                    return float(tv[k]), k
    return None, None


rows = []
for fm in glob.glob(os.path.join(OUTROOT, "**", "fpp_metrics.json"), recursive=True):
    run_dir = os.path.dirname(fm)
    try:
        m = json.load(open(fm))
    except Exception:
        continue
    # parse path: .../<model>/<task>/subblock_<S>/<runtag>/fpp_metrics.json
    parts = run_dir.split(os.sep)
    try:
        sub = int([p for p in parts if p.startswith("subblock_")][0].split("_")[1])
        task = parts[[i for i, p in enumerate(parts) if p.startswith("subblock_")][0] - 1]
        model = parts[[i for i, p in enumerate(parts) if p.startswith("subblock_")][0] - 2]
    except Exception:
        sub, task, model = m.get("small_block_size"), "?", "?"
    acc, acck = find_accuracy(run_dir, task)
    rows.append({
        "model": model, "task": task, "sub_block_size": sub,
        "selector": m.get("selector"), "delta": m.get("frechet_delta"),
        "factor": m.get("factor"), "threshold": m.get("threshold"),
        "accuracy": acc, "acc_metric": acck,
        "throughput_tok_s": m.get("throughput_tok_s"), "total_nfe": m.get("total_nfe"),
        "tokens_per_nfe": m.get("tokens_per_nfe"), "total_tokens": m.get("total_tokens"),
        "avg_selected_n": m.get("avg_selected_n"), "avg_candidate_k": m.get("avg_candidate_k"),
        "mean_G_n": m.get("mean_G_n"), "mean_B_n": m.get("mean_B_n"),
        "bonus_active_step_rate": m.get("bonus_active_step_rate"),
        "mean_extra_tokens_vs_matched_factor": m.get("mean_extra_tokens_vs_matched_factor"),
    })

rows.sort(key=lambda r: (str(r["task"]), r["sub_block_size"] or 0, str(r["selector"]), r["delta"] or -1))
cols = ["model", "task", "sub_block_size", "selector", "delta", "factor", "threshold",
        "accuracy", "acc_metric", "throughput_tok_s", "total_nfe", "tokens_per_nfe",
        "total_tokens", "avg_selected_n", "avg_candidate_k", "mean_G_n", "mean_B_n",
        "bonus_active_step_rate", "mean_extra_tokens_vs_matched_factor"]
transfer_csv = os.path.join(TABLES, "v2_subblock_transfer.csv")
with open(transfer_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# matched-factor pairs: frechet(delta) vs matched_factor(delta)
def key(r):
    return (r["task"], r["sub_block_size"], r["delta"])
fre = {key(r): r for r in rows if r["selector"] == "frechet"}
mat = {key(r): r for r in rows if r["selector"] == "matched_factor"}
pairs = []
for k, fr in fre.items():
    mf = mat.get(k)
    if not mf:
        continue
    def pct(a, b):
        return None if (a is None or b in (None, 0)) else round(100.0 * (a - b) / b, 2)
    pairs.append({
        "model": fr["model"], "task": fr["task"], "sub_block_size": fr["sub_block_size"],
        "delta": fr["delta"], "factor_matched": round(1 - fr["delta"], 2) if fr["delta"] is not None else None,
        "frechet_accuracy": fr["accuracy"], "factor_accuracy": mf["accuracy"],
        "delta_accuracy": None if (fr["accuracy"] is None or mf["accuracy"] is None) else round(fr["accuracy"] - mf["accuracy"], 4),
        "frechet_tok_s": fr["throughput_tok_s"], "factor_tok_s": mf["throughput_tok_s"],
        "delta_tok_s_percent": pct(fr["throughput_tok_s"], mf["throughput_tok_s"]),
        "frechet_nfe": fr["total_nfe"], "factor_nfe": mf["total_nfe"],
        "delta_nfe_percent": pct(fr["total_nfe"], mf["total_nfe"]),
        "bonus_active_step_rate": fr["bonus_active_step_rate"],
        "mean_extra_tokens": fr["mean_extra_tokens_vs_matched_factor"],
    })
pairs.sort(key=lambda r: (r["task"], r["sub_block_size"], r["delta"]))
pcols = ["model", "task", "sub_block_size", "delta", "factor_matched", "frechet_accuracy",
         "factor_accuracy", "delta_accuracy", "frechet_tok_s", "factor_tok_s",
         "delta_tok_s_percent", "frechet_nfe", "factor_nfe", "delta_nfe_percent",
         "bonus_active_step_rate", "mean_extra_tokens"]
pairs_csv = os.path.join(TABLES, "v2_matched_factor_pairs.csv")
with open(pairs_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=pcols)
    w.writeheader()
    for r in pairs:
        w.writerow(r)

print(f"wrote {transfer_csv} ({len(rows)} rows)")
print(f"wrote {pairs_csv} ({len(pairs)} pairs)")
print("\n=== Fréchet(δ) vs matched Factor(1−δ): NFE% (neg=fewer=better), acc delta, bonus rate ===")
print(f"{'task':6} {'sub':>3} {'δ':>5} {'NFE%':>7} {'tok/s%':>7} {'Δacc':>7} {'bonus':>6} {'xtra':>5}")
for r in pairs:
    print(f"{r['task']:6} {r['sub_block_size']:>3} {r['delta']:>5.2f} "
          f"{('' if r['delta_nfe_percent'] is None else r['delta_nfe_percent']):>7} "
          f"{('' if r['delta_tok_s_percent'] is None else r['delta_tok_s_percent']):>7} "
          f"{('' if r['delta_accuracy'] is None else r['delta_accuracy']):>7} "
          f"{('' if r['bonus_active_step_rate'] is None else round(r['bonus_active_step_rate'],3)):>6} "
          f"{('' if r['mean_extra_tokens'] is None else round(r['mean_extra_tokens'],3)):>5}")
