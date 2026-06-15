"""
Offline GPT judging for MathVerse predictions saved by lmms-eval.

Loads a `mathverse_<split>_<problem_version>_results.json` file (the snapshot
written at LLaDA-V/eval/lmms-eval/lmms_eval/tasks/mathverse/utils.py before GPT
scoring) and runs the same `extract_answer` + `score_answer` pipeline as
`MathVerseEvaluator.eval_results`, then writes the scored results JSON and the
scores JSON in the exact format lmms-eval would produce.

Adds two things lmms-eval lacks:
  - per-item JSONL cache so a 429 / Ctrl-C does not lose progress
  - resumable: re-running picks up at the first un-judged sample_index

Usage:
  export OPENAI_API_KEY=<your_openai_api_key>
  python LLaDA-V/eval/scripts/mathverse_offline_judge.py \
      --input  exp/llava_v_eval/LLaDA-V/submissions/mathverse_testmini_vision_intensive_results.json \
      --output-dir exp/llava_v_eval/LLaDA-V/submissions \
      [--gpt-model gpt-4o] [--trunk-response 30] [--quick-match] [--workers 4]
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# this file lives at LLaDA-V/eval/scripts/ ; lmms-eval is one level up
LMMS_EVAL_ROOT = Path(__file__).resolve().parents[1] / "lmms-eval"
sys.path.insert(0, str(LMMS_EVAL_ROOT))

from lmms_eval.tasks.mathverse.mathverse_evals import MathVerseEvaluator  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to mathverse_<split>_<problem_version>_results.json")
    p.add_argument("--output-dir", default=None, help="Where to write _results.json and _scores.json (defaults to input dir)")
    p.add_argument("--cache", default=None, help="Per-item JSONL cache file (defaults to <input>.judge_cache.jsonl)")
    p.add_argument("--gpt-model", default="gpt-4o")
    p.add_argument("--trunk-response", type=int, default=30, help="Keep last N whitespace-split tokens of prediction; -1 to disable. Matches mathverse.yaml default of 30.")
    p.add_argument("--quick-match", action="store_true", help="Skip GPT scoring; literal string compare extraction==answer")
    p.add_argument("--quick-extract", action="store_true", help="Forwarded to MathVerseEvaluator (currently unused upstream)")
    p.add_argument("--workers", type=int, default=1, help="Concurrent GPT calls. >1 trades RPM headroom for speed.")
    p.add_argument("--backoff-base", type=float, default=1.0, help="Initial 429 backoff in seconds (doubles each consecutive rate-limit). Sets LMMS_OPENAI_BACKOFF_BASE.")
    p.add_argument("--backoff-cap",  type=float, default=60.0, help="Max 429 backoff cap in seconds. Sets LMMS_OPENAI_BACKOFF_CAP.")
    return p.parse_args()


def load_cache(path: Path) -> dict:
    cache = {}
    if not path.exists():
        return cache
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "sample_index" in obj:
                cache[str(obj["sample_index"])] = obj
    return cache


def append_cache(path: Path, record: dict, lock: threading.Lock):
    with lock:
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def judge_one(evaluator: MathVerseEvaluator, inst: dict, trunk_response: int, quick_match: bool):
    full_prediction = (inst.get("prediction") or "").strip()
    if trunk_response > 0:
        prediction = " ".join(full_prediction.split(" ")[-trunk_response:])
    else:
        prediction = full_prediction

    extraction = evaluator.extract_answer(prediction)
    if inst.get("answer") is not None:
        true_false = evaluator.score_answer(inst["question"], inst["answer"], extraction, quick_match)
    else:
        true_false = False

    return {
        "sample_index": inst["sample_index"],
        "extraction": extraction,
        "prediction": prediction,
        "true_false": bool(true_false),
    }


def aggregate_scores(results: list) -> dict:
    """Replicates MathVerseEvaluator.eval_results scoring output exactly."""
    total = len(results)
    correct = sum(1 for r in results if r["true_false"])
    accuracy = round(correct / total * 100, 2) if total else 0.0
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    # mirror eval_results: flatten metadata fields onto each row
    flat = []
    for r in results:
        row = dict(r)
        meta = row.pop("metadata", {}) or {}
        row.update(meta)
        flat.append(row)

    df = pd.DataFrame({r["sample_index"]: r for r in flat}).T
    target_keys = ["problem_version", "subfield"]

    def acc_with_condition(res_pd, key, value):
        total_pd = res_pd[res_pd[key] == value]
        correct_pd = total_pd[total_pd["true_false"] == True]  # noqa: E712
        acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100) if len(total_pd) > 0 else "0.00"
        return len(correct_pd), len(total_pd), acc

    for key in target_keys:
        if key not in df.columns:
            continue
        scores[key] = {}
        for value in df[key].unique():
            c, t, acc = acc_with_condition(df, key, value)
            if t > 0:
                scores[key][value] = {"accuracy": acc, "correct": c, "total": t}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))

    results_dict = {r["sample_index"]: r for r in flat}
    return results_dict, scores


def main():
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY":
        sys.exit("OPENAI_API_KEY env var is not set")

    # Forwarded to MathVerseEvaluator.get_chat_response (read at call time)
    os.environ["LMMS_OPENAI_BACKOFF_BASE"] = str(args.backoff_base)
    os.environ["LMMS_OPENAI_BACKOFF_CAP"]  = str(args.backoff_cap)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache).resolve() if args.cache else input_path.with_suffix(".judge_cache.jsonl")

    with input_path.open() as f:
        results = json.load(f)
    if not isinstance(results, list):
        sys.exit("Input JSON must be a list of result dicts")

    print(f"[load] {len(results)} predictions from {input_path}")
    cache = load_cache(cache_path)
    print(f"[cache] {len(cache)} already-judged items at {cache_path}")

    evaluator = MathVerseEvaluator(api_key=api_key, gpt_model=args.gpt_model, quick_extract=args.quick_extract)
    cache_lock = threading.Lock()

    pending = [inst for inst in results if str(inst["sample_index"]) not in cache]
    print(f"[plan] {len(pending)} items still need judging  (workers={args.workers})")

    def run(inst):
        rec = judge_one(evaluator, inst, args.trunk_response, args.quick_match)
        append_cache(cache_path, rec, cache_lock)
        return rec

    try:
        if args.workers <= 1:
            for inst in tqdm(pending, desc="judge"):
                rec = run(inst)
                cache[str(rec["sample_index"])] = rec
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(run, inst) for inst in pending]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="judge"):
                    rec = fut.result()
                    cache[str(rec["sample_index"])] = rec
    except KeyboardInterrupt:
        print("\n[interrupt] partial progress saved to cache; rerun the same command to resume.")
        sys.exit(130)

    # merge cache back onto the original list (preserves ordering and all other fields)
    for inst in results:
        rec = cache.get(str(inst["sample_index"]))
        if rec is None:
            print(f"[warn] missing judgement for sample_index={inst['sample_index']}")
            continue
        inst["extraction"] = rec["extraction"]
        inst["prediction"] = rec["prediction"]
        inst["true_false"] = rec["true_false"]

    # derive output names from input filename
    stem = input_path.stem  # e.g. mathverse_testmini_vision_intensive_results
    base = stem[:-len("_results")] if stem.endswith("_results") else stem
    results_out = output_dir / f"{base}_results.json"
    scores_out = output_dir / f"{base}_scores.json"

    results_dict, scores = aggregate_scores(results)
    with results_out.open("w") as f:
        json.dump(results_dict, f, indent=4)
    with scores_out.open("w") as f:
        json.dump(scores, f, indent=4)

    print(f"\n[write] {results_out}")
    print(f"[write] {scores_out}")
    print("\n=== Scores ===")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
