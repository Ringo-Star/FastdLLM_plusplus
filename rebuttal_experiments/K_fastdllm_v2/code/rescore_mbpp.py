#!/usr/bin/env python3
"""Offline (CPU-only) re-score of Phase A mbpp accuracy with the code sanitizer.

Phase A mbpp accuracy was floored because base `mbpp` scores the RAW completion and the
chat model wraps code in fences/prose. lm-eval did not save structured generations, BUT
each run.log prints every (question, answer): the question embeds the target's 3 assert
tests, and the answer is the generated code. So we can re-score on CPU with the same
`evaluate` code_eval lm-eval uses. No GPU, no re-generation.

Self-validation: we FIRST re-score the RAW answers and check it reproduces the archived
pass_at_1 from results_*.json (within tolerance). If that matches, the parser+executor are
trustworthy, and we report the SANITIZED pass_at_1 as the corrected number.

Usage: rescore_mbpp.py <phaseA_root> [--one <run_dir>] [--workers N]
"""
import os, re, sys, json, glob, argparse
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
import evaluate

CODE_EVAL = evaluate.load("code_eval")


def sanitize_code(text):
    if not text:
        return text
    text = text.split("[DONE]")[0]
    m = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    m = re.search(r"```(?:python|py)?\s*\n?(.*)", text, re.DOTALL)
    if m:
        return m.group(1).split("```")[0].strip("\n")
    return text.split("```")[0]


def parse_log(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    pairs = []
    # blocks are delimited by lines of '=' (len 20); each block has question:  ... answer:  ...
    for block in raw.split("=" * 20):
        qi = block.find("question:  ")
        ai = block.find("\nanswer:  ")
        if qi == -1 or ai == -1 or ai < qi:
            continue
        question = block[qi + len("question:  "):ai]
        answer = block[ai + len("\nanswer:  "):]
        pairs.append((question, answer))
    return pairs


def extract_asserts(question):
    seg = question.rsplit("pass these tests:", 1)[-1]  # target block is the last one
    asserts = []
    for line in seg.splitlines():
        s = line.strip()
        if s.startswith("assert "):
            asserts.append(s)
        elif (s.startswith("[BEGIN]") or "<|im_end|>" in s) and asserts:
            break
    return asserts


def score(pairs, workers, do_sanitize):
    preds, refs = [], []
    for q, a in pairs:
        asserts = extract_asserts(q)
        if not asserts:
            continue
        code = sanitize_code(a) if do_sanitize else a
        preds.append([code])
        refs.append("\n".join(asserts))
    if not preds:
        return None, 0
    res, _ = CODE_EVAL.compute(predictions=preds, references=refs, k=[1], num_workers=workers)
    return res["pass@1"], len(preds)


def run_dir_meta(run_dir):
    parts = run_dir.split(os.sep)
    sub = int([p for p in parts if p.startswith("subblock_")][0].split("_")[1])
    tag = parts[-1]  # e.g. frechet_d0p25_fx_th0p9
    sel = tag.split("_d")[0]
    return sub, sel, tag


def archived_pass1(run_dir):
    for p in glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True):
        try:
            d = json.load(open(p))
            for tk, tv in d.get("results", {}).items():
                for k, v in tv.items():
                    if k.startswith("pass_at_1") and "stderr" not in k:
                        return float(v)
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--one", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.one:
        dirs = [args.one]
    else:
        dirs = sorted({os.path.dirname(p) for p in glob.glob(
            os.path.join(args.root, "**", "mbpp", "**", "run.log"), recursive=True)})
    rows = []
    for rd in dirs:
        log = os.path.join(rd, "run.log")
        if not os.path.exists(log):
            continue
        pairs = parse_log(log)
        raw_p1, n = score(pairs, args.workers, do_sanitize=False)
        san_p1, _ = score(pairs, args.workers, do_sanitize=True)
        arch = archived_pass1(rd)
        sub, sel, tag = run_dir_meta(rd)
        rows.append(dict(sub=sub, selector=sel, tag=tag, n=n,
                         archived=arch, raw_rescore=raw_p1, sanitized_rescore=san_p1))
        print(f"sub{sub:>2} {tag:<34} n={n} archived={arch} raw_rescore={raw_p1:.4f} "
              f"sanitized={san_p1:.4f}")
    if args.out and rows:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
