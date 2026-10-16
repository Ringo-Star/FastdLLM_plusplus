# WINO Frechet INSERTION (pipeline 4): GSM8K n=300, ec2_p5 GPU 7

Date: 2026-08-02. Repo: Feng-Hong/WINO-DLLM @ 9549a1d (box copy ~/baselines/WINO-DLLM).
Env: ~/miniforge3/envs/wino. Model: LLaDA-8B-Instruct @ 08b83a6. GPU 7 ONLY (verified 0 MiB before each launch).

## What was inserted
`decoding_wino_frechet` added to LLaDA/decoding.py (baseline `decoding_wino` byte-identical, verified
via .orig backups): WINO's per-step acceptance decision on the wide draft: the
(confidence > threshold=0.6) test + max_accept top-k cap (min(max(0.7*remaining,5),20)) +
at-least-one-argmax fallback: is replaced by the Frechet-profile certificate ported VERBATIM from
~/anchors/generate.py::get_transfer_index_frechet_profile (sorted desc confidences; L_n = cumsum-(n-1);
U_n = 1-c_(n); score = L_n-U_n; score[:,0]=inf; commit largest prefix with score > delta).
KEPT unchanged: wide-draft lookahead block, attention/position setup, verification back-remasking
(threshold_back=0.9, same as baseline), block scheduling, step counter definition, forward passes.
WINO has no KV cache in this evaluator; both arms cache-free (fair).
Patch: patches/wino_frechet_insertion.patch. Dispatch: eval.py method 'wino_frechet'.
Originals on box: decoding.py.orig_20260802T192800Z, eval.py.orig_20260802T192800Z.

## Validation gate (limit-8)
Modified (delta=0.25): acc 0.75, avg steps 64.75. Unmodified: acc 0.75, avg steps 49.62.
IDENTICAL per-sample correctness pattern [T,T,F,T,T,T,T,F]; coherent CoT + <answer> blocks. PASS.

## Sweep (n=300, first 300 GSM8K test docs, gen 256 / block 128, temp 0, 1 rep)
tok/s = 300*256 canvas tokens / final tqdm elapsed (uniform convention incl. baseline row;
eval.py emits no native tok/s). Steps = WINO's OWN "Average Steps" counter (decoding iterations
per sample, same counter as the baseline row; NOT our NFE).

| delta | acc | avg steps | tok/s | wall_s |
|---|---|---|---|---|
| baseline (wino thr=0.6) | 0.7767 | 46.19 | 143.6 | 544 |
| 0.10 | 0.7700 | 53.32 | 125.7 | 621 |
| 0.20 | 0.7600 | 56.22 | 117.4 | 664 |
| 0.25 | 0.7800 | 58.04 | 113.8 | 684 |
| 0.30 | 0.7733 | 60.41 | 111.6 | 698 |
| 0.40 | 0.7600 | 65.76 | 102.4 | 760 |

Baseline row = ~/matrix/wino/rep1_20260801T215321Z (not rerun; tqdm elapsed 535 s).

## Reading (honest)
Accuracy parity across the sweep (0.76 to 0.78 vs baseline 0.7767; delta=0.25 nominally +0.3pp).
Frechet inside WINO does NOT cut steps: WINO's native rule (threshold 0.6 + aggressive up-to-20-token
cap + draft-verify remasking) is already more aggressive per step than the distribution-free
certificate, so the certificate commits fewer tokens per step (+15% to +42% steps, -13% to -29% tok/s).
WINO is a case where the certificate is SAFE (accuracy preserved at every delta) but not additive on
speed: consistent with the certificate being a conservative guarantee, and with WINO's own
verification loop already policing over-commits. Report as: acceptance-rule transfer works
(gate + parity), speed bonus does not transfer to this draft-and-verify scheduler.

## Provenance
Commands: driver run_ins_wino.sh (gate|sweep|one), gpu7_guard preflight, per-run DONE sentinels,
screen mx_ins_wino_sweep. Sweep chain log: sweep/chain.log. Config per run: sweep/*/config.yaml.
Raw per-sample jsonl per run: sweep/*/gsm8k_wino_frechet_samples300.jsonl.
