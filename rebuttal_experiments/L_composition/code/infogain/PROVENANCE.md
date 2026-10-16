# PROVENANCE: Info-Gain Sampler breadth runs (Fast-dLLM++ rebuttal)

Date: 2026-08-02 (UTC). Box: ec2_p5 ip-172-31-7-19 (p5.48xlarge, 8x H100). GPU: CUDA_VISIBLE_DEVICES=3 ONLY.
Operator: rebuttal automation (orchestrated from the dev desktop; all compute on the GPU box).

## Repo
- Upstream: https://github.com/yks23/Information-Gain-Sampler
- Checked out commit: ff721bd99f59050a7d1cfccca89c4df0cdfca52f (requested "codex commit ff721bd9";
  it is the merge of PR#3 readme restructure).
- dllm submodule DEVIATION: .gitmodules pins yks23/dllm @ 75e188539cb44053bf77d438d3b2f911f5491cca,
  which is NOT reachable in the public repo (fetch fails: "not our ref"). Used branch
  origin/info-gain @ 0306f52bb9edb4d97a5887463ae041b09ec69f76 (head of the public Info-Gain pipeline
  work: "feat: support weighted Info-Gain entropy cost"), the closest public equivalent.

## Environment
- Conda env: `infogain` (miniforge3), python 3.10, created from the repo's own requirements.txt.
- Key pins: transformers 4.49.0, lm_eval 0.4.8, accelerate 0.34.2, torch 2.13.0+cu130 (unpinned in
  repo spec; latest at install time), math_verify 0.9.0, antlr4-python3-runtime 4.9.3.
  Full log: ~/baselines2/infogain/env_create.log
- Model: GSAI-ML/LLaDA-8B-Instruct from ~/hf_cache (HF_HUB_OFFLINE=1),
  snapshot 08b83a6feb34df1a6011b80c3c00c7563e963b07.
- Data: data/gsm8k.jsonl (1319 rows, question/answer) materialized from the cached HF dataset
  openai/gsm8k main test split (repo expects this file but does not ship it).

## Config note (IMPORTANT)
The author-default configs/gsm8k_info_gain.yaml sets use_cache: prefix. In this public dllm branch,
InfoGainLLaDASampler with use_cache != none DELEGATES ENTIRELY to the FastdLLM parent sampler
(the repo's own scripts/tests/test_infogain_fastdllm_cache_gpu.py asserts output identity with
FastdLLMLLaDASampler). The actual Info-Gain candidate-lookahead algorithm only runs at
use_cache=none. Therefore ALL runs here use --use_cache none; run (a) is otherwise
author-default (candidate_number 8, position_temperature 0.2, threshold 0.8, temperature 0,
gen_length 256, steps 256, block_length 32, 2-shot repo prompt).

## Patches (all marked FRECHET_INSERTION_PATCH; applier: apply_patches.py)
1. dllm/dllm/pipelines/infogain/llada/sampler.py: added `frechet_margin` config field, kwargs
   read, `_frechet_select()` (port of get_transfer_index_frechet_profile from ~/anchors/generate.py:
   L_n = sum c_(j) - (n-1); U_n = 1 - c_(n); commit largest n with L_n - U_n > margin; >=1 token;
   float64 confidences; greedy at temp 0), and a selection branch that replaces the Info-Gain
   candidate lookahead when frechet_margin is set.
2. scripts/eval.py: (i) module path fix dllm.pipelines.info_gain -> .infogain (public branch
   layout); (ii) frechet_margin plumb-through into sampler config; (iii) mask_token_id fix (LLaDA
   tokenizer lacks it; copy adapter-detected 126336); (iv) compute-accounting instrumentation:
   per-sample wall time (cuda-synchronized), model_calls, row_forwards, row_token_forwards,
   flushed to <result_path>.metrics.json every 10 samples.
3. src/compat.py: tolerate absent torchvision (repo bug: unguarded import; torchvision is not in
   the repo's requirements.txt).

## Compute accounting (why raw NFE is not comparable)
Info-Gain evaluates up-to-8-candidate batched lookahead per committed token: raw forward-call
counts under-represent its compute. Recorded per sample:
- model_calls: forward invocations (native NFE analog; the repo has no native NFE counter in the
  info-gain path).
- row_forwards: sum of batch rows over calls = effective forward work (an 8-candidate batch = 8).
- row_token_forwards: rows x seq_len summed.
- wall_s: cuda-synchronized wall time.
Primary ranking axis: wall time (and row_forwards); never raw model_calls/steps.
tok/s convention: repo has none; we report canvas tokens (gen_length) / wall_s as
canvas_tok_per_s in metrics.json.

## Gate (limit-8 native smoke, GPU3)
- Accuracy 0.75 (6/8): plausible for LLaDA-8B GSM8K with the repo's 2-shot prompt + flexible
  numeric extraction (#### -> "answer is" -> last number).
- Leakage grep: 0 hits (no ground-truth rationale fragments, no #### markers in outputs).
- 202.8 s wall, 3028 model_calls, 8896 row_forwards for 8 samples (~25 s/sample).
- Frechet-branch smoke (limit-4, delta .25): acc 3/4, 248 calls = 248 row_forwards (single-row,
  no lookahead), ~2.5 s/sample.

## Launch (screen ig_gpu3_queue, started 2026-08-02T22:30:02Z)
Sequential queue on GPU3 (single GPU; serialized so wall-time is uncontaminated):
  ~/baselines2/infogain/ig_queue.sh via `screen -dmS ig_gpu3_queue`
  (a) ig_native_headon : python run.py --config configs/gsm8k_info_gain.yaml
        --model GSAI-ML/LLaDA-8B-Instruct --use_cache none --max_samples 300
  (b) ig_frechet_d{0.1,0.2,0.25,0.3,0.4} : same + --frechet_margin <delta> --threshold null
        (threshold bypass disabled: the certificate alone decides the commit set)
n=300 first GSM8K test questions, gen256/steps256/block32, temperature 0.
Outputs: results_ig/<run>/gsm8k.txt{,.raw_results.json,.metrics.json}, DONE|FAILED sentinel,
logs/<run>.log, driver log logs/ig_queue_driver.log.
Results stay ON-BOX (no dev_new/S3 pull), per task instruction.
