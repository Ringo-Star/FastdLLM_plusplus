# DSB baseline provenance (Fast-dLLM++ rebuttal breadth run)

- Date: 2026-08-02 (UTC)
- Box: ec2_p5 (ip-172-31-7-19), GPU 1 only (CUDA_VISIBLE_DEVICES=1)
- Repo: https://github.com/lizhuo-luo/DSB @ codex commit
  114c12182c8c158b23d5efa07fbc2b691e155677 ("update the script and readme")
- Env: conda env `dsb` (python 3.10) built from the repo`s own requirements.txt:
  torch 2.5.1+cu121, transformers 4.49.0, accelerate 0.34.2, lm_eval 0.4.8,
  datasets 4.4.1, numpy 2.1.2. No training deps, no cross-env pip.
- Model: GSAI-ML/LLaDA-8B-Instruct, HF_HOME=~/hf_cache, cached snapshot rev
  08b83a6feb34df1a6011b80c3c00c7563e963b07 (repo has native LLaDA adapter: llada/).
- Native arm config (author default flagship, eval_instruct.sh "dsb_cache + parallel"):
  gen_length=256, steps=256, block_length=32, use_cache=True, dsb=True,
  threshold=0.9, pwl=24, swl=0, max_block_length=-1, temperature=0.
- Frechet arm: SAME dsb_cache path and cache settings; only the per-step commit
  rule get_transfer_index(threshold) is replaced by
  get_transfer_index_frechet_profile(margin=delta), delta in {0.1,0.2,0.25,0.3,0.4}.
  Rule (ported from ~/anchors/generate.py): sort masked confidences desc;
  L_n = cumsum - (n-1); U_n = 1 - c_(n); score = L_n - U_n; commit largest
  prefix with score > delta; always >= 1 token.
- Patch diff: frechet_insertion.diff (files: llada/generate.py, llada/eval_llada.py).
  Dispatch precedence: frechet_margin > factor > threshold/quota. Both
  generate_dsb and generate_dsb_cache patched at every selection site.
- Task: GSM8K, 5-shot, first 300 docs (--limit 300), lm-eval 0.4.8.
- Metrics: acc = exact_match,flexible-extract; NFE/tok-s/wall from the repo`s own
  show_speed counters (same counter both arms), also dumped to runs/<tag>/speed.jsonl.
- Runs under runs/<tag>/ with run.log, lm-eval results json, speed.jsonl, DONE sentinel.
- Gate: limit-8 smoke (native + frechet d=0.25) + oracle/leakage grep (clean:
  no oracle/gold/ground-truth/target usage in the generation path).
- Results stay ON-BOX per instruction (no dev_new/S3 pull).
