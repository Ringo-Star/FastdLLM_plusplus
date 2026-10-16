# SOAR breadth run provenance (Fast-dLLM++ NeurIPS 2026 rebuttal)

Date: 2026-08-02 (UTC). Box: ec2_p5 (i-0c587d9fd7a664a98, us-east-2, p5.48xlarge). GPU: CUDA_VISIBLE_DEVICES=4 only.
GPU4 preflight 22:02Z: 0 MiB used (clean). GPU2 in use by another job (DUS sweep), untouched.

## Repo
- https://github.com/duterscmy/SOAR cloned at ~/baselines2/soar
- Requested commit ec3ec400 DOES NOT EXIST in the repo; checked out main tip
  ec3eb400e41a43dc05db20a49c0219b9a968d28e ("add: eval llada instruct") -- assumed 1-char typo.
- LLaDA-8B adapter present: eval_llada8b/{eval_llada.soar.py,generate.py,eval_soar_llada.instruct.sh}

## Environment (conda env: soar)
- python 3.10, torch 2.5.1+cu121, transformers 4.46.2, accelerate 1.12.0 (repo README spec)
- lm_eval 0.4.8, datasets 5.0.1 (harness needed by eval_llada.soar.py; eval-only, no training deps)
- Model: GSAI-ML/LLaDA-8B-Instruct from ~/hf_cache, snapshot rev 08b83a6feb34df1a6011b80c3c00c7563e963b07

## Deviations from repo scripts (all necessary to run)
1. Repo scripts call eval_llada.py; file is eval_llada.soar.py -- invoked directly.
2. Task gsm8k_zeroshot_cot does not exist in lm_eval 0.4.8 -> gsm8k_cot_zeroshot (same 0-shot CoT GSM8K).
3. --num_few_shot -> --num_fewshot; batch_size moved to CLI (--batch_size 1) to avoid duplicate-kwarg crash.
4. AutoConfig.from_pretrained needed trust_remote_code=True (one-line patch).

## Patched files (marker: FDPP_REBUTTAL_PATCH; diff vs ec3eb400 via git diff)
- eval_llada8b/generate.py: added frechet_certified_count() (port of ~/anchors/generate.py::
  get_transfer_index_frechet_profile counting rule: sorted conf c_(1)>=..., L_n=sum c-(n-1), U_n=1-c_(n),
  commit largest n with L_n-U_n>delta; NO forced min commit -> 0 certified = SOAR beam-search fallback).
  generate_soar(frechet_delta=None): delta set -> Frechet certificate REPLACES the native accept rule
  (conf>0.95, cap 5/step); delta None -> native rule byte-identical behavior. Added fw_stats counters.
- eval_llada8b/eval_llada.soar.py: frechet_delta model_arg; num_nfe = forward work; re-enabled non-EOS
  token count (id 126081); prints Total NFE is / decode steps / max beam / avg forwards per step.

## Metrics conventions
- acc: exact_match, flexible-extract (NOT strict-match, NOT stderr).
- Total forward work (PRIMARY, "Total NFE is N"): one unit per SEQUENCE ROW per model forward call --
  a step forwarding a beam of k costs k units, so beam expansion is fully charged. cfg=0 (no doubling).
- Also: decode steps (model calls), max beam width, wall time, tok/s = repo convention (non-EOS
  answer tokens after stop-seq truncation / total wall time).
- COMPUTE FRAMING: SOAR is a quality/compute frontier point (dynamic beam search spends EXTRA forward
  work for quality), NOT a pure accelerator. Never claim speedup without matching forward work.
- Native SOAR author defaults: confidence_threshold=0.95, min_parallel=1, max_parallel=5, max_beam_size=2.

## Gate (limit-8 GSM8K, GPU4)
- native: flexible-extract 0.625 (5/8), NFE 1122, decode steps 845, max beam 2, 62.2 tok/s. DONE 22:24Z.
- frechet delta=0.25: flexible-extract 0.625, NFE 462, steps 422, max beam 2, 125.8 tok/s. DONE 22:30Z.
- Leakage: code grep clean (gen path reads only question + stop tokens, no doc.answer/gold access);
  samples grep clean (no #### or gold rationale in any prompt), 0 leaks.

## Full runs (n=300 GSM8K test[:300], gen256/block32, steps=256, temp 0, batch 1, GPU4)
Launched 22:36:35Z as ONE sequential queue (screen soar_full_queue) so wall times are uncontended:
order = native, delta 0.10, 0.20, 0.25, 0.30, 0.40. Baseline row = native.
Commands: ~/baselines2/soar/run_queue.sh -> ./run_soar.sh <tag> 300 <delta|native>
Outputs: results/<tag>/ (lm-eval results json + samples jsonl + DONE sentinel), logs/<tag>.log,
results/QUEUE_ALL_DONE when queue finishes. Do NOT overwrite; reruns get _rerun suffixes.
