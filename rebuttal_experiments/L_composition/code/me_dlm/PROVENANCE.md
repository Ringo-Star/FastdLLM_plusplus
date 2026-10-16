# PROVENANCE: ME-DLM (sp6 breadth run, GPU6)

Date: 2026-08-02 (UTC). Runner: sp6 breadth-run agent (Fast-dLLM++ rebuttal).

## Paper / code
- Paper: Edit-Based Refinement for Parallel Masked Diffusion Language Models, arXiv 2605.09603, ICML 2026.
- Official repo: https://github.com/renhouxing/ME-DLM
- Cloned at commit: 11d1053b6e56da456a245c9996065b7dde0daa15 (master HEAD, pushed 2026-05-12).
- Code link source: listed directly on the arXiv abstract page.

## Model
- ME-DLM is a TRAINED method: the released checkpoint IS the method. No training performed here (inference only).
- Checkpoint: renhouxing/ME-DLM-Stage3 (HF), snapshot d705e5fa3425053516af916e3586ac329d6cd07f, cached under ~/hf_cache.
- Base model lineage: GSAI-ML/LLaDA-8B-Base (per HF model card tags). NOTE: this is NOT the stock
  LLaDA-8B-Instruct rev 08b83a6 used elsewhere in the rebuttal; the authors rewrote the LLaDA codebase
  (models/llada/, custom special tokens <|del|> etc., varlen flash-attention packing) and finetuned 3 stages.
  Cross-method comparisons must treat this as a different checkpoint, not a decoding-only delta on stock LLaDA.

## Environment
- Box: ec2_p5 (ip-172-31-7-19), GPU 6 only (CUDA_VISIBLE_DEVICES=6). GPU2 belongs to another job (DUS), untouched.
- Conda env: medlm (fresh, dedicated). python 3.10, torch 2.5.1+cu121, transformers 4.57.6,
  flash_attn 2.7.4.post1 (prebuilt wheel cu12/torch2.5/cp310), evalplus, pylatexenc, word2number, latex2sympy2.
- No cross-env pip; no training.

## Harness (test_sp6.py = patched copy of author test.py; original untouched)
Patch summary (all changes additive, author paths preserved):
1. --limit N: truncate task samples (gate runs).
2. --frechet_margin DELTA: replaces the author fixed per-step quota in mask_diffusion with the
   Frechet-profile commit rule (port of ~/anchors/generate.py get_transfer_index_frechet_profile):
   sorted confidences c_(1)>=...>=c_(m), L_n = sum_j<=n c_(j) - (n-1), U_n = 1 - c_(n),
   commit largest n with L_n - U_n > margin (always >=1 token). Confidence convention identical to the
   author arm (softmax prob of argmax token, del/pad banned). Edit-refinement phase left untouched in both arms.
3. NFE counter: counts model forward passes in mask_diffusion (+frechet variant) and edit_diffusion.
   Same counter in both arms. NOTE: one forward = one packed varlen batch (multiple sequences per forward,
   ~4096 tokens per pack), so NFE is per-batch-forward, not per-sample; compare within this harness only.
4. --out_root: results directory separated from model path.
5. run_meta.json per run: arm, margin, steps, NFE, wall seconds.

## Protocol
- Task: GSM8K, author-native protocol: 0-shot boxed-answer chat prompt, author math_reward_fn scorer
  (NOT lm-eval flexible-extract; do not mix with lm-eval numbers). Data: repo data/math/gsm8k.jsonl,
  1319 rows = GSM8K test split (leakage grep: prompts carry question only; verified).
- Author defaults (head-on arm): init_num 512, mask_diffusion_step 48, edit_diffusion_step 16
  (README-recommended eval command), 1 rep, full 1319 samples.
- Frechet insertion arm: delta in {0.1, 0.25, 0.4} (reduced set), same edit_diffusion_step 16, full set.
- tok/s convention: repo's own inference_time per sample (author convention) + wall_seconds in run_meta.
- Accuracy: repo-reported mask/edit accuracies (test-set exact match via math_reward_fn).

## Gate (limit-8, GPU6, passed 2026-08-02 ~22:20 UTC)
- quota (author default): rc=0, mask acc 0.375, edit acc 0.500, NFE 96 mask + 32 edit forwards, wall 12.1s.
- frechet delta=0.25: rc=0, mask acc 0.875, edit acc 0.875, NFE 497 mask + 32 edit forwards, wall 51.1s.
- Reading: author default quota is very aggressive (~10.7 committed tokens/step); Frechet certificate
  commits fewer tokens/step at these margins, trading NFE for accuracy. The delta sweep traces this frontier.

## Runs (screen sp6_gsm8k, sequential on GPU6)
Queue: (a) quota head-on; (b) frechet 0.1; (c) frechet 0.25; (d) frechet 0.4. Full GSM8K each.
Completion keys off run_meta.json per run dir; SP6_DONE sentinel at queue end.
Results root: ~/baselines2/ME-DLM/results_stage3/sp6_results-<arm>-n_512-mds_48-eds_16-limit_0/
Log: ~/baselines2/ME-DLM/sp6_run.log

## Backfill (screen medlm_backfill, sequential on GPU0): 2026-08-02 ~23:46 UTC
Additive extension of the Frechet-insertion delta sweep from {0.1, 0.25, 0.4} to 5 points: adds
delta=0.2 then delta=0.3, launched by the rebuttal backfill agent while the original sp6_gsm8k
queue was still running on GPU6 (untouched). Verbatim replica of the sp6_run.sh config
(same env medlm, same checkpoint snapshot d705e5fa, test_sp6.py, init_num 512,
mask_diffusion_step 48, edit_diffusion_step 16, full GSM8K 1319, author math_reward_fn scorer);
changed ONLY: CUDA_VISIBLE_DEVICES 6->0, torchrun master_port 29631->29632, delta set, log file.
Launcher: ~/baselines2/ME-DLM/medlm_backfill.sh. Log: ~/baselines2/ME-DLM/medlm_backfill.log.
Results: results_stage3/sp6_results-frechet0.2-n_512-mds_48-eds_16-limit_0/ and
results_stage3/sp6_results-frechet0.3-n_512-mds_48-eds_16-limit_0/ (run_meta.json + per-run DONE
sentinel each; MEDLM_BACKFILL_DONE at queue end). Same NFE counters/scorer: within-harness only,
do not mix with lm-eval rows.

## 2026-08-03 01:20Z: restructure to n=300 protocol (explicit user decision, 2026-08-03 01:12Z)

**Decision:** keep whatever fully ran; stop remaining full-set (n=1319) arms; run all remaining ME-DLM work at n=300 only (user chose not to wait for full-set passes).

**Inventory at decision time (results_stage3/):**
- COMPLETE at n=1319 (KEPT): quota (`sp6_results-quota-...-limit_0`, finished 2026-08-02 22:53Z, edit_acc 0.6300) and frechet d0.1 (`sp6_results-frechet0.1-...-limit_0`, finished 2026-08-03 01:13Z, edit_acc 0.8537). Plus the two limit_8 gate smoke runs.
- MID-RUN (STOPPED): d0.2 full-set in medlm_backfill at 836/1319 (~63%); d0.25 full-set in sp6_gsm8k at ~24/1319 (just started).
- NOT STARTED: d0.3 (backfill queue), d0.4 (sp6 queue).
- Partial run dirs were EMPTY on disk (harness writes gsm8k.jsonl only at run end; per-rank output not flushed incrementally), so no raw partial outputs existed to preserve. For the record: `partial_frechet0.2_limit0_20260803.tgz` and `partial_frechet0.25_limit0_20260803.tgz` (empty dirs + log snapshots `*.log.snapshot_20260803_0117` showing progress lines). Nothing was deleted.

**What was stopped (PID-verified, no pattern kills):** identity of every PID confirmed via `ps -o pid,ppid,etime,cmd -p ...` immediately before the kill (all were bash sp6_run.sh / medlm_backfill.sh or torchrun/python test_sp6.py children of OUR two screens). Killed by exact PID: 1401708 (bash sp6_run.sh) + children 1525164/1525236 (torchrun+python test_sp6.py --frechet_margin 0.25); 1464470 (bash medlm_backfill.sh) + children 1464491/1464561 (torchrun+python test_sp6.py --frechet_margin 0.2). Drivers killed first so no next arm launched. Verified after: all 8 target PIDs gone, screens sp6_gsm8k + medlm_backfill closed, GPUs 0 and 6 at 0 MiB. `ig_gpu3_queue` (Info-Gain, GPU3, PID 1412562) untouched and still running.

**n=300 subset method:** the harness (`test_sp6.py`) implements `--limit N` as `samples[:N]` on `data/math/gsm8k.jsonl`, and runs single-rank (nproc_per_node=1) so the merged output jsonl preserves dataset order. Therefore first-300-of-full-run == a `--limit 300` run for accuracy purposes.
- n=300 QUOTA BASELINE: derived by subset-rescore (`subset_first300.py`) of the completed n=1319 quota run -> `sp6_results-quota-...-limit_300_subset1319` (mask 0.4333, edit 0.6633, real_edit_steps 14.86). Same derivation applied to d0.1 -> `sp6_results-frechet0.1-...-limit_300_subset1319` (edit 0.8767) as a free cross-check. CAVEAT recorded in each derived run_meta.json: NFE/wall_seconds are inherited from the full run and are NOT valid at n=300; use fresh-run NFE for efficiency numbers.
- FRESH n=300 Frechet arms (all five deltas, protocol-uniform NFE): GPU0 chain `medlm300_g0` (d0.1, d0.25, d0.4, port 29641, log medlm300_g0.log); GPU6 chain `medlm300_g6` (d0.2, d0.3, port 29642, log medlm300_g6.log, launch staggered 60s per OOM guard). Config verbatim from sp6_run.sh except `--limit 300`. Output dirs `...-limit_300`; DONE sentinel per arm + per-chain sentinel MEDLM300_G0_DONE / MEDLM300_G6_DONE.

**Full-set vs subset disclosure:** the completed n=1319 quota and d0.1 runs are KEPT as stronger-sample references; the n=300 block (subset-derived quota baseline + five fresh Frechet arms) is the protocol-uniform comparison set. Do not mix n=1319 and n=300 rows in one series without labeling the n.

**Not yet done:** pull to dev_new/S3 (later coordinated pass).
