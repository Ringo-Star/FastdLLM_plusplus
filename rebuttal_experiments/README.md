# Rebuttal and camera-ready appendix experiments

This directory packages the additional experiments that appear in the camera-ready appendix of
Fast-dLLM++ (Frechet profile decoding). Each subdirectory corresponds to one appendix section and
contains two things:

- `results/`: the result tables exactly as printed in the paper, one CSV per table, values verbatim
  (header row added). Files named `raw_*` are the underlying run outputs for the tables where the raw
  artifact reproduces the printed numbers; they are compact CSV or JSON extracts, never sample dumps.
  Files named `*_prose_values.csv` capture numbers that appear only in the appendix text.
- `code/`: the scripts, patches and analysis code used to produce the section.

All numbers are from the NeurIPS 2026 author response and were independently reproduced by the
collaborators before being placed in the camera-ready. Throughput (tok/s) is measured on a single
NVIDIA H100 unless a table states a pipeline-specific counter (wall time, steps, forward work).

Conventions shared by every section: F denotes Frechet with margin delta, C denotes the matched factor
selector with f = 1 - delta (the two coincide when all selected confidences are equal, so the pair
isolates the gain from a heterogeneous confidence profile). GSM8K is 5-shot with flexible-extract
scoring (8-shot at generation length 1024 and above), MATH is 4-shot scored with math_verify, HumanEval
is 0-shot pass@1 after `postprocess_code.py`, MBPP is 3-shot pass@1. Steps equal gen_length /
block_length for PrefixCache and no-cache runs and gen_length for DualCache runs. NFE counts model
forward passes. Sample counts are recorded in each section's run scripts and raw result files.

Path placeholders: every copied script had absolute personal and cluster paths replaced by `$ROOT`
(for example `$ROOT/fast-dllm/v1/llada`, `$ROOT/hf_cache/...`, `$ROOT/models/...`). Home-relative
`~/...` paths were left as written. Set `ROOT` (or edit the path) before running anything. The scripts
reference the evaluation harness in `../v1/llada` of this repository (`eval_llada.py`, `generate.py`,
`postprocess_code.py`), the public Fast-dLLM-v2 code, and the public repositories of the baselines named
in Section L; none of those external repositories are vendored here.

## Section index

| Appendix section | Directory | Tables (CSV) | Hardware |
|---|---|---|---|
| H. Matched Frechet versus Factor sweeps | `H_matched_sweeps/` | `tab_app_matched_sweeps.csv`, `tab_app_commit_more.csv`, `mbpp_prose_values.csv` | H100 |
| I. Commit-level calibration and robustness | `I_calibration/` | `tab_app_commit_correct.csv`, `tab_app_eta.csv`, `tab_app_mbpp.csv`, `mbpp_diagnostic_prose_values.csv` | H100 |
| J. Generality: backbones, open-ended generation, length | `J_generality/` | `tab_app_llada15.csv`, `tab_app_ifeval.csv`, `long_generation_prose_values.csv` | H100 |
| K. Fast-dLLM-v2 sub-block transfer | `K_fastdllm_v2/` | `tab_app_v2.csv` | H100 |
| L. Composition with concurrent accelerators | `L_composition/` | `00_main_text_tab_composition.csv` plus `01_klass.csv` to `15_dapd_prose_values.csv` | H100 (one GPU per run) |

## H. Matched Frechet versus Factor sweeps (`H_matched_sweeps/`)

Tables. `results/tab_app_matched_sweeps.csv` is the appendix table of F versus C accuracy, NFE cut and
throughput gain across delta in {0.00, 0.10, 0.20, 0.25, 0.30} for LLaDA-8B-Instruct on MATH
(PrefixCache), HumanEval (PrefixCache and DualCache) and for Fast-dLLM-v2 7B on GSM8K at sub-block 4.
`results/tab_app_commit_more.csv` gives, per task, the fraction of decoding steps on which Frechet
delta = 0.25 commits strictly more tokens than matched factor f = 0.75 and the mean number of extra tokens
on those steps. `results/mbpp_prose_values.csv` holds the MBPP full-test-set numbers quoted in the text.
`results/raw_v2_gsm8k_subblock4_matched_pairs.csv` is the raw per-run extract behind the Fast-dLLM-v2
rows (accuracy and NFE per arm).

Code. `code/sweep_matched_300.sh` runs the GSM8K matched grid (PrefixCache and DualCache, gen 256,
block 32, first 300 examples) with one serial queue per GPU. `code/sweep_math_300.sh` is the same grid on
MATH. `code/sweep_code_matched.sh` and `code/sweep_code_matched_west.sh` run the MBPP and HumanEval grids
(the second is the rerun used for the published rows; it uses one GPU serially because the HumanEval
code_eval scorer cannot run concurrently). `code/postprocess_humaneval.sh` re-scores every HumanEval
sample file with `postprocess_code.py` to obtain the pass@1 values in the table. The commit-more
statistics are computed from the per-step traces by `I_calibration/code/commit_level_analysis.py`
(Exp E block), and the Fast-dLLM-v2 rows come from the Section K pipeline (`aggregate_v2.py`).

Protocol. Both arms of every pair use the same model, cache mode, generation length, block length and
prompts; only the selector differs. NFE cut is 1 - NFE(F) / NFE(C); throughput gain is tok/s(F) /
tok/s(C) - 1, both over the same 300 examples. HumanEval pass@1 is reported after post-processing;
MBPP in the prose paragraph uses the full test set.

## I. Commit-level calibration and robustness (`I_calibration/`)

Tables. `results/tab_app_commit_correct.csv` reports, for GSM8K and MBPP under PrefixCache, the fraction
of multi-token commits that reproduce the sequential greedy reference and the AUC with which the margin
G_n ranks commits by correctness, for Factor f = 1.0, Factor f = 0.75 and Frechet delta = 0.25.
`results/tab_app_eta.csv` is the calibration-robust variant (every confidence discounted by eta before the
certificate) with accuracy and commit error. `results/tab_app_mbpp.csv` is the full MBPP test-set table
(Threshold, Factor f = 1.0, matched factor f = 0.75, Frechet delta = 0.25 and its robust variants at eta in
{0.04, 0.06, 0.08}). `results/mbpp_diagnostic_prose_values.csv` collects the commit-error, argmax-flip
and commit-set-size numbers quoted in the text. The `raw_exp*.csv` files are the analysis outputs the
tables were read from: `raw_expA_prefix_token_calibration.csv` (token-level agreement and ECE),
`raw_expB_prefix_commit_reliability.csv` (commit success, commit error, mean L_n and G_n, AUC and
lower-bound violations per selector), `raw_expD_coupling.csv` (argmax-flip probe), `raw_expE_prefix_sextra.csv`
(safety of the extra tokens Frechet commits beyond matched factor) and `raw_prefix_accuracy.txt`
(task accuracies of the same runs).

Code. `code/instrumented_llada/eval_llada.py` and `code/instrumented_llada/generate.py` are the
instrumented versions of `../v1/llada/eval_llada.py` and `generate.py`. Relative to the released
files they add the model arguments `trace_jsonl` and `trace_topk` (write a per-step JSONL trace of the
sorted confidences and the selected set), `frechet_eta` (the uniform confidence discount of the robust
variant, applied as max(0, c - eta) before the certificate), `calib_verify` (after every multi-token
commit, decode the same positions one token at a time from the same decoder state, most confident first,
and record whether the parallel commit equals that sequential reference), `coupling_diag` (reveal a
position's selected neighbours and recompute its argmax, the flip probe) and `verify_max`. The new
helper functions are `_sequential_greedy_verify`, `_coupling_reveal_diag`, `_build_selector_summary`
and `_append_jsonl`; with all new arguments at their defaults the decoding path is identical to the
released code. `code/instrumented_llada/calib_smoke.py` is a one-prompt smoke test of the
instrumentation. `code/sweep_calib_v1.sh` launches the calibration battery (Frechet delta = 0.25 at
several eta, Factor f = 1.0 and f = 0.75, first 200 examples, `calib_verify=True`, trace per run).
`code/commit_level_analysis.py` and `code/commit_level_prefix_analysis.py` read the traces on CPU and
produce the `raw_exp*` CSVs (commit correctness, AUC of G_n, lower-bound violation counts by margin
decile, coupling flips, extra-token safety).

Protocol. Main setting, LLaDA-8B-Instruct with PrefixCache, gen 256, block 32. A commit is correct if it
equals the sequential greedy decode of the same positions; commit error is one minus that fraction over
multi-token commits. L_n, U_n and G_n = L_n - U_n are computed from the confidences alone and logged per
commit; the lower-bound check compares the observed correct fraction in each G_n decile against L_n.
The MBPP table runs the full test set with the standard 3-shot prompt.

## J. Generality: backbones, open-ended generation, length (`J_generality/`)

Tables. `results/tab_app_llada15.csv` is the LLaDA-1.5 GSM8K table (gen 256, 512, 1024; PrefixCache and
DualCache; the five canonical arms Threshold tau = 0.9, Factor f = 1.0, Frechet delta = 0, Factor f = 0.75,
Frechet delta = 0.25) with accuracy, tok/s and NFE per example; `results/raw_llada15_gsm8k_matched.csv` is
the per-run source (accuracy as a fraction, tok/s, total NFE over 300 examples). `results/tab_app_ifeval.csv`
is the IFEval table (prompt-level and instruction-level strict accuracy, NFE per example, tok/s);
`results/raw_ifeval_threshold_0p9_results.json` and `raw_ifeval_threshold_0p9_headline.csv` are the raw
output of the Threshold arm. `results/long_generation_prose_values.csv` holds the GSM8K 1024 and 2048
numbers quoted in the text.

Code. `code/sweep_llada15_gsm8k.sh`, `code/sweep_llada15_frechet0.sh` and `code/sweep_llada15_threshold.sh`
together run the five arms on LLaDA-1.5 (`model_path=GSAI-ML/LLaDA-1.5`) at the three generation lengths
and two cache modes. `code/sweep_ifeval.sh` runs the Factor and Frechet arms on IFEval (0-shot, gen 512,
block 32, PrefixCache, first 300 prompts) and `code/run_ifeval_threshold.sh` adds the Threshold arm.
`code/sweep_longgen_gsm8k.sh` runs GSM8K at gen 1024 and 2048 (DualCache, 8-shot) and
`code/sweep_blocksize_gsm8k.sh` varies the block length at gen 1024 (the block-size dependence discussed
in the text).

Protocol. Same harness and metrics as Section H, with LLaDA-1.5 substituted for LLaDA-8B-Instruct in the
first table. IFEval uses the lm-eval `ifeval` task with strict prompt-level and instruction-level
accuracy; it has no reference answer, so throughput and NFE are the comparison axes and accuracy is
checked for parity. For the long-generation rows the NFE reduction and accuracy difference compare
Frechet delta = 0.25 with matched factor f = 0.75 on the same 300 problems.

## K. Fast-dLLM-v2 sub-block transfer (`K_fastdllm_v2/`)

Tables. `results/tab_app_v2.csv` is the appendix table: for GSM8K and MATH and sub-block sizes 4, 8, 16, 32,
the native decoder's accuracy and total NFE (printed once, in the delta = 0.25 block, exactly as in the
paper) and, for Frechet at delta = 0.25, 0.10 and 0.30, the absolute accuracy change, NFE cut and
throughput gain relative to the native decoder. `results/raw_v2_subblock_runs.csv` is the per-run extract
(selector, delta, accuracy, total NFE, total tokens, tokens per NFE, mean selected set size) for the 7B
model on GSM8K and MATH.

Code. `code/patch_v2.py` applies an assertion-guarded patch to the public Fast-dLLM-v2 code: in
`generation_functions.py` the intra-sub-block selection is routed through `code/frechet_selectors.py`,
and `eval.py` gains `selector`, `frechet_delta`, `factor` and `log_frechet_stats` arguments plus an NFE
counter wrapped around `forward()`. `code/frechet_selectors.py` implements the threshold, factor, matched
factor and Frechet profile selectors as drop-in functions over the sub-block confidence window.
`code/check_selectors.py` unit-checks the selectors against the reference implementation.
`code/run_v2_single.sh` runs one (task, sub-block, selector) cell; `code/sweep_phaseA.sh` and
`code/sweep_pool.py` dispatch the full grid over free GPUs; `code/aggregate_v2.py` collects the lm-eval
outputs into the per-run and matched-pair tables; `code/rescore_mbpp.py` and
`code/patch_mbpp_sanitize.py` are the MBPP re-scoring helpers used in the same sweep.

Protocol. Public Fast-dLLM-v2 7B checkpoint, block-diffusion decoding unchanged, 5-shot GSM8K and
4-shot MATH, first 300 examples (MATH: 100 per subtask), block size 32, sub-block size swept over
{4, 8, 16, 32}. Only the intra-sub-block selector is replaced. Accuracy change is in absolute percentage
points relative to the unmodified native decoder on the same examples; NFE cut and throughput gain are
relative to the same native run. These runs exercise the selector inside the block-diffusion decoder and
do not vary the v2 cache configuration.

## L. Composition with concurrent accelerators (`L_composition/`)

Tables. `results/00_main_text_tab_composition.csv` is the main-text summary table (native accuracy versus
Frechet accuracy at the stated delta, and the efficiency change in each pipeline's own counter). The
appendix blocks are one CSV each, numbered in paper order: `01_klass.csv`, `02_dllm_cache.csv`,
`03_streaming_dllm.csv`, `04_dynamic_dllm.csv`, `05_dsb.csv`, `06_dc_leap.csv`, `07_swordsman.csv`,
`08_ccd_dream.csv`, `09_wino.csv`, `10_soar.csv`, `11_info_gain.csv`, `12_me_dlm.csv`, `13_dparallel.csv`,
`14_d3llm.csv`, and `15_dapd_prose_values.csv` for the DAPD boundary case described in the text. Raw
extracts are included where they reproduce the printed values: `raw_klass_block_all_runs.csv` and
`raw_klass_block_metrics_summary.json` (KLASS block, three timed repetitions per arm);
`raw_insertion_runs_vet_extracted.json` and `raw_insertion_runs_compact.csv` (Dynamic-dLLM, DSB, DC-Leap,
Swordsman, CCD, WINO and SOAR runs with each pipeline's own counters); `raw_native_baselines_extract.csv`
(native DAPD, DC-Leap and WINO reference rows and the Frechet delta = 0 anchor quoted for DAPD);
`raw_infogain_runs.csv` (accuracy, model calls, row forwards, wall time per arm);
`raw_medlm_runs.csv` (mask and edit accuracy and seconds per sample per arm, including the full-set
reference runs); `raw_d3llm_native_gsm8k_results.json` and its headline CSV (native d3LLM reference row).

Code, one subdirectory per pipeline under `code/`:

- `klass/`: `KLASS_frechet_port_generate_klass.py` is the KLASS selector (confidence 0.6, KL 0.015,
  history 2) ported into our LLaDA harness, `eval_llada_patched_with_klass.py` is the harness with that
  selector added, `manifest.txt` records the upstream commits used for validation.
- `dllm_cache/`: `utils/frechet_hook.py` adds the matched-factor and Frechet selectors and a dynamic
  commit loop that mirrors dLLM-Cache's own `generate` while leaving every model call and cache refresh
  unchanged; `utils/generate_function.py` dispatches to it when `factor` or `frechet_margin` is set and
  returns NFE; `eval_model/LLaDA.py` threads the arguments through the evaluation harness. The `.orig`
  files are the unmodified upstream versions for diffing.
- `streaming_dllm/`: `generate.py` and `eval_llada.py` patch Streaming-dLLM's LLaDA-1.5 path so that
  `get_transfer_index_dynamic` offers the Frechet rule while the pipeline's suffix pruning (window and
  confidence alpha passed through unchanged as model arguments) is retained; `.orig` files are upstream.
- `dynamic_dllm/`: `frechet_insertion.patch` plus the fully patched `patched/generate_function.py` and
  `patched/LLaDA.py`, `run_insertion_dyn.sh`, `PROVENANCE.md`.
- `dsb/`: `frechet_insertion.diff`, `run_dsb.sh`, `PROVENANCE.md`.
- `dcleap/`: `frechet_insertion.patch`, `run_ins_dcleap.sh`, `git_sha.txt`, `PROVENANCE.md`.
- `swordsman/`: `generate.py.patch`, `eval_llada.py.patch`, `import_fix_generate_sep.patch`, the patch
  applier `swordsman_frechet_patch.py`, and the native and Frechet run scripts.
- `ccd/`: `frechet_insertion.patch` against CCD's Dream `model.py`.
- `wino/`: `wino_frechet_insertion.patch`, `run_ins_wino.sh`, `README.md`.
- `soar/`: `soar_patch.py`, the patched `eval_llada.soar.py`, and the run scripts.
- `infogain/`: `infogain_dllm_submodule_20260803.patch`, `infogain_worktree_20260803.patch`,
  `apply_patches.py`, `ig_queue.sh`, `PROVENANCE.md`.
- `me_dlm/`: the harness `test_sp6.py` with the Frechet quota replacement, `sp6_run.sh`,
  `medlm300_g0.sh`, `medlm300_g6.sh`, `subset_first300.py`, `PROVENANCE.md`.
- `dapd/`: `frechet_insertion.patch`, `run_ins_dapd.sh`, `PROVENANCE.md`.

dParallel and d3LLM hooks. The patched `generate.py` files for the two distilled checkpoints are not
included because they lived only on the compute hosts. The modification was the same one-line selector
swap used in every other block: inside each repository's confidence-based generate loop
(`dParallel/LLaDA/generate.py` and `d3LLM/utils/utils_LLaDA/generate.py`, both derived from the
Fast-dLLM LLaDA loop), the call that picks the positions to commit from the sorted confidences was
replaced by the matched-factor or Frechet profile rule copied from `../v1/llada/generate.py`
(`get_transfer_index_dynamic` and `get_transfer_index_frechet_profile`), and `factor` / `frechet_margin`
were threaded through each repository's `eval_llada.py` as model arguments. The native rows in
`13_dparallel.csv` and `14_d3llm.csv` use each checkpoint's shipped decoder unchanged.

Protocol. Each baseline's pipeline, checkpoint, cache setting, decoding loop and evaluation harness are
left as released; only its commit or acceptance rule is replaced, in place, by the Frechet certificate.
Unless a block states otherwise the setting is LLaDA-8B-Instruct, GSM8K 5-shot with flexible-extract
scoring, first 300 test examples, one H100. CCD is run on Dream-7B-Instruct because its code is
Dream-only; SOAR and the information-gain sampler use their released zero-shot or two-shot GSM8K scripts;
ME-DLM uses its released finetuned checkpoint and scorer; dParallel and d3LLM use their released distilled
checkpoints. Efficiency counters follow each pipeline's own definition (NFE, steps, forward work charging
every beam or lookahead row, or wall time) and are identical in both arms of a block; they are never
compared across blocks. NFE cut is relative to the native row of the block, except for the distilled
checkpoints where it is relative to the matched factor selector f = 1 - delta on the same checkpoint and
decode loop.
