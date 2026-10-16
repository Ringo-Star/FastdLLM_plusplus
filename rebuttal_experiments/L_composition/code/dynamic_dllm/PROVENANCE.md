# Frechet INSERTION into Dynamic-dLLM full-native pipeline
Date: 2026-08-02 (UTC). Box: ec2_p5 (ip-172-31-7-19, acct 889474144632). GPU: 1 ONLY (CUDA_VISIBLE_DEVICES=1), one run at a time.
Repo: ~/baselines/DYNAMIC-DLLM, git SHA 43cb93b7805842d6017e1ee4be4e34345f1ae032 (clean except this patch).
Patch: frechet_insertion.patch (utils/generate_function.py + eval_model/LLaDA.py). Backups: *.orig_insertion_20260802.
Insertion: generate_pd gains frechet_margin=None; when set, ONLY the commit rule (PD threshold update + get_transfer_index_pd) is replaced by get_transfer_index_frechet_profile, ported VERBATIM from ~/anchors/generate.py (L177-247). Model, dynamic cache (DCU), block loop, scheduling untouched. Instrumentation ([INSERTION_METRICS]: pipeline-native nfe counter, CUDA-synced decode seconds, non-EOS gen tokens, tok/s) applied identically to baseline and inserted runs.
Env: ~/miniforge3/envs/dyndllm. Model: LLaDA-8B-Instruct snapshot 08b83a6feb34df1a6011b80c3c00c7563e963b07 (local HF cache).
Config (all runs): is_feature_cache=True,window_size=32,layer_budget=32,select_from=v,prompt_interval_steps=5000,gen_interval_steps=5000,generate_mode=pd,pd_mode=1; gen_kwargs block_length=32,gen_length=256,steps=256,cfg_scale=0.0; gsm8k_local (--include_path ./lm_eval_tasks), num_fewshot 4, batch_size 1, apply_chat_template, fewshot_as_multiturn, limit 300 (first 300 test docs, same as matrix) / limit 8 for gate.
Driver: ~/run_insertion_dyn.sh (copied into this dir). Screens: mx_ins_dyn_<tag>, sequential, GPU-1-busy guard before each launch.
Accuracy metric: exact_match,flexible-extract (strict also recorded). NFE: generate_pd own counter, summed over 300 docs.
