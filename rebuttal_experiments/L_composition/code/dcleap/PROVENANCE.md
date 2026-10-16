# DC-Leap Frechet INSERTION (pipeline 3) provenance
Date: 2026-08-02 (UTC). Box: ec2_p5 (ip-172-31-7-19), acct 889474144632. GPU: 6 ONLY (CUDA_VISIBLE_DEVICES=6, per-run gpu_preflight.txt).
Repo: ~/baselines/DC-Leap, git SHA e105d45 (clean before patch). Env: ~/miniforge3/envs/dcleap.
Patch: frechet_insertion.patch (generate.py: verify_and_commit_frechet ported verbatim from ~/anchors/generate.py get_transfer_index_frechet_profile L_n/U_n/score, constrained to DC-Leap contiguous masked prefix; eval_llada.py: frechet_margin plumb + NFE forward-pass counter, instrumentation only). Originals: *.orig_20260802.
Baseline (unmodified matrix row, NOT rerun): ~/matrix/dcleap/rep1_20260801T215500Z acc 0.7700 flex, 96.25 tok/s, wall 826s (predates NFE counter).
Model: LLaDA-8B-Instruct hf_cache snapshot 08b83a6f. Task: gsm8k 5-shot, limit 300, gen_length=256, steps=256, block_length=32, commit_thres=0.70, draft_thres=0.98, max_widow_size=128, method=dc_leap, apply_chat_template=True. Per-run command in command.txt.
NFE = count of model forward passes in generate_with_dc_leap decoding loop (1/window-iteration).
