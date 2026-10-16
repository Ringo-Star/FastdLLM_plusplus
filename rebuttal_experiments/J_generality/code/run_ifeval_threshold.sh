#!/bin/bash
# ISSUE-13 add-on: IFEval Threshold tau=0.9 arm (completes the 5-arm canonical set).
# Mirrors Kevin sweep_ifeval.sh exactly: LLaDA-8B pcache, 0-shot, gen 512, block 32, steps 16, limit 300.
cd ~/fastdllmpp_rebuttal/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true PYTHONPATH=~/pyextras:$PYTHONPATH
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_ifeval_west; mkdir -p $BASE/logs
name=ifeval_pcache_threshold_0p9
out=$BASE/$name
if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "SKIP done"; exit 0; fi
CUDA_VISIBLE_DEVICES=0 $PY eval_llada.py --tasks ifeval --num_fewshot 0   --model llada_dist --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,show_speed=True,gen_length=512,steps=16,block_length=32,use_cache=True,threshold=0.9"   --limit 300 --confirm_run_unsafe_code --output_path "$out" --log_samples > $BASE/logs/$name.log 2>&1
echo "DONE rc=$?" >> $BASE/logs/$name.log
