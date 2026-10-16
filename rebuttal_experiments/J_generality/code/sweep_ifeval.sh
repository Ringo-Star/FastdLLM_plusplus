#!/bin/bash
# ISSUE-13 Exp1: IFEval (open-ended, verifiable instruction following), LLaDA-8B, PrefixCache.
# 0-shot, gen_length 512, block 32, pcache steps=16. First 300 prompts.
# Methods: Factor f=1.0 (ref), matched Factor f=0.75, Frechet d=0.25 (submitted), Frechet d=0.0.
# Idle H100s only (GPUs 0,1,2,4,5,6,7); GPU 3 was reserved by another user on the shared box.
cd ~/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_ifeval; mkdir -p $BASE/logs
MP="model_path=GSAI-ML/LLaDA-8B-Instruct,show_speed=True"
LIMIT=300

run_one(){ # gpu name margs
  local gpu="$1" name="$2" margs="$3"
  local log="$BASE/logs/$name.log"; local out="$BASE/$name"
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "SKIP done: $name"; return; fi
  echo "[$(date +%H:%M:%S)] START $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY eval_llada.py --tasks ifeval --num_fewshot 0 \
    --model llada_dist --model_args "$MP,$margs" --limit $LIMIT \
    --confirm_run_unsafe_code --output_path "$out" --log_samples > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name rc=$?"
}

JOBS=(
 "ifeval_pcache_factor_f1p00|gen_length=512,steps=16,block_length=32,use_cache=True,factor=1.0"
 "ifeval_pcache_factor_f0p75|gen_length=512,steps=16,block_length=32,use_cache=True,factor=0.75"
 "ifeval_pcache_frechet_d0p25|gen_length=512,steps=16,block_length=32,use_cache=True,frechet_margin=0.25"
 "ifeval_pcache_frechet_d0p00|gen_length=512,steps=16,block_length=32,use_cache=True,frechet_margin=0.0"
)

GPUS=(0 1 2 4)
n=0
for spec in "${JOBS[@]}"; do
  g=${GPUS[$((n % ${#GPUS[@]}))]}
  name="${spec%%|*}"; margs="${spec##*|}"
  ( run_one $g "$name" "$margs" ) > "$BASE/logs/gpu_${g}_chain.log" 2>&1 &
  n=$((n+1))
done
wait
echo "ALL_IFEVAL_DONE $(date -u)"
