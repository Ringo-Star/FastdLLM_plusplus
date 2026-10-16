#!/bin/bash
# ISSUE-09: long-generation GSM8K, LLaDA-8B, DualCache, 8-shot, first 300 samples.
# gen_length in {1024, 2048}, block=32, dcache steps=gen_length.
# Methods: Factor f=1.0 (ref), matched Factor f=0.75, Frechet d=0.25 (submitted), Frechet d=0.0 (aggressive).
# Idle H100s only (GPUs 0,1,2,4,5,6,7); GPU 3 was reserved by another user on the shared box.
cd ~/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_longgen_gsm8k; mkdir -p $BASE/logs
MP="model_path=GSAI-ML/LLaDA-8B-Instruct,show_speed=True"
LIMIT=300

run_one(){ # gpu name margs
  local gpu="$1" name="$2" margs="$3"
  local log="$BASE/logs/$name.log"; local out="$BASE/$name"
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "SKIP done: $name"; return; fi
  echo "[$(date +%H:%M:%S)] START $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY eval_llada.py --tasks gsm8k --num_fewshot 8 \
    --model llada_dist --model_args "$MP,$margs" --limit $LIMIT \
    --confirm_run_unsafe_code --output_path "$out" --log_samples > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name rc=$?"
}

# name|margs  (dcache steps = gen_length)
JOBS=(
 "longgen1024_dcache_factor_f1p00|gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,factor=1.0"
 "longgen1024_dcache_factor_f0p75|gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,factor=0.75"
 "longgen1024_dcache_frechet_d0p25|gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,frechet_margin=0.25"
 "longgen1024_dcache_frechet_d0p00|gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,frechet_margin=0.0"
 "longgen2048_dcache_factor_f1p00|gen_length=2048,steps=2048,block_length=32,use_cache=True,dual_cache=True,factor=1.0"
 "longgen2048_dcache_factor_f0p75|gen_length=2048,steps=2048,block_length=32,use_cache=True,dual_cache=True,factor=0.75"
 "longgen2048_dcache_frechet_d0p25|gen_length=2048,steps=2048,block_length=32,use_cache=True,dual_cache=True,frechet_margin=0.25"
 "longgen2048_dcache_frechet_d0p00|gen_length=2048,steps=2048,block_length=32,use_cache=True,dual_cache=True,frechet_margin=0.0"
)

GPUS=(0 1 2 4 5 6 7)
declare -A Q
n=0
for spec in "${JOBS[@]}"; do
  g=${GPUS[$((n % ${#GPUS[@]}))]}
  name="${spec%%|*}"; margs="${spec##*|}"
  Q[$g]+="run_one $g ${name} \"${margs}\""$'\n'
  n=$((n+1))
done
echo "Total jobs: ${#JOBS[@]} across ${#GPUS[@]} GPUs"
for g in "${GPUS[@]}"; do
  ( eval "${Q[$g]}" ) > "$BASE/logs/gpu_${g}_chain.log" 2>&1 &
done
wait
echo "ALL_LONGGEN_DONE $(date -u)"
