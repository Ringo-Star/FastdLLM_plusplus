#!/bin/bash
# ISSUE-18: LLaDA-1.5 on GSM8K across generation lengths, all caches EXCEPT no-cache.
# gen in {256,512,1024}; pcache steps=gen/32, dcache steps=gen; block 32; limit 300.
# fewshot: 5-shot for 256/512, 8-shot for 1024 (eval convention).
# Methods: Factor f=1.0 (ref), matched Factor f=0.75, Frechet delta=0.25.
# Shared west box: use GPUs 0-5, leave 6,7 free for the user. One run per GPU (serial chain).
cd ~/fastdllmpp_rebuttal/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_llada15_gsm8k; mkdir -p $BASE/logs
MP="model_path=GSAI-ML/LLaDA-1.5,show_speed=True"
LIMIT=300

run_one(){ # gpu name fewshot margs
  local gpu="$1" name="$2" fs="$3" margs="$4"
  local log="$BASE/logs/$name.log"; local out="$BASE/$name"
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "SKIP done: $name"; return; fi
  echo "[$(date -u +%H:%M:%S)] START $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY eval_llada.py --tasks gsm8k --num_fewshot "$fs" \
    --model llada_dist --model_args "$MP,$margs" --limit $LIMIT \
    --confirm_run_unsafe_code --output_path "$out" --log_samples > "$log" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE $name rc=$?"
}

declare -a JOBS
for gen in 256 512 1024; do
  if [ "$gen" = 1024 ]; then fs=8; else fs=5; fi
  for cache in pcache dcache; do
    if [ "$cache" = pcache ]; then C="use_cache=True"; STEPS=$((gen/32)); else C="use_cache=True,dual_cache=True"; STEPS=$gen; fi
    JOBS+=("llada15_gsm8k_${gen}_${cache}_factor_f1p00|${fs}|gen_length=${gen},steps=${STEPS},block_length=32,${C},factor=1.0")
    JOBS+=("llada15_gsm8k_${gen}_${cache}_factor_f0p75|${fs}|gen_length=${gen},steps=${STEPS},block_length=32,${C},factor=0.75")
    JOBS+=("llada15_gsm8k_${gen}_${cache}_frechet_d0p25|${fs}|gen_length=${gen},steps=${STEPS},block_length=32,${C},frechet_margin=0.25")
  done
done

GPUS=(0 1 2 3 4 5)
declare -A Q
n=0
for spec in "${JOBS[@]}"; do
  g=${GPUS[$((n % ${#GPUS[@]}))]}
  IFS='|' read -r name fs margs <<< "$spec"
  Q[$g]+="run_one $g ${name} ${fs} \"${margs}\""$'\n'
  n=$((n+1))
done
echo "Total jobs: ${#JOBS[@]} across ${#GPUS[@]} GPUs (6,7 left free)"
for g in "${GPUS[@]}"; do
  ( eval "${Q[$g]}" ) > "$BASE/logs/gpu_${g}_chain.log" 2>&1 &
done
wait
echo "ALL_LLADA15_DONE $(date -u)"
