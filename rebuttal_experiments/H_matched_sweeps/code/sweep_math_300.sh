#!/bin/bash
# Matched delta-sweep for MATH: Frechet(delta) vs Factor(f=1-delta), LLaDA-8B minerva_math.
# minerva_math is a 7-subtask group; --limit is PER subtask, so 43 -> ~301 total (300 max).
# gen_length=256, block=32. pcache steps=8, dcache steps=256. 4-shot, math_verify metric.
# Idle H100s only (GPUs 0,1,2,4,5,6,7); GPU 3 was reserved by another user on the shared box.
cd ~/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_math_300; mkdir -p $BASE/logs
MP="model_path=GSAI-ML/LLaDA-8B-Instruct,show_speed=True"
LIMIT=43

run_one(){ # gpu name margs
  local gpu="$1" name="$2" margs="$3"
  local log="$BASE/logs/$name.log"; local out="$BASE/$name"
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "SKIP done: $name"; return; fi
  echo "[$(date +%H:%M:%S)] START $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY eval_llada.py --tasks minerva_math --num_fewshot 4 \
    --model llada_dist --model_args "$MP,$margs" --limit $LIMIT \
    --confirm_run_unsafe_code --output_path "$out" --log_samples > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name rc=$?"
}

DELTAS=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50)
FS=(1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.60 0.50)
DTAG=(0p00 0p05 0p10 0p15 0p20 0p25 0p30 0p40 0p50)
FTAG=(1p00 0p95 0p90 0p85 0p80 0p75 0p70 0p60 0p50)

JOBS=()
for i in "${!DELTAS[@]}"; do
  d="${DELTAS[$i]}"; f="${FS[$i]}"; dt="${DTAG[$i]}"; ft="${FTAG[$i]}"
  JOBS+=("math_pcache_frechet_d${dt}|gen_length=256,steps=8,block_length=32,use_cache=True,frechet_margin=${d}")
  JOBS+=("math_pcache_factor_f${ft}|gen_length=256,steps=8,block_length=32,use_cache=True,factor=${f}")
  JOBS+=("math_dcache_frechet_d${dt}|gen_length=256,steps=256,block_length=32,use_cache=True,dual_cache=True,frechet_margin=${d}")
  JOBS+=("math_dcache_factor_f${ft}|gen_length=256,steps=256,block_length=32,use_cache=True,dual_cache=True,factor=${f}")
done

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
echo "ALL_MATH_SWEEP_DONE $(date -u)"
