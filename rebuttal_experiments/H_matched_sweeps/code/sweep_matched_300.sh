#!/bin/bash
# Matched delta-sweep: Frechet(delta) vs Factor(f=1-delta), LLaDA-8B GSM8K, first 300 samples.
# PrefixCache + DualCache. gen_length=256, block=32. pcache steps=8, dcache steps=256.
# Runs on idle H100s only (GPUs 0,1,2,4,5,6,7); GPU 3 was reserved by another user on the shared box.
cd ~/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_matched_300; mkdir -p $BASE/logs
MP="model_path=GSAI-ML/LLaDA-8B-Instruct,show_speed=True"
LIMIT=300

run_one(){ # gpu name margs
  local gpu="$1" name="$2" margs="$3"
  local log="$BASE/logs/$name.log"; local out="$BASE/$name"
  if grep -qE "exact_match" "$log" 2>/dev/null; then echo "SKIP done: $name"; return; fi
  echo "[$(date +%H:%M:%S)] START $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY eval_llada.py --tasks gsm8k --num_fewshot 5 \
    --model llada_dist --model_args "$MP,$margs" --limit $LIMIT \
    --confirm_run_unsafe_code --output_path "$out" --log_samples > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name rc=$?"
}

# delta grid and matched factor f=1-delta
DELTAS=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50)
FS=(1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.60 0.50)
# name-safe tags
DTAG=(0p00 0p05 0p10 0p15 0p20 0p25 0p30 0p40 0p50)
FTAG=(1p00 0p95 0p90 0p85 0p80 0p75 0p70 0p60 0p50)

JOBS=()
for i in "${!DELTAS[@]}"; do
  d="${DELTAS[$i]}"; f="${FS[$i]}"; dt="${DTAG[$i]}"; ft="${FTAG[$i]}"
  # PrefixCache
  JOBS+=("gsm8k_pcache_frechet_d${dt}|gen_length=256,steps=8,block_length=32,use_cache=True,frechet_margin=${d}")
  JOBS+=("gsm8k_pcache_factor_f${ft}|gen_length=256,steps=8,block_length=32,use_cache=True,factor=${f}")
  # DualCache
  JOBS+=("gsm8k_dcache_frechet_d${dt}|gen_length=256,steps=256,block_length=32,use_cache=True,dual_cache=True,frechet_margin=${d}")
  JOBS+=("gsm8k_dcache_factor_f${ft}|gen_length=256,steps=256,block_length=32,use_cache=True,dual_cache=True,factor=${f}")
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
echo "ALL_SWEEP_DONE $(date -u)"
