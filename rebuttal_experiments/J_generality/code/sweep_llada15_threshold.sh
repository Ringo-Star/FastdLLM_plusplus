#!/bin/bash
# ISSUE-18 add-on: LLaDA-1.5 GSM8K threshold (tau=0.9) reference arm across gen lengths + caches.
# gen {256,512,1024}; pcache steps=gen/32, dcache steps=gen; block 32; limit 300; no no-cache.
# fewshot 5 (256/512) / 8 (1024). One run per GPU (GPUs 0-5), 6,7 left free.
cd ~/fastdllmpp_rebuttal/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_llada15_gsm8k; mkdir -p $BASE/logs   # same base so it joins the sweep
MP="model_path=GSAI-ML/LLaDA-1.5,show_speed=True"
LIMIT=300

run_one(){ local gpu="$1" name="$2" fs="$3" margs="$4"
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
    JOBS+=("llada15_gsm8k_${gen}_${cache}_threshold_0p9|${fs}|gen_length=${gen},steps=${STEPS},block_length=32,${C},threshold=0.9")
  done
done

GPUS=(0 1 2 3 4 5)
n=0
for spec in "${JOBS[@]}"; do
  g=${GPUS[$((n % ${#GPUS[@]}))]}
  IFS='|' read -r name fs margs <<< "$spec"
  ( run_one $g "$name" "$fs" "$margs" ) > "$BASE/logs/thr_gpu_${g}_chain.log" 2>&1 &
  n=$((n+1))
done
wait
echo "ALL_LLADA15_THRESHOLD_DONE $(date -u)"
