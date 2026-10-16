#!/bin/bash
# Block-size check: does the matched Frechet(0.25) vs Factor(0.75) NFE reduction grow with block size?
# GSM8K gen_length 1024, DualCache, 8-shot, limit 300. dcache steps=gen_length=1024.
# block_length in {32, 64, 128}; methods {frechet d=0.25, factor f=0.75}. 6 runs.
# Idle H100s only (GPUs 0,1,2,4,5,6,7); GPU 3 was reserved by another user on the shared box.
cd ~/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_blocksize_gsm8k; mkdir -p $BASE/logs
MP="model_path=GSAI-ML/LLaDA-8B-Instruct,show_speed=True"
LIMIT=300

run_one(){ local gpu="$1" name="$2" margs="$3"
  local log="$BASE/logs/$name.log"; local out="$BASE/$name"
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "SKIP done: $name"; return; fi
  echo "[$(date +%H:%M:%S)] START $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY eval_llada.py --tasks gsm8k --num_fewshot 8 \
    --model llada_dist --model_args "$MP,$margs" --limit $LIMIT \
    --confirm_run_unsafe_code --output_path "$out" --log_samples > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name rc=$?"
}

JOBS=(
 "bs32_frechet_d0p25|gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,frechet_margin=0.25"
 "bs32_factor_f0p75|gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,factor=0.75"
 "bs64_frechet_d0p25|gen_length=1024,steps=1024,block_length=64,use_cache=True,dual_cache=True,frechet_margin=0.25"
 "bs64_factor_f0p75|gen_length=1024,steps=1024,block_length=64,use_cache=True,dual_cache=True,factor=0.75"
 "bs128_frechet_d0p25|gen_length=1024,steps=1024,block_length=128,use_cache=True,dual_cache=True,frechet_margin=0.25"
 "bs128_factor_f0p75|gen_length=1024,steps=1024,block_length=128,use_cache=True,dual_cache=True,factor=0.75"
)
GPUS=(0 1 2 4 5 6)
n=0
for spec in "${JOBS[@]}"; do
  g=${GPUS[$((n % ${#GPUS[@]}))]}
  name="${spec%%|*}"; margs="${spec##*|}"
  ( run_one $g "$name" "$margs" ) > "$BASE/logs/gpu_${g}_chain.log" 2>&1 &
  n=$((n+1))
done
wait
echo "ALL_BLOCKSIZE_DONE $(date -u)"
