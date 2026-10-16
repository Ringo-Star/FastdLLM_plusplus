#!/bin/bash
# DC-Leap Frechet INSERTION driver. GPU 6 ONLY, sequential.
set -u
SNAP=$ROOT/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
RES=~/results/insertion_dcleap
BASEARGS="model_path=$SNAP,gen_length=256,steps=256,block_length=32,commit_thres=0.70,draft_thres=0.98,max_widow_size=128,method=dc_leap,apply_chat_template=True"
run_one() { # $1 tag  $2 limit  $3 extra_model_args (may be empty)
  tag=$1; limit=$2; extra=$3
  out=$RES/${tag}_$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$out"
  # GPU-6 preflight
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$out/gpu_preflight.txt"
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6)
  if [ "$used" -gt 2000 ]; then echo "GPU6_BUSY $used MiB, abort" > "$out/ABORT"; return 1; fi
  export CUDA_VISIBLE_DEVICES=6
  cd ~/baselines/DC-Leap/llada
  margs="$BASEARGS"; [ -n "$extra" ] && margs="$BASEARGS,$extra"
  echo "CMD: accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --limit $limit --model_args $margs" > "$out/command.txt"
  t0=$(date +%s)
  ~/miniforge3/envs/dcleap/bin/python -m accelerate.commands.launch --num_processes 1 --num_machines 1 --mixed_precision no \
    eval_llada.py --tasks gsm8k --model llada_dist --confirm_run_unsafe_code --limit "$limit" \
    --model_args "$margs" --output_path "$out" --log_samples > "$out/run.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "WALL_SECONDS:$((t1-t0))" > "$out/wall_time.txt"
  echo "RC:$rc" >> "$out/wall_time.txt"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$out/gpu_postflight.txt" 2>/dev/null
  touch "$out/DONE_$tag"
  echo "DONE $tag rc=$rc wall=$((t1-t0))s"
}
case "${1:-}" in
  gate)
    run_one gate_unmod8 8 ""
    run_one gate_frechet8_d0.25 8 "frechet_margin=0.25"
    echo "GATE_DONE $(date -u +%FT%TZ)"
    ;;
  sweep)
    for d in 0.1 0.2 0.25 0.3 0.4; do
      run_one sweep_d${d} 300 "frechet_margin=${d}"
    done
    echo "SWEEP_DONE $(date -u +%FT%TZ)"
    ;;
esac
