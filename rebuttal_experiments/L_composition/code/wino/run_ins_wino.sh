#!/bin/bash
# WINO Frechet-insertion driver (rebuttal pipeline 4). GPU 7 ONLY.
# Usage: run_ins_wino.sh gate            -> limit-8 validation gate (delta=0.25)
#        run_ins_wino.sh sweep           -> full delta sweep, n=300 (run in screen)
#        run_ins_wino.sh one <delta> <n> -> single run
set -u
export HF_HOME=$HOME/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7
R=$(cat $HOME/insertion/CURRENT_WINO)
REPO=$HOME/baselines/WINO-DLLM/LLaDA
PY=$HOME/miniforge3/envs/wino/bin/python

gpu7_guard() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)
  if [ "$used" -gt 1000 ]; then
    echo "ABORT: GPU 7 busy (${used} MiB used)"; exit 42
  fi
}

run_one() { # $1 = delta, $2 = num_samples, $3 = outdir
  local delta="$1" n="$2" outdir="$3"
  mkdir -p "$outdir"
  gpu7_guard
  local cfg="$outdir/config.yaml"
  cat > "$cfg" <<EOF
dataset_name: 'gsm8k'
model_path: "$ROOT/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
data_root: ""

dataset_config:
  data_loader: 'huggingface'
  loader_fn: null
  load_dataset_args: { path: "gsm8k", name: "main" }
  split: "test"
  doc_to_text_fn: "gsm8k_doc_to_text"
  extract_answer_fn: "gsm8k_extract_answer"
  is_correct_fn: "gsm8k_is_correct"
  eval_mode: "binary"
  prompt_suffix: "<reasoning>"

generation_args:
  gen_length: 256
  block_length: 128
  temperature: 0.0

method: 'wino_frechet'
method_args:
  wino_frechet:
    frechet_margin: $delta
    threshold_back: 0.9
num_samples: $n
output_path: $outdir/wino_frechet_results.json
EOF
  echo "[ins] START delta=$delta n=$n at $(date -u +%FT%TZ) outdir=$outdir"
  cd "$REPO"
  local t0=$SECONDS
  $PY eval.py --config "$cfg" > "$outdir/run.log" 2>&1
  local rc=$?
  local wall=$((SECONDS - t0))
  echo "WALL_SECONDS:$wall" > "$outdir/wall_time.txt"
  # rescue the jsonl before the next delta overwrites it (same method+samples name)
  cp -f "$REPO/results/gsm8k_wino_frechet_samples${n}.jsonl" "$outdir/" 2>/dev/null
  echo "[ins] END delta=$delta rc=$rc wall=${wall}s at $(date -u +%FT%TZ)"
  if [ $rc -eq 0 ]; then touch "$outdir/DONE"; else touch "$outdir/FAILED_rc$rc"; fi
  return $rc
}

case "${1:-}" in
  gate)
    run_one 0.25 8 "$R/gate/frechet_d0.25_limit8"
    ;;
  one)
    run_one "$2" "$3" "$R/sweep/frechet_d$2_n$3_$(date -u +%Y%m%dT%H%M%SZ)"
    ;;
  sweep)
    for d in 0.1 0.2 0.25 0.3 0.4; do
      run_one "$d" 300 "$R/sweep/frechet_d${d}_n300"
    done
    echo "SWEEP_DONE $(date -u +%FT%TZ)"
    touch "$R/sweep/SWEEP_DONE"
    ;;
  *) echo "usage: $0 gate|sweep|one <delta> <n>"; exit 1;;
esac
