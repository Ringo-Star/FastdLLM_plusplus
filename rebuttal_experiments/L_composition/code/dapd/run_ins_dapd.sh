#!/bin/bash
# Frechet insertion runs for DAPD (pipeline 2). GPU 2 ONLY.
# Usage: run_ins_dapd.sh <alg> <margin-or-NA> <limit> <label>
set -u
export HF_HOME=$HOME/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
SNAP=$HOME/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
ALG=$1; MARGIN=$2; LIMIT=$3; LABEL=$4
BASE=$HOME/results/insertion_ext/dapd
ts=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR=$BASE/${LABEL}_${ts}
mkdir -p "$OUTDIR"
MA="pretrained=$SNAP,alg=$ALG,tau_min=0.005,tau_max=0.05,layer_ratio=0.3,max_gen_toks=256,summary_file=$OUTDIR/dapd_summary.csv,task_name=gsm8k,block_length=0"
if [ "$MARGIN" != "NA" ]; then MA="$MA,frechet_margin=$MARGIN"; fi
export PYTHONPATH=$HOME/baselines/DAPD
cd $HOME/baselines/DAPD/evaluation/lm-evaluation-harness
t0=$SECONDS
OUTPUT_PATH="$OUTDIR" $HOME/miniforge3/envs/dapd/bin/python -m lm_eval --model llada-dapd \
  --model_args "$MA" \
  --tasks gsm8k --batch_size 1 --limit "$LIMIT" \
  --output_path "$OUTDIR" --confirm_run_unsafe_code --log_samples > "$OUTDIR/run.log" 2>&1
rc=$?
echo "WALL_SECONDS:$((SECONDS-t0))" > "$OUTDIR/wall_time.txt"
if [ $rc -eq 0 ]; then touch "$OUTDIR/DONE_${LABEL}"; else touch "$OUTDIR/FAILED_${LABEL}_rc${rc}"; fi
echo "$OUTDIR rc=$rc"
