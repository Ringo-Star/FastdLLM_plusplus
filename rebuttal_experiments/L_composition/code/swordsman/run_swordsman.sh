#!/bin/bash
# Swordsman GSM8K runner. Usage: run_swordsman.sh <limit> <outdir>
set -x
LIMIT=$1
OUTDIR=$2
mkdir -p "$OUTDIR"
export HF_HOME=$HOME/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=5
cd $ROOT/baselines/SWORDSMAN/code_unzipped/Swordsman/llada
MODEL_PATH=$ROOT/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
GEN_LEN=256
STEPS=8   # test.sh convention: gen_length/block_length(32)
CMD="$ROOT/miniforge3/envs/fast-dllm-eval/bin/python eval_llada.py --tasks gsm8k --num_fewshot 5 --batch_size 1 --limit $LIMIT --confirm_run_unsafe_code --model llada_dist --output_path $OUTDIR --model_args model_path=$MODEL_PATH,gen_length=$GEN_LEN,steps=$STEPS,adaptive=jump,threshold=0.9,show_speed=True,min_block_len=1,max_block_len=32,dynamic_threshold_method=power_decay,base_threshold=0.9,confidence_boost=0.5,use_cache=True"
echo "$CMD" > "$OUTDIR/command.txt"
START=$(date +%s)
eval $CMD > "$OUTDIR/run.log" 2>&1
RC=$?
END=$(date +%s)
echo "$((END-START)) seconds (rc=$RC)" > "$OUTDIR/wall_time.txt"
[ $RC -eq 0 ] && touch "$OUTDIR/DONE" || touch "$OUTDIR/FAILED"
