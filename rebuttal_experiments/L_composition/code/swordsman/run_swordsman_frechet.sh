#!/bin/bash
# Swordsman + Frechet-profile commit rule. Usage: run_swordsman_frechet.sh <limit> <delta> <outdir>
set -x
LIMIT=$1; DELTA=$2; OUTDIR=$3
mkdir -p "$OUTDIR"
export HF_HOME=$HOME/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=5
cd $ROOT/baselines/SWORDSMAN/code_unzipped/Swordsman/llada
MODEL_PATH=$ROOT/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
CMD="$ROOT/miniforge3/envs/fast-dllm-eval/bin/python eval_llada.py --tasks gsm8k --num_fewshot 5 --batch_size 1 --limit $LIMIT --confirm_run_unsafe_code --model llada_dist --output_path $OUTDIR --model_args model_path=$MODEL_PATH,gen_length=256,steps=8,adaptive=jump,show_speed=True,min_block_len=1,max_block_len=32,use_cache=True,frechet_margin=$DELTA"
echo "$CMD" > "$OUTDIR/command.txt"
START=$(date +%s)
eval $CMD > "$OUTDIR/run.log" 2>&1
RC=$?
END=$(date +%s)
echo "$((END-START)) seconds (rc=$RC)" > "$OUTDIR/wall_time.txt"
[ $RC -eq 0 ] && touch "$OUTDIR/DONE" || touch "$OUTDIR/FAILED"
exit $RC
