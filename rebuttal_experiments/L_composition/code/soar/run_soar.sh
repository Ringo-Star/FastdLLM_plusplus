#!/bin/bash
# usage: run_soar.sh <tag> <limit> <frechet_delta_or_native>
set -u
TAG=$1; LIMIT=$2; DELTA=$3
source ~/miniforge3/etc/profile.d/conda.sh
conda activate soar
export HF_HOME=~/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/baselines2/soar/eval_llada8b
MA="model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,show_speed=True,enable_soar=True"
if [ "$DELTA" != "native" ]; then MA="$MA,frechet_delta=$DELTA"; fi
OUT=~/baselines2/soar/results/$TAG
mkdir -p $OUT
echo "LAUNCH $(date -u +%FT%TZ) tag=$TAG limit=$LIMIT delta=$DELTA" 
python eval_llada.soar.py --tasks gsm8k_cot_zeroshot --model llada_dist \
    --confirm_run_unsafe_code --trust_remote_code --num_fewshot 0 --batch_size 1 \
    --limit $LIMIT --output_path $OUT --log_samples \
    --model_args "$MA"
RC=$?
echo "RC=$RC $(date -u +%FT%TZ)"
if [ $RC -eq 0 ]; then echo DONE > $OUT/DONE; else echo "FAILED rc=$RC" > $OUT/FAILED; fi
