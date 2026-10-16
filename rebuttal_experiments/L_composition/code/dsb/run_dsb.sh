#!/bin/bash
# DSB run driver: GPU1 ONLY. Usage: run_dsb.sh <tag> <limit> <selector_args>
#   selector_args like "threshold=0.9" or "frechet_margin=0.25"
set -u
TAG=$1; LIMIT=$2; SEL=$3
cd ~/baselines2/dsb/llada
export HF_HOME=~/hf_cache
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVBIN=~/miniforge3/envs/dsb/bin
OUT=~/baselines2/dsb/runs/${TAG}
mkdir -p ${OUT}
if [ -f ${OUT}/DONE ]; then echo "SKIP ${TAG}: DONE exists"; exit 0; fi
echo "=== ${TAG} start $(date -u +%FT%TZ) sel=${SEL} limit=${LIMIT} ===" | tee -a ${OUT}/run.log
CUDA_VISIBLE_DEVICES=1 ${ENVBIN}/accelerate launch --num_processes 1 eval_llada.py \
  --tasks gsm8k --num_fewshot 5 --limit ${LIMIT} \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,use_cache=True,dsb=True,${SEL},show_speed=True,pwl=24,swl=0,outp_path=${OUT}/speed.jsonl \
  --output_path ${OUT} >> ${OUT}/run.log 2>&1
RC=$?
echo "=== ${TAG} end $(date -u +%FT%TZ) rc=${RC} ===" | tee -a ${OUT}/run.log
if [ ${RC} -eq 0 ] && ls ${OUT}/*/results_*.json >/dev/null 2>&1; then
  touch ${OUT}/DONE
fi
exit ${RC}
