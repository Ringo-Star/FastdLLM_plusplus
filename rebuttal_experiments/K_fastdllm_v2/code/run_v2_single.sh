#!/usr/bin/env bash
# Single Fast-dLLM-v2 + Fréchet eval run. One GPU per invocation.
# Usage: run_v2_single.sh GPU TASK SUBBLOCK SELECTOR DELTA FACTOR THRESHOLD NFEWSHOT LIMIT OUTROOT
#   DELTA/FACTOR/LIMIT = -1 to omit. SELECTOR in default|threshold|factor|matched_factor|frechet
# Env overrides: MODEL_PATH MODEL_NAME BDSIZE BATCH MAXNEW USE_BLOCK_CACHE
set -uo pipefail
GPU=$1; TASK=$2; SUB=$3; SELECTOR=$4; DELTA=$5; FACTOR=$6; THRESHOLD=$7; NFEWSHOT=$8; LIMIT=$9; OUTROOT=${10}

MODEL_PATH=${MODEL_PATH:-$ROOT/models/Fast_dLLM_v2_1.5B}
MODEL_NAME=${MODEL_NAME:-Fast_dLLM_v2_1.5B}
BDSIZE=${BDSIZE:-32}
BATCH=${BATCH:-8}
MAXNEW=${MAXNEW:-256}
USE_BLOCK_CACHE=${USE_BLOCK_CACHE:-False}
V2=~/fastdllmpp_rebuttal/fast-dllm/v2

# tag encodes the run identity; d/f use 'x' for omitted
DTAG=${DELTA/./p}; [ "$DELTA" = "-1" ] && DTAG=x
FTAG=${FACTOR/./p}; [ "$FACTOR" = "-1" ] && FTAG=x
OUT="$OUTROOT/$MODEL_NAME/$TASK/subblock_${SUB}/${SELECTOR}_d${DTAG}_f${FTAG}_th${THRESHOLD/./p}"
mkdir -p "$OUT"
cd "$V2"

source ~/envs/fastdllm-v2/bin/activate
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
export FPP_SUMMARY="$OUT/fpp_metrics.json"
export TOKENIZERS_PARALLELISM=false
# Per-run isolation so concurrent code_eval (mbpp/humaneval) runs never share the
# metric-module cache or the sandbox tmpdir (steering: code_eval collides if shared).
# TMPDIR MUST be short: multiprocessing.Manager (used by code_eval) binds an AF_UNIX
# socket with a 108-char path limit, so a deep $OUT/tmp path breaks it. Use /tmp.
export TMPDIR=$(mktemp -d /tmp/fpp.XXXXXX)
export HF_MODULES_CACHE="$OUT/hfmod"
_cleanup_tmp() { rm -rf "$TMPDIR" 2>/dev/null; }
trap _cleanup_tmp EXIT

MARGS="model_path=$MODEL_PATH,small_block_size=$SUB,bd_size=$BDSIZE,threshold=$THRESHOLD,selector=$SELECTOR,use_block_cache=$USE_BLOCK_CACHE,max_new_tokens=$MAXNEW,show_speed=True,log_frechet_stats=True"
[ "$DELTA" != "-1" ] && MARGS="$MARGS,frechet_delta=$DELTA"
[ "$FACTOR" != "-1" ] && MARGS="$MARGS,factor=$FACTOR"
LIMARG=""; [ "$LIMIT" != "-1" ] && LIMARG="--limit $LIMIT"

echo "=== RUN $(date -u) GPU=$GPU TASK=$TASK SUB=$SUB SEL=$SELECTOR d=$DELTA f=$FACTOR th=$THRESHOLD ==="
echo "MARGS=$MARGS"
CUDA_VISIBLE_DEVICES=$GPU python eval.py --tasks "$TASK" --num_fewshot "$NFEWSHOT" --batch_size "$BATCH" \
  --confirm_run_unsafe_code --model fast_dllm_v2 --apply_chat_template --fewshot_as_multiturn \
  --model_args "$MARGS" --output_path "$OUT" $LIMARG > "$OUT/run.log" 2>&1
RC=$?
echo "EXIT=$RC OUT=$OUT"
if [ $RC -eq 0 ]; then touch "$OUT/RUN_DONE"; else touch "$OUT/RUN_FAILED"; fi
exit $RC
