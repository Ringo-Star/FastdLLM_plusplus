#!/bin/bash
# ISSUE-15 commit-level calibration battery. LLaDA-8B, NO-CACHE (instrumented path),
# gen_length=256 block=32, limit 200, calib_verify=True + per-run trace_jsonl.
# Per task: raw Frechet d=0.25 at eta in {0.00,0.04,0.08} (Exp A/B/C/E), factor f=1.0,
# matched factor f=0.75. HumanEval jobs serialized on one GPU (code_eval collision).
# Idle H100s only: 0,1,2,4,5,6,7. GPU 3 was reserved by another user on the shared box.
cd ~/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/calib_runs; mkdir -p $BASE/logs
MP="model_path=GSAI-ML/LLaDA-8B-Instruct,show_speed=True"
COMMON="gen_length=256,steps=8,block_length=32,calib_verify=True"
LIMIT=200

run_one(){ # gpu name task fewshot code margs
  local gpu="$1" name="$2" task="$3" fewshot="$4" code="$5" margs="$6"
  local out="$BASE/$name"; local log="$BASE/logs/$name.log"
  if [ -f "$out/DONE" ]; then echo "SKIP done: $name"; return; fi
  mkdir -p "$out/tmp"
  local codeflag=""; [ "$code" = "1" ] && codeflag="--confirm_run_unsafe_code"
  echo "[$(date +%H:%M:%S)] START $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu TMPDIR="$out/tmp" HF_MODULES_CACHE="$out/tmp/hfmod" \
    $PY eval_llada.py --tasks $task --num_fewshot $fewshot \
    --model llada_dist \
    --model_args "$MP,$COMMON,trace_jsonl=$out/trace.jsonl,$margs" \
    --limit $LIMIT $codeflag --output_path "$out" --log_samples > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] FINISH $name rc=$rc"
  if [ $rc -eq 0 ] && [ -s "$out/trace.jsonl" ]; then touch "$out/DONE"; fi
}

# job spec: name|task|fewshot|code|margs
NONCODE=()   # gsm8k + mbpp (mbpp code but parallel-safe with unique TMPDIR)
CODE=()      # humaneval (serialized)
for T in gsm8k mbpp humaneval; do
  case $T in
    gsm8k) fs=5; code=0;;
    mbpp)  fs=3; code=1;;
    humaneval) fs=0; code=1;;
  esac
  for e in 0p00 0p04 0p08; do
    ev=$(echo $e | sed 's/p/./')
    spec="calib_${T}_nocache_frechet_d0p25_eta${e}|$T|$fs|$code|frechet_margin=0.25,frechet_eta=${ev}"
    if [ "$T" = "humaneval" ]; then CODE+=("$spec"); else NONCODE+=("$spec"); fi
  done
  # baselines
  for b in "factor_f1p00|factor=1.0" "matched_f0p75|factor=0.75"; do
    bn="${b%%|*}"; bm="${b##*|}"
    spec="calib_${T}_nocache_${bn}|$T|$fs|$code|$bm"
    if [ "$T" = "humaneval" ]; then CODE+=("$spec"); else NONCODE+=("$spec"); fi
  done
done

GPUS=(0 1 2 4 5 6)   # humaneval reserved to GPU 7 (serial)
declare -A Q
n=0
for spec in "${NONCODE[@]}"; do
  g=${GPUS[$((n % ${#GPUS[@]}))]}
  IFS='|' read -r name task fs code margs <<< "$spec"
  Q[$g]+="run_one $g \"$name\" \"$task\" \"$fs\" \"$code\" \"$margs\""$'\n'
  n=$((n+1))
done
# all humaneval on GPU 7, serial
for spec in "${CODE[@]}"; do
  IFS='|' read -r name task fs code margs <<< "$spec"
  Q[7]+="run_one 7 \"$name\" \"$task\" \"$fs\" \"$code\" \"$margs\""$'\n'
done

echo "Total jobs: $(( ${#NONCODE[@]} + ${#CODE[@]} )) (noncode ${#NONCODE[@]} on GPUs ${GPUS[*]}, humaneval ${#CODE[@]} serial on GPU 7)"
for g in 0 1 2 4 5 6 7; do
  [ -n "${Q[$g]}" ] && ( eval "${Q[$g]}" ) > "$BASE/logs/gpu_${g}_chain.log" 2>&1 &
done
wait
echo "ALL_CALIB_DONE $(date -u)"
