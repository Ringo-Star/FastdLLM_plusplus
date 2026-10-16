#!/bin/bash
# ISSUE-16: complete LLaDA matched grid on MBPP + HumanEval (Kevin box, ~4h window).
# Matched Frechet(delta) vs Factor(f=1-delta), delta in {0.0,0.1,0.2,0.3}, f in {1.0,0.9,0.8,0.7}.
# gen_length 256, block 32; pcache steps=8, dcache steps=256. limit 300.
# ONE run per GPU (per-GPU serial chain, no concurrent runs on a GPU). GPU 3 was reserved by another user on the shared box.
# code_eval isolation: unique SHORT TMPDIR per run (AF_UNIX 108-char limit; thread-117 gotcha).
cd ~/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_code_matched; mkdir -p $BASE/logs
MP="model_path=GSAI-ML/LLaDA-8B-Instruct,show_speed=True"
LIMIT=300

run_one(){ # gpu name task fewshot margs
  local gpu="$1" name="$2" task="$3" fs="$4" margs="$5"
  local log="$BASE/logs/$name.log"; local out="$BASE/$name"
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "SKIP done: $name"; return; fi
  local TD; TD=$(mktemp -d /tmp/fpp.XXXXXX)
  echo "[$(date +%H:%M:%S)] START $name on GPU $gpu (TMPDIR=$TD)"
  CUDA_VISIBLE_DEVICES=$gpu TMPDIR="$TD" $PY eval_llada.py --tasks "$task" --num_fewshot "$fs" \
    --model llada_dist --model_args "$MP,$margs" --limit $LIMIT \
    --confirm_run_unsafe_code --output_path "$out" --log_samples > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name rc=$?"
  rm -rf "$TD" 2>/dev/null || true
}

DTAG=(0p00 0p10 0p20 0p30); DVAL=(0.0 0.1 0.2 0.3)
FTAG=(1p00 0p90 0p80 0p70); FVAL=(1.0 0.9 0.8 0.7)

JOBS=()
# pcache first (primary cache), then dcache; mbpp then humaneval
for cache in pcache dcache; do
  if [ "$cache" = pcache ]; then C="use_cache=True"; STEPS=8; else C="use_cache=True,dual_cache=True"; STEPS=256; fi
  for TASKSPEC in "mbpp:3" "humaneval:0"; do
    task="${TASKSPEC%%:*}"; fs="${TASKSPEC##*:}"
    for i in "${!DVAL[@]}"; do
      d="${DVAL[$i]}"; f="${FVAL[$i]}"; dt="${DTAG[$i]}"; ft="${FTAG[$i]}"
      JOBS+=("${task}_${cache}_frechet_d${dt}|${task}|${fs}|gen_length=256,steps=${STEPS},block_length=32,${C},frechet_margin=${d}")
      JOBS+=("${task}_${cache}_factor_f${ft}|${task}|${fs}|gen_length=256,steps=${STEPS},block_length=32,${C},factor=${f}")
    done
  done
done

GPUS=(0 1 2 4 5 6 7)
declare -A Q
n=0
for spec in "${JOBS[@]}"; do
  g=${GPUS[$((n % ${#GPUS[@]}))]}
  IFS='|' read -r name task fs margs <<< "$spec"
  Q[$g]+="run_one $g ${name} ${task} ${fs} \"${margs}\""$'\n'
  n=$((n+1))
done
echo "Total jobs: ${#JOBS[@]} across ${#GPUS[@]} GPUs (one run per GPU at a time)"
for g in "${GPUS[@]}"; do
  ( eval "${Q[$g]}" ) > "$BASE/logs/gpu_${g}_chain.log" 2>&1 &
done
wait
echo "ALL_CODE_MATCHED_DONE $(date -u)"
