#!/bin/bash
# ISSUE-16 RE-RUN on ec2_p5_west (window to Jul 29). LLaDA MBPP + HumanEval matched grid.
# Matched Frechet(delta) vs Factor(f=1-delta), delta in {0.0,0.1,0.2,0.3}, f in {1.0,0.9,0.8,0.7}.
# gen 256, block 32; pcache steps=8, dcache steps=256. limit 300. NO no-cache.
# ONE dedicated GPU (serial), shared-box-safe. Default flags => decoding identical to Kevin runs
# (frechet_eta=0 no-op, calib_verify off). code_eval: unique short TMPDIR per run.
GPU="${GPU:-0}"
cd ~/fastdllmpp_rebuttal/fast-dllm/v1/llada
export HF_HOME=~/.cache/huggingface HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
PY=~/envs/fast-dllm-eval/bin/python
BASE=~/sweep_code_matched_west; mkdir -p $BASE/logs
MP="model_path=GSAI-ML/LLaDA-8B-Instruct,show_speed=True"
LIMIT=300

run_one(){ # name task fewshot margs
  local name="$1" task="$2" fs="$3" margs="$4"
  local log="$BASE/logs/$name.log"; local out="$BASE/$name"
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "SKIP done: $name"; return; fi
  local TD; TD=$(mktemp -d /tmp/fpp.XXXXXX)
  echo "[$(date -u +%H:%M:%S)] START $name on GPU $GPU (TMPDIR=$TD)"
  CUDA_VISIBLE_DEVICES=$GPU TMPDIR="$TD" $PY eval_llada.py --tasks "$task" --num_fewshot "$fs" \
    --model llada_dist --model_args "$MP,$margs" --limit $LIMIT \
    --confirm_run_unsafe_code --output_path "$out" --log_samples > "$log" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE $name rc=$?"
  rm -rf "$TD" 2>/dev/null || true
}

DTAG=(0p00 0p10 0p20 0p30); DVAL=(0.0 0.1 0.2 0.3)
FTAG=(1p00 0p90 0p80 0p70); FVAL=(1.0 0.9 0.8 0.7)

# pcache first (faster, primary cache), then dcache; mbpp then humaneval
for cache in pcache dcache; do
  if [ "$cache" = pcache ]; then C="use_cache=True"; STEPS=8; else C="use_cache=True,dual_cache=True"; STEPS=256; fi
  for TASKSPEC in "mbpp:3" "humaneval:0"; do
    task="${TASKSPEC%%:*}"; fs="${TASKSPEC##*:}"
    for i in "${!DVAL[@]}"; do
      d="${DVAL[$i]}"; f="${FVAL[$i]}"; dt="${DTAG[$i]}"; ft="${FTAG[$i]}"
      run_one "${task}_${cache}_frechet_d${dt}" "$task" "$fs" "gen_length=256,steps=${STEPS},block_length=32,${C},frechet_margin=${d}"
      run_one "${task}_${cache}_factor_f${ft}"  "$task" "$fs" "gen_length=256,steps=${STEPS},block_length=32,${C},factor=${f}"
    done
  done
done
echo "ALL_CODE_MATCHED_WEST_DONE $(date -u)"
