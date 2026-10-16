#!/bin/bash
# Frechet INSERTION into Dynamic-dLLM full-native pipeline (cache ON + PD).
# Pipeline 1 of the corrected insertion design. GPU 1 ONLY.
#
# Usage: run_insertion_dyn.sh gate            -> limit-8 validation pair (native, frechet 0.25)
#        run_insertion_dyn.sh one <tag> <margin|native> <limit>  -> single run (used by screens)
#        run_insertion_dyn.sh sweep           -> baseline_cacheon + deltas, sequential via screens
set -u
export HF_HOME=$HOME/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$HOME/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
OUT=$HOME/results/insertion_dyn
PY=$HOME/miniforge3/envs/dyndllm/bin/python
mkdir -p $OUT

gpu1_guard() {
  # Abort if any process is on GPU 1 (never co-locate, never touch others' GPUs).
  local procs
  procs=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | \
          grep -f <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==1{print $2}') | wc -l)
  if [ "$procs" -gt 0 ]; then
    echo "[guard] GPU 1 is BUSY, refusing to launch" >&2
    return 1
  fi
  return 0
}

run_single() { # $1=tag $2=margin(native|<float>) $3=limit
  local tag="$1" margin="$2" limit="$3"
  local ts=$(date -u +%Y%m%dT%H%M%SZ)
  local outdir="$OUT/${tag}_${ts}"
  mkdir -p "$outdir"
  gpu1_guard || { touch "$outdir/FAILED_${tag}_gpubusy"; return 1; }
  local margs="pretrained=$SNAP,is_feature_cache=True,window_size=32,layer_budget=32,select_from=v,prompt_interval_steps=5000,gen_interval_steps=5000,generate_mode=pd,pd_mode=1"
  if [ "$margin" != "native" ]; then
    margs="$margs,frechet_margin=$margin"
  fi
  echo "[run] START $tag margin=$margin limit=$limit at $(date -u +%FT%TZ) outdir=$outdir"
  local t0=$SECONDS
  ( cd $HOME/baselines/DYNAMIC-DLLM && \
    CUDA_VISIBLE_DEVICES=1 $PY evaluation_script.py -m lm_eval --model LLaDA \
      --model_args "$margs" \
      --gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0" \
      --include_path ./lm_eval_tasks --tasks gsm8k_local --num_fewshot 4 \
      --batch_size 1 --limit "$limit" \
      --output_path "$outdir" --log_samples --apply_chat_template --fewshot_as_multiturn \
  ) > "$outdir/run.log" 2>&1
  local rc=$?
  local wall=$((SECONDS - t0))
  echo "WALL_SECONDS:$wall" > "$outdir/wall_time.txt"
  echo "[run] END $tag rc=$rc wall=${wall}s"
  if [ $rc -eq 0 ]; then touch "$outdir/DONE_${tag}"; else touch "$outdir/FAILED_${tag}_rc${rc}"; fi
  return $rc
}

wait_screen() { # $1 = screen name; poll until it exits
  while screen -ls | grep -q "\.$1[[:space:]]"; do sleep 30; done
}

case "${1:-}" in
  one)
    run_single "$2" "$3" "$4"
    ;;
  gate)
    run_single gate_native8 native 8
    run_single gate_frechet025_8 0.25 8
    ;;
  sweep)
    # sequential chain on GPU 1: cache-ON native baseline then frechet deltas
    for spec in "baseline_cacheon:native" "d010:0.1" "d020:0.2" "d025:0.25" "d030:0.3" "d040:0.4"; do
      tag="${spec%%:*}"; margin="${spec##*:}"
      screen -dmS "mx_ins_dyn_${tag}" bash "$HOME/run_insertion_dyn.sh" one "$tag" "$margin" 300
      sleep 10
      wait_screen "mx_ins_dyn_${tag}"
    done
    echo "[sweep] ALL DONE at $(date -u +%FT%TZ)"
    touch "$OUT/DONE_SWEEP_ALL"
    ;;
  *)
    echo "usage: $0 gate | one <tag> <margin|native> <limit> | sweep"; exit 2
    ;;
esac
