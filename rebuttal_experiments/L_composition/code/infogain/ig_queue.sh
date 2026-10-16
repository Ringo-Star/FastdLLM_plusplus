#!/bin/bash
# Info-Gain breadth runs, GPU3, sequential queue (single GPU => serialized so
# wall-time is uncontaminated). Each run: own log, own results dir, DONE sentinel.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate infogain
cd ~/baselines2/infogain
export HF_HOME=~/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3

run_one () {
  local name="$1"; shift
  local outdir="results_ig/${name}"
  mkdir -p "$outdir"
  if [ -f "$outdir/DONE" ]; then echo "[queue] $name already DONE, skip"; return; fi
  echo "[queue] START $name  $(date -u +%FT%TZ)"
  python run.py --config configs/gsm8k_info_gain.yaml \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --use_cache none --max_samples 300 \
    --result_path "$outdir/gsm8k.txt" \
    "$@" > "logs/${name}.log" 2>&1
  local rc=$?
  echo "RC=$rc" >> "logs/${name}.log"
  if [ $rc -eq 0 ]; then touch "$outdir/DONE"; else touch "$outdir/FAILED"; fi
  echo "[queue] END $name rc=$rc  $(date -u +%FT%TZ)"
}

# (a) HEAD-ON: native Info-Gain, author-default hyperparams (candidate_number 8,
#     position_temperature 0.2, threshold 0.8, temp 0, gen256/steps256/block32).
#     use_cache none because in this public dllm branch use_cache=prefix
#     delegates to plain Fast-dLLM (info-gain not exercised) -- verified by
#     the repo's own test_infogain_fastdllm_cache_gpu.py assertion.
run_one ig_native_headon

# (b) FRECHET-INSERTION: same loop, selection rule replaced by the Frechet
#     certificate; threshold bypass disabled (certificate decides alone).
for d in 0.1 0.2 0.25 0.3 0.4; do
  run_one "ig_frechet_d${d}" --frechet_margin "$d" --threshold null
done

echo "[queue] ALL DONE $(date -u +%FT%TZ)"
touch results_ig/QUEUE_DONE
