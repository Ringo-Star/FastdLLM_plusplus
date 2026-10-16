#!/bin/bash
# n=300 Frechet chain per user decision 2026-08-03 01:12Z. GPU 6, port 29642.
# Verbatim sp6_run.sh config except --limit 300 and delta set.
set -uo pipefail
exec > ~/baselines2/ME-DLM/medlm300_g6.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate medlm
export HF_HOME=~/hf_cache
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/baselines2/ME-DLM
CKPT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('renhouxing/ME-DLM-Stage3'))")
OUT=~/baselines2/ME-DLM/results_stage3
PORT=29642
run_one () {
  local tag="$1"; shift
  local outdir="$1"; shift
  if [ -f "$outdir/run_meta.json" ]; then echo "[SKIP] $tag"; return 0; fi
  echo "=== [$tag] start $(date -u) ==="
  torchrun --nproc_per_node 1 --master_port $PORT test_sp6.py \
    --path "$CKPT" --out_root "$OUT" --task gsm8k --mask_diffusion_step 48 --edit_diffusion_step 16 --limit 300 "$@"
  local rc=$?
  echo "=== [$tag] rc=$rc end $(date -u) ==="
  if [ $rc -eq 0 ] && [ -f "$outdir/run_meta.json" ]; then date -u > "$outdir/DONE"; fi
}
run_one f0.2_300  "$OUT/sp6_results-frechet0.2-n_512-mds_48-eds_16-limit_300"  --frechet_margin 0.2
run_one f0.3_300  "$OUT/sp6_results-frechet0.3-n_512-mds_48-eds_16-limit_300"  --frechet_margin 0.3
date -u > ~/baselines2/ME-DLM/MEDLM300_G6_DONE
echo "=== G6 CHAIN DONE $(date -u) ==="
