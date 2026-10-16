#!/bin/bash
# sp6 ME-DLM GSM8K queue: head-on (author quota default) + Frechet-insertion delta {0.1,0.25,0.4}
# Sequential on GPU6 only. Each run keys completion off run_meta.json; DONE sentinel at end.
set -uo pipefail
exec > ~/baselines2/ME-DLM/sp6_run.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate medlm
export HF_HOME=~/hf_cache
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/baselines2/ME-DLM
CKPT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('renhouxing/ME-DLM-Stage3'))")
OUT=~/baselines2/ME-DLM/results_stage3
mkdir -p "$OUT"
echo "CKPT=$CKPT"
PORT=29631

run_one () {
  local tag="$1"; shift
  local outdir="$1"; shift
  if [ -f "$outdir/run_meta.json" ]; then
    echo "[SKIP] $tag already has $outdir/run_meta.json"
    return 0
  fi
  echo "=== [$tag] start $(date -u) ==="
  torchrun --nproc_per_node 1 --master_port $PORT test_sp6.py \
    --path "$CKPT" --out_root "$OUT" --task gsm8k --mask_diffusion_step 48 --edit_diffusion_step 16 "$@"
  local rc=$?
  echo "=== [$tag] rc=$rc end $(date -u) ==="
}

# (a) head-on, author default quota schedule, full GSM8K, 1 rep
run_one quota    "$OUT/sp6_results-quota-n_512-mds_48-eds_16-limit_0"
# (b) Frechet insertion, reduced delta set
run_one f0.1     "$OUT/sp6_results-frechet0.1-n_512-mds_48-eds_16-limit_0"  --frechet_margin 0.1
run_one f0.25    "$OUT/sp6_results-frechet0.25-n_512-mds_48-eds_16-limit_0" --frechet_margin 0.25
run_one f0.4     "$OUT/sp6_results-frechet0.4-n_512-mds_48-eds_16-limit_0"  --frechet_margin 0.4

date -u > ~/baselines2/ME-DLM/SP6_DONE
echo "=== ALL DONE $(date -u) ==="
