#!/usr/bin/env bash
# Phase A sweep dispatcher for Fast-dLLM-v2 + Fréchet on ec2_p5_west (8x H100).
# One eval per GPU, 8-way concurrency via a wait -n job pool. gsm8k is safe to
# parallelize (no code_eval). Completion keyed off each run's RUN_DONE/RUN_FAILED flag.
#
# Usage: sweep_phaseA.sh "gsm8k" 500        # TASKS (space-sep) and LIMIT (-1=full)
set -uo pipefail
TASKS_IN=${1:-gsm8k}
LIMIT=${2:-500}
OUTROOT=${OUTROOT:-~/fastdllmpp_rebuttal/phaseA}
OUTROOT=$(eval echo "$OUTROOT")
RUN=~/fastdllmpp_rebuttal/v2_frechet/run_v2_single.sh
NGPU=${NGPU:-8}
IFS=' ' read -ra GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3 4 5 6 7}"
NGPU=${#GPUS[@]}
SUBS=(4 8 16 32)
DELTAS=(0.00 0.10 0.20 0.25 0.30 0.40 0.50)
mkdir -p "$OUTROOT/logs"
export MAXNEW=${MAXNEW:-256}

# Build job list (each: TASK SUB SELECTOR DELTA FACTOR THRESHOLD NFEWSHOT)
JOBS=()
for T in $TASKS_IN; do
  NF=0; [ "$T" = "mbpp" ] && NF=3
  for S in "${SUBS[@]}"; do
    JOBS+=("$T $S default -1 -1 0.9 $NF")
    JOBS+=("$T $S threshold -1 -1 0.9 $NF")
    JOBS+=("$T $S factor -1 1.0 0.9 $NF")
    for D in "${DELTAS[@]}"; do
      JOBS+=("$T $S frechet $D -1 0.9 $NF")
      JOBS+=("$T $S matched_factor $D -1 0.9 $NF")
    done
  done
done
echo "PhaseA: ${#JOBS[@]} jobs, tasks=[$TASKS_IN] limit=$LIMIT ngpu=$NGPU maxnew=$MAXNEW  $(date -u)"

i=0
for J in "${JOBS[@]}"; do
  read -r T S SEL D F TH NF <<< "$J"
  G=${GPUS[$(( i % NGPU ))]}
  tag="${T}_s${S}_${SEL}_d${D}_f${F}"
  setsid bash "$RUN" "$G" "$T" "$S" "$SEL" "$D" "$F" "$TH" "$NF" "$LIMIT" "$OUTROOT" \
    > "$OUTROOT/logs/${tag}.log" 2>&1 < /dev/null &
  i=$(( i + 1 ))
  # throttle to NGPU concurrent
  while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do wait -n; done
done
wait
echo "PhaseA DONE tasks=[$TASKS_IN]  $(date -u)"
touch "$OUTROOT/PHASEA_${TASKS_IN// /_}_COMPLETE"
