#!/bin/bash
# Sequential queue on GPU4: wall time is a primary metric, so runs never share the GPU.
set -u
cd ~/baselines2/soar
for spec in "full300_native native" "full300_f010 0.10" "full300_f020 0.20" "full300_f025 0.25" "full300_f030 0.30" "full300_f040 0.40"; do
  set -- $spec
  TAG=$1; DELTA=$2
  if [ -f results/$TAG/DONE ]; then echo "SKIP $TAG (DONE exists)"; continue; fi
  echo "QUEUE_START $TAG $(date -u +%FT%TZ)"
  ./run_soar.sh $TAG 300 $DELTA &> logs/$TAG.log
  echo "QUEUE_END $TAG $(date -u +%FT%TZ)"
done
echo ALL_DONE > results/QUEUE_ALL_DONE
