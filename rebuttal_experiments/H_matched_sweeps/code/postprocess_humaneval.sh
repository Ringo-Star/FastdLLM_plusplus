#!/bin/bash
# Postprocess (sanitize + re-score) all HumanEval matched runs -> corrected pass@1.
# Parallel across CPU (code_eval is CPU-bound), each with its own short TMPDIR (AF_UNIX 108-char).
cd ~/fastdllmpp_rebuttal/fast-dllm/v1/llada
export HF_ALLOW_CODE_EVAL=1
PY=~/envs/fast-dllm-eval/bin/python
OUT=~/humaneval_postproc.txt; : > "$OUT"
for f in $(find ~/sweep_code_matched_west -path "*humaneval*" -name "samples_*.jsonl" | sort); do
  rid=$(echo "$f" | sed 's#.*/sweep_code_matched_west/##; s#/.*##')
  (
    TD=$(mktemp -d /tmp/pp.XXXXXX)
    acc=$(TMPDIR="$TD" $PY postprocess_code.py "$f" 2>/dev/null | tail -1)
    echo "$rid $acc" >> "$OUT"
    rm -rf "$TD"
  ) &
done
wait
echo "POSTPROC_DONE $(date -u)" >> "$OUT"
