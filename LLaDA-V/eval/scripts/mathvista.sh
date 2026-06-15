#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLADA_V_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LLADA_V_ROOT/.." && pwd)"

export PYTHONPATH="$LLADA_V_ROOT/train:$LLADA_V_ROOT/eval/lmms-eval:${PYTHONPATH:-}"

if [[ "${HF_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

MODEL_PATH="${LLADA_V_MODEL_PATH:-GSAI-ML/LLaDA-V}"
MODEL="llava_onevision_llada"
MODEL_NAME="llava_llada"
CONV_TEMPLATE="llava_llada"
TASK_NAME="${TASK_NAME:-mathvista_testmini}"
GPU_ID="${GPU_ID:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-23456}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USE_FAST_DLLM="${USE_FAST_DLLM:-false}"

GEN_KWARGS="${GEN_KWARGS:-{\"temperature\":0,\"cfg\":0,\"remasking\":\"low_confidence\",\"gen_length\":96,\"block_length\":96,\"gen_steps\":48,\"think_mode\":\"think\"}}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/exp/llava_v_eval/LLaDA-V/mathvista}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-${TASK_NAME}}"
LOG_DIR="${LOG_DIR:-$OUTPUT_PATH/logs}"
LOG_PATH="$LOG_DIR/${TASK_NAME}.log"

MODEL_ARGS="pretrained=$MODEL_PATH,conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME,load_4bit=false,use_fast_dllm=$USE_FAST_DLLM"

mkdir -p "$OUTPUT_PATH" "$LOG_DIR"

if [[ -z "${OPENAI_API_KEY:-}" && -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
    echo "Warning: MathVista scoring uses a GPT evaluator. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY before running if you need final scores." >&2
fi

echo "Running LLaDA-V MathVista 4-bit evaluation"
echo "Model path: $MODEL_PATH"
echo "Task: $TASK_NAME"
echo "GPU: $GPU_ID"
echo "Output path: $OUTPUT_PATH"
echo "Log path: $LOG_PATH"
echo "Generation args: $GEN_KWARGS"
echo "Model args: $MODEL_ARGS"
echo "----------------------------------------"

CMD=(
    accelerate launch
    --num_processes=1
    --main_process_port "$MAIN_PROCESS_PORT"
    -m lmms_eval
    --model "$MODEL"
    --gen_kwargs "$GEN_KWARGS"
    --model_args "$MODEL_ARGS"
    --tasks "$TASK_NAME"
    --batch_size "$BATCH_SIZE"
    --log_samples
    --log_samples_suffix "$LOG_SAMPLES_SUFFIX"
    --output_path "$OUTPUT_PATH"
)

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
