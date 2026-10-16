
#!/bin/bash

lengths=(256 512)
block=32
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
script=eval_llada.py

mkdir logs

for length in "${lengths[@]}"; do
    python $script --tasks humaneval --model llada_dist \
        --confirm_run_unsafe_code \
        --trust_remote_code \
        --num_few_shot 0 \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=$length,steps=$length,block_length=$block,low_cpu_mem_usage=True,device_map='auto',torch_dtype=torch.bfloat16,enable_soar=True \
        &> logs/humaneval.instruct.len${length}.soar.log

    python $script --tasks mbpp --model llada_dist \
        --confirm_run_unsafe_code \
        --trust_remote_code \
        --num_few_shot 0 \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=$length,steps=$length,block_length=$block,low_cpu_mem_usage=True,device_map='auto',torch_dtype=torch.bfloat16,enable_soar=True \
        &> logs/mbpp.instruct.len${length}.soar.log

    python $script --tasks gsm8k_zeroshot_cot --model llada_dist \
        --confirm_run_unsafe_code \
        --trust_remote_code \
        --num_few_shot 0 \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=$length,steps=$length,block_length=$block,low_cpu_mem_usage=True,device_map='auto',torch_dtype=torch.bfloat16,enable_soar=True \
        &> logs/gsm8k.instruct.len${length}.soar.log
    
done