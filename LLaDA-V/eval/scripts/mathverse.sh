# MathVerse scoring uses a GPT judge. Export your own key before running:
#   export OPENAI_API_KEY=<your_openai_api_key>

accelerate launch --num_processes=1 --main_process_port 23456 -m lmms_eval \
--model llava_onevision_llada \
--gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":64,"block_length":64,"gen_steps":32,"think_mode":"think","threshold": 1, "prefix_refresh_interval": 32}' \
--model_args pretrained=GSAI-ML/LLaDA-V,conv_template=llava_llada,model_name=llava_llada,use_fast_dllm=true,load_4bit=false \
--tasks mathverse_testmini_vision \
--batch_size 1 \
--log_samples \
--log_samples_suffix mathverse_testmini_vision \
--output_path exp/llava_v_eval/LLaDA-V
