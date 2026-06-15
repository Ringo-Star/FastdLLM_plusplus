# Fast-dLLM++ on Multimodal LLaDA-V

Multimodal evaluation for Fast-dLLM++, reproducing the **MathVista / MathVerse** results with [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), a diffusion vision-language model.
Evaluation uses the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) harness. The model is downloaded automatically from Hugging Face (`GSAI-ML/LLaDA-V`).

## Setup

```bash
cd LLaDA-V
bash init_env.sh   # installs eval/lmms-eval and train/ (the llava package)
pip install levenshtein loguru
```

## Run evaluation

MathVista / MathVerse, with Fast-dLLM++ enabled (`use_fast_dllm=true`); the Fréchet margin and other
decoding controls are passed through `--gen_kwargs`:

```bash
cd eval

bash scripts/mathvista.sh     # MathVista (single GPU)
bash scripts/mathverse.sh               # MathVerse
```

MathVista is scored by a GPT evaluator and MathVerse by an offline GPT judge, both of which read
`OPENAI_API_KEY` from the environment — export your own key, do **not** commit it:

```bash
export OPENAI_API_KEY=<your_openai_api_key>

# score MathVerse submissions
python scripts/mathverse_offline_judge.py --workers 2 \
       --input exp/llava_v_eval/LLaDA-V/submissions/mathverse_testmini_vision_results.json
```

`scripts/evaluate.sh` is a general multi-task runner if you want to evaluate on the broader lmms-eval
task suite.

## Acknowledgements & citation

The model package and harness are based on [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V),
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT),
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and
[Fast-dLLM](https://github.com/NVlabs/Fast-dLLM). We thank the authors for their work.

```bibtex
@article{you2025llada,
  title={LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning},
  author={You, Zebin and Nie, Shen and Zhang, Xiaolu and Hu, Jun and Zhou, Jun and Lu, Zhiwu and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2505.16933},
  year={2025}
}
```
