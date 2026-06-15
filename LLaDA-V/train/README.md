# LLaDA-V model package (`llava`)

Inference-only subset of the [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) /
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) model package, kept here so the
multimodal evaluation in [`../eval`](../eval) can load `GSAI-ML/LLaDA-V`.

The Fréchet / Fast-dLLM++ parallel-decoding logic for the multimodal model lives in
[`llava/hooks/`](llava/hooks/) (`fast_dllm_hook.py`, `cache_hook_LLaDA_V.py`) and is
enabled from the eval scripts via the `use_fast_dllm=true` model argument.

Install (editable) as part of the LLaDA-V environment setup — see
[`../README.md`](../README.md):

```bash
pip install -e ".[train]"
```
