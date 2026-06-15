# Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference

> Training-free parallel decoding for diffusion LLMs that commits more tokens per denoising step by using the **full sorted confidence profile** — not just the weakest token.

[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2606.02955)

Diffusion LLMs (dLLMs) can decode many tokens in parallel, but inference is bottlenecked by deciding *which* masked tokens can be committed together. [Fast-dLLM](https://arxiv.org/abs/2505.22618) solved this with KV caching and a confidence **factor** rule that accepts a candidate set of size *n* when `(n+1)(1 − c₍ₙ₎) < f`, where `c₍ₙ₎` is the **weakest** confidence among the top-*n* tokens. That rule assumes all selected tokens share the same confidence — conservative when, in practice, the confidence profile is highly heterogeneous.

**Fast-dLLM++** replaces the weakest-token selector with **Fréchet profile decoding**, which uses the entire sorted profile `(c₍₁₎, …, c₍ₙ₎)`:

- **Fréchet lower bound** on joint correctness: `Lₙ = max(0, Σⱼ c₍ⱼ₎ − (n−1))`
- **Competing-event upper bound:** `Uₙ = 1 − c₍ₙ₎`
- **Fréchet score:** `Gₙ = Lₙ − Uₙ`

Given a margin `δ ≥ 0`, it commits the largest prefix with `Gₙ > δ`.

**Why it works.** If `Lₙ > Uₙ`, the parallel greedy commit provably matches the true joint greedy decision. In the equal-confidence case it reduces *exactly* to factor decoding under `f = 1 − δ`, so it is a strict generalization. The score decomposes into the factor core plus a non-negative *heterogeneity bonus* `Bₙ = Σⱼ<ₙ (c₍ⱼ₎ − c₍ₙ₎)` — the profile information the weakest-token rule discards — which is what lets it commit more tokens per step. The model, diffusion schedule, and cache are untouched: the only change is the token-selection rule (sorting + prefix sums over the active block, no extra parameters or memory), making it a true drop-in replacement.

## Key results (Same-Hardware Comparisons)

All comparisons use our own reproductions of threshold/factor on the same GPU.

### LLaDA-8B-Instruct, GSM8K 256, Prefix Cache (L40S)

| Method | Accuracy | Tok/s | NFE |
|--------|----------|-------|-----|
| Threshold=0.9 | 78.47% | 53.6 | 107,053 |
| Factor=1.0 | 78.32% | 67.9 | 79,387 |
| **Fréchet d=0.30** | **78.54%** | **70.1** | **75,448** |

### LLaDA-8B-Instruct, HumanEval 256, Prefix Cache (L40S)

| Method | pass@1 | Tok/s | NFE |
|--------|--------|-------|-----|
| Threshold=0.9 | 40.24% | 70.3 | 13,930 |
| **Fréchet d=0.25** | **40.85%** | **96.1** | **9,771** |

### LLaDA-8B-Instruct, GSM8K 1024, Dual Cache (A100)

| Method | Accuracy | Tok/s | NFE |
|--------|----------|-------|-----|
| Factor=1.0 | 74.45% | 21.6 | 118,581 |
| **Fréchet d=0.02** | **74.98%** | **22.7** | **101,002** |

## Installation

```bash
git clone https://github.com/Ringo-Star/FastdLLM_plusplus
cd FastdLLM_plusplus
pip install -r v1/requirements.txt # LLaDA/Dream text evaluation
```

Models download automatically from Hugging Face (`GSAI-ML/LLaDA-8B-Instruct`, `Dream-org/Dream-v0-Base-7B`). Multimodal experiments use a separate environment — see [LLaDA-V/README.md](LLaDA-V/README.md).

## Usage

Cache regimes: add `use_cache=True` for **PrefixCache**, or `use_cache=True,dual_cache=True` for **DualCache** (dual cache uses `steps=gen_length`).

### LLaDA-8B

```bash
cd v1/llada
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true

accelerate launch eval_llada.py \
  --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist \
  --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=8,block_length=32,use_cache=True,frechet_margin=0.25,show_speed=True" \
  --output_path results/gsm8k_pcache_frechet
```

Swap `--tasks` for `minerva_math` (MATH), `humaneval`, or `mbpp`.

### Dream-7B

```bash
cd v1/dream
accelerate launch eval.py \
  --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model dream \
  --model_args "pretrained=Dream-org/Dream-v0-Base-7B,max_new_tokens=256,diffusion_steps=256,block_length=32,use_cache=True,alg=confidence_threshold,frechet_margin=0.25,show_speed=True" \
  --output_path results/dream_gsm8k_frechet
```

### Sanity check

Verufy that Fréchet recovers factor decoding in the equal-confidence case and stays within bounds:

```bash
python v1/llada/test_frechet_rule.py
```

### Multimodal — LLaDA-V

See [LLaDA-V/README.md](LLaDA-V/README.md) for environment setup and the MathVista / MathVerse scripts.

## Choosing the margin δ

Smaller `δ` commits more tokens per step (faster, riskier); larger `δ` is more conservative. `δ = 0.25` is the global default used in the main tables (`f = 1 − δ` maps to Fast-dLLM's factor).

| Task | Recommended δ | Notes
|---|---|---|
| GSM8K | 0.25 – 0.30 | Conservative preserves accuracy |
| HumanEval | 0.20 – 0.25 | Code needs conservative margins |
| MBPP | 0.02 – 0.20 | Simpler code tolerates aggressive |
| MATH | 0.02 – 0.30 | Margin-insensitive |

## Citation

```bibtex
@inproceedings{kasa2026fastdllmpp,
  title = {Fast-dLLM++: Fr\'{e}chet Profile Decoding for Faster Diffusion LLM Inference},
  author = {Kasa, Siva Rajesh and Dai, Yasong and Negi, Sumit and Li, Hongdong},
  booktitle = {ICML 2026 Workshop on Structured Probabilistic Inference and Generative Modeling (SPIGM)},
  year = {2026},
  eprint = {2606.02955},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2606.02955}
}
```

## License & acknowledgements

Released under the Apache 2.0 License — see [LICENSE](LICENSE). This work builds on [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), [LLaDA](https://github.com/ML-GSAI/LLaDA), [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), and [Dream](https://github.com/HKUNLP/Dream). We thank the authors for releasing their code and models.
