#!/usr/bin/env python3
"""Insert Fréchet-profile commit rule into Swordsman prefix-cache adaptive path.

Replaces ONLY the accept/commit rule over the per-step candidate confidences in
generate_with_prefix_cache_adaptive. Model, passes, KV cache handling, and the
entropy-driven block partition (determine_next_block_length) are untouched.
"""
import re, sys

LLADA = "$ROOT/baselines/SWORDSMAN/code_unzipped/Swordsman/llada"

# ---- 1. generate.py ----
gp = f"{LLADA}/generate.py"
src = open(gp).read()

# 1a. Append the anchor function verbatim.
frechet_fn = open("/tmp/frechet_fn.py").read()
if "get_transfer_index_frechet_profile" not in src:
    src += "\n\n# ==== Fréchet-profile commit rule (ported VERBATIM from $ROOT/anchors/generate.py L177-245) ====\n" + frechet_fn

# 1b. Add frechet_margin kwarg to generate_with_prefix_cache_adaptive signature.
old_sig = """    dynamic_threshold_method=None, base_threshold=0.9, min_threshold=0.7,confidence_boost=1.0,
    threshold=None, factor=None, prompt_id="0"
):
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)"""
new_sig = """    dynamic_threshold_method=None, base_threshold=0.9, min_threshold=0.7,confidence_boost=1.0,
    threshold=None, factor=None, frechet_margin=None, prompt_id="0"
):
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)"""
assert src.count(old_sig) == 1, f"sig match count {src.count(old_sig)}"
src = src.replace(old_sig, new_sig)

# 1c. Replace the commit dispatch inside generate_with_prefix_cache_adaptive only.
old_dispatch = """            if factor is None:
                x0, transfer_index = get_transfer_index_origin(logits, temperature, remasking, mask_index, 
                                                                x[:, current_idx:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                                x[:, current_idx:], None, factor)
            x[:, current_idx:][transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, current_idx:block_end] == mask_id).sum() == 0:
                break
            if i >= current_steps and threshold is None:
                break
        current_idx = block_end
    return x, nfe

@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,"""
new_dispatch = """            if frechet_margin is not None:
                x0, transfer_index = get_transfer_index_frechet_profile(logits, temperature, remasking, mask_index,
                                                                x[:, current_idx:], margin=frechet_margin)
            elif factor is None:
                x0, transfer_index = get_transfer_index_origin(logits, temperature, remasking, mask_index, 
                                                                x[:, current_idx:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                                x[:, current_idx:], None, factor)
            x[:, current_idx:][transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, current_idx:block_end] == mask_id).sum() == 0:
                break
            if i >= current_steps and threshold is None and frechet_margin is None:
                break
        current_idx = block_end
    return x, nfe

@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,"""
assert src.count(old_dispatch) == 1, f"dispatch match count {src.count(old_dispatch)}"
src = src.replace(old_dispatch, new_dispatch)
open(gp, "w").write(src)

# ---- 2. eval_llada.py ----
ep = f"{LLADA}/eval_llada.py"
src = open(ep).read()

old = "        threshold=None,\n        factor=None,\n"
assert src.count(old) == 1
src = src.replace(old, "        threshold=None,\n        factor=None,\n        frechet_margin=None,\n")

old = "        self.adaptive = adaptive \n"
assert src.count(old) == 1
src = src.replace(old, "        self.adaptive = adaptive \n        self.frechet_margin = float(frechet_margin) if frechet_margin is not None else None\n")

old = """                        generated_answer, nfe = generate_with_prefix_cache_adaptive(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, 
                            min_block_len=self.min_block_len, max_block_len=self.max_block_len,metric=self.metric,topk_entropy_k=self.topk_entropy_k,
                            dynamic_threshold_method=self.dynamic_threshold_method,base_threshold=self.base_threshold, min_threshold=self.min_threshold,
                            temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)  """
new = """                        generated_answer, nfe = generate_with_prefix_cache_adaptive(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, 
                            min_block_len=self.min_block_len, max_block_len=self.max_block_len,metric=self.metric,topk_entropy_k=self.topk_entropy_k,
                            dynamic_threshold_method=self.dynamic_threshold_method,base_threshold=self.base_threshold, min_threshold=self.min_threshold,
                            temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor, frechet_margin=self.frechet_margin)  """
assert src.count(old) == 1
src = src.replace(old, new)
open(ep, "w").write(src)

import ast
ast.parse(open(gp).read()); ast.parse(open(ep).read())
print("PATCH OK")
