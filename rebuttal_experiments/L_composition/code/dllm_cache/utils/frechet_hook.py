# Fréchet / factor commit-selector hook for dLLM-Cache.
#
# ADDITIVITY EXPERIMENT (ISSUE-07): this module inserts the Fast-dLLM++ token-commit
# selectors (matched-factor and Fréchet profile) into dLLM-Cache's LLaDA generation
# loop WITHOUT touching the caching machinery. Every model(...) call is identical to
# dLLM-Cache's own generate(); only the choice of WHICH masked positions to commit
# each step changes. The dLLMCache feature-cache hooks (registered on the model)
# therefore behave exactly as in the base cache arm: one model call per iteration,
# same refresh cadence: so any speed/NFE difference is attributable to the selector,
# not to a change in caching. This is the clean cache-vs-selector additivity test.
#
# Selectors are ported verbatim (math-identical) from the paper's
# Fast-dLLM/v1/llada/generate.py (get_transfer_index_dynamic, get_transfer_index_frechet_profile).

import torch
import numpy as np
import torch.nn.functional as F
from dllm_cache.cache import dLLMCache


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits.exp()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, factor=1.0):
    """Fast-dLLM matched-factor rule. Commit largest prefix n of sorted confidences
    with c_(top_i) >= 1 - factor/(top_i+1). Reduces to factor decoding of the paper."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_masked = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        num_tokens = int(num_masked[j].item())
        if num_tokens == 0:
            continue
        ns = list(range(1, num_tokens + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]
        threshs[0] = -1  # at least one token
        sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
        top_i = 0
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break
        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1
        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True
    return x0, transfer_index, int(transfer_index.sum().item())


def get_transfer_index_frechet_profile(logits, temperature, remasking, mask_index, x, margin=0.0):
    """Fréchet-profile rule. For sorted selected confidences c_(1)>=...>=c_(n):
        L_n = sum_{j<=n} c_(j) - (n-1);  U_n = 1 - c_(n);  score_n = L_n - U_n.
    Commit largest n with score_n > margin. Reduces to factor under equal confidence."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    B, L = confidence.shape
    values, idx = torch.sort(confidence, dim=1, descending=True)
    counts = mask_index.sum(dim=1)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    valid = cols < counts.unsqueeze(1)
    sorted_conf = torch.where(valid, values, torch.zeros_like(values))
    cum = torch.cumsum(sorted_conf, dim=1)
    frechet_lb = cum - cols.to(cum.dtype)          # L_n, n 1-indexed via 0-indexed cols
    competitor_ub = 1.0 - values                   # U_n
    score = frechet_lb - competitor_ub
    score = torch.where(valid, score, torch.full_like(score, -torch.inf))
    score[:, 0] = torch.inf                         # always >=1 token
    choose = score > margin
    topn = choose.sum(dim=1).clamp(min=1)
    select_sorted = cols < topn.unsqueeze(1)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index, int(transfer_index.sum().item())


@torch.no_grad()
def generate_dynamic(
    input_ids, attention_mask, model,
    steps=128, gen_length=128, block_length=128, temperature=0.0,
    cfg_scale=0.0, remasking="low_confidence", mask_id=126336,
    factor=None, frechet_margin=None,
):
    """Mirror of dLLM-Cache utils.generate (cfg_scale==0 path) but with a dynamic
    while-until-unmasked commit loop driven by the factor / Fréchet selector.
    The dLLMCache feature cache is reset/used exactly as in the base generate, so the
    only difference vs the base cache arm is the per-step commit rule. Returns
    (tokens, nfe, avg_committed_per_step)."""
    assert cfg_scale == 0.0, "dynamic Fréchet/factor path assumes cfg_scale==0 (our eval convention)"
    assert (factor is None) != (frechet_margin is None), "set exactly one of factor / frechet_margin"

    batch_size, prompt_length = input_ids.shape
    x = torch.full((batch_size, prompt_length + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    feature_cache = dLLMCache()
    feature_cache.reset_cache(prompt_length)

    nfe = 0
    committed = 0
    n_commit_events = 0
    for num_block in range(num_blocks):
        block_end = prompt_length + (num_block + 1) * block_length
        while True:
            block_slice = x[:, prompt_length + num_block * block_length: block_end]
            if (block_slice == mask_id).sum() == 0:
                break
            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits[:, prompt_length:]
            nfe += 1
            # restrict commits to current-and-earlier response region
            resp_mask = mask_index[:, prompt_length:].clone()
            resp_mask[:, (num_block + 1) * block_length:] = False
            if frechet_margin is not None:
                x0, transfer_index, nsel = get_transfer_index_frechet_profile(
                    logits, temperature, remasking, resp_mask, x[:, prompt_length:], frechet_margin)
            else:
                x0, transfer_index, nsel = get_transfer_index_dynamic(
                    logits, temperature, remasking, resp_mask, x[:, prompt_length:], factor)
            x[:, prompt_length:][transfer_index] = x0[transfer_index]
            committed += nsel
            n_commit_events += 1
    avg_sel = committed / max(n_commit_events, 1)
    return x[:, prompt_length:], nfe, avg_sel
