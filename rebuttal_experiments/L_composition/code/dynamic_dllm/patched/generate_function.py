import torch
from dynamic_dllm_cache.cache import DynamicDLLMCache
import torch.nn.functional as F
import numpy as np


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits.exp()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)


def generate(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, :prompt_length] = input_ids

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        feature_cache = DynamicDLLMCache()
        feature_cache.reset_cache(prompt_length)
        feature_cache.set_parameter(
            (feature_cache.select_from, feature_cache.window_size, feature_cache.layer_budget)
        )
        feature_cache.set_cur_prompt(torch.arange(prompt_length, device=model.device))
        for num_block in range(num_blocks):
            start_idx = prompt_length + num_block * block_length
            end_idx = prompt_length + (num_block + 1) * block_length

            block_x = x[:, start_idx:end_idx]
            block_mask_index = block_x == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    if hasattr(feature_cache, "cfg_interval_steps"):
                        feature_cache.update_step(layer_id=33)
                        if feature_cache.refresh_cfg(layer_id=33):
                            cfg_x = x.clone()
                            cfg_x[prompt_index] = mask_id
                            logits = model(x, attention_mask=attention_mask).logits[
                                :, prompt_length:
                            ]
                            feature_cache.cache_type = "cfg"
                            cfg_logits = model(
                                cfg_x, attention_mask=attention_mask
                            ).logits[:, prompt_length:]
                            cfg_residual = logits - cfg_logits
                            feature_cache.set_cache(
                                layer_id=33,
                                feature_name="cfg_residual",
                                features=cfg_residual,
                                cache_type="gen",
                            )
                            feature_cache.cache_type = "no_cfg"
                        else:
                            feature_cache.cache_type = "cfg"
                            cfg_residual = feature_cache.get_cache(
                                layer_id=33,
                                feature_name="cfg_residual",
                                cache_type="gen",
                            )
                            feature_cache.cache_type = "no_cfg"
                            logits = model(x, attention_mask=attention_mask).logits[
                                :, prompt_length:
                            ]
                    else:
                        cfg_x = x.clone()
                        cfg_x[prompt_index] = mask_id
                        logits = model(x, attention_mask=attention_mask).logits[
                            :, prompt_length:
                        ]
                        cfg_logits = model(cfg_x, attention_mask=attention_mask).logits[
                            :, prompt_length:
                        ]
                        cfg_residual = logits - cfg_logits
                    logits = (logits - cfg_residual) + (cfg_scale + 1) * cfg_residual
                else:
                    logits = model(x, attention_mask=attention_mask).logits[
                        :, prompt_length:
                    ]
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, (num_block + 1) * block_length :] = -np.inf

                x0 = torch.where(
                    mask_index[:, prompt_length:], x0, x[:, prompt_length:]
                )
                confidence = torch.where(mask_index[:, prompt_length:], x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    ).indices
                    transfer_index[j, select_index] = True
                x[:, prompt_length:][transfer_index] = x0[transfer_index]

                # Update cur_prompt for sliding window
                selected_positions = torch.where(transfer_index.any(dim=0))[0] + prompt_length
                if selected_positions.numel() > 0:
                    feature_cache.set_cur_prompt(selected_positions)
        return x[:, prompt_length:]


# ===========================================================================
# Prediction Dynamics (PD) — adaptive threshold generation
# ===========================================================================

def update_pd_threshold(logits, prev_probs, current_threshold, mask_index,
                         pd_mode, alpha, beta, global_step_counter):
    """
    Update PD (Prediction Dynamics) threshold based on model confidence.

    Args:
        logits: Current logits for the relevant region (B, L, V).
        prev_probs: Softmax probs from previous step (B, L, V), or None for first step.
        current_threshold: scalar (mode 1) or tensor [B,L] (mode 2).
        mask_index: Bool tensor (B, L), True for masked positions.
        pd_mode: 1 = global scalar, 2 = per-token tensor.
        alpha: Weight for peak-confidence term.
        beta: Weight for distribution-shift term.
        global_step_counter: 1-based step count.

    Returns:
        (updated_threshold, current_probs)
    """
    if pd_mode == 0:
        return current_threshold, None

    probabilities = F.softmax(logits.to(torch.float64), dim=-1)

    if global_step_counter > 1 and prev_probs is not None:
        sorted_values, _ = torch.sort(probabilities, dim=-1)
        peak_confidence = 1 - sorted_values[:, :, -2]

        distribution_similarity = 1 - F.cosine_similarity(
            probabilities, prev_probs, dim=-1
        )

        if pd_mode == 1:
            pc_mean = torch.mean(peak_confidence[mask_index]) if mask_index.any() else torch.tensor(0.0, device=logits.device)
            ds_mean = torch.mean(distribution_similarity[mask_index]) if mask_index.any() else torch.tensor(0.0, device=logits.device)
            updated_threshold = current_threshold - alpha * pc_mean + beta * ds_mean
        else:  # pd_mode == 2
            updated_threshold = current_threshold - alpha * peak_confidence + beta * distribution_similarity

        return updated_threshold, probabilities
    else:
        if pd_mode == 2:
            current_threshold = torch.full(
                size=logits.shape[:2], fill_value=float(current_threshold),
                device=logits.device, dtype=torch.float64
            )
        return current_threshold, probabilities


def get_transfer_index_pd(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    threshold,
    pd_mode: int,
):
    """
    PD-aware token selection. Computes confidence the same way as get_transfer_index,
    then selects tokens whose confidence >= the adaptive PD threshold.

    At least one token (max confidence) is always transferred per batch row.

    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
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

    transfer_index = mask_index & (confidence >= threshold)

    max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
    force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
    transfer_index = transfer_index | force_mask
    transfer_index = transfer_index & mask_index

    return x0, transfer_index



# ===========================================================================
# Frechet-profile commit rule (INSERTION for rebuttal experiment).
# Ported VERBATIM from ~/anchors/generate.py (Fast-dLLM++ reference
# implementation used by eval_llada.py). Only the commit/parallel-token-
# selection rule; model, cache, drafting, scheduling untouched.
# ===========================================================================

def get_transfer_index_frechet_profile(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    margin: float = 0.0,
):
    """
    Profile-aware Fréchet decoding.

    For sorted selected confidences c_(1) >= ... >= c_(n), define
        L_n = sum_{j=1}^n c_(j) - (n - 1)
        U_n = 1 - c_(n)
        score_n = L_n - U_n
    Decode the largest n with score_n > margin.

    When all confidences equal 1-eps, this reduces to the paper's factor
    condition (n+1)*eps < 1, so this is a strict generalization.

    Returns:
        x0: (B, L) long
        transfer_index: (B, L) bool
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    B, L = confidence.shape
    values, idx = torch.sort(confidence, dim=1, descending=True)

    counts = mask_index.sum(dim=1)  # (B,)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    valid = cols < counts.unsqueeze(1)

    sorted_conf = torch.where(valid, values, torch.zeros_like(values))
    cum = torch.cumsum(sorted_conf, dim=1)

    # n is 1-indexed: L_n = cum_n - (n-1) = cum - cols  (cols is 0-indexed)
    frechet_lb = cum - cols.to(cum.dtype)
    competitor_ub = 1.0 - values
    score = frechet_lb - competitor_ub

    score = torch.where(valid, score, torch.full_like(score, -torch.inf))

    # always transfer at least one token whenever masked tokens exist
    score[:, 0] = torch.inf

    choose = score > margin
    topn = choose.sum(dim=1).clamp(min=1)

    select_sorted = cols < topn.unsqueeze(1)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index

    return x0, transfer_index


@torch.no_grad()
def generate_pd(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                remasking='low_confidence', mask_id=126336, threshold=None,
                pd_mode=1, pd_threshold=1.0, alpha=0.01, beta=0.15,
                use_cache=False, cfg_scale=0.0, attention_mask=None,
                frechet_margin=None):
    """
    Block-wise generation with Prediction Dynamics (PD) adaptive threshold.

    Compatible with window_budget cache mode and no-cache mode.
    When use_cache=True, the DynamicDLLMCache singleton must already be
    configured (via DynamicDLLMCache.new_instance) and the model hooks must be registered
    (via register_cache_LLaDA) before calling this function.

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length for semi-autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] (default 126336).
        threshold: Fixed threshold for non-PD fallback (ignored when pd_mode > 0).
        pd_mode: 0=disabled, 1=global scalar threshold, 2=per-token threshold.
        pd_threshold: Initial threshold value (default 1.0).
        alpha: PD update weight for peak confidence (default 0.01).
        beta: PD update weight for distribution shift (default 0.15).
        use_cache: Enable dynamic dllm (default False).
        cfg_scale: Unsupervised classifier-free guidance scale (default 0.0).
        attention_mask: Attention mask for the model forward pass.
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    prompt_len = prompt.shape[1]

    # --- Cache initialization ---
    if use_cache:
        feature_cache = DynamicDLLMCache()
        feature_cache.reset_cache(prompt_len)
        feature_cache.set_parameter(
            (feature_cache.select_from, feature_cache.window_size, feature_cache.layer_budget)
        )
        feature_cache.set_cur_prompt(torch.arange(prompt_len, device=model.device))

    # --- PD state ---
    prev_probs = None
    current_pd_threshold = pd_threshold
    global_step_counter = 0

    nfe = 0
    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        i = 0
        while True:
            nfe += 1
            global_step_counter += 1

            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits
            mask_index[:, block_end:] = 0

            # --- Compute PD on full generation-region logits ---
            gen_logits = logits[:, prompt_len:, :]
            gen_mask = mask_index[:, prompt_len:]

            if frechet_margin is not None:
                # --- INSERTION: Frechet-profile commit rule replaces the PD
                # adaptive-threshold commit rule (threshold update + selection).
                # Everything else (model forward, cache, block loop) unchanged.
                x0, transfer_index = get_transfer_index_frechet_profile(
                    logits, temperature, remasking, mask_index, x,
                    margin=frechet_margin,
                )
            else:
                # --- Update PD threshold ---
                updated_threshold, cur_probs_full = update_pd_threshold(
                    gen_logits, prev_probs, current_pd_threshold,
                    gen_mask, pd_mode=pd_mode, alpha=alpha, beta=beta,
                    global_step_counter=global_step_counter,
                )
                current_pd_threshold = updated_threshold
                if cur_probs_full is not None:
                    prev_probs = cur_probs_full

                # --- Build threshold for get_transfer_index_pd ---
                if pd_mode == 1:
                    use_threshold = float(current_pd_threshold) if not isinstance(current_pd_threshold, (float, int)) else current_pd_threshold
                elif pd_mode == 2:
                    full_threshold = torch.full(
                        (prompt.shape[0], prompt_len + gen_length),
                        float('inf'), device=current_pd_threshold.device, dtype=current_pd_threshold.dtype
                    )
                    full_threshold[:, prompt_len:] = current_pd_threshold
                    use_threshold = full_threshold
                else:
                    use_threshold = current_pd_threshold

                x0, transfer_index = get_transfer_index_pd(
                    logits, temperature, remasking, mask_index, x,
                    threshold=use_threshold, pd_mode=pd_mode,
                )
            x[transfer_index] = x0[transfer_index]

            # --- Update sliding window center ---
            if use_cache:
                selected_positions = torch.where(transfer_index.any(dim=0))[0]
                if selected_positions.numel() > 0:
                    feature_cache.set_cur_prompt(selected_positions)

            i += 1

            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

    return x, nfe
