"""
KLASS (KL-Adaptive Stability Sampling) commit selector ported into the Fast-dLLM
PrefixCache decode path, for a same-cache controlled comparison against Factor/Frechet.

Selector logic mirrors shkim0116/KLASS@ad87ecb  src/model/llada_klass.py::
stable_confident_decode  with unmask_strategy="all":
  commit masked positions that are BOTH KL-stable over the last H steps
  (all step-to-step KL < kl_threshold) AND high-confidence (top-1 prob > conf_threshold);
  if none qualify for a row, fall back to top-k-by-confidence using KLASS's
  get_num_transfer_tokens quota schedule (guarantees progress).

Only the SELECTION differs from generate_with_prefix_cache; the model forwards and
KV-cache mechanics are copied verbatim, so KLASS is measured under the SAME PrefixCache
and backbone as the Factor/Frechet arms.

Caveat (documented): under PrefixCache only the active block suffix is recomputed each
step, so KLASS cross-block KL accumulation is limited vs its official no-cache decoder
(which recomputes the full sequence every step). Validate accuracy against the official
no-cache smoke before trusting the port.

Returns (x, nfe) matching generate_with_prefix_cache.
"""
import torch
import torch.nn.functional as F
from generate import get_num_transfer_tokens, add_gumbel_noise

_EPS = 1e-12


@torch.no_grad()
def generate_with_prefix_cache_klass(
    model, prompt, steps=256, gen_length=256, block_length=32, temperature=0.,
    remasking="low_confidence", mask_id=126336,
    klass_conf=0.6, klass_kl=0.015, klass_hist=2, alg="klass",
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    kl_hist = int(klass_hist)

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt
    nfe = 0

    for num_block in range(num_blocks):
        cbs = Lp + num_block * block_length
        cbe = cbs + block_length
        block_mask_index = (x[:, cbs:cbe] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        kl_history = torch.zeros((B, block_length, kl_hist), dtype=torch.float64, device=x.device)
        p_prev = None

        # ---- step 0: full forward, seed KV ----
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        block_logits = output.logits[:, cbs:cbe, :]
        kl_history, p_prev = _klass_step(
            x, cbs, cbe, block_logits, kl_history, p_prev, 0, steps_per_block,
            num_transfer_tokens, klass_conf, klass_kl, kl_hist, alg, temperature, mask_id)
        nfe += 1

        # slice KV down to the committed prefix
        new_past = []
        for i_ in range(len(past_key_values)):
            new_past.append(tuple(past_key_values[i_][j][:, :, :cbs] for j in range(len(past_key_values[i_]))))
        past_key_values = new_past

        step = 1
        while True:
            if (x[:, cbs:cbe] == mask_id).sum() == 0:
                break
            logits = model(x[:, cbs:], past_key_values=past_key_values, use_cache=True).logits
            block_logits = logits[:, :block_length, :]
            kl_history, p_prev = _klass_step(
                x, cbs, cbe, block_logits, kl_history, p_prev, step, steps_per_block,
                num_transfer_tokens, klass_conf, klass_kl, kl_hist, alg, temperature, mask_id)
            nfe += 1
            step += 1

    return x, nfe


def _klass_step(x, cbs, cbe, block_logits, kl_history, p_prev, step, steps_per_block,
                num_transfer_tokens, conf_threshold, kl_threshold, kl_hist, alg,
                temperature, mask_id):
    """One KLASS selection+commit step on block [cbs:cbe]."""
    if temperature > 0:
        block_logits = add_gumbel_noise(block_logits, temperature)
    p_curr = F.softmax(block_logits.to(torch.float64), dim=-1)          # (B, Lb, V)
    x0 = torch.argmax(p_curr, dim=-1)                                   # (B, Lb)

    if alg == "topk_margin":
        sp, _ = torch.sort(p_curr, dim=-1, descending=True)
        curr_conf = sp[..., 0] - sp[..., 1]
    elif alg == "entropy":
        curr_conf = -torch.sum(p_curr * torch.log(p_curr + 1e-10), dim=-1)
    else:
        curr_conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)

    if p_prev is None:
        kl_cur = torch.full_like(curr_conf, float("inf"), dtype=torch.float64)
    else:
        kl_cur = (p_curr * (torch.log(p_curr + _EPS) - torch.log(p_prev + _EPS))).sum(dim=-1)
    kl_history = torch.roll(kl_history, shifts=-1, dims=-1)
    kl_history[..., -1] = kl_cur
    p_prev_new = p_curr.clone()

    block_mask = (x[:, cbs:cbe] == mask_id)                             # (B, Lb)

    if step >= kl_hist - 1:
        stable = torch.all(kl_history < kl_threshold, dim=-1)
    else:
        stable = torch.zeros_like(curr_conf, dtype=torch.bool)
    conf_mask = curr_conf > conf_threshold
    ready = stable & conf_mask & block_mask                            # (B, Lb)

    transfer = torch.zeros_like(x0, dtype=torch.bool)
    for j in range(x0.shape[0]):
        ri = torch.where(ready[j])[0]
        if len(ri) > 0:
            transfer[j, ri] = True                                     # unmask_strategy="all"
        else:
            n_masked = int(block_mask[j].sum().item())
            if n_masked == 0:
                continue
            k = int(num_transfer_tokens[j, min(step, steps_per_block - 1)].item())
            k = max(1, min(k, n_masked))
            neg_inf = torch.tensor(float("-inf"), dtype=curr_conf.dtype, device=curr_conf.device)
            conf_j = torch.where(block_mask[j], curr_conf[j], neg_inf)
            _, sel = torch.topk(conf_j, k=k)
            transfer[j, sel] = True

    x[:, cbs:cbe][transfer] = x0[transfer]
    return kl_history, p_prev_new
