# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from torch.cuda import nvtx

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity
    score but reduces generation quality. Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens


# ---------------------------------------------------------------------------
# Token-selection helpers
# ---------------------------------------------------------------------------

def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
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

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)

        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = transfer_index | force_mask
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded

    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index

    return x0, transfer_index


def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue

        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # at least one token is transferred
        threshs[0] = -1
        sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


def get_transfer_index_frechet_profile(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    margin: float = 0.0,
    eta: float = 0.0,
):
    """
    Profile-aware Fréchet decoding.

    eta > 0 activates the calibration-robust variant (Appendix C): each observed
    confidence is shrunk to c_lower = max(0, c_hat - eta) before the certificate is
    formed, so a systematic overconfidence budget eta is discounted. eta = 0 is an
    exact no-op (max(c,0) = c for c in [0,1]) and reproduces the paper's rule. Token
    identities (argmax x0) are taken from the raw logits and are unaffected by eta;
    eta only tightens the commit DECISION.

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

    # Robust variant (Appendix C): shrink observed confidences by eta before the
    # certificate. clamp(v - 0, min=0) == v for v in [0,1], so eta=0 is a no-op.
    values_r = torch.clamp(values - eta, min=0.0) if eta else values

    sorted_conf = torch.where(valid, values_r, torch.zeros_like(values_r))
    cum = torch.cumsum(sorted_conf, dim=1)

    # n is 1-indexed: L_n = cum_n - (n-1) = cum - cols  (cols is 0-indexed)
    frechet_lb = cum - cols.to(cum.dtype)
    competitor_ub = 1.0 - values_r
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


# ---------------------------------------------------------------------------
# Trace / logging helpers (Workstream A)
# ---------------------------------------------------------------------------

def _append_jsonl(path, record):
    """Append a single JSON record to a JSONL file."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _selected_confidence_vector(logits, x0, mask_index, remasking):
    """Recompute per-token confidences for the current proposal x0."""
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        conf = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        conf = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    neg_inf = torch.tensor(torch.finfo(conf.dtype).min, device=conf.device, dtype=conf.dtype)
    return torch.where(mask_index, conf, neg_inf)



@torch.no_grad()
def _sequential_greedy_verify(model, x, mask_index, transfer_index, x0, step_logits,
                              mask_id, remasking="low_confidence", max_verify=None):
    """
    Sequential-greedy verifier for commit-level calibration (ISSUE-15; BZkr-Q1/xMJT-Q2).

    At the current decode state x (the selected set S is still masked), decode the
    positions in S ONE AT A TIME, greedily, in parallel-confidence-descending order,
    each conditioned on the previously revealed members of S. Compare this sequential
    reference to the parallel proposal x0. Yields:
      commit_success   : 1 iff every position in S agrees with the sequential decode
      per_pos_agree    : token-calibration label per selected position (Exp A)
      (agreement of a position given its higher-confidence peers are revealed is also
       the coupling-flip proxy for Exp D)

    Full-recompute forward passes, independent of the caller's KV cache (correct for
    LLaDA since a full forward recomputes bidirectional attention anyway). batch=1.
    Cost is |S| extra forwards per step, so gate behind a flag and a small --limit.
    """
    tf = transfer_index[0] if transfer_index.dim() == 2 else transfer_index
    sel = tf.nonzero(as_tuple=False).squeeze(-1)
    n = int(sel.numel())
    if n == 0:
        return {"verify_n": 0}
    conf_vec = _selected_confidence_vector(step_logits, x0, mask_index, remasking)
    conf0 = conf_vec[0] if conf_vec.dim() == 2 else conf_vec
    sel_conf = conf0[sel]
    order = torch.argsort(sel_conf, descending=True)
    ordered = sel[order].tolist()
    if isinstance(ordered, int):
        ordered = [ordered]
    run = ordered if max_verify is None else ordered[:max_verify]

    x_seq = x.clone()
    x0_0 = x0[0] if x0.dim() == 2 else x0
    agree, positions, seq_conf = [], [], []
    for p in run:
        out = model(x_seq)
        logits_s = out.logits if hasattr(out, "logits") else out
        l0 = logits_s[0] if logits_s.dim() == 3 else logits_s
        probs_p = F.softmax(l0[p].to(torch.float64), dim=-1)
        tok = int(torch.argmax(l0[p]).item())
        agree.append(int(tok == int(x0_0[p].item())))
        seq_conf.append(float(probs_p.max().item()))
        positions.append(int(p))
        x_seq[0, p] = tok

    full = (max_verify is None) or (len(run) == n)
    order_list = order[:len(run)].tolist()
    if isinstance(order_list, int):
        order_list = [order_list]
    return {
        "verify_n": n,
        "verify_ran": len(run),
        "commit_success": int(all(agree)) if full else None,
        "commit_agree_frac": float(sum(agree) / len(agree)) if agree else None,
        "per_pos_agree": agree,
        "per_pos_positions": positions,
        "per_pos_parallel_conf": [float(sel_conf[o].item()) for o in order_list],
        "per_pos_seq_conf": seq_conf,
    }


@torch.no_grad()
def _sequential_greedy_verify_abs(model, x_full, sel_abs, x0_row, conf_row,
                                  mask_id, remasking="low_confidence", max_verify=None):
    """Absolute-position sequential-greedy verifier (cache-agnostic, full forwards).

    For the prefix/dual-cache paths the main decode loop runs on a slice with a KV
    cache, but the verifier must condition on the FULL sequence (prefix included), so
    it always does a full forward on x_full. Inputs use absolute indices into x_full:
      sel_abs : list[int] absolute positions of the selected set S
      x0_row  : 1D long tensor (len L_full) of proposed tokens (only S used)
      conf_row: 1D tensor (len L_full) parallel confidence per position
    Returns the same fields as _sequential_greedy_verify (commit_success, per_pos_*)."""
    n = len(sel_abs)
    if n == 0:
        return {"verify_n": 0}
    order = sorted(range(n), key=lambda k: -float(conf_row[sel_abs[k]].item()))
    ordered = [sel_abs[k] for k in order]
    run = ordered if max_verify is None else ordered[:max_verify]
    x_seq = x_full.clone()
    agree, positions, seq_conf, ppar = [], [], [], []
    for p in run:
        out = model(x_seq)
        logits_s = out.logits if hasattr(out, "logits") else out
        l0 = logits_s[0] if logits_s.dim() == 3 else logits_s
        probs_p = F.softmax(l0[p].to(torch.float64), dim=-1)
        tok = int(torch.argmax(l0[p]).item())
        agree.append(int(tok == int(x0_row[p].item())))
        seq_conf.append(float(probs_p.max().item()))
        positions.append(int(p)); ppar.append(float(conf_row[p].item()))
        x_seq[0, p] = tok
    full = (max_verify is None) or (len(run) == n)
    return {
        "verify_n": n, "verify_ran": len(run),
        "commit_success": int(all(agree)) if full else None,
        "commit_agree_frac": float(sum(agree) / len(agree)) if agree else None,
        "per_pos_agree": agree, "per_pos_positions": positions,
        "per_pos_parallel_conf": ppar, "per_pos_seq_conf": seq_conf,
    }


@torch.no_grad()
def _coupling_reveal_diag(model, x_full, sel_abs, x0_row, mask_id,
                          remasking="low_confidence", max_diag=None):
    """Exp D coupling diagnostic (ISSUE-17).

    For each selected position i in S, reveal the OTHER selected tokens S\\i (their
    proposed values x0) and recompute p_i; measure whether the argmax flips and the
    Jensen-Shannon divergence (nats) between p_i(.|E) and p_i(.|E, x_{S\\i}). This is
    the direct measurement of the token coupling a marginal-only certificate cannot
    see. Coupling is only defined for |S|>=2. Cost ~ |S|+1 full forwards/step; gate
    behind a flag + small --limit."""
    n = len(sel_abs)
    if n < 2:
        return {"coupling_n": n}
    out = model(x_full)
    l0 = out.logits if hasattr(out, "logits") else out
    l0 = l0[0] if l0.dim() == 3 else l0
    base_p = {p: F.softmax(l0[p].to(torch.float64), dim=-1) for p in sel_abs}
    base_am = {p: int(torch.argmax(l0[p]).item()) for p in sel_abs}
    run = sel_abs if max_diag is None else sel_abs[:max_diag]
    flips, jsds, positions = [], [], []
    def _kl(a, b):
        m = a > 0
        return float(torch.sum(a[m] * (torch.log(a[m]) - torch.log(b[m]))).item())
    for p in run:
        x_rev = x_full.clone()
        for q in sel_abs:
            if q != p:
                x_rev[0, q] = int(x0_row[q].item())
        out2 = model(x_rev)
        l2 = out2.logits if hasattr(out2, "logits") else out2
        l2 = l2[0] if l2.dim() == 3 else l2
        pnew = F.softmax(l2[p].to(torch.float64), dim=-1)
        pb = base_p[p]
        mix = 0.5 * (pb + pnew)
        jsd = 0.5 * _kl(pb, mix) + 0.5 * _kl(pnew, mix)
        flips.append(int(int(torch.argmax(l2[p]).item()) != base_am[p]))
        jsds.append(float(jsd)); positions.append(int(p))
    return {
        "coupling_n": n, "coupling_ran": len(run),
        "coupling_flip_frac": float(sum(flips) / len(flips)) if flips else None,
        "coupling_jsd_mean": float(sum(jsds) / len(jsds)) if jsds else None,
        "per_pos_coupling_flip": flips, "per_pos_coupling_jsd": jsds,
        "per_pos_coupling_positions": positions,
    }


def _build_selector_summary(
    logits,
    x0,
    mask_index,
    transfer_index,
    remasking,
    threshold,
    factor,
    frechet_margin,
    topk=64,
):
    """
    Compute theory-side quantities for a single decode step.
    Works on the first batch element only (batch_size=1 in eval).
    """
    confidence = _selected_confidence_vector(logits, x0, mask_index, remasking)

    # Flatten to first batch element for simplicity (eval is batch_size=1)
    if confidence.dim() == 2:
        confidence_1d = confidence[0]
        mask_1d = mask_index[0] if mask_index.dim() == 2 else mask_index
        transfer_1d = transfer_index[0] if transfer_index.dim() == 2 else transfer_index
    else:
        confidence_1d = confidence
        mask_1d = mask_index
        transfer_1d = transfer_index

    conf_masked = confidence_1d[mask_1d]
    if conf_masked.numel() == 0:
        return {"masked_count": 0, "selected_count": 0}

    sorted_conf, sorted_idx = torch.sort(conf_masked, descending=True)
    m = int(sorted_conf.numel())
    cols = torch.arange(m, device=sorted_conf.device, dtype=sorted_conf.dtype)
    cum = torch.cumsum(sorted_conf, dim=0)

    # Fréchet quantities for every possible n
    L = cum - cols                    # L_n = cumsum - (n-1), cols is 0-indexed
    U = 1.0 - sorted_conf            # U_n = 1 - c_(n)
    G = L - U                         # G_n = L_n - U_n

    selected_count = int(transfer_1d.sum().item())
    selected_count = max(selected_count, 1)
    sc = min(selected_count, m) - 1   # 0-indexed into sorted arrays

    c_n = float(sorted_conf[sc].item())
    L_n = float(L[sc].item())
    U_n = float(U[sc].item())
    G_n = float(G[sc].item())
    factor_core = float(((selected_count + 1) * c_n) - selected_count)
    heterogeneity_bonus = float(G_n - factor_core)

    ns = torch.arange(1, m + 1, device=sorted_conf.device, dtype=sorted_conf.dtype)

    # Counterfactual: factor with f = 1.0
    factor_thresholds_f1 = 1.0 - (1.0 / (ns + 1.0))
    factor_n_f1 = int((sorted_conf >= factor_thresholds_f1).sum().item())
    factor_n_f1 = max(factor_n_f1, 1)

    # Counterfactual: matched factor for Fréchet margin delta
    delta = 0.0 if frechet_margin is None else float(frechet_margin)
    f_match = 1.0 - delta
    factor_thresholds_match = 1.0 - (f_match / (ns + 1.0))
    matched_factor_n = int((sorted_conf >= factor_thresholds_match).sum().item())
    matched_factor_n = max(matched_factor_n, 1)

    selected_conf = confidence_1d[transfer_1d]
    mean_selected_conf = float(selected_conf.mean().item()) if selected_conf.numel() else None
    min_selected_conf = float(selected_conf.min().item()) if selected_conf.numel() else None

    selected_positions = transfer_1d.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(selected_positions, int):
        selected_positions = [selected_positions]

    return {
        "masked_count": m,
        "selected_count": selected_count,
        "sorted_conf_topk": [float(x) for x in sorted_conf[:topk].tolist()],
        "selected_conf_topk": [float(x) for x in selected_conf[:topk].tolist()],
        "selected_positions_topk": selected_positions[:topk],
        "c_n": c_n,
        "L_n": L_n,
        "U_n": U_n,
        "G_n": G_n,
        "factor_core": factor_core,
        "heterogeneity_bonus": heterogeneity_bonus,
        "factor_n_f1": factor_n_f1,
        "matched_factor_n": matched_factor_n,
        "mean_selected_conf": mean_selected_conf,
        "min_selected_conf": min_selected_conf,
    }


# ---------------------------------------------------------------------------
# Unified token-selection dispatcher
# ---------------------------------------------------------------------------

def _select_tokens(logits, temperature, remasking, mask_index, x,
                   num_transfer_tokens_i, threshold, factor, frechet_margin,
                   frechet_eta=0.0):
    """Unified dispatch for token selection across all generation paths."""
    n_set = sum(v is not None for v in (threshold, factor, frechet_margin))
    assert n_set <= 1, (
        f"At most one of threshold, factor, frechet_margin may be set, "
        f"got threshold={threshold}, factor={factor}, frechet_margin={frechet_margin}"
    )
    if frechet_margin is not None:
        return get_transfer_index_frechet_profile(
            logits, temperature, remasking, mask_index, x, frechet_margin, frechet_eta
        )
    elif factor is not None:
        return get_transfer_index_dynamic(
            logits, temperature, remasking, mask_index, x, None, factor
        )
    else:
        return get_transfer_index(
            logits, temperature, remasking, mask_index, x,
            num_transfer_tokens_i if threshold is None else None,
            threshold,
        )


def _determine_decode_mode(threshold, factor, frechet_margin):
    """Return a string label for the active decode mode."""
    if frechet_margin is not None:
        return "frechet"
    elif factor is not None:
        return "factor"
    elif threshold is not None:
        return "threshold"
    else:
        return "quota"


# ---------------------------------------------------------------------------
# Generation functions (with optional trace instrumentation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None,
             frechet_margin=None, trace_jsonl=None, trace_meta=None, trace_topk=64,
             frechet_eta=0.0, calib_verify=False, verify_max=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Confidence threshold for parallel decoding.
        factor: Factor for dynamic parallel decoding.
        frechet_margin: Safety margin delta for Frechet-profile decoding.
        trace_jsonl: Path to JSONL file for per-step trace output (None = no tracing).
        trace_meta: Dict of metadata fields to include in every trace record.
        trace_topk: Number of top-k confidence values to log per step.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    global_step = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0

            masked_before = int(mask_index.sum().item())

            quota_i = num_transfer_tokens[:, min(i, steps - 1)] if (threshold is None and factor is None and frechet_margin is None) else None
            x0, transfer_index = _select_tokens(
                logits, temperature, remasking, mask_index, x,
                quota_i, threshold, factor, frechet_margin, frechet_eta
            )

            # --- trace ---
            if trace_jsonl is not None:
                summary = _build_selector_summary(
                    logits, x0, mask_index, transfer_index,
                    remasking, threshold, factor, frechet_margin, topk=trace_topk,
                )
                if calib_verify:
                    summary.update(_sequential_greedy_verify(
                        model, x, mask_index, transfer_index, x0, logits,
                        mask_id, remasking, verify_max,
                    ))
                    summary["frechet_eta"] = float(frechet_eta)
                x[transfer_index] = x0[transfer_index]
                masked_after = int((x == mask_id).sum().item())
                prompt_len = int(prompt.shape[1])
                decoded_count = int((x[:, prompt_len:] != mask_id).sum().item())
                record = {
                    **(trace_meta or {}),
                    "cache_mode": "none",
                    "block_idx": int(num_block),
                    "step_idx_within_block": int(i),
                    "global_step_idx": global_step,
                    "masked_remaining_before": masked_before,
                    "masked_remaining_after": masked_after,
                    "fraction_decoded_after": float(decoded_count / gen_length),
                    **summary,
                }
                _append_jsonl(trace_jsonl, record)
            else:
                x[transfer_index] = x0[transfer_index]

            global_step += 1
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe


@torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                               remasking='low_confidence', mask_id=126336, threshold=None, factor=None,
                               frechet_margin=None, trace_jsonl=None, trace_meta=None, trace_topk=64,
                               frechet_eta=0.0, calib_verify=False, coupling_diag=False, verify_max=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Confidence threshold for parallel decoding.
        factor: Factor for dynamic parallel decoding.
        frechet_margin: Safety margin delta for Frechet-profile decoding.
        trace_jsonl: Path to JSONL file for per-step trace output (None = no tracing).
        trace_meta: Dict of metadata fields to include in every trace record.
        trace_topk: Number of top-k confidence values to log per step.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    global_step = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0

        masked_before = int(mask_index.sum().item())

        quota_0 = num_transfer_tokens[:, 0] if (threshold is None and factor is None and frechet_margin is None) else None
        x0, transfer_index = _select_tokens(
            output.logits, temperature, remasking, mask_index, x,
            quota_0, threshold, factor, frechet_margin, frechet_eta
        )

        # --- trace (step 0) ---
        if trace_jsonl is not None:
            summary = _build_selector_summary(
                output.logits, x0, mask_index, transfer_index,
                remasking, threshold, factor, frechet_margin, topk=trace_topk,
            )
            if calib_verify or coupling_diag:
                conf_full = _selected_confidence_vector(output.logits, x0, mask_index, remasking)
                conf_row = conf_full[0] if conf_full.dim() == 2 else conf_full
                tf0 = transfer_index[0] if transfer_index.dim() == 2 else transfer_index
                sel_abs = tf0.nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(sel_abs, int): sel_abs = [sel_abs]
                x0_row = x0[0] if x0.dim() == 2 else x0
                if calib_verify:
                    summary.update(_sequential_greedy_verify_abs(
                        model, x, sel_abs, x0_row, conf_row, mask_id, remasking, verify_max))
                if coupling_diag:
                    summary.update(_coupling_reveal_diag(
                        model, x, sel_abs, x0_row, mask_id, remasking, verify_max))
                summary["frechet_eta"] = float(frechet_eta)
            x[transfer_index] = x0[transfer_index]
            masked_after = int((x == mask_id).sum().item())
            prompt_len = int(prompt.shape[1])
            decoded_count = int((x[:, prompt_len:] != mask_id).sum().item())
            record = {
                **(trace_meta or {}),
                "cache_mode": "prefix",
                "block_idx": int(num_block),
                "step_idx_within_block": 0,
                "global_step_idx": global_step,
                "masked_remaining_before": masked_before,
                "masked_remaining_after": masked_after,
                "fraction_decoded_after": float(decoded_count / gen_length),
                **summary,
            }
            _append_jsonl(trace_jsonl, record)
        else:
            x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)

        past_key_values = new_past_key_values
        nfe += 1
        global_step += 1

        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            masked_before = int(mask_index.sum().item())

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            quota_i = num_transfer_tokens[:, min(i, steps - 1)] if (threshold is None and factor is None and frechet_margin is None) else None
            x0, transfer_index = _select_tokens(
                logits, temperature, remasking, mask_index, x[:, current_block_start:],
                quota_i, threshold, factor, frechet_margin, frechet_eta
            )

            # --- trace (inner loop) ---
            if trace_jsonl is not None:
                summary = _build_selector_summary(
                    logits, x0, mask_index, transfer_index,
                    remasking, threshold, factor, frechet_margin, topk=trace_topk,
                )
                if calib_verify or coupling_diag:
                    cbs = current_block_start
                    conf_slice = _selected_confidence_vector(logits, x0, mask_index, remasking)
                    conf_srow = conf_slice[0] if conf_slice.dim() == 2 else conf_slice
                    x0_srow = x0[0] if x0.dim() == 2 else x0
                    tfi = transfer_index[0] if transfer_index.dim() == 2 else transfer_index
                    sel_slice = tfi.nonzero(as_tuple=False).squeeze(-1).tolist()
                    if isinstance(sel_slice, int): sel_slice = [sel_slice]
                    sel_abs = [int(s) + cbs for s in sel_slice]
                    conf_row = torch.full((x.shape[1],), float(torch.finfo(conf_srow.dtype).min),
                                          dtype=conf_srow.dtype, device=conf_srow.device)
                    conf_row[cbs:cbs + conf_srow.shape[0]] = conf_srow
                    x0_row = x[0].clone()
                    x0_row[cbs:cbs + x0_srow.shape[0]] = x0_srow
                    if calib_verify:
                        summary.update(_sequential_greedy_verify_abs(
                            model, x, sel_abs, x0_row, conf_row, mask_id, remasking, verify_max))
                    if coupling_diag:
                        summary.update(_coupling_reveal_diag(
                            model, x, sel_abs, x0_row, mask_id, remasking, verify_max))
                    summary["frechet_eta"] = float(frechet_eta)
                x[:, current_block_start:][transfer_index] = x0[transfer_index]
                masked_after = int((x == mask_id).sum().item())
                prompt_len = int(prompt.shape[1])
                decoded_count = int((x[:, prompt_len:] != mask_id).sum().item())
                record = {
                    **(trace_meta or {}),
                    "cache_mode": "prefix",
                    "block_idx": int(num_block),
                    "step_idx_within_block": int(i),
                    "global_step_idx": global_step,
                    "masked_remaining_before": masked_before,
                    "masked_remaining_after": masked_after,
                    "fraction_decoded_after": float(decoded_count / gen_length),
                    **summary,
                }
                _append_jsonl(trace_jsonl, record)
            else:
                x[:, current_block_start:][transfer_index] = x0[transfer_index]

            global_step += 1
            i += 1

    return x, nfe


@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    frechet_margin=None, trace_jsonl=None, trace_meta=None, trace_topk=64
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0
    global_step = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        global_mask_index[:, e:] = False

        masked_before = int(global_mask_index.sum().item())

        quota_0 = num_transfer_tokens[:, 0] if (threshold is None and factor is None and frechet_margin is None) else None
        x0, transfer_index = _select_tokens(
            out_full.logits, temperature, remasking, global_mask_index, x,
            quota_0, threshold, factor, frechet_margin
        )

        # --- trace (step 0) ---
        if trace_jsonl is not None:
            summary = _build_selector_summary(
                out_full.logits, x0, global_mask_index, transfer_index,
                remasking, threshold, factor, frechet_margin, topk=trace_topk,
            )
            x = torch.where(transfer_index, x0, x)
            masked_after = int((x == mask_id).sum().item())
            decoded_count = int((x[:, Lp:] != mask_id).sum().item())
            record = {
                **(trace_meta or {}),
                "cache_mode": "dual",
                "block_idx": int(nb),
                "step_idx_within_block": 0,
                "global_step_idx": global_step,
                "masked_remaining_before": masked_before,
                "masked_remaining_after": masked_after,
                "fraction_decoded_after": float(decoded_count / gen_length),
                **summary,
            }
            _append_jsonl(trace_jsonl, record)
        else:
            x = torch.where(transfer_index, x0, x)

        global_step += 1

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits

            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)
            masked_before = int(mask_blk.sum().item())

            quota_i = num_transfer_tokens[:, min(i, steps_per_block - 1)] if (threshold is None and factor is None and frechet_margin is None) else None
            x0_blk, transfer_idx_blk = _select_tokens(
                logits_blk, temperature, remasking, mask_blk, x[:, s:e],
                quota_i, threshold, factor, frechet_margin
            )

            # --- trace (inner loop) ---
            if trace_jsonl is not None:
                summary = _build_selector_summary(
                    logits_blk, x0_blk, mask_blk, transfer_idx_blk,
                    remasking, threshold, factor, frechet_margin, topk=trace_topk,
                )
                blk_old = x[:, s:e]
                blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
                x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
                masked_after = int((x == mask_id).sum().item())
                decoded_count = int((x[:, Lp:] != mask_id).sum().item())
                record = {
                    **(trace_meta or {}),
                    "cache_mode": "dual",
                    "block_idx": int(nb),
                    "step_idx_within_block": int(i),
                    "global_step_idx": global_step,
                    "masked_remaining_before": masked_before,
                    "masked_remaining_after": masked_after,
                    "fraction_decoded_after": float(decoded_count / gen_length),
                    **summary,
                }
                _append_jsonl(trace_jsonl, record)
            else:
                blk_old = x[:, s:e]
                blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
                x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

            global_step += 1
            nfe += 1

    return x, nfe


def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
