# Fast-dLLM-v2 + Fréchet Profile Decoding: drop-in selector at the
# intra-sub-block confidence-aware selection step of generation_functions.batch_sample.
#
# The ONLY thing this module changes is WHICH masked positions in the current
# sub-block window get committed this step. It does not touch model loading,
# tokenizer, attention, block-diffusion schedule, or the (hierarchical) cache.
#
# Contract (matches the v2 hook):
#   x1_p : FloatTensor [B, W]  confidence (prob of the argmax/top-p token) per
#          position in the current sub-block window [start:end]; NON-candidate
#          positions are already set to -inf by the caller
#          (x1_p = torch.where(mask_idx[:, start:end], x1_p, -inf)).
#   Returns unmask_idx : BoolTensor [B, W]: positions to commit this step.
#
# Every selector commits at least the per-row argmax (>=1 token/step), exactly
# like v2's default rule, so decoding always makes progress.
#
# Selectors:
#   default / threshold : v2's native rule  unmask = (x1_p > threshold)
#   factor              : Fast-dLLM factor rule, largest n with (n+1)(1-c_(n)) < f
#   matched_factor      : factor with f = 1 - frechet_delta  (the paper's matched baseline)
#   frechet             : largest n with  G_n = L_n - U_n > delta
#                         L_n = max(0, sum_{j<=n} c_(j) - (n-1)),  U_n = 1 - c_(n)
#
# Reduction (verified): under equal confidence, frechet(delta) == factor(1-delta).

import torch

NEG_INF = float("-inf")
# Deterministic tie-break at the exact decision boundary. The Fréchet(δ)==Factor(1−δ)
# reduction is exact in real arithmetic, but the two selectors reach the boundary via
# different fp paths (cumsum vs (n+1)(1−c)). This eps (>> fp noise ~1e-9) makes BOTH
# consistently REJECT the exact-margin token, matching the paper's strict inequalities,
# so the equivalence holds token-for-token. Negligible for real (non-tied) confidences.
_TIE_EPS = 1e-6


def _prefix_boundary_to_unmask(order, best_n, mask_win):
    """Given descending sort order [B,W], per-row commit count best_n [B] (>=1),
    return a bool [B,W] committing the top-best_n positions, ANDed with mask_win."""
    B, W = order.shape
    # rank[b, pos] = descending-rank of position pos in row b (0 = highest conf)
    rank = torch.empty_like(order)
    ar = torch.arange(W, device=order.device).unsqueeze(0).expand(B, W)
    rank.scatter_(1, order, ar)
    unmask = rank < best_n.clamp(min=1).unsqueeze(1)
    return unmask & mask_win


def select_threshold(x1_p, mask_win, threshold):
    """v2 native rule: commit conf > threshold, always include the argmax."""
    unmask = x1_p > threshold
    max_prob_idx = x1_p.argmax(dim=-1)
    unmask[torch.arange(x1_p.shape[0], device=x1_p.device), max_prob_idx] = True
    return unmask & mask_win


def _factor_best_n(vals, f):
    """vals: descending-sorted conf [B,W] (-inf padded). Return best_n [B] (>=1):
    largest n (1-based) with (n+1)*(1 - c_(n)) < f. Non-candidate (-inf) ranks
    give (1-(-inf))=+inf so the condition is False there."""
    B, W = vals.shape
    n = torch.arange(1, W + 1, device=vals.device).unsqueeze(0)  # [1,W]
    cond = (n + 1).float() * (1.0 - vals) < f - _TIE_EPS                    # [B,W]
    idx = torch.where(cond, n, torch.zeros_like(n))              # 1-based where true else 0
    best_n = idx.max(dim=1).values
    return best_n.clamp(min=1)


def _frechet_best_n(vals, delta):
    """vals: descending-sorted conf [B,W] (-inf padded). Return best_n [B] (>=1):
    largest n with G_n = L_n - U_n > delta."""
    B, W = vals.shape
    # cumulative sum of top-n confidences; -inf padding poisons downstream ranks (G=-inf)
    safe = torch.where(torch.isinf(vals), torch.zeros_like(vals), vals)
    prefix = torch.cumsum(safe, dim=1)                           # sum_{j<=n} c_(j)
    n = torch.arange(1, W + 1, device=vals.device).unsqueeze(0).float()
    L_n = torch.clamp(prefix - (n - 1.0), min=0.0)
    U_n = 1.0 - vals                                            # +inf at padded ranks
    G_n = L_n - U_n                                             # -inf at padded ranks
    n_int = torch.arange(1, W + 1, device=vals.device).unsqueeze(0)
    cond = G_n > delta + _TIE_EPS
    idx = torch.where(cond, n_int, torch.zeros_like(n_int))
    best_n = idx.max(dim=1).values
    return best_n.clamp(min=1)


def _stats_at_n(vals, best_n):
    """Per-row diagnostic stats at the selected commit count best_n [B].
    Returns dict of [B] tensors: G_n, F_n (factor core), B_n (heterogeneity bonus),
    c_n (weakest committed conf). B_n = G_n - (factor-core certificate) is reported
    via the task's explicit formulas for interpretability."""
    B, W = vals.shape
    safe = torch.where(torch.isinf(vals), torch.zeros_like(vals), vals)
    prefix = torch.cumsum(safe, dim=1)
    gather_i = (best_n - 1).clamp(min=0, max=W - 1).unsqueeze(1)
    c_n = torch.gather(vals, 1, gather_i).squeeze(1)
    sum_n = torch.gather(prefix, 1, gather_i).squeeze(1)
    nf = best_n.float()
    L_n = torch.clamp(sum_n - (nf - 1.0), min=0.0)
    U_n = 1.0 - c_n
    G_n = L_n - U_n
    F_n = (nf + 1.0) * c_n - nf                    # factor-core certificate
    B_n = sum_n - nf * c_n                          # sum_{j<=n}(c_(j) - c_(n)) = heterogeneity bonus
    return {"G_n": G_n, "F_n": F_n, "B_n": B_n, "c_n": c_n}


def select(x1_p, mask_win, selector="default", threshold=0.9,
           factor=None, frechet_delta=None, collect_stats=False):
    """Unified entry. Returns (unmask_idx [B,W] bool, stats_or_None).

    stats (when collect_stats): dict of aggregate scalars for this step, incl.
    frechet vs matched-factor commit-count comparison (the heterogeneity bonus).
    """
    if selector in ("default", "threshold"):
        return select_threshold(x1_p, mask_win, threshold), None

    vals, order = torch.sort(x1_p, dim=1, descending=True)

    if selector == "factor":
        assert factor is not None, "factor selector needs factor f"
        best_n = _factor_best_n(vals, float(factor))
    elif selector == "matched_factor":
        assert frechet_delta is not None, "matched_factor needs frechet_delta"
        best_n = _factor_best_n(vals, 1.0 - float(frechet_delta))
    elif selector == "frechet":
        assert frechet_delta is not None, "frechet selector needs frechet_delta"
        best_n = _frechet_best_n(vals, float(frechet_delta))
    else:
        raise ValueError(f"unknown selector {selector!r}")

    unmask = _prefix_boundary_to_unmask(order, best_n, mask_win)

    stats = None
    if collect_stats:
        # candidate count per row (finite entries only)
        k = torch.isfinite(vals).sum(dim=1)
        active = k > 0
        st = _stats_at_n(vals, best_n)
        # matched-factor commit count on the SAME profile, for the bonus comparison
        if selector == "frechet":
            mf_best_n = _factor_best_n(vals, 1.0 - float(frechet_delta))
        elif selector == "matched_factor":
            mf_best_n = best_n
        else:
            mf_best_n = best_n
        diff = (best_n - mf_best_n).float()
        stats = {
            "rows": int(active.sum().item()),
            "sum_selected_n": float(best_n[active].float().sum().item()) if active.any() else 0.0,
            "sum_candidate_k": float(k[active].float().sum().item()) if active.any() else 0.0,
            "sum_G_n": float(st["G_n"][active].sum().item()) if active.any() else 0.0,
            "sum_B_n": float(st["B_n"][active].sum().item()) if active.any() else 0.0,
            "sum_F_n": float(st["F_n"][active].sum().item()) if active.any() else 0.0,
            "bonus_active_rows": int(((diff > 0) & active).sum().item()),
            "sum_extra_tokens": float(diff[active].clamp(min=0).sum().item()) if active.any() else 0.0,
        }
    return unmask, stats
