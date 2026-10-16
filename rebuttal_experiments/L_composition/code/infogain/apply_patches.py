#!/usr/bin/env python3
"""Apply Frechet-insertion + instrumentation patches to Information-Gain-Sampler repo.

Idempotent: each patch checks for its marker before applying.
Run from repo root: python apply_patches.py
"""
import re
import sys

MARKER = "FRECHET_INSERTION_PATCH"

# ---------------------------------------------------------------------------
# Patch 1: dllm infogain llada sampler: add frechet_margin config + selection
# ---------------------------------------------------------------------------
SAMPLER = "dllm/dllm/pipelines/infogain/llada/sampler.py"

FRECHET_HELPER = '''

# --- FRECHET_INSERTION_PATCH: profile-aware Frechet commit-set selection ---
# Ported from ~/anchors/generate.py::get_transfer_index_frechet_profile
# (Fast-dLLM++ Frechet certificate). For sorted selected confidences
# c_(1) >= ... >= c_(n): L_n = sum_j c_(j) - (n-1); U_n = 1 - c_(n);
# decode the largest n with L_n - U_n > margin. Always commits >= 1 token.
# temperature=0 greedy path only (Gumbel noise is a no-op at temp 0).
def _frechet_select(logits_row: torch.Tensor, x_row: torch.Tensor,
                    block_mask: torch.Tensor, margin: float):
    """logits_row [1,T,V]; x_row [1,T]; block_mask [1,T] bool (masked positions
    of the CURRENT block only). Returns (x0, transfer_index [1,T] bool)."""
    p = F.softmax(logits_row.to(torch.float64), dim=-1)
    x0 = torch.argmax(logits_row, dim=-1)
    x0 = torch.where(block_mask, x0, x_row)
    conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
    neg_inf = torch.finfo(conf.dtype).min
    conf = torch.where(block_mask, conf, torch.full_like(conf, neg_inf))

    B, L = conf.shape
    values, idx = torch.sort(conf, dim=1, descending=True)
    counts = block_mask.sum(dim=1)
    cols = torch.arange(L, device=conf.device).unsqueeze(0).expand(B, L)
    valid = cols < counts.unsqueeze(1)

    sorted_conf = torch.where(valid, values, torch.zeros_like(values))
    cum = torch.cumsum(sorted_conf, dim=1)
    frechet_lb = cum - cols.to(cum.dtype)
    competitor_ub = 1.0 - values
    score = frechet_lb - competitor_ub
    score = torch.where(valid, score, torch.full_like(score, float("-inf")))
    score[:, 0] = float("inf")  # always transfer at least one token

    choose = score > margin
    topn = choose.sum(dim=1).clamp(min=1)
    select_sorted = cols < topn.unsqueeze(1)
    transfer = torch.zeros(B, L, dtype=torch.bool, device=conf.device)
    transfer = transfer.scatter(1, idx, select_sorted)
    transfer = transfer & block_mask
    return x0, transfer
# --- end FRECHET_INSERTION_PATCH helper ---
'''


def patch_sampler():
    src = open(SAMPLER).read()
    if MARKER in src:
        print(f"[skip] {SAMPLER} already patched")
        return

    # 1a. config field
    anchor = '    info_gain_future_weight: float = 1.0\n'
    assert anchor in src, "config anchor not found"
    src = src.replace(anchor, anchor +
        '    frechet_margin: Optional[float] = None  # FRECHET_INSERTION_PATCH\n')

    # 1b. helper function after imports (before @dataclass)
    anchor = '@dataclass\nclass InfoGainLLaDASamplerConfig'
    assert anchor in src
    src = src.replace(anchor, FRECHET_HELPER + '\n\n' + anchor)

    # 1c. kwargs read inside _sample_info_gain_no_cache
    anchor = '''        info_gain_future_weight = float(
            kwargs.get("info_gain_future_weight", config.info_gain_future_weight)
        )
'''
    assert anchor in src
    src = src.replace(anchor, anchor +
        '        frechet_margin = kwargs.get(  # FRECHET_INSERTION_PATCH\n'
        '            "frechet_margin", getattr(config, "frechet_margin", None)\n'
        '        )\n')

    # 1d. selection branch: right after the threshold-bypass `continue` block,
    # before `k = 1`
    anchor = '''                    k = 1
                    result = ig.generate_candidates('''
    assert anchor in src
    frechet_branch = '''                    # FRECHET_INSERTION_PATCH: replace Info-Gain candidate
                    # lookahead selection with the Frechet certificate commit set.
                    if frechet_margin is not None:
                        x0f, tri = _frechet_select(lx, xx, ma, float(frechet_margin))
                        if tri.any():
                            x_row = x[row].clone()
                            sel = tri[0]
                            x_row[sel] = x0f[0][sel]
                            x[row] = x_row
                            changed = True
                        continue

'''
    src = src.replace(anchor, frechet_branch + anchor)

    open(SAMPLER, "w").write(src)
    print(f"[ok] patched {SAMPLER}")


# ---------------------------------------------------------------------------
# Patch 2: scripts/eval.py: frechet_margin plumb-through + instrumentation
# ---------------------------------------------------------------------------
EVAL = "scripts/eval.py"


def patch_eval():
    src = open(EVAL).read()
    if MARKER in src:
        print(f"[skip] {EVAL} already patched")
        return

    # 2a. pass frechet_margin into the sampler ConfigClass (only when set,
    # so Dream/SDAR config classes without the field are unaffected)
    anchor = '''            # Create config
            config = ConfigClass(
'''
    assert anchor in src
    src = src.replace(anchor, '''            # FRECHET_INSERTION_PATCH: forward frechet_margin only when set
            _extra_cfg = {}
            _fm = kwargs.get('frechet_margin', None)
            if _fm is not None:
                _extra_cfg['frechet_margin'] = float(_fm)
            # Create config
            config = ConfigClass(
                **_extra_cfg,
''')

    # 2b. add frechet_margin to info-gain gen_kwargs
    anchor = """                'variant': getattr(args, 'variant', 'info_gain'),
            })"""
    assert anchor in src
    src = src.replace(anchor, """                'variant': getattr(args, 'variant', 'info_gain'),
                'frechet_margin': getattr(args, 'frechet_margin', None),  # FRECHET_INSERTION_PATCH
            })""")

    # 2c. instrumentation: forward-work counters + per-sample wall time.
    # Wrap adapter.model.forward right after adapter creation.
    anchor = '''    adapter = get_model_adapter(args.model_name, device=args.device)
'''
    assert anchor in src
    src = src.replace(anchor, anchor + '''
    # --- FRECHET_INSERTION_PATCH: compute-accounting instrumentation ---
    # Info-Gain does batched candidate lookahead: raw call counts are NOT
    # compute-comparable to single-forward methods. We record:
    #   model_calls      = number of model forward invocations (native NFE analog)
    #   row_forwards     = sum over calls of input batch rows (effective forward
    #                      work; a call with an 8-candidate batch counts 8)
    #   row_token_forwards = sum over calls of rows*seq_len (token-level work)
    # plus wall time per sample. Primary ranking axis = wall time / row_forwards.
    import time as _time
    import json as _json
    _COUNT = {"model_calls": 0, "row_forwards": 0, "row_token_forwards": 0}
    _orig_forward = adapter.model.forward
    def _counting_forward(*_a, **_kw):
        _ids = _kw.get("input_ids", _a[0] if _a else None)
        if _ids is not None and hasattr(_ids, "shape") and _ids.dim() == 2:
            _COUNT["model_calls"] += 1
            _COUNT["row_forwards"] += int(_ids.shape[0])
            _COUNT["row_token_forwards"] += int(_ids.shape[0]) * int(_ids.shape[1])
        return _orig_forward(*_a, **_kw)
    adapter.model.forward = _counting_forward
    _PER_SAMPLE = []
    # --- end instrumentation setup ---
''')

    # 2d. per-sample timing around generation
    anchor = '''        # Generate
        with torch.no_grad():
            output_ids = generate_with_algorithm(
                adapter, prompt_ids, args.algorithm, **gen_kwargs
            )
'''
    assert anchor in src
    src = src.replace(anchor, '''        # Generate  (FRECHET_INSERTION_PATCH: timed + counted)
        _snap0 = dict(_COUNT)
        _t0 = _time.perf_counter()
        with torch.no_grad():
            output_ids = generate_with_algorithm(
                adapter, prompt_ids, args.algorithm, **gen_kwargs
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _dt = _time.perf_counter() - _t0
        _PER_SAMPLE.append({
            "idx": idx,
            "wall_s": round(_dt, 4),
            "model_calls": _COUNT["model_calls"] - _snap0["model_calls"],
            "row_forwards": _COUNT["row_forwards"] - _snap0["row_forwards"],
            "row_token_forwards": _COUNT["row_token_forwards"] - _snap0["row_token_forwards"],
            "prompt_len": int(prompt_ids.shape[1]),
            "gen_length": int(gen_kwargs.get("gen_length", 0)),
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
            _tot_w = sum(s["wall_s"] for s in _PER_SAMPLE)
            _met = {
                "config": {k: v for k, v in vars(args).items()
                           if isinstance(v, (int, float, str, bool, type(None)))},
                "n_samples_done": len(_PER_SAMPLE),
                "totals": {
                    "wall_s": round(_tot_w, 2),
                    "model_calls": sum(s["model_calls"] for s in _PER_SAMPLE),
                    "row_forwards": sum(s["row_forwards"] for s in _PER_SAMPLE),
                    "row_token_forwards": sum(s["row_token_forwards"] for s in _PER_SAMPLE),
                    "gen_tokens_canvas": sum(s["gen_length"] for s in _PER_SAMPLE),
                    "canvas_tok_per_s": round(sum(s["gen_length"] for s in _PER_SAMPLE) / max(_tot_w, 1e-9), 3),
                },
                "definitions": {
                    "model_calls": "model forward invocations (native NFE analog for this repo; no native NFE counter exists in the info-gain path)",
                    "row_forwards": "sum of input batch rows over all forward calls = effective forward work (candidate lookahead batches counted at full width)",
                    "row_token_forwards": "sum of rows*seq_len over all forward calls",
                    "canvas_tok_per_s": "gen_length canvas tokens / total wall seconds (repo has no native tok/s convention)",
                },
                "per_sample": _PER_SAMPLE,
            }
            try:
                with open(str(args.result_path) + ".metrics.json", "w") as _mf:
                    _json.dump(_met, _mf, indent=1)
            except Exception as _e:
                print(f"[metrics] write failed: {_e}")
''')

    open(EVAL, "w").write(src)
    print(f"[ok] patched {EVAL}")


if __name__ == "__main__":
    patch_sampler()
    patch_eval()
    import py_compile
    py_compile.compile(SAMPLER, doraise=True)
    py_compile.compile(EVAL, doraise=True)
    print("py_compile OK for both files")
