#!/usr/bin/env python3
"""Surgical, assertion-guarded patch of Fast-dLLM v2 for Fréchet Profile Decoding.

Edits exactly:
  1. generation_functions.py: import frechet_selectors; replace the intra-sub-block
     selection block inside batch_sample ONLY (matched via the unique `start:end` where-line).
  2. eval.py: add selector/frechet_delta/factor/log_frechet_stats args; cast + attach to
     self.model; wrap forward() for NFE counting; reset+dump diagnostics in generate_until.

Idempotent: if the file already contains the patch marker, that file is skipped.
Aborts loudly if any anchor is not found exactly once (never silently mis-patches).
"""
import io, sys, os

V2 = os.path.expanduser("~/fastdllmpp_rebuttal/fast-dllm/v2")
GF = os.path.join(V2, "generation_functions.py")
EV = os.path.join(V2, "eval.py")
MARK = "Fast-dLLM++ drop-in selector"


def _read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def _write(p, s):
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(s)


def _replace_once(text, old, new, label):
    c = text.count(old)
    if c != 1:
        raise SystemExit(f"[patch ABORT] anchor {label!r} found {c} times (expected 1)")
    return text.replace(old, new)


# ---------------- generation_functions.py ----------------
gf = _read(GF)
if MARK in gf:
    print("[skip] generation_functions.py already patched")
else:
    # import
    gf = _replace_once(
        gf,
        "import types\nfrom transformers.utils import auto_docstring, logging\n",
        "import types\nfrom transformers.utils import auto_docstring, logging\nimport frechet_selectors as _frechet_selectors\n",
        "gf-import",
    )
    old_sel = (
        "                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)\n"
        "\n"
        "                        unmask_idx = (x1_p > threshold)\n"
        "                        max_prob_idx = x1_p.argmax(dim=-1)\n"
        "                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True\n"
        "                        unmask_idx = unmask_idx & mask_idx[:, start:end]\n"
        "\n"
        "                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]\n"
    )
    new_sel = (
        "                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)\n"
        "\n"
        "                        # === Fast-dLLM++ drop-in selector (only the selection rule changes) ===\n"
        "                        unmask_idx, _fstat = _frechet_selectors.select(\n"
        "                            x1_p, mask_idx[:, start:end],\n"
        "                            selector=getattr(self, \"_fpp_selector\", \"default\"),\n"
        "                            threshold=threshold,\n"
        "                            factor=getattr(self, \"_fpp_factor\", None),\n"
        "                            frechet_delta=getattr(self, \"_fpp_delta\", None),\n"
        "                            collect_stats=getattr(self, \"_fpp_log\", False),\n"
        "                        )\n"
        "                        if _fstat is not None and getattr(self, \"_fpp_stats\", None) is not None:\n"
        "                            self._fpp_stats.append(_fstat)\n"
        "\n"
        "                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]\n"
    )
    gf = _replace_once(gf, old_sel, new_sel, "gf-selection-block")
    _write(GF, gf)
    print("[ok] patched generation_functions.py")

# ---------------- eval.py ----------------
ev = _read(EV)

# (0) lm_eval >=0.4.2 makes LM.device a getter-only property; the repo (written for
# 0.4.1) assigns self.device directly -> AttributeError. Assign the backing attr instead.
# Idempotent + runs regardless of the selector-patch marker.
if "        self.device = torch.device(device)\n" in ev:
    ev = ev.replace(
        "        self.device = torch.device(device)\n",
        "        self._device = torch.device(device)\n",
    ).replace(
        "            self.device = torch.device(f'{self.accelerator.device}')\n",
        "            self._device = torch.device(f'{self.accelerator.device}')\n",
    )
    _write(EV, ev)
    print("[ok] device-property fix applied to eval.py")

if MARK in ev:
    print("[skip] eval.py already patched")
else:
    # (A) __init__ new args
    ev = _replace_once(
        ev,
        "        bd_size=32,\n        threshold=0.9,\n        **kwargs,\n    ):\n",
        "        bd_size=32,\n        threshold=0.9,\n        selector=\"default\",\n"
        "        frechet_delta=-1.0,\n        factor=-1.0,\n        log_frechet_stats=False,\n"
        "        **kwargs,\n    ):\n",
        "ev-init-args",
    )
    # (B) attach fpp config + NFE-counting forward wrapper at end of __init__
    ev = _replace_once(
        ev,
        "        self.bd_size = bd_size\n",
        "        self.bd_size = bd_size\n\n"
        "        # === Fast-dLLM++ drop-in selector wiring ===\n"
        "        def _f_or_none(v):\n"
        "            try:\n"
        "                v = float(v)\n"
        "            except (TypeError, ValueError):\n"
        "                return None\n"
        "            return None if v < 0 else v\n"
        "        self.model._fpp_selector = str(selector)\n"
        "        self.model._fpp_delta = _f_or_none(frechet_delta)\n"
        "        self.model._fpp_factor = _f_or_none(factor)\n"
        "        self.model._fpp_log = str(log_frechet_stats).lower() in (\"1\", \"true\", \"yes\")\n"
        "        self.model._fpp_stats = [] if self.model._fpp_log else None\n"
        "        self.model._nfe = 0\n"
        "        _orig_forward = self.model.forward\n"
        "        def _counting_forward(*a, **k):\n"
        "            self.model._nfe += 1\n"
        "            return _orig_forward(*a, **k)\n"
        "        self.model.forward = _counting_forward\n",
        "ev-init-wiring",
    )
    # (C) reset counters at start of generate_until
    ev = _replace_once(
        ev,
        "    def generate_until(self, requests):\n"
        "        output = [None] * len(requests)  # pre-allocate output list\n"
        "        num_tokens = 0\n",
        "    def generate_until(self, requests):\n"
        "        output = [None] * len(requests)  # pre-allocate output list\n"
        "        num_tokens = 0\n"
        "        self.model._nfe = 0\n"
        "        if getattr(self.model, \"_fpp_log\", False):\n"
        "            self.model._fpp_stats = []\n",
        "ev-genreset",
    )
    # (D) count tokens ALWAYS (not only under show_speed) for NFE math
    ev = _replace_once(
        ev,
        "                # count token number\n"
        "                if self.show_speed:\n"
        "                    num_tokens += (generated_ids[batch_pos][seq_len[batch_pos]:] != self.mask_id).sum()\n",
        "                # count token number (always, for NFE accounting)\n"
        "                num_tokens += (generated_ids[batch_pos][seq_len[batch_pos]:] != self.mask_id).sum()\n",
        "ev-tokencount",
    )
    # (E) dump NFE + selector diagnostics before return
    ev = _replace_once(
        ev,
        "        return output\n",
        "        _nfe = int(getattr(self.model, \"_nfe\", 0))\n"
        "        _tok = int(num_tokens.item()) if torch.is_tensor(num_tokens) else int(num_tokens)\n"
        "        print(f\"Total NFE is {_nfe}\")\n"
        "        if _nfe > 0:\n"
        "            print(f\"Avg tokens per NFE: {_tok / _nfe}\")\n"
        "        _fpp_summary(self, _nfe, _tok, (end_time - start_time))\n"
        "        return output\n",
        "ev-genreturn",
    )
    # (F) module-level summary helper (before __main__)
    ev = _replace_once(
        ev,
        "if __name__ == \"__main__\":\n    cli_evaluate()\n",
        "def _fpp_summary(lm, nfe, tokens, wall):\n"
        "    # === Fast-dLLM++ drop-in selector diagnostics dump ===\n"
        "    path = os.environ.get(\"FPP_SUMMARY\")\n"
        "    stats = getattr(lm.model, \"_fpp_stats\", None)\n"
        "    agg = {\n"
        "        \"selector\": getattr(lm.model, \"_fpp_selector\", \"default\"),\n"
        "        \"frechet_delta\": getattr(lm.model, \"_fpp_delta\", None),\n"
        "        \"factor\": getattr(lm.model, \"_fpp_factor\", None),\n"
        "        \"small_block_size\": lm.small_block_size,\n"
        "        \"bd_size\": lm.bd_size,\n"
        "        \"threshold\": lm.threshold,\n"
        "        \"total_nfe\": nfe,\n"
        "        \"total_tokens\": tokens,\n"
        "        \"tokens_per_nfe\": (tokens / nfe) if nfe else None,\n"
        "        \"wall_time_sec\": wall,\n"
        "        \"throughput_tok_s\": (tokens / wall) if wall else None,\n"
        "    }\n"
        "    if stats:\n"
        "        rows = sum(s[\"rows\"] for s in stats)\n"
        "        agg[\"decode_steps\"] = len(stats)\n"
        "        agg[\"avg_selected_n\"] = (sum(s[\"sum_selected_n\"] for s in stats) / rows) if rows else None\n"
        "        agg[\"avg_candidate_k\"] = (sum(s[\"sum_candidate_k\"] for s in stats) / rows) if rows else None\n"
        "        agg[\"mean_G_n\"] = (sum(s[\"sum_G_n\"] for s in stats) / rows) if rows else None\n"
        "        agg[\"mean_B_n\"] = (sum(s[\"sum_B_n\"] for s in stats) / rows) if rows else None\n"
        "        agg[\"mean_F_n\"] = (sum(s[\"sum_F_n\"] for s in stats) / rows) if rows else None\n"
        "        agg[\"bonus_active_step_rate\"] = (sum(s[\"bonus_active_rows\"] for s in stats) / rows) if rows else None\n"
        "        agg[\"mean_extra_tokens_vs_matched_factor\"] = (sum(s[\"sum_extra_tokens\"] for s in stats) / rows) if rows else None\n"
        "    if path:\n"
        "        with open(path, \"w\") as f:\n"
        "            json.dump(agg, f, indent=2)\n"
        "    print(\"FPP_SUMMARY:\", json.dumps(agg))\n"
        "    return agg\n\n\n"
        "if __name__ == \"__main__\":\n    cli_evaluate()\n",
        "ev-summary-helper",
    )
    _write(EV, ev)
    print("[ok] patched eval.py")

print("[done] patch complete")
