#!/usr/bin/env python3
"""Patch SOAR (commit ec3eb400) for the Fast-dLLM++ rebuttal breadth run.

1. generate.py:
   - add frechet_certified_count() (port of ~/anchors/generate.py::
     get_transfer_index_frechet_profile, 1-D batch-1 form, NO forced min commit)
   - generate_soar(): new kwarg frechet_delta=None. When set, the parallel
     accept/commit rule (confidence > 0.95, cap 5) is REPLACED by the Frechet
     certificate (commit the certified prefix, uncapped); certificate fails
     (n_cert < min_parallel_tokens) -> original beam-search fallback.
   - forward-work counters: fw_stats counts one unit per SEQUENCE ROW per model
     forward (beam of k forwarded in one batched call = k units; cfg would
     double, cfg=0 here). Also decode_steps and max beam width used.
   - generate_soar now returns (best_sequence, fw_stats).
2. eval_llada.soar.py:
   - new model_arg frechet_delta (float or None), passed to generate_soar
   - accumulates fw_stats across requests; num_nfe = total forward work
   - re-enables non-EOS token counting (repo convention, token id 126081)
     for the non-humaneval path
   - prints: Total NFE is <int> (newline-terminated), decode steps, max beam
     width, avg sequence-forwards per step.
Idempotent: skips if marker already present.
"""
import sys

MARKER = "# FDPP_REBUTTAL_PATCH"

# ---------------- generate.py ----------------
gp = "generate.py"
src = open(gp).read()
if MARKER in src:
    print("generate.py already patched")
else:
    helper = '''

''' + MARKER + '''
def frechet_certified_count(confidences, delta):
    """Frechet-profile commit certificate, ported from
    ~/anchors/generate.py::get_transfer_index_frechet_profile (1-D, batch-1).

    For candidate confidences sorted descending c_(1) >= ... >= c_(n):
        L_n = sum_{j<=n} c_(j) - (n - 1)
        U_n = 1 - c_(n)
        score_n = L_n - U_n
    Certified commit count = (score > delta).sum(), the anchor's counting rule.
    Unlike the anchor there is NO forced minimum commit: returns 0 when nothing
    certifies, in which case the SOAR caller falls back to position beam search.
    """
    if confidences.numel() == 0:
        return 0
    vals, _ = torch.sort(confidences.to(torch.float64), descending=True)
    n0 = torch.arange(vals.numel(), device=vals.device, dtype=torch.float64)
    cum = torch.cumsum(vals, dim=0)
    score = (cum - n0) - (1.0 - vals)
    return int((score > delta).sum().item())

'''
    anchor_def = "def generate_soar(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,\n                        cfg_scale=0., remasking='low_confidence', mask_id=126336, max_beam_size=2, log=False):"
    assert src.count(anchor_def) == 1, "generate_soar signature anchor not found"
    head, tail = src.split(anchor_def, 1)
    head = head + helper
    new_def = "def generate_soar(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,\n                        cfg_scale=0., remasking='low_confidence', mask_id=126336, max_beam_size=2, log=False,\n                        frechet_delta=None):"

    # counters init (anchor unique to generate_soar body)
    a = "    confidence_threshold = 0.95\n    min_parallel_tokens = 1\n    max_parallel_tokens = 5"
    assert tail.count(a) == 1
    tail = tail.replace(a, a + "\n    fw_stats = {'forward_work': 0, 'decode_steps': 0, 'max_beam': 0}  " + MARKER, 1)

    # count forward work right before gumbel (unique: batch_logits)
    a = "        logits_with_noise = add_gumbel_noise(batch_logits, temperature=temperature)"
    assert tail.count(a) == 1
    cnt = ("        fw_stats['forward_work'] += int(batch_sequences.shape[0]) * (2 if cfg_scale > 0. else 1)  " + MARKER + "\n"
           "        fw_stats['decode_steps'] += 1\n"
           "        fw_stats['max_beam'] = max(fw_stats['max_beam'], len(beam_sequences))\n")
    tail = tail.replace(a, cnt + a, 1)

    # replace the accept/commit rule
    old_rule = """            high_confidence_mask = block_mask_confidence > confidence_threshold
            high_confidence_indices = torch.where(high_confidence_mask)[0]
            
            if len(high_confidence_indices) >= min_parallel_tokens:
                num_to_unmask = min(len(high_confidence_indices), max_parallel_tokens)
                top_probs, top_indices = torch.topk(block_mask_confidence[high_confidence_indices], num_to_unmask)
                selected_indices = high_confidence_indices[top_indices]"""
    assert tail.count(old_rule) == 1, "accept/commit rule anchor not found"
    new_rule = """            """ + MARKER + """: accept/commit rule dispatch
            if frechet_delta is not None:
                n_cert = frechet_certified_count(block_mask_confidence, frechet_delta)
                do_parallel_commit = n_cert >= min_parallel_tokens
            else:
                high_confidence_mask = block_mask_confidence > confidence_threshold
                high_confidence_indices = torch.where(high_confidence_mask)[0]
                do_parallel_commit = len(high_confidence_indices) >= min_parallel_tokens

            if do_parallel_commit:
                if frechet_delta is not None:
                    num_to_unmask = n_cert
                    top_probs, top_indices = torch.topk(block_mask_confidence, num_to_unmask)
                    selected_indices = top_indices
                else:
                    num_to_unmask = min(len(high_confidence_indices), max_parallel_tokens)
                    top_probs, top_indices = torch.topk(block_mask_confidence[high_confidence_indices], num_to_unmask)
                    selected_indices = high_confidence_indices[top_indices]"""
    # fix comment syntax (MARKER line must be a comment; it already starts with #)
    tail = tail.replace(old_rule, new_rule, 1)

    # return stats (last occurrence in soar body; anchor within tail after split is
    # unique because tail starts at generate_soar and pbs is above the split)
    a = "    if beam:\n        best_sequence, best_score, _, best_records = beam[0]\n    else:\n        best_sequence = x\n        best_records = []\n    \n    return best_sequence"
    assert tail.count(a) == 1, f"return anchor count = {tail.count(a)}"
    tail = tail.replace(a, a.replace("return best_sequence", "return best_sequence, fw_stats  " + MARKER), 1)

    open(gp, "w").write(head + new_def + tail)
    print("generate.py patched")

# ---------------- eval_llada.soar.py ----------------
ep = "eval_llada.soar.py"
src = open(ep).read()
if MARKER in src:
    print("eval_llada.soar.py already patched")
else:
    a = "        enable_pbs=False,\n        enable_soar=False,"
    assert src.count(a) == 1
    src = src.replace(a, a + "\n        frechet_delta=None,  " + MARKER, 1)

    a = "        self.enable_pbs = enable_pbs\n        self.enable_soar = enable_soar"
    assert src.count(a) == 1
    src = src.replace(a, a + "\n        self.frechet_delta = None if frechet_delta is None else float(frechet_delta)  " + MARKER, 1)

    old_call = """            elif self.enable_soar:
                from generate import generate_soar
                generated_answer = generate_soar(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                    temperature=0, remasking=self.remasking, mask_id=self.mask_id)"""
    assert src.count(old_call) == 1
    new_call = """            elif self.enable_soar:
                from generate import generate_soar
                generated_answer, fw_stats = generate_soar(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                    temperature=0, remasking=self.remasking, mask_id=self.mask_id, frechet_delta=self.frechet_delta)  """ + MARKER + """
                num_nfe += fw_stats['forward_work']
                self._decode_steps = getattr(self, '_decode_steps', 0) + fw_stats['decode_steps']
                self._max_beam = max(getattr(self, '_max_beam', 0), fw_stats['max_beam'])"""
    src = src.replace(old_call, new_call, 1)

    a = """                    # if self.show_speed:
                    #     num_tokens += (generated_answer_ids != 126081).sum()
                    #     num_nfe += nfe"""
    assert src.count(a) == 1
    src = src.replace(a, """                    if self.show_speed:  """ + MARKER + """
                        num_tokens += (generated_answer_ids != 126081).sum()""", 1)

    a = '            print(f"Total NFE is {num_nfe}")'
    assert src.count(a) == 1
    src = src.replace(a, a + """
            _ds = getattr(self, '_decode_steps', 0)  """ + MARKER + """
            print(f"Total decode steps: {_ds}")
            print(f"Max beam width used: {getattr(self, '_max_beam', 0)}")
            print(f"Avg sequence-forwards per decode step: {num_nfe / _ds if _ds else -1}")""", 1)

    open(ep, "w").write(src)
    print("eval_llada.soar.py patched")

print("PATCH_OK")
