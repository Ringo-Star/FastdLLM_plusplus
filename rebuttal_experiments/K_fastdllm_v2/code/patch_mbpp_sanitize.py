#!/usr/bin/env python3
"""Add a task-gated code sanitizer for MBPP to Fast-dLLM v2 eval.py.

Base `mbpp` scores the RAW completion with the test asserts, so the chat model's
markdown fences / prose / trailing text make valid code fail to compile -> pass@1=0.
This adds `_sanitize_code` (module level) and calls it in generate_until ONLY for
tasks whose name starts with "mbpp". gsm8k/MATH (flexible-extract/math_verify scan
anywhere) and humaneval_instruct (its own create_test filter) are left untouched.

Idempotent + assertion-guarded.
"""
import io, os

EV = os.path.expanduser("~/fastdllmpp_rebuttal/fast-dllm/v2/eval.py")
MARK = "_sanitize_code"


def _replace_once(text, old, new, label):
    c = text.count(old)
    if c != 1:
        raise SystemExit(f"[patch ABORT] anchor {label!r} found {c} times (expected 1)")
    return text.replace(old, new)


ev = io.open(EV, encoding="utf-8").read()
if MARK in ev:
    print("[skip] eval.py already has _sanitize_code")
    raise SystemExit(0)

# (1) module-level sanitizer, inserted just before the harness class registration
sanitizer = '''
def _sanitize_code(text):
    """Extract runnable Python from a chat model's completion for code tasks.
    Handles: fenced ```python...``` blocks, an opening fence with no close, a stray
    closing fence after bare code, and a trailing [DONE] marker. Falls back to the
    raw text when no fence/marker is present."""
    import re
    if not text:
        return text
    # drop an explicit [DONE] terminator and anything after it
    text = text.split("[DONE]")[0]
    # complete fenced block -> inner content
    m = re.search(r"```(?:python|py)?\\s*\\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip("\\n")
    # opening fence, no close -> everything after it, cut at any later stray fence
    m = re.search(r"```(?:python|py)?\\s*\\n?(.*)", text, re.DOTALL)
    if m:
        return m.group(1).split("```")[0].strip("\\n")
    # no fences -> cut at a stray closing fence if present
    return text.split("```")[0]


'''
ev = _replace_once(
    ev,
    '@register_model("fast_dllm_v2")\n',
    sanitizer + '@register_model("fast_dllm_v2")\n',
    "sanitizer-def",
)

# (2) apply it in generate_until, right after the answer is decoded, gated on mbpp
old_decode = (
    "                generated_answer = self.tokenizer.decode(\n"
    "                    generated_ids[batch_pos][seq_len[batch_pos]:], \n"
    "                    skip_special_tokens=True\n"
    "                )\n"
)
new_decode = old_decode + (
    "                if getattr(req, \"task_name\", \"\") and req.task_name.startswith(\"mbpp\"):\n"
    "                    generated_answer = _sanitize_code(generated_answer)\n"
)
ev = _replace_once(ev, old_decode, new_decode, "mbpp-sanitize-call")

io.open(EV, "w", encoding="utf-8").write(ev)
print("[ok] added _sanitize_code + mbpp-gated call to eval.py")
