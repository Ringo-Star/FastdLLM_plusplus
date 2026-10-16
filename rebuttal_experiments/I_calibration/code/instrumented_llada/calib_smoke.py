import sys, json, torch
sys.path.insert(0, "$ROOT/fast-dllm/v1/llada")
from transformers import AutoTokenizer, AutoConfig
from model.modeling_llada import LLaDAModelLM
from generate import generate

MP = "GSAI-ML/LLaDA-8B-Instruct"
cfg = AutoConfig.from_pretrained(MP)
cfg.flash_attention = True
model = LLaDAModelLM.from_pretrained(MP, trust_remote_code=True, torch_dtype=torch.bfloat16, config=cfg).to("cuda").eval()
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)

q = ("Natalia sold clips to 48 friends in April, then half as many in May. "
     "How many clips did she sell altogether? Answer step by step.")
ids = tok(q)["input_ids"]
input_ids = torch.tensor(ids, device="cuda").unsqueeze(0)

OUT = "/tmp/smoke_calib.jsonl"
open(OUT, "w").close()
out, nfe = generate(model, input_ids, steps=32, gen_length=64, block_length=32, temperature=0.,
                    remasking="low_confidence", mask_id=126336, frechet_margin=0.25,
                    calib_verify=True, trace_jsonl=OUT,
                    trace_meta={"run": "smoke", "task": "gsm8k", "delta": 0.25}, trace_topk=64)
print("NFE:", nfe)
print("DECODED:", repr(tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)[:200]))

recs = [json.loads(l) for l in open(OUT)]
print("STEPS_LOGGED:", len(recs))
ok = True
for i, r in enumerate(recs[:8]):
    keys = ("step_idx_within_block", "selected_count", "L_n", "G_n",
            "verify_n", "verify_ran", "commit_success", "commit_agree_frac",
            "per_pos_agree", "frechet_eta")
    print(i, {k: r.get(k) for k in keys})
    # sanity checks
    if r.get("verify_n") != r.get("selected_count"):
        print("  WARN verify_n != selected_count", r.get("verify_n"), r.get("selected_count"))
    pa = r.get("per_pos_agree") or []
    if r.get("verify_ran") is not None and len(pa) != r.get("verify_ran"):
        print("  WARN per_pos_agree len != verify_ran"); ok = False
    cs = r.get("commit_success")
    if cs is not None and pa:
        if cs != int(all(pa)):
            print("  WARN commit_success != all(per_pos_agree)"); ok = False
cs_all = [r["commit_success"] for r in recs if r.get("commit_success") is not None]
pa_all = [a for r in recs for a in (r.get("per_pos_agree") or [])]
print("COMMIT_SUCCESS_MEAN:", (sum(cs_all)/len(cs_all)) if cs_all else None, "n_steps=", len(cs_all))
print("PER_TOKEN_AGREE_MEAN:", (sum(pa_all)/len(pa_all)) if pa_all else None, "n_tokens=", len(pa_all))
print("SANITY_OK:", ok)
