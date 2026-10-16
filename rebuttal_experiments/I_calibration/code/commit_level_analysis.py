#!/usr/bin/env python3
"""Commit-level calibration analysis (ISSUE-20, no-cache LLaDA-8B, west battery).
Exp B: commit-level reliability (commit_success vs L_n / G_n / n / B_n; AUC; violation).
Exp A: token-level calibration of committed tokens (conf vs agreement w/ sequential verifier).
Exp E: S_extra -- safety of the tokens Fréchet commits BEYOND matched factor.
Pure numpy+matplotlib, no GPU. Reads synced trace.jsonl dirs.
"""
import json, glob, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.expanduser("~/work_code/records/fast-dllm/rebuttal/results/calib_west_nocache_20260724")
OUT = os.path.expanduser("~/work_code/records/fast-dllm/rebuttal/paper_figures/calibration_commit_level_v1")
os.makedirs(OUT, exist_ok=True)
# Tol bright, colorblind-friendly
TOL = {"blue":"#4477AA","cyan":"#66CCEE","green":"#228833","yellow":"#CCBB44","red":"#EE6677","purple":"#AA3377","grey":"#BBBBBB"}

def method_label(rows):
    r = rows[0]
    if r.get("decode_mode") == "frechet":
        return f"frechet_eta{r.get('frechet_eta',0.0):.2f}"
    f = r.get("factor")
    return f"factor_f{float(f):.2f}" if f is not None else "unknown"

def load():
    runs = {}
    for d in sorted(glob.glob(f"{SRC}/calib_*")):
        tf = os.path.join(d, "trace.jsonl")
        if not os.path.exists(tf): continue
        rows = []
        for l in open(tf):
            try: rows.append(json.loads(l))
            except: pass
        v = [r for r in rows if r.get("commit_success") is not None and r.get("verify_ran")]
        if not v: continue
        task = v[0].get("task")
        runs[os.path.basename(d)] = {"task": task, "method": method_label(v), "rows": v}
    return runs

def auc(scores, labels):
    # AUC of scores predicting label==1 (Mann-Whitney U). labels in {0,1}.
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos = s[y==1]; neg = s[y==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avgrank = (start + csum + 1) / 2.0
    ranks = avgrank[inv]
    r_pos = ranks[y==1].sum()
    U = r_pos - len(pos)*(len(pos)+1)/2.0
    return U/(len(pos)*len(neg))

def spearman(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    def rank(x):
        o=np.argsort(x); r=np.empty(len(x)); r[o]=np.arange(len(x)); 
        u,inv,c=np.unique(x,return_inverse=True,return_counts=True); cs=np.cumsum(c); st=cs-c
        return ((st+cs-1)/2.0)[inv]
    ra,rb=rank(a),rank(b)
    if ra.std()==0 or rb.std()==0: return float("nan")
    return float(np.corrcoef(ra,rb)[0,1])

def ece(conf, correct, nb=10):
    conf=np.asarray(conf,float); correct=np.asarray(correct,float)
    edges=np.linspace(0,1,nb+1); e=0.0; n=len(conf)
    for i in range(nb):
        m=(conf>=edges[i])&(conf<edges[i+1] if i<nb-1 else conf<=edges[i+1])
        if m.sum()==0: continue
        e += (m.sum()/n)*abs(correct[m].mean()-conf[m].mean())
    return e

runs = load()
print(f"Loaded {len(runs)} runs")

# ---------- Exp B: commit-level reliability ----------
expB = []
for name, R in runs.items():
    rows=R["rows"]; task=R["task"]; meth=R["method"]
    Ln=np.array([r["L_n"] for r in rows]); Gn=np.array([r["G_n"] for r in rows])
    n=np.array([r["selected_count"] for r in rows]); succ=np.array([r["commit_success"] for r in rows],int)
    Bn=np.array([r.get("heterogeneity_bonus",0.0) for r in rows])
    multi = n>=2
    row=dict(run=name, task=task, method=meth, n_steps=len(rows),
             frac_n1=float((n==1).mean()), n_multi=int(multi.sum()),
             commit_succ_all=float(succ.mean()),
             commit_succ_multi=float(succ[multi].mean()) if multi.sum() else float("nan"),
             mean_Ln=float(Ln.mean()), mean_Gn=float(Gn.mean()),
             auc_Gn_multi=auc(Gn[multi], succ[multi]) if multi.sum()>5 else float("nan"),
             auc_Ln_multi=auc(Ln[multi], succ[multi]) if multi.sum()>5 else float("nan"),
             spearman_Gn_multi=spearman(Gn[multi], succ[multi]) if multi.sum()>5 else float("nan"))
    # lower-bound violation: among multi-token commits, does empirical success >= L_n hold per L_n decile?
    viol=0; nb=0
    if multi.sum()>=20:
        lm=Ln[multi]; sm=succ[multi]; edges=np.quantile(lm,np.linspace(0,1,6))
        for i in range(len(edges)-1):
            m=(lm>=edges[i])&(lm<=edges[i+1])
            if m.sum()<5: continue
            nb+=1
            if sm[m].mean() < lm[m].mean()-1e-9: viol+=1
    row["lb_violation_bins"]=f"{viol}/{nb}"
    expB.append(row)

# ---------- Exp A: token-level calibration (committed tokens vs sequential verifier) ----------
expA=[]
tok_by_task={}  # task -> (conf[], agree[]) pooled over factor f=1.0 (widest commit) + frechet eta0
for name,R in runs.items():
    meth=R["method"]; task=R["task"]
    if meth not in ("factor_f1.00","frechet_eta0.00"): continue
    conf=[]; agree=[]
    for r in R["rows"]:
        pc=r.get("per_pos_parallel_conf") or []; ag=r.get("per_pos_agree") or []
        for c,a in zip(pc,ag):
            conf.append(c); agree.append(a)
    conf=np.array(conf,float); agree=np.array(agree,int)
    if len(conf)==0: continue
    hi=conf>=0.9
    expA.append(dict(task=task, method=meth, n_tok=len(conf),
        acc=float(agree.mean()), mean_conf=float(conf.mean()),
        ECE=ece(conf,agree), Brier=float(((conf-agree)**2).mean()),
        overconf_gap=float((conf-agree).mean()),
        highconf_err=float(1-agree[hi].mean()) if hi.sum() else float("nan"),
        highconf_frac=float(hi.mean())))
    tok_by_task.setdefault(task,[[],[]])
    if meth=="frechet_eta0.00":
        tok_by_task[task][0]+=conf.tolist(); tok_by_task[task][1]+=agree.tolist()

# ---------- Exp E: S_extra safety (frechet eta0 runs) ----------
expE=[]
for name,R in runs.items():
    if R["method"]!="frechet_eta0.00": continue
    task=R["task"]
    ex_ag=[]; ma_ag=[]; ex_conf=[]; n_extra_steps=0
    for r in R["rows"]:
        m=r.get("matched_factor_n"); nf=r.get("selected_count")
        pc=r.get("per_pos_parallel_conf") or []; ag=r.get("per_pos_agree") or []
        if not pc or m is None or nf is None: continue
        order=np.argsort(-np.array(pc))  # confidence descending
        ag_s=np.array(ag)[order]; pc_s=np.array(pc)[order]
        m=min(m,nf)
        ma_ag += ag_s[:m].tolist()
        if nf>m:
            n_extra_steps+=1
            ex_ag += ag_s[m:nf].tolist(); ex_conf += pc_s[m:nf].tolist()
    expE.append(dict(task=task, steps=len(R["rows"]), steps_with_extra=n_extra_steps,
        matched_tok=len(ma_ag), matched_agree=float(np.mean(ma_ag)) if ma_ag else float("nan"),
        extra_tok=len(ex_ag), extra_agree=float(np.mean(ex_ag)) if ex_ag else float("nan"),
        extra_mean_conf=float(np.mean(ex_conf)) if ex_conf else float("nan")))

# ---------- write CSVs ----------
def wcsv(path, rows, cols):
    with open(path,"w") as f:
        f.write(",".join(cols)+"\n")
        for r in rows: f.write(",".join(str(r.get(c,"")) for c in cols)+"\n")
wcsv(f"{OUT}/expB_commit_reliability.csv", expB,
     ["run","task","method","n_steps","frac_n1","n_multi","commit_succ_all","commit_succ_multi","mean_Ln","mean_Gn","auc_Gn_multi","auc_Ln_multi","spearman_Gn_multi","lb_violation_bins"])
wcsv(f"{OUT}/expA_token_calibration.csv", expA,
     ["task","method","n_tok","acc","mean_conf","ECE","Brier","overconf_gap","highconf_err","highconf_frac"])
wcsv(f"{OUT}/expE_sextra_safety.csv", expE,
     ["task","steps","steps_with_extra","matched_tok","matched_agree","extra_tok","extra_agree","extra_mean_conf"])

# ---------- figures ----------
# Fig A: token reliability diagram per task (frechet eta0)
plt.figure(figsize=(5,4))
cols={"gsm8k":TOL["blue"],"mbpp":TOL["red"],"humaneval":TOL["green"]}
for task,(conf,agree) in tok_by_task.items():
    conf=np.array(conf); agree=np.array(agree)
    edges=np.linspace(0,1,11); xs=[]; ys=[]
    for i in range(10):
        m=(conf>=edges[i])&(conf<=edges[i+1] if i==9 else conf<edges[i+1])
        if m.sum()<20: continue
        xs.append(conf[m].mean()); ys.append(agree[m].mean())
    plt.plot(xs,ys,marker="o",color=cols.get(task,TOL["grey"]),label=f"{task} (n={len(conf)})")
plt.plot([0,1],[0,1],color=TOL["grey"],linewidth=1,label="perfect calibration")
plt.xlabel("committed-token confidence"); plt.ylabel("empirical agreement w/ sequential verifier")
plt.title("Token calibration of committed tokens (raw Frechet, $\\delta$=0.25)")
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(f"{OUT}/figA_token_calibration.png",dpi=200,bbox_inches="tight")
plt.savefig(f"{OUT}/figA_token_calibration.pdf",dpi=300,bbox_inches="tight"); plt.close()

# Fig B: commit success vs L_n (multi-token) per task, frechet eta0
plt.figure(figsize=(5,4))
for name,R in runs.items():
    if R["method"]!="frechet_eta0.00": continue
    task=R["task"]; rows=R["rows"]
    n=np.array([r["selected_count"] for r in rows]); Ln=np.array([r["L_n"] for r in rows]); su=np.array([r["commit_success"] for r in rows],int)
    mm=n>=2
    if mm.sum()<20: continue
    lm=Ln[mm]; sm=su[mm]; edges=np.quantile(lm,np.linspace(0,1,7)); xs=[];ys=[]
    for i in range(len(edges)-1):
        m=(lm>=edges[i])&(lm<=edges[i+1])
        if m.sum()<5: continue
        xs.append(lm[m].mean()); ys.append(sm[m].mean())
    plt.plot(xs,ys,marker="o",color=cols.get(task,TOL["grey"]),label=f"{task} (n$\\geq$2: {int(mm.sum())})")
plt.plot([0,1],[0,1],color=TOL["grey"],linewidth=1,label="lower-bound line (y=x)")
plt.xlabel("Frechet certificate $L_n$"); plt.ylabel("empirical commit success")
plt.title("Commit-level conservativeness (multi-token commits)")
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(f"{OUT}/figB_commit_success_vs_Ln.png",dpi=200,bbox_inches="tight")
plt.savefig(f"{OUT}/figB_commit_success_vs_Ln.pdf",dpi=300,bbox_inches="tight"); plt.close()

# print summaries
print("\n=== Exp A token calibration ===")
for r in expA: print(r)
print("\n=== Exp B commit reliability ===")
for r in expB: print(r)
print("\n=== Exp E S_extra ===")
for r in expE: print(r)
print(f"\nFigures + CSVs -> {OUT}")
