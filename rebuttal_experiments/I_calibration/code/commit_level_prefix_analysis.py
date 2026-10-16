#!/usr/bin/env python3
# PrefixCache commit-level (I16) + coupling (I17) analysis. CPU-only, reads trace.jsonl.
# Outputs summary CSVs. No GPU, no writes to run dirs.
import json, glob, os, sys
import numpy as np
I16 = os.path.expanduser("~/calib_runs_i16")
I17 = os.path.expanduser("~/calib_runs_i17")
OUT = os.path.expanduser("~/calib_prefix_analysis"); os.makedirs(OUT, exist_ok=True)

def rows_of(d):
    f=os.path.join(d,"trace.jsonl")
    if not os.path.exists(f): return []
    out=[]
    for l in open(f):
        try: out.append(json.loads(l))
        except: pass
    return out

def method_label(r):
    if r.get("decode_mode")=="frechet": return f"frechet_eta{r.get('frechet_eta',0.0):.2f}"
    fac=r.get("factor"); return f"factor_f{float(fac):.2f}" if fac is not None else "?"

def auc(scores,labels):
    s=np.asarray(scores,float); y=np.asarray(labels,int)
    if (y==1).sum()==0 or (y==0).sum()==0: return float("nan")
    u,inv,c=np.unique(s,return_inverse=True,return_counts=True); cs=np.cumsum(c); st=cs-c
    ar=((st+cs+1)/2.0)[inv]; rp=ar[y==1].sum(); U=rp-(y==1).sum()*((y==1).sum()+1)/2.0
    return U/((y==1).sum()*(y==0).sum())

def ece(conf,corr,nb=10):
    conf=np.asarray(conf,float); corr=np.asarray(corr,float); e=0.0; n=len(conf); ed=np.linspace(0,1,nb+1)
    for i in range(nb):
        m=(conf>=ed[i])&(conf<(ed[i+1] if i<nb-1 else ed[i+1]+1e-9))
        if m.sum(): e+=(m.sum()/n)*abs(corr[m].mean()-conf[m].mean())
    return e

def wcsv(path,rows,cols):
    with open(path,"w") as f:
        f.write(",".join(cols)+"\n")
        for r in rows: f.write(",".join(str(r.get(c,"")) for c in cols)+"\n")

# ---------- Exp B + Exp C (I16 verifier runs) ----------
expB=[]; tok={}  # tok[task]=(conf[],agree[]) for Exp A
for d in sorted(glob.glob(I16+"/calib_*")):
    rws=[r for r in rows_of(d) if r.get("commit_success") is not None]
    if not rws: continue
    task=rws[0]["task"]; meth=method_label(rws[0])
    Ln=np.array([r["L_n"] for r in rws]); Gn=np.array([r["G_n"] for r in rws])
    n=np.array([r["selected_count"] for r in rws]); su=np.array([r["commit_success"] for r in rws],int)
    mm=n>=2; viol=0; nb=0
    if mm.sum()>=20:
        lm=Ln[mm]; sm=su[mm]; ed=np.quantile(lm,np.linspace(0,1,6))
        for i in range(len(ed)-1):
            m=(lm>=ed[i])&(lm<=ed[i+1])
            if m.sum()>=5:
                nb+=1
                if sm[m].mean()<lm[m].mean()-1e-9: viol+=1
    expB.append(dict(task=task,method=meth,n_steps=len(rws),frac_n1=round(float((n==1).mean()),3),
        n_multi=int(mm.sum()),commit_err_multi=round(float(1-su[mm].mean()),4) if mm.sum() else "",
        commit_succ_multi=round(float(su[mm].mean()),4) if mm.sum() else "",
        mean_Ln=round(float(Ln.mean()),4),mean_Gn=round(float(Gn.mean()),4),
        auc_Gn_multi=round(auc(Gn[mm],su[mm]),4) if mm.sum()>5 else "",
        lb_violation=f"{viol}/{nb}"))
    if meth in ("factor_f1.00","frechet_eta0.00"):
        tok.setdefault(task,[[],[]])
        for r in rws:
            for c,a in zip(r.get("per_pos_parallel_conf") or [], r.get("per_pos_agree") or []):
                tok[task][0].append(c); tok[task][1].append(a)
wcsv(f"{OUT}/expB_prefix_commit_reliability.csv",expB,
     ["task","method","n_steps","frac_n1","n_multi","commit_succ_multi","commit_err_multi","mean_Ln","mean_Gn","auc_Gn_multi","lb_violation"])

# ---------- Exp A token calibration (prefix) ----------
expA=[]
for task,(cf,ag) in tok.items():
    cf=np.array(cf,float); ag=np.array(ag,int); hi=cf>=0.9
    expA.append(dict(task=task,n_tok=len(cf),agree=round(float(ag.mean()),4),mean_conf=round(float(cf.mean()),4),
        ECE=round(ece(cf,ag),4),overconf_gap=round(float((cf-ag).mean()),4),
        highconf_err=round(float(1-ag[hi].mean()),5) if hi.sum() else ""))
wcsv(f"{OUT}/expA_prefix_token_calibration.csv",expA,["task","n_tok","agree","mean_conf","ECE","overconf_gap","highconf_err"])

# ---------- Exp E S_extra (I16 frechet_eta0) ----------
expE=[]
for d in sorted(glob.glob(I16+"/calib_*frechet_d0p25_eta0p00")):
    rws=[r for r in rows_of(d) if r.get("commit_success") is not None]
    if not rws: continue
    task=rws[0]["task"]; exa=[]; maa=[]; exc=[]
    for r in rws:
        m=r.get("matched_factor_n"); nf=r.get("selected_count")
        pc=r.get("per_pos_parallel_conf") or []; ag=r.get("per_pos_agree") or []
        if not pc or m is None or nf is None: continue
        o=np.argsort(-np.array(pc)); ags=np.array(ag)[o]; pcs=np.array(pc)[o]; m=min(m,nf)
        maa+=ags[:m].tolist()
        if nf>m: exa+=ags[m:nf].tolist(); exc+=pcs[m:nf].tolist()
    expE.append(dict(task=task,matched_tok=len(maa),matched_agree=round(float(np.mean(maa)),4) if maa else "",
        extra_tok=len(exa),extra_agree=round(float(np.mean(exa)),4) if exa else "",
        extra_mean_conf=round(float(np.mean(exc)),4) if exc else ""))
wcsv(f"{OUT}/expE_prefix_sextra.csv",expE,["task","matched_tok","matched_agree","extra_tok","extra_agree","extra_mean_conf"])

# ---------- Exp D coupling (I17) ----------
expD=[]
for d in sorted(glob.glob(I17+"/couple_*")):
    rws=[r for r in rows_of(d) if r.get("coupling_jsd_mean") is not None]
    if not rws: continue
    task=rws[0]["task"]; meth=method_label(rws[0])
    flips=np.array([r["coupling_flip_frac"] for r in rws],float)
    jsds=np.array([r["coupling_jsd_mean"] for r in rws],float)
    ncoup=np.array([r.get("coupling_n",0) for r in rws])
    # also per-position flip/jsd pooled
    pf=[]; pj=[]
    for r in rws:
        pf+=r.get("per_pos_coupling_flip") or []; pj+=r.get("per_pos_coupling_jsd") or []
    expD.append(dict(task=task,method=meth,n_multi_steps=len(rws),mean_coupling_n=round(float(ncoup.mean()),2),
        step_flip_frac=round(float(flips.mean()),4),step_jsd_mean=round(float(jsds.mean()),4),
        pos_flip_frac=round(float(np.mean(pf)),4) if pf else "",pos_jsd_mean=round(float(np.mean(pj)),4) if pj else "",
        n_pos=len(pf)))
wcsv(f"{OUT}/expD_coupling.csv",expD,["task","method","n_multi_steps","mean_coupling_n","step_flip_frac","step_jsd_mean","pos_flip_frac","pos_jsd_mean","n_pos"])

for name,rows in [("expB",expB),("expA",expA),("expE",expE),("expD",expD)]:
    print(f"===== {name} ====="); 
    for r in rows: print(r)
print("OUT",OUT)
